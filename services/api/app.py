"""FastAPI analyze service (M2).

POST /analyze  — pose clip → ONNX probe + geometric rules
GET  /health   — liveness + model path
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import sys
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.adapters.mediapipe_to_coco17 import mediapipe_to_coco17  # noqa: E402
from pipeline.metrics.geometric import compute_geometric_metrics  # noqa: E402
from pipeline.model.skeleton_probe import (  # noqa: E402
    CLASS_IDS,
    COCO17_V,
    DEFAULT_T,
    NUM_CLASSES,
    load_taxonomy_labels,
    preprocess_tvx_to_nctv,
)
from pipeline.pose.load_mediapipe_json import load_mediapipe_json  # noqa: E402
from pipeline.rules_engine import evaluate_rules  # noqa: E402

DEFAULT_ONNX = REPO_ROOT / "models" / "youth_football_probe.onnx"

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    load_taxonomy_labels()
    try:
        _get_session()
    except FileNotFoundError:
        # Allow process to start; /analyze will 503 until model exists
        pass
    yield


app = FastAPI(
    title="AI Football Coach Analyze API",
    version="0.2.0",
    description="M2: ONNX skeleton probe + geometric rules dual-track",
    lifespan=_lifespan,
)

_session = None
_onnx_path: Path | None = None


def _get_session():
    global _session, _onnx_path
    if _session is not None:
        return _session
    import onnxruntime as ort

    path = Path(
        __import__("os").environ.get("AIFC_ONNX_PATH", str(DEFAULT_ONNX))
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"ONNX model not found at {path}. "
            "Run: python scripts/train_linear_probe.py && python scripts/export_onnx.py"
        )
    _session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    _onnx_path = path
    return _session


class AnalyzeRequest(BaseModel):
    """Pose clip for analysis.

    Provide one of:
      - keypoints: T×V×C nested list (V=17 COCO or V=33 MediaPipe)
      - pose_path: path to pose JSON under the repo (relative or absolute)
    """

    keypoints: list[list[list[float]]] | None = None
    pose_path: str | None = None
    topology: str | None = Field(
        default=None,
        description="coco17 | mediapipe33 — inferred from V if omitted",
    )
    label_id: str | None = Field(
        default=None,
        description="Optional hint; if omitted, model argmax is used for rules",
    )
    fps: float = 30.0
    T: int | None = None
    V: int | None = None


def _to_coco17(keypoints: np.ndarray, topology: str | None) -> np.ndarray:
    arr = np.asarray(keypoints, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"keypoints must be T×V×C, got {arr.shape}")
    topo = (topology or "").lower()
    if arr.shape[1] == COCO17_V or topo.startswith("coco"):
        return arr
    if arr.shape[1] == 33 or topo.startswith("mediapipe"):
        return mediapipe_to_coco17(arr)
    raise ValueError(f"Cannot map topology={topology!r} shape={arr.shape}")


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _scores_from_confidence(confidence: float) -> dict[str, int]:
    """Heuristic form/timing/balance placeholders until a multi-head model exists."""
    base = int(round(100 * float(np.clip(confidence, 0.0, 1.0))))
    return {
        "form": base,
        "timing": max(0, base - 4),
        "balance": max(0, base - 2),
        "overall": max(0, base - 2),
    }


def analyze_coco17(
    coco17: np.ndarray,
    *,
    label_id_hint: str | None = None,
    fps: float = 30.0,
) -> dict[str, Any]:
    """Run ONNX probe + geometric rules on a COCO-17 clip."""
    sess = _get_session()
    nctv = preprocess_tvx_to_nctv(coco17, t_out=DEFAULT_T, normalize=True)
    logits = sess.run(None, {"pose": nctv})[0]  # (1, 4)
    probs = _softmax(logits)[0]
    class_id = int(np.argmax(probs))
    predicted_label = CLASS_IDS[class_id]

    # Rules use hint if provided (coach-selected technique), else model label
    label_for_rules = label_id_hint or predicted_label
    if label_for_rules not in CLASS_IDS:
        raise ValueError(f"Unknown label_id {label_for_rules!r}; expected one of {CLASS_IDS}")
    rules_class_id = CLASS_IDS.index(label_for_rules)
    # Confidence matches the label we report (hint class if provided, else argmax)
    confidence = float(probs[rules_class_id])

    metrics_raw = compute_geometric_metrics(coco17)
    metrics = {
        k: (None if (isinstance(v, float) and v != v) else float(v))
        for k, v in metrics_raw.items()
    }
    # rules engine needs finite floats; drop NaNs
    metrics_for_rules = {
        k: v for k, v in metrics_raw.items() if isinstance(v, float) and v == v
    }
    signals = {"confidence": confidence}
    triggered = evaluate_rules(
        label_for_rules,
        metrics_for_rules,
        signals=signals,
    )

    topk = sorted(
        [
            {
                "label_id": CLASS_IDS[i],
                "class_id": i,
                "confidence": float(probs[i]),
            }
            for i in range(NUM_CLASSES)
        ],
        key=lambda x: -x["confidence"],
    )

    # Response shape compatible with fixtures/demo_pass_inside.json
    # plus metrics + triggered_rules aliases for the M2 acceptance wording.
    rules_fired = [
        {
            "id": r["id"],
            "from": r["from"],
            "severity": r.get("severity"),
            "feedback_zh": r.get("feedback_zh"),
            "feedback_en": r.get("feedback_en"),
            **(
                {"observed": r["observed"]}
                if r.get("observed") is not None
                else {}
            ),
        }
        for r in triggered
    ]

    return {
        "label_id": label_for_rules,
        "class_id": rules_class_id,
        "confidence": confidence,
        "model_label_id": predicted_label,
        "model_class_id": class_id,
        "scores": _scores_from_confidence(confidence),
        "topk": topk,
        "metrics": metrics,
        "triggered_rules": rules_fired,
        "rules_fired": rules_fired,  # alias for demo fixture / coach-ui
        "pose": {
            "topology": "coco17",
            "T": int(coco17.shape[0]),
            "V": COCO17_V,
            "fps": fps,
            "model_T": DEFAULT_T,
        },
    }


@app.get("/health")
def health() -> dict[str, Any]:
    path = Path(
        __import__("os").environ.get("AIFC_ONNX_PATH", str(DEFAULT_ONNX))
    )
    ok = path.is_file()
    return {
        "status": "ok" if ok else "degraded",
        "onnx_path": str(path),
        "onnx_present": ok,
        "num_classes": NUM_CLASSES,
        "class_ids": list(CLASS_IDS),
    }


@app.post("/analyze")
def analyze(body: AnalyzeRequest) -> dict[str, Any]:
    try:
        if body.keypoints is not None:
            kps = np.asarray(body.keypoints, dtype=np.float64)
            topo = body.topology
        elif body.pose_path:
            path = Path(body.pose_path)
            if not path.is_file():
                alt = REPO_ROOT / body.pose_path
                path = alt if alt.is_file() else path
            if not path.is_file():
                raise HTTPException(status_code=400, detail=f"pose_path not found: {body.pose_path}")
            kps, meta = load_mediapipe_json(path)
            topo = body.topology or meta.get("topology")
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide keypoints (T×V×C) or pose_path",
            )

        coco = _to_coco17(kps, topo)
        return analyze_coco17(coco, label_id_hint=body.label_id, fps=body.fps)
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
