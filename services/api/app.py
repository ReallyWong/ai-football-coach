"""FastAPI analyze service (M2).

POST /analyze  — pose clip → ONNX probe + geometric rules
GET  /health   — liveness + model path
"""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
import threading
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.adapters.mediapipe_to_coco17 import mediapipe_to_coco17  # noqa: E402
from pipeline.metrics.geometric import (  # noqa: E402
    compute_geometric_metrics,
    plant_foot_angle_series,
    torso_lean_series,
)
from pipeline.model.skeleton_probe import (  # noqa: E402
    CLASS_IDS,
    COCO17_V,
    DEFAULT_T,
    NUM_CLASSES,
    load_taxonomy_labels,
    preprocess_tvx_to_nctv,
)
from pipeline.pose.load_mediapipe_json import load_mediapipe_json  # noqa: E402
from pipeline.pose.video_to_mp33 import video_to_mp33  # noqa: E402
from pipeline.rules_engine import evaluate_rules  # noqa: E402
from pipeline.pose.shot_clip_limits import (  # noqa: E402
    MAX_SHOT_BYTES,
    ShotClipLimitError,
    release_shot_memory,
    shot_too_large_detail,
)
from pipeline.shot.scoring import SHOT_LABEL_ID, score_shot_clip  # noqa: E402
from services.api.shot_s3 import (  # noqa: E402
    ShotStorageError,
    download_shot_object,
    shot_bucket,
    shot_s3_status,
)

# One decoded shot clip at a time. A second caller is rejected instead of
# stacking another frame buffer on the 512Mi instance.
_shot_video_lock = threading.Lock()
SHOT_VIDEO_BUSY_DETAIL = "busy: shot analysis already running"
_UPLOAD_CHUNK_BYTES = 1024 * 1024


@contextmanager
def _shot_video_slot() -> Iterator[None]:
    """Hold the single shot-video slot, then return freed native memory to the OS."""
    if not _shot_video_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail=SHOT_VIDEO_BUSY_DETAIL)
    try:
        yield
    finally:
        try:
            release_shot_memory()
        finally:
            _shot_video_lock.release()


def _write_upload_capped(upload: UploadFile, dest: Path) -> None:
    """Copy an upload to disk, stopping once it passes the shot size cap."""
    declared = getattr(upload, "size", None)
    if declared is not None and int(declared) > MAX_SHOT_BYTES:
        raise HTTPException(status_code=413, detail=shot_too_large_detail())
    total = 0
    with dest.open("wb") as out:
        while True:
            chunk = upload.file.read(_UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_SHOT_BYTES:
                raise HTTPException(status_code=413, detail=shot_too_large_detail())
            out.write(chunk)
    if total <= 0:
        raise HTTPException(status_code=400, detail="Empty video upload.")

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
    version="0.4.0",
    description=(
        "Youth technique coaching API / 青少年技术纠错 API\n\n"
        "EN: pose or video → skill label + geometric correction tips + evidence frame.\n"
        "中文：姿态或视频 → 技术动作类别 + 几何纠错建议 + 证据帧。\n\n"
        "Shooting slice / 射门: POST /analyze/shot returns score 0–100 and one correction.\n\n"
        "Friendly docs: `/docs` · Classic Swagger: `/swagger` · OpenAPI: `/openapi.json`"
    ),
    lifespan=_lifespan,
    docs_url=None,
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    # "*" covers the local coach demo (127.0.0.1:8765 / localhost:8765)
    # and hosted deploys (Render + GitHub Pages).
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/coach-ui/")


_STATIC = Path(__file__).resolve().parent / "static"
if _STATIC.is_dir():
    app.mount("/docs-static", StaticFiles(directory=str(_STATIC)), name="docs-static")


@app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
def bilingual_docs() -> HTMLResponse:
    """Human-friendly bilingual API docs (not raw Swagger)."""
    page = _STATIC / "api_docs.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="api_docs.html missing")
    return HTMLResponse(page.read_text(encoding="utf-8"))


@app.get("/swagger", include_in_schema=False)
def classic_swagger():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="AI Football Coach · Swagger")


# Coach-ui and fixtures mounts are registered after API routes (see bottom of file).

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
        description="coco17 | mediapipe33 (EN/中文: topology; inferred from V if omitted)",
    )
    label_id: str | None = Field(
        default=None,
        description="Optional skill hint / 可选技术动作提示; else model argmax",
    )
    fps: float = 30.0
    T: int | None = None
    V: int | None = None


def _load_keypoints_array(
    *,
    keypoints: list | np.ndarray | None,
    pose_path: str | None,
    topology: str | None,
) -> tuple[np.ndarray, str | None]:
    """Resolve a pose body into an array plus topology. Shared by /analyze and /analyze/shot."""
    if keypoints is not None:
        return np.asarray(keypoints, dtype=np.float64), topology
    if pose_path:
        path = Path(pose_path)
        if not path.is_file():
            alt = REPO_ROOT / pose_path
            path = alt if alt.is_file() else path
        if not path.is_file():
            raise HTTPException(status_code=400, detail=f"pose_path not found: {pose_path}")
        kps, meta = load_mediapipe_json(path)
        return kps, topology or meta.get("topology")
    raise HTTPException(
        status_code=400,
        detail="Provide keypoints (T×V×C) or pose_path",
    )


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
    series_map = {
        "plant_foot_angle_deg": plant_foot_angle_series(coco17),
        "torso_lean_deg": torso_lean_series(coco17),
    }
    triggered = evaluate_rules(
        label_for_rules,
        metrics_for_rules,
        signals=signals,
        metric_series_map=series_map,
        fps=fps,
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
    rules_fired = []
    for r in triggered:
        item = {
            "id": r["id"],
            "from": r["from"],
            "severity": r.get("severity"),
            "feedback_zh": r.get("feedback_zh"),
            "feedback_en": r.get("feedback_en"),
        }
        if r.get("observed") is not None:
            item["observed"] = r["observed"]
        if r.get("metric"):
            item["metric"] = r["metric"]
        if r.get("evidence_frame"):
            item["evidence_frame"] = r["evidence_frame"]
        rules_fired.append(item)

    # Top-level evidence: first geometric rule with evidence_frame (coach-ui jump)
    evidence = None
    for r in rules_fired:
        if r.get("from") == "geometric" and r.get("evidence_frame"):
            evidence = dict(r["evidence_frame"])
            break

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
        "evidence": evidence,
        "pose": {
            "topology": "coco17",
            "T": int(coco17.shape[0]),
            "V": COCO17_V,
            "fps": fps,
            "model_T": DEFAULT_T,
        },
        # For coach-ui stick figure (finite coords; NaN → 0 for JSON)
        "keypoints": np.nan_to_num(coco17, nan=0.0).tolist(),
    }


@app.get("/health", summary="Health / 健康检查", tags=["core"])
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
        "shot": _shot_health(),
    }


def _shot_health() -> dict[str, Any]:
    """Pose-file presence only. Do not import cv2 here — deploy has no OpenCV."""
    from pipeline.pose.opencv_dnn_pose import find_pose_model

    model = find_pose_model()
    return {
        "label_id": SHOT_LABEL_ID,
        "routes": ["/analyze/shot", "/analyze/shot/video"],
        "pose_model_present": model is not None,
        "pose_model_path": str(model) if model else None,
        "s3": shot_s3_status(),
    }


def _shot_confidence(coco: np.ndarray) -> float | None:
    """Probability of shot.instep. Missing ONNX does not block geometric scoring."""
    try:
        sess = _get_session()
    except FileNotFoundError:
        return None
    nctv = preprocess_tvx_to_nctv(coco, t_out=DEFAULT_T, normalize=True)
    logits = sess.run(None, {"pose": nctv})[0]
    probs = _softmax(logits)[0]
    return float(probs[CLASS_IDS.index(SHOT_LABEL_ID)])


@app.post("/analyze", summary="Analyze pose JSON / 分析姿态 JSON", tags=["core"])
def analyze(body: AnalyzeRequest) -> dict[str, Any]:
    try:
        kps, topo = _load_keypoints_array(
            keypoints=body.keypoints,
            pose_path=body.pose_path,
            topology=body.topology,
        )
        coco = _to_coco17(kps, topo)
        return analyze_coco17(coco, label_id_hint=body.label_id, fps=body.fps)
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post(
    "/analyze/video",
    summary="Analyze uploaded video / 上传视频分析",
    tags=["core"],
)
async def analyze_video(
    file: UploadFile = File(..., description="Short training clip (mp4/webm/mov)"),
    label_id: str | None = Form(default=None),
    max_frames: int = Form(default=120),
) -> dict[str, Any]:
    """Upload a video → MediaPipe pose → same analyze payload as /analyze."""
    suffix = Path(file.filename or "clip.mp4").suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".webm", ".mov", ".avi", ".mkv"}:
        raise HTTPException(status_code=400, detail=f"Unsupported video type: {suffix}")
    tmp_dir = Path(tempfile.mkdtemp(prefix="aifc_vid_"))
    dest = tmp_dir / f"upload{suffix}"
    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        try:
            mp33, fps = video_to_mp33(dest, max_frames=max(16, min(int(max_frames), 240)))
        except ImportError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        coco = _to_coco17(mp33, "mediapipe33")
        result = analyze_coco17(coco, label_id_hint=label_id or None, fps=fps)
        result["source"] = {"type": "video", "filename": file.filename, "fps": fps}
        return result
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


class ShotAnalyzeRequest(BaseModel):
    """Shooting-only body. The skill is always shot.instep.

    Provide keypoints, pose_path, or ``s3_key`` (an ``uploads/…`` object).
    When ``s3_key`` is set it is fetched from ``SHOT_BUCKET`` and scored as a
    video; keypoints in the same body are ignored. There is no label_id field:
    this path cannot return a pass, dribble, or first-touch result.
    """

    keypoints: list[list[list[float]]] | None = None
    pose_path: str | None = None
    topology: str | None = Field(
        default=None,
        description="coco17 | mediapipe33. Inferred from V when omitted.",
    )
    fps: float = 30.0
    s3_key: str | None = Field(
        default=None,
        description="S3 object key under uploads/. Requires SHOT_BUCKET and AWS credentials.",
    )
    max_seconds: float = Field(default=10.0, gt=0, le=30)


def _score_shot_response(
    coco: np.ndarray,
    *,
    fps: float,
    image_quality: dict[str, Any] | None = None,
    frame_wh: tuple[int, int] | None = None,
    opencv_meta: dict[str, Any] | None = None,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = score_shot_clip(
        coco,
        fps=fps,
        confidence=_shot_confidence(coco),
        image_quality=image_quality,
        frame_wh=frame_wh,
        opencv_meta=opencv_meta,
    )
    if source is not None:
        result["source"] = source
    return result


def _score_shot_file(
    dest: Path,
    *,
    filename: str | None,
    max_seconds: float,
    source: dict[str, Any],
) -> dict[str, Any]:
    """OpenCV decode + DNN pose, then the shooting score. Shared by upload and S3."""
    from pipeline.pose.opencv_dnn_pose import PoseModelMissingError, video_to_coco17_opencv

    if max_seconds <= 0 or max_seconds > 30:
        raise HTTPException(status_code=400, detail="max_seconds must be in (0, 30]")
    coco = None
    try:
        try:
            coco, fps, meta = video_to_coco17_opencv(
                dest,
                max_frames=120,
                max_seconds=float(max_seconds),
            )
        except ShotClipLimitError as exc:
            raise HTTPException(status_code=413, detail=exc.detail) from exc
        except PoseModelMissingError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ImportError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        wh = meta.get("frame_wh") or [0, 0]
        frame_wh = (int(wh[0]), int(wh[1])) if wh and wh[0] and wh[1] else None
        opencv_meta = {
            "version": meta.get("opencv_version"),
            "pose_backend": meta.get("pose_backend"),
            "frames_used": int(meta.get("frames") or coco.shape[0]),
            "compliant": meta.get("opencv_compliant"),
            "dnn_engine": meta.get("dnn_engine"),
            "pinned": "5.0.0.93",
        }
        source = dict(source)
        source["fps"] = fps
        return _score_shot_response(
            coco,
            fps=fps,
            image_quality=meta.get("quality"),
            frame_wh=frame_wh,
            opencv_meta=opencv_meta,
            source=source,
        )
    finally:
        # Keypoints are copied into the JSON result. Drop the array before
        # the slot returns memory to the OS.
        del coco


def _analyze_shot_from_s3(s3_key: str, *, max_seconds: float) -> dict[str, Any]:
    """Download uploads/<id>.<ext> from SHOT_BUCKET, then score it as a shot video."""
    suffix = Path(s3_key.strip()).suffix.lower() or ".mp4"
    tmp_dir = Path(tempfile.mkdtemp(prefix="aifc_shot_s3_"))
    dest = tmp_dir / f"object{suffix}"
    try:
        try:
            key = download_shot_object(s3_key, dest)
        except ShotStorageError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        return _score_shot_file(
            dest,
            filename=Path(key).name,
            max_seconds=max_seconds,
            source={"type": "s3", "s3_key": key, "bucket": shot_bucket()},
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post(
    "/analyze/shot",
    summary="Score an instep shot / 射门评分（仅射门）",
    tags=["shot"],
)
def analyze_shot(body: ShotAnalyzeRequest) -> dict[str, Any]:
    """Return 0–100 and one coaching sentence for shot.instep.

    Keypoints work on the Render deploy image. ``s3_key`` downloads the object
    with boto3 and then uses the OpenCV video path. That path needs
    requirements-comp.txt, which the live deploy image does not install.
    """
    s3_key = (body.s3_key or "").strip()
    if s3_key:
        with _shot_video_slot():
            return _analyze_shot_from_s3(s3_key, max_seconds=body.max_seconds)
    try:
        kps, topo = _load_keypoints_array(
            keypoints=body.keypoints,
            pose_path=body.pose_path,
            topology=body.topology,
        )
        coco = _to_coco17(kps, topo)
        return _score_shot_response(coco, fps=body.fps)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post(
    "/analyze/shot/video",
    summary="Score a shot video / 上传射门视频评分",
    tags=["shot"],
)
async def analyze_shot_video(
    file: UploadFile = File(..., description="Side-view instep shot, up to 10 seconds"),
    max_seconds: float = Form(default=10.0),
) -> dict[str, Any]:
    """OpenCV 5 decode + DNN pose, then the same shooting score as /analyze/shot.

    MediaPipe is not the primary backend. It runs only when the process has
    AIFC_ALLOW_MP_FALLBACK=1 and the ONNX pose file is missing.

    One analysis runs at a time. A concurrent caller gets 429. Clips over
    20 MB, 12 seconds, or 720p get 413. Pose runs frame by frame so the
    decoded clip is not kept in RAM.
    """
    suffix = Path(file.filename or "clip.mp4").suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".webm", ".mov", ".avi", ".mkv"}:
        raise HTTPException(status_code=400, detail=f"Unsupported video type: {suffix}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="aifc_shot_"))
    dest = tmp_dir / f"upload{suffix}"
    try:
        _write_upload_capped(file, dest)
        # Score off the event loop so a second request can observe the lock
        # and return 429 while this one is still inside OpenCV.
        with _shot_video_slot():
            return await asyncio.to_thread(
                _score_shot_file,
                dest,
                filename=file.filename,
                max_seconds=float(max_seconds),
                source={"type": "video", "filename": file.filename},
            )
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _load_signature_catalog() -> dict[str, Any]:
    path = REPO_ROOT / "labels" / "signature_moves.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_signature_rules() -> dict[str, Any]:
    path = REPO_ROOT / "rules" / "signature_rules.json"
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/modes", summary="List product modes / 产品模式", tags=["product"])
def list_modes() -> dict[str, Any]:
    return {
        "modes": [
            {
                "id": "fundamentals",
                "name_zh": "基础技术纠错",
                "name_en": "Youth fundamentals",
                "status": "live",
                "api": ["/analyze", "/analyze/video"],
            },
            {
                "id": "shot",
                "name_zh": "射门纠错",
                "name_en": "Instep shot",
                "status": "competition",
                "api": ["/analyze/shot", "/analyze/shot/video"],
                "scope_zh": "仅射门。分数 0–100，一句纠错。",
                "scope_en": "Shooting only. Score 0–100 and one correction sentence.",
            },
            {
                "id": "signature_move",
                "name_zh": "球星招牌动作",
                "name_en": "Signature Move",
                "status": "preseed",
                "api": ["/signature-moves"],
                "hook_zh": "从马赛回旋、牛尾巴、踩单车开始练——小孩本来就在学球星。",
                "hook_en": "Train Marseille turn, elastico, step-overs — kids copy stars first.",
            },
        ]
    }


@app.get(
    "/signature-moves",
    summary="Signature Move catalog / 招牌动作目录",
    tags=["product"],
)
def signature_moves() -> dict[str, Any]:
    catalog = _load_signature_catalog()
    rules = _load_signature_rules()
    by_label: dict[str, list] = {}
    for r in rules.get("rules", []):
        by_label.setdefault(r.get("label_id") or "", []).append(r)
    moves = []
    for m in catalog.get("moves", []):
        item = dict(m)
        item["rules"] = by_label.get(m["id"], []) + by_label.get("sig.*", [])
        moves.append(item)
    return {
        "version": catalog.get("version"),
        "status": catalog.get("status"),
        "hook_zh": catalog.get("product_hook_zh"),
        "hook_en": catalog.get("product_hook_en"),
        "moves": moves,
    }


# Serve coach UI + fixtures (Docker / Render). Mounts last so API routes take precedence.
_FIXTURES = REPO_ROOT / "fixtures"
_COACH_UI = REPO_ROOT / "coach-ui"
if _FIXTURES.is_dir():
    app.mount("/fixtures", StaticFiles(directory=str(_FIXTURES)), name="fixtures")
if _COACH_UI.is_dir():
    app.mount("/coach-ui", StaticFiles(directory=str(_COACH_UI), html=True), name="coach-ui")
