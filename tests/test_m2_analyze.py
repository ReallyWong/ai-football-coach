"""M2 tests: ONNX probe inference + FastAPI /analyze."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.adapters.mediapipe_to_coco17 import mediapipe_to_coco17
from pipeline.model.skeleton_probe import (
    CLASS_IDS,
    COCO17_V,
    DEFAULT_T,
    NUM_CLASSES,
    load_taxonomy_labels,
    preprocess_tvx_to_nctv,
)
from pipeline.pose.load_mediapipe_json import load_mediapipe_json
from scripts.run_offline_demo import make_synthetic_mp33
from scripts.train_linear_probe import (
    SYNTH_FNS,
    generate_dataset,
)

ONNX_PATH = REPO_ROOT / "models" / "youth_football_probe.onnx"


@pytest.fixture(scope="module")
def ensure_onnx():
    if not ONNX_PATH.is_file():
        from scripts.export_onnx import export_onnx
        from scripts.train_linear_probe import train
        import torch

        models = REPO_ROOT / "models"
        models.mkdir(exist_ok=True)
        model, meta = train(n_per_class=40, epochs=15, t=DEFAULT_T, seed=0)
        pt = models / "youth_football_probe.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "meta": meta,
                "model_kwargs": {"num_classes": NUM_CLASSES, "t": DEFAULT_T, "v": COCO17_V},
            },
            pt,
        )
        export_onnx(pt, ONNX_PATH, t=DEFAULT_T)
    assert ONNX_PATH.is_file()
    return ONNX_PATH


def test_taxonomy_aligned():
    labels = load_taxonomy_labels()
    assert [l["id"] for l in labels] == list(CLASS_IDS)
    assert len(CLASS_IDS) == NUM_CLASSES == 4


def test_preprocess_shape():
    clip = np.random.randn(20, 17, 3).astype(np.float64)
    # hips/shoulders need some spread for scale
    clip[:, 11, :] = [95, 160, 1]
    clip[:, 12, :] = [125, 160, 1]
    clip[:, 5, :] = [90, 80, 1]
    clip[:, 6, :] = [130, 80, 1]
    nctv = preprocess_tvx_to_nctv(clip, t_out=DEFAULT_T)
    assert nctv.shape == (1, 3, DEFAULT_T, 17)
    assert nctv.dtype == np.float32


def test_onnx_infer_label_in_taxonomy(ensure_onnx):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(ensure_onnx), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(1)
    clip = SYNTH_FNS[0](DEFAULT_T, rng)  # pass.inside
    nctv = preprocess_tvx_to_nctv(clip, t_out=DEFAULT_T)
    logits = sess.run(None, {"pose": nctv})[0]
    assert logits.shape == (1, NUM_CLASSES)
    pred = int(np.argmax(logits[0]))
    assert 0 <= pred < NUM_CLASSES
    assert CLASS_IDS[pred] in CLASS_IDS


def test_onnx_synthetic_classes_mostly_correct(ensure_onnx):
    """Probe should get most synthetic class signatures right (not 100% required)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(ensure_onnx), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(7)
    correct = 0
    total = 0
    for class_id, fn in enumerate(SYNTH_FNS):
        for _ in range(5):
            clip = fn(DEFAULT_T, rng)
            nctv = preprocess_tvx_to_nctv(clip, t_out=DEFAULT_T)
            logits = sess.run(None, {"pose": nctv})[0]
            if int(np.argmax(logits[0])) == class_id:
                correct += 1
            total += 1
    assert correct / total >= 0.6, f"only {correct}/{total} correct on synth"


def test_analyze_api_fixture_plant_foot(ensure_onnx):
    from fastapi.testclient import TestClient
    from services.api.app import app

    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["onnx_present"] is True

    pose_path = "fixtures/demo_pass_inside_pose.json"
    resp = client.post(
        "/analyze",
        json={"pose_path": pose_path, "label_id": "pass.inside"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["label_id"] == "pass.inside"
    assert "confidence" in data
    assert "metrics" in data
    assert "plant_foot_angle_deg" in data["metrics"]
    rules = data.get("triggered_rules") or data.get("rules_fired")
    assert any(r["id"] == "pass.inside.plant_foot" for r in rules)
    tip = next(r for r in rules if r["id"] == "pass.inside.plant_foot")
    assert "支撑脚" in tip["feedback_zh"]
    # Compatible with demo fixture keys
    assert "scores" in data
    assert "topk" in data
    demo = json.loads((REPO_ROOT / "fixtures" / "demo_pass_inside.json").read_text())
    for key in ("label_id", "confidence", "scores", "topk", "rules_fired", "pose"):
        assert key in data or (key == "rules_fired" and "triggered_rules" in data)


def test_analyze_api_keypoints_body(ensure_onnx):
    from fastapi.testclient import TestClient
    from services.api.app import app

    mp = make_synthetic_mp33(bad_plant_foot=True, t=16)
    coco = mediapipe_to_coco17(mp)
    client = TestClient(app)
    resp = client.post(
        "/analyze",
        json={
            "keypoints": coco.tolist(),
            "topology": "coco17",
            "label_id": "pass.inside",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["class_id"] in range(NUM_CLASSES)
    assert any(
        r["id"] == "pass.inside.plant_foot"
        for r in (data.get("triggered_rules") or data["rules_fired"])
    )


def test_generate_dataset_shapes():
    x, y = generate_dataset(n_per_class=2, t=32, seed=0)
    assert x.shape[0] >= 8 and x.shape[1:] == (3, 32, 17)
    assert set(y.tolist()) == {0, 1, 2, 3}
