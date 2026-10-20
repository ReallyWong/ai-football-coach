"""M3: evidence_frame on geometric rules + synthetic dataset manifest."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.adapters.mediapipe_to_coco17 import mediapipe_to_coco17
from pipeline.metrics.geometric import (
    compute_geometric_metrics,
    plant_foot_angle_series,
    torso_lean_series,
)
from pipeline.rules_engine import evaluate_rules
from scripts.run_offline_demo import make_synthetic_mp33


def test_evidence_frame_picks_peak_violation():
    mp = make_synthetic_mp33(bad_plant_foot=True, t=12)
    coco = mediapipe_to_coco17(mp)
    coco[7, 15, 0] = coco[7, 13, 0] + 80
    coco[7, 15, 1] = coco[7, 13, 1] + 20
    series = plant_foot_angle_series(coco)
    metrics = compute_geometric_metrics(coco)
    fired = evaluate_rules(
        "pass.inside",
        metrics,
        metric_series_map={
            "plant_foot_angle_deg": series,
            "torso_lean_deg": torso_lean_series(coco),
        },
        fps=30.0,
    )
    tip = next(r for r in fired if r["id"] == "pass.inside.plant_foot")
    assert "evidence_frame" in tip
    assert tip["evidence_frame"]["frame_index"] == 7
    assert tip["evidence_frame"]["metric"] == "plant_foot_angle_deg"
    assert tip["evidence_frame"]["rule_id"] == "pass.inside.plant_foot"


def test_analyze_api_returns_evidence():
    from fastapi.testclient import TestClient
    from services.api.app import app

    onnx = REPO_ROOT / "models" / "youth_football_probe.onnx"
    if not onnx.is_file():
        pytest.skip("ONNX missing")

    mp = make_synthetic_mp33(bad_plant_foot=True, t=10)
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
    assert data.get("evidence") is not None
    assert "frame_index" in data["evidence"]
    rules = data["rules_fired"]
    plant = next(r for r in rules if r["id"] == "pass.inside.plant_foot")
    assert "evidence_frame" in plant


def test_manifest_has_80_per_class():
    manifest = REPO_ROOT / "datasets" / "youth_football_synth" / "manifest.jsonl"
    assert manifest.is_file()
    counts: dict[str, int] = {}
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            counts[row["label_id"]] = counts.get(row["label_id"], 0) + 1
    assert len(counts) == 4
    for lid, n in counts.items():
        assert n >= 80, f"{lid} has {n}"


def test_demo_analyze_fixture_has_evidence():
    path = REPO_ROOT / "fixtures" / "demo_analyze_response.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["evidence"]["frame_index"] >= 0
    assert data["rules_fired"][0]["evidence_frame"]["frame_index"] == data["evidence"]["frame_index"]
    assert "keypoints" in data
    assert data["demo"]["scenario_id"] == "pass.inside"
    assert "支撑脚" in data["rules_fired"][0]["feedback_zh"]
    assert "纠正后" in data["demo"]["after_zh"]


def test_demo_shot_fixture_matches_torso_rule():
    path = REPO_ROOT / "fixtures" / "demo_shot_instep_response.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["label_id"] == "shot.instep"
    assert data["demo"]["scenario_id"] == "shot.instep"
    kps = np.asarray(data["keypoints"], dtype=np.float64)
    assert kps.ndim == 3 and kps.shape[1] == 17
    assert kps.shape[0] == data["pose"]["T"]
    metrics = compute_geometric_metrics(kps)
    fired = evaluate_rules(
        "shot.instep",
        metrics,
        signals={"confidence": float(data["confidence"])},
        metric_series_map={
            "torso_lean_deg": torso_lean_series(kps),
            "plant_foot_angle_deg": plant_foot_angle_series(kps),
        },
        fps=float(data["pose"]["fps"]),
    )
    tip = next(r for r in fired if r["id"] == "shot.instep.lean")
    assert tip["evidence_frame"]["frame_index"] == data["evidence"]["frame_index"]
    assert data["rules_fired"][0]["id"] == "shot.instep.lean"
    assert "上身" in data["rules_fired"][0]["feedback_zh"]
    assert data["pose"]["keypoints"]


def test_signature_catalog_fixture_is_preseed():
    path = REPO_ROOT / "fixtures" / "signature_catalog.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["status"] == "preseed"
    assert data["moves"][0]["id"] == "sig.marseille_turn"
    assert data["moves"][0]["name_zh"] == "马赛回旋"
    assert data["moves"][0]["rules"], "catalog joins cue rules for the product cards"
    assert all(m.get("status") == "preseed" for m in data["moves"])


def test_coach_ui_demo_mode_is_static():
    html = (REPO_ROOT / "coach-ui" / "index.html").read_text(encoding="utf-8")
    assert "演示模式" in html
    assert "正在分析" in html
    assert "demo_shot_instep_response.json" in html
    assert "signature_catalog.json" in html
    assert "http://127.0.0.1:8000/signature-moves" not in html
    assert "/analyze/video" in html
