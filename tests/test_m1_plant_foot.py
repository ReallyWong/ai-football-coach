"""M1 unit tests: adapter, plant_foot metric, rules engine (numpy only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.adapters.mediapipe_to_coco17 import MP_TO_COCO17, mediapipe_to_coco17
from pipeline.metrics.geometric import plant_foot_angle_deg, torso_lean_deg
from pipeline.pose.load_mediapipe_json import load_mediapipe_json
from pipeline.rules_engine import evaluate_rules, load_rules
from scripts.run_offline_demo import make_synthetic_mp33, run_pipeline


def test_mp_to_coco17_shape_and_map():
    t, c = 4, 3
    mp = np.zeros((t, 33, c))
    for coco_i, mp_i in enumerate(MP_TO_COCO17):
        mp[:, mp_i, 0] = coco_i + 100
        mp[:, mp_i, 1] = coco_i + 200
        mp[:, mp_i, 2] = 0.9
    coco = mediapipe_to_coco17(mp)
    assert coco.shape == (t, 17, 3)
    for coco_i in range(17):
        assert np.allclose(coco[:, coco_i, 0], coco_i + 100)
        assert np.allclose(coco[:, coco_i, 1], coco_i + 200)


def test_bad_plant_foot_triggers_rule():
    mp = make_synthetic_mp33(bad_plant_foot=True)
    coco = mediapipe_to_coco17(mp)
    angle = plant_foot_angle_deg(coco, plant_side="left")
    assert angle < 70 or angle > 110
    metrics, fired = run_pipeline(coco, label_id="pass.inside")
    ids = [r["id"] for r in fired]
    assert "pass.inside.plant_foot" in ids
    tip = next(r for r in fired if r["id"] == "pass.inside.plant_foot")
    assert "支撑脚" in tip["feedback_zh"]
    # feedback must come from rules.json, not a hardcoded engine copy
    rules = load_rules()
    expected = next(r for r in rules["rules"] if r["id"] == "pass.inside.plant_foot")
    assert tip["feedback_zh"] == expected["feedback_zh"]
    assert tip["feedback_en"] == expected["feedback_en"]


def test_good_plant_foot_does_not_trigger():
    mp = make_synthetic_mp33(bad_plant_foot=False)
    coco = mediapipe_to_coco17(mp)
    angle = plant_foot_angle_deg(coco, plant_side="left")
    assert 70 <= angle <= 110
    _, fired = run_pipeline(coco, label_id="pass.inside")
    assert "pass.inside.plant_foot" not in [r["id"] for r in fired]


def test_fixture_pose_json_triggers_plant_foot():
    path = REPO_ROOT / "fixtures" / "demo_pass_inside_pose.json"
    kps, meta = load_mediapipe_json(path)
    assert meta["topology"] == "mediapipe33"
    assert kps.shape[1] == 33
    coco = mediapipe_to_coco17(kps)
    _, fired = run_pipeline(coco, label_id="pass.inside")
    assert any(r["id"] == "pass.inside.plant_foot" for r in fired)


def test_demo_fixture_payload_matches_rules_tone():
    demo = json.loads((REPO_ROOT / "fixtures" / "demo_pass_inside.json").read_text())
    fired = demo["rules_fired"][0]
    assert fired["id"] == "pass.inside.plant_foot"
    assert "支撑脚" in fired["feedback_zh"]


def test_torso_lean_upright_near_zero():
    mp = make_synthetic_mp33(bad_plant_foot=False)
    coco = mediapipe_to_coco17(mp)
    lean = torso_lean_deg(coco)
    assert lean < 15  # synthetic torso is nearly vertical


def test_rules_engine_outside_op_direct():
    fired = evaluate_rules(
        "pass.inside",
        {"plant_foot_angle_deg": 45.0},
        track="geometric",
    )
    assert len(fired) == 1
    assert fired[0]["id"] == "pass.inside.plant_foot"

    quiet = evaluate_rules(
        "pass.inside",
        {"plant_foot_angle_deg": 90.0},
        track="geometric",
    )
    assert quiet == []
