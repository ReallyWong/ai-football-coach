"""Shooting-only score, quality gates, and OpenCV DNN pose scaffolding.

The Render deploy image has no OpenCV and no torch. Tests that need cv2
import it inside the test and skip when this environment has not installed
requirements-comp.txt. Scoring itself stays numpy-only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.pose.opencv_dnn_pose import (  # noqa: E402
    PoseModelMissingError,
    decode_movenet_output,
    decode_yolo_pose_output,
    preferred_engine_name,
)
from pipeline.pose.quality_gates import merge_quality, pose_quality  # noqa: E402
from pipeline.rules_engine import load_rules  # noqa: E402
from pipeline.shot.scoring import (  # noqa: E402
    POSITIVE_ZH,
    RERECORD_ZH,
    decide_next_action,
    score_shot_clip,
)
from pipeline.shot.synthetic import make_shot_coco17  # noqa: E402

FIXTURE = REPO_ROOT / "fixtures" / "demo_shot_instep_pose.json"


def _one_zh(text: str) -> None:
    assert text.count("。") == 1
    assert text.endswith("。")


def _rule_zh(rule_id: str) -> str:
    rules = load_rules()
    return next(rule["feedback_zh"] for rule in rules["rules"] if rule["id"] == rule_id)


def test_upright_shot_scores_100_and_one_reinforcement():
    clip = make_shot_coco17(lean="upright", plant="good")
    result = score_shot_clip(clip)
    assert result["label_id"] == "shot.instep"
    assert result["score"] == 100
    assert result["coaching_zh"] == POSITIVE_ZH
    _one_zh(result["coaching_zh"])
    assert result["agent"]["next_action"] == "accept_and_coach"
    assert result["agent"]["drill_id"] is None
    assert result["triggered_rules"] == []
    assert result["model"]["forced"] is True


def test_back_lean_assigns_one_drill_sentence():
    clip = make_shot_coco17(lean="back", plant="good")
    result = score_shot_clip(clip)
    assert result["label_id"] == "shot.instep"
    assert 0 <= result["score"] < 100
    assert result["coaching_zh"] == _rule_zh("shot.instep.lean")
    assert "支撑脚" not in result["coaching_zh"]
    _one_zh(result["coaching_zh"])
    assert result["agent"]["next_action"] == "assign_drill"
    assert result["agent"]["drill_id"] == "drill.lean_wall"
    assert "torso_lean_high" in result["agent"]["reason_codes"]
    assert result["evidence"]["metric"] == "torso_lean_deg"
    assert result["penalties"]["torso_lean_deg"] > 0


def test_plant_foot_rule_beats_a_smaller_lean_penalty():
    plant = score_shot_clip(make_shot_coco17(lean="upright", plant="off"))
    both = score_shot_clip(make_shot_coco17(lean="back", plant="off"))
    lean = score_shot_clip(make_shot_coco17(lean="back", plant="good"))
    assert plant["coaching_zh"] == _rule_zh("shot.instep.plant_foot")
    assert plant["agent"]["drill_id"] == "drill.plant_target"
    _one_zh(plant["coaching_zh"])
    # Both faults still produce one sentence: the larger plant-foot penalty.
    assert both["coaching_zh"] == plant["coaching_zh"]
    assert "上身后仰" not in both["coaching_zh"]
    _one_zh(both["coaching_zh"])
    assert both["score"] < plant["score"] < lean["score"] < 100
    assert len(both["triggered_rules"]) == 2


def test_low_pose_confidence_rerecords_without_a_score():
    clip = make_shot_coco17(conf=0.05)
    result = score_shot_clip(clip)
    assert result["score"] is None
    assert result["coaching_zh"] == RERECORD_ZH
    _one_zh(result["coaching_zh"])
    assert result["agent"]["next_action"] == "request_re_record"
    assert result["agent"]["drill_id"] is None
    assert "pose_low_conf" in result["agent"]["reason_codes"]
    assert result["triggered_rules"] == []


def test_blur_overrides_an_otherwise_good_skeleton():
    clip = make_shot_coco17(lean="back", plant="good")
    result = score_shot_clip(
        clip,
        image_quality={"blur_var": 4.0, "brightness": 128.0, "frames": clip.shape[0]},
    )
    assert result["score"] is None
    assert result["coaching_zh"] == RERECORD_ZH
    assert result["agent"]["next_action"] == "request_re_record"
    assert "blur" in result["quality"]["fail_reasons"]
    assert "上身后仰" not in result["coaching_zh"]


def test_too_few_frames_and_tiny_person():
    short = score_shot_clip(make_shot_coco17(t=4))
    assert short["agent"]["next_action"] == "request_re_record"
    assert "too_few_frames" in short["quality"]["fail_reasons"]
    far = pose_quality(make_shot_coco17(), frame_wh=(1920, 1080))
    assert far["person_frac"] < 0.05
    merged = merge_quality(None, far)
    assert merged["pass"] is False
    assert "person_small" in merged["fail_reasons"]


def test_decide_next_action_matches_the_agent_policy():
    assert decide_next_action(quality_pass=False, selected_rule_id="shot.instep.lean") == (
        "request_re_record",
        None,
    )
    assert decide_next_action(quality_pass=True, selected_rule_id="shot.instep.lean") == (
        "assign_drill",
        "drill.lean_wall",
    )
    assert decide_next_action(quality_pass=True, selected_rule_id=None) == (
        "accept_and_coach",
        None,
    )


def test_confidence_prior_does_not_change_the_sentence_or_label():
    clip = make_shot_coco17(lean="upright", plant="good")
    low = score_shot_clip(clip, confidence=0.1)
    high = score_shot_clip(clip, confidence=0.9)
    assert low["label_id"] == high["label_id"] == "shot.instep"
    assert low["coaching_zh"] == high["coaching_zh"] == POSITIVE_ZH
    assert low["agent"]["next_action"] == "accept_and_coach"
    assert 0 <= low["score"] < high["score"] <= 100


def test_decode_movenet_yx_and_xy():
    raw = np.zeros((1, 1, 17, 3), dtype=np.float32)
    raw[0, 0, :, 0] = 0.25
    raw[0, 0, :, 1] = 0.50
    raw[0, 0, :, 2] = 0.80
    yx = decode_movenet_output(raw, 200, 100, coord_order="yx")
    assert yx.shape == (17, 3)
    assert np.allclose(yx[:, 0], 100.0)
    assert np.allclose(yx[:, 1], 25.0)
    xy = decode_movenet_output(raw, 200, 100, coord_order="xy")
    assert np.allclose(xy[:, 0], 50.0)
    assert np.allclose(xy[:, 1], 50.0)


def test_decode_yolo_pose_picks_the_higher_score():
    raw = np.zeros((1, 56, 2), dtype=np.float32)
    raw[0, 4, 0] = 0.1
    raw[0, 4, 1] = 0.95
    # Joint 0 of person 1: x=320, y=160, conf=0.7 in a 640 input, stretch to 320x240.
    raw[0, 5, 1] = 320
    raw[0, 6, 1] = 160
    raw[0, 7, 1] = 0.7
    kps = decode_yolo_pose_output(raw, 320, 240, 640, {"mode": "stretch"})
    assert kps.shape == (17, 3)
    assert kps[0, 0] == pytest.approx(160.0)
    assert kps[0, 1] == pytest.approx(60.0)
    assert kps[0, 2] == pytest.approx(0.7)
    assert np.all(kps[1:, 2] == 0.0)


def test_requirements_comp_pins_opencv5_and_deploy_does_not():
    comp = (REPO_ROOT / "requirements-comp.txt").read_text(encoding="utf-8")
    assert "opencv-python-headless==5.0.0.93" in comp
    deploy_lines = []
    for line in (REPO_ROOT / "requirements-deploy.txt").read_text(encoding="utf-8").splitlines():
        deploy_lines.append(line.split("#", 1)[0].strip().lower())
    deploy = "\n".join(deploy_lines)
    assert "opencv" not in deploy
    assert "torch" not in deploy
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "requirements-deploy.txt" in dockerfile
    assert "requirements-comp.txt" not in dockerfile
    render = (REPO_ROOT / "render.yaml").read_text(encoding="utf-8")
    assert "dockerfilePath: ./Dockerfile" in render
    assert "branch: main" in render


def test_analyze_shot_api_and_existing_paths(tmp_path: Path):
    from fastapi.testclient import TestClient
    from services.api.app import app

    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["onnx_present"] is True
    assert body["shot"]["label_id"] == "shot.instep"
    assert body["shot"]["pose_model_present"] is False

    modes = client.get("/modes").json()["modes"]
    assert modes[0]["id"] == "fundamentals"
    assert any(mode["id"] == "shot" for mode in modes)

    passed = client.post(
        "/analyze",
        json={"pose_path": "fixtures/demo_pass_inside_pose.json", "label_id": "pass.inside"},
    )
    assert passed.status_code == 200, passed.text
    assert passed.json()["label_id"] == "pass.inside"
    assert any(rule["id"] == "pass.inside.plant_foot" for rule in passed.json()["triggered_rules"])

    upright = make_shot_coco17(lean="upright", plant="good")
    good = client.post(
        "/analyze/shot",
        json={"keypoints": upright.tolist(), "topology": "coco17", "fps": 30},
    )
    assert good.status_code == 200, good.text
    good_body = good.json()
    assert good_body["label_id"] == "shot.instep"
    assert good_body["coaching_zh"] == POSITIVE_ZH
    _one_zh(good_body["coaching_zh"])
    assert good_body["agent"]["next_action"] == "accept_and_coach"
    assert 0 <= good_body["score"] <= 100

    leaned = make_shot_coco17(lean="back", plant="good")
    bad = client.post("/analyze/shot", json={"keypoints": leaned.tolist(), "topology": "coco17"})
    assert bad.status_code == 200, bad.text
    bad_body = bad.json()
    assert bad_body["coaching_zh"] == _rule_zh("shot.instep.lean")
    _one_zh(bad_body["coaching_zh"])
    assert bad_body["agent"]["drill_id"] == "drill.lean_wall"
    assert "pass.inside" not in json.dumps(bad_body["label_id"])

    missing = client.post("/analyze/shot", json={})
    assert missing.status_code == 400
    s3 = client.post("/analyze/shot", json={"s3_key": "uploads/demo.mp4"})
    assert s3.status_code == 501
    assert "PR-B" in s3.json()["detail"]

    assert FIXTURE.is_file()
    from_file = client.post("/analyze/shot", json={"pose_path": "fixtures/demo_shot_instep_pose.json"})
    assert from_file.status_code == 200, from_file.text
    assert from_file.json()["coaching_zh"] == _rule_zh("shot.instep.lean")


def _opencv_installed() -> bool:
    try:
        import cv2  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _opencv_installed(), reason="opencv-python-headless is not installed")
def test_opencv5_quality_overlay_and_injected_video_pose(tmp_path: Path):
    import cv2

    from pipeline.pose.opencv_dnn_pose import (
        analyze_bgr_frames,
        opencv_runtime_info,
        render_skeleton_bgr,
        video_to_coco17_opencv,
    )
    from pipeline.pose.quality_gates import quality_gates

    info = opencv_runtime_info()
    assert info["compliant"] is True
    assert info["version"].startswith("5.")
    assert preferred_engine_name(cv2) == "ENGINE_NEW"

    rng = np.random.default_rng(0)
    sharp = [rng.integers(0, 255, size=(360, 320, 3), dtype=np.uint8) for _ in range(8)]
    sharp_q = merge_quality(quality_gates(sharp), None)
    assert sharp_q["pass"] is True
    assert sharp_q["blur_var"] >= 30.0

    blurry = [np.full((360, 320, 3), 18, dtype=np.uint8) for _ in range(8)]
    blur_q = merge_quality(quality_gates(blurry), None)
    assert blur_q["pass"] is False
    assert "blur" in blur_q["fail_reasons"]
    assert "brightness" in blur_q["fail_reasons"]

    def _forward(_frame: np.ndarray) -> np.ndarray:
        return make_shot_coco17(lean="back", plant="good", t=1)[0]

    coco, meta = analyze_bgr_frames(sharp, forward_fn=_forward)
    assert coco.shape == (8, 17, 3)
    assert meta["pose_backend"] == "injected"
    assert meta["opencv_compliant"] is True
    assert meta["quality"]["pass"] is True
    scored = score_shot_clip(
        coco,
        image_quality=meta["quality"],
        frame_wh=tuple(meta["frame_wh"]),
        opencv_meta={
            "version": meta["opencv_version"],
            "pose_backend": meta["pose_backend"],
            "compliant": meta["opencv_compliant"],
            "frames_used": meta["frames"],
        },
    )
    assert scored["coaching_zh"] == _rule_zh("shot.instep.lean")
    assert scored["opencv"]["compliant"] is True

    blank = np.zeros((360, 320, 3), dtype=np.uint8)
    drawn = render_skeleton_bgr(blank, coco[0])
    assert drawn.shape == blank.shape
    assert not np.array_equal(drawn, blank)

    with pytest.raises(PoseModelMissingError):
        analyze_bgr_frames(sharp, model_path=tmp_path / "missing.onnx")

    path = _write_clip(tmp_path / "shot.avi", sharp)
    with pytest.raises(PoseModelMissingError):
        video_to_coco17_opencv(path, allow_mediapipe_fallback=False)
    with pytest.raises(ImportError, match="[Mm]ediapipe|[Mm]ediaPipe"):
        video_to_coco17_opencv(path, allow_mediapipe_fallback=True)


@pytest.mark.skipif(not _opencv_installed(), reason="opencv-python-headless is not installed")
def test_dnn_engine_new_runs_constant_movenet_graph():
    """Proves readNetFromONNX(ENGINE_NEW) and the MoveNet decoder, not a real detector."""
    from pipeline.pose.opencv_dnn_pose import analyze_bgr_frames, infer_frame, load_pose_net

    model = REPO_ROOT / "tests" / "fixtures" / "constant_movenet.onnx"
    net, spec, engine = load_pose_net(model)
    assert engine == "ENGINE_NEW"
    assert spec.family == "movenet"
    assert spec.input_size == 8
    frame = np.zeros((40, 100, 3), dtype=np.uint8)
    kps = infer_frame(net, spec, frame)
    assert kps.shape == (17, 3)
    assert np.allclose(kps[:, 0], 60.0)
    assert np.allclose(kps[:, 1], 16.0)
    assert np.allclose(kps[:, 2], 0.9)

    frames = [np.zeros((40, 100, 3), dtype=np.uint8) for _ in range(8)]
    coco, meta = analyze_bgr_frames(frames, model_path=model)
    assert coco.shape == (8, 17, 3)
    assert meta["pose_backend"] == "dnn_movenet"
    assert meta["dnn_engine"] == "ENGINE_NEW"
    assert meta["opencv_compliant"] is True
    assert np.allclose(coco[0, :, 0], 60.0)


def _write_clip(path: Path, frames: list[np.ndarray]) -> Path:
    import cv2

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (width, height))
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path
