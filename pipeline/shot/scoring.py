"""Shooting-only score: 0–100 and one coaching sentence.

The label is always ``shot.instep``. Passing, dribbling, and first touch are
out of scope on this path. The sentence is chosen by a deterministic policy
so a later AWS agent can be audited against the same observation:

- quality gate fails → re-record sentence, no score
- one or more geometric faults → the single largest penalty's rule text
- otherwise → one reinforcement sentence

Rule text is copied from ``rules.json``. The re-record and reinforcement
lines are product copy for outcomes that are not geometric faults.

TODO(PR-B): POST the observation to AGENT_API_URL. Bedrock may rewrite
``coaching_zh`` only. It must not choose ``next_action``.
TODO(PR-B): write trace JSON to s3://$SHOT_BUCKET/traces/ and set trace_s3_key.
TODO(PR-C): coach-ui shows score + coaching_zh + next_action, nothing else.
TODO(PR-D): fold this contract into docs/opencv26_report.md.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from pipeline.metrics.geometric import (
    compute_geometric_metrics,
    plant_foot_angle_series,
    torso_lean_series,
)
from pipeline.pose.quality_gates import merge_quality, pose_quality
from pipeline.rules_engine import evaluate_rules

SHOT_LABEL_ID = "shot.instep"

LEAN_LIMIT_DEG = 25.0
PLANT_RANGE = (70.0, 110.0)
LEAN_PENALTY_PER_DEG = 1.6
LEAN_PENALTY_CAP = 45.0
PLANT_PENALTY_PER_DEG = 0.8
PLANT_PENALTY_CAP = 35.0
# Geometry stays in charge. The four-class probe is only a small prior.
CONFIDENCE_WEIGHT = 0.2

DRILL_BY_RULE = {
    "shot.instep.lean": "drill.lean_wall",
    "shot.instep.plant_foot": "drill.plant_target",
}
DEFAULT_DRILL = "drill.shot_form"

RERECORD_ZH = "画面不够清楚或人太小，请换成侧面、把全身拍进画面后再录一次射门。"
RERECORD_EN = (
    "The picture is too unclear or the player is too small — "
    "record the shot again from the side with the full body in frame."
)
POSITIVE_ZH = "支撑和上身都稳，保持微微前倾，用脚背把球压住再打。"
POSITIVE_EN = (
    "Plant and torso are stable — keep a slight forward lean and "
    "strike through the ball with the instep."
)

_SEVERITY_RANK = {"critical": 3, "warn": 2, "tip": 1}


def geometry_penalties(metrics: dict[str, float]) -> dict[str, float]:
    """Point deductions for torso lean and plant-foot angle. Empty if clean."""
    penalties: dict[str, float] = {}
    lean = metrics.get("torso_lean_deg")
    if lean is not None and lean == lean and float(lean) > LEAN_LIMIT_DEG:
        over = float(lean) - LEAN_LIMIT_DEG
        penalties["torso_lean_deg"] = float(min(LEAN_PENALTY_CAP, over * LEAN_PENALTY_PER_DEG))
    plant = metrics.get("plant_foot_angle_deg")
    if plant is not None and plant == plant:
        lo, hi = PLANT_RANGE
        if float(plant) < lo:
            dist = lo - float(plant)
        elif float(plant) > hi:
            dist = float(plant) - hi
        else:
            dist = 0.0
        if dist > 0.0:
            penalties["plant_foot_angle_deg"] = float(min(PLANT_PENALTY_CAP, dist * PLANT_PENALTY_PER_DEG))
    return penalties


def decide_next_action(
    *,
    quality_pass: bool,
    selected_rule_id: str | None,
) -> tuple[str, str | None]:
    """Map a visual observation to one agent action. Pure and side-effect free.

    TODO(PR-B): call this from the agent Lambda. Do not reimplement it in a prompt.
    """
    if not quality_pass:
        return "request_re_record", None
    if selected_rule_id:
        return "assign_drill", DRILL_BY_RULE.get(selected_rule_id, DEFAULT_DRILL)
    return "accept_and_coach", None


def _json_metric(value: float) -> float | None:
    if isinstance(value, float) and value != value:
        return None
    return float(value)


def _select_rule(
    fired: list[dict[str, Any]],
    penalties: dict[str, float],
) -> dict[str, Any] | None:
    geometric = [rule for rule in fired if rule.get("from") == "geometric"]
    if not geometric:
        return None

    def rank(rule: dict[str, Any]) -> tuple[float, int]:
        metric = str(rule.get("metric") or "")
        penalty = float(penalties.get(metric, 0.0))
        severity = _SEVERITY_RANK.get(str(rule.get("severity") or ""), 0)
        return (penalty, severity)

    return max(geometric, key=rank)


def _reason_codes(quality: dict[str, Any], penalties: dict[str, float]) -> list[str]:
    if not quality.get("pass", False):
        reasons = list(quality.get("fail_reasons") or [])
        return reasons or ["quality_fail"]
    codes = ["quality_ok"]
    if penalties.get("torso_lean_deg", 0.0) > 0.0:
        codes.append("torso_lean_high")
    if penalties.get("plant_foot_angle_deg", 0.0) > 0.0:
        codes.append("plant_foot_off")
    return codes


def score_shot_clip(
    coco17: np.ndarray,
    *,
    fps: float = 30.0,
    confidence: float | None = None,
    image_quality: dict[str, Any] | None = None,
    frame_wh: tuple[int, int] | None = None,
    opencv_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one instep shot. ``label_id`` is always ``shot.instep``."""
    arr = np.asarray(coco17, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != 17:
        raise ValueError(f"Expected (T, 17, C) COCO-17, got {arr.shape}")

    pose_q = pose_quality(arr, frame_wh=frame_wh)
    quality = merge_quality(image_quality, pose_q)
    metrics_raw = compute_geometric_metrics(arr)
    metrics = {key: _json_metric(value) for key, value in metrics_raw.items()}
    finite_metrics = {
        key: value
        for key, value in metrics_raw.items()
        if isinstance(value, float) and value == value
    }
    if quality.get("pass") and not finite_metrics:
        quality = dict(quality)
        reasons = list(quality.get("fail_reasons") or [])
        reasons.append("pose_unreadable")
        quality["fail_reasons"] = reasons
        quality["pass"] = False

    penalties = geometry_penalties(finite_metrics) if quality.get("pass") else {}
    # Confidence is a score prior only. It is not a second coaching sentence,
    # and it does not relabel the clip as pass / dribble / first touch.
    fired = evaluate_rules(
        SHOT_LABEL_ID,
        finite_metrics,
        metric_series_map={
            "plant_foot_angle_deg": plant_foot_angle_series(arr),
            "torso_lean_deg": torso_lean_series(arr),
        },
        fps=fps,
        track="geometric",
    )
    selected = _select_rule(fired, penalties) if quality.get("pass") else None
    next_action, drill_id = decide_next_action(
        quality_pass=bool(quality.get("pass")),
        selected_rule_id=selected.get("id") if selected else None,
    )

    score: int | None
    if next_action == "request_re_record":
        score = None
        coaching_zh, coaching_en = RERECORD_ZH, RERECORD_EN
        evidence = None
        penalty_out: dict[str, float] = {}
    else:
        geo = 100.0 - float(sum(penalties.values()))
        if confidence is not None and confidence == confidence:
            conf_score = 100.0 * float(np.clip(confidence, 0.0, 1.0))
            blended = (1.0 - CONFIDENCE_WEIGHT) * geo + CONFIDENCE_WEIGHT * conf_score
        else:
            blended = geo
        score = int(np.clip(int(round(blended)), 0, 100))
        penalty_out = {key: round(value, 4) for key, value in penalties.items()}
        if selected is not None:
            coaching_zh = str(selected.get("feedback_zh") or POSITIVE_ZH)
            coaching_en = str(selected.get("feedback_en") or POSITIVE_EN)
            evidence = selected.get("evidence_frame")
        else:
            coaching_zh, coaching_en = POSITIVE_ZH, POSITIVE_EN
            evidence = None

    opencv = {
        "version": None,
        "pose_backend": "keypoints",
        "frames_used": int(arr.shape[0]),
        "compliant": None,
        "dnn_engine": None,
        "pinned": "5.0.0.93",
    }
    if opencv_meta:
        opencv.update(opencv_meta)
        opencv["frames_used"] = int(opencv_meta.get("frames_used", arr.shape[0]))

    return {
        "label_id": SHOT_LABEL_ID,
        "score": score,
        "coaching_zh": coaching_zh,
        "coaching_en": coaching_en,
        "agent": {
            "next_action": next_action,
            "drill_id": drill_id,
            "reason_codes": _reason_codes(quality, penalty_out),
            "policy": "deterministic_v1",
            # TODO(PR-B): fill after the agent Lambda writes the S3 trace.
            "trace_s3_key": None,
        },
        "opencv": opencv,
        "quality": quality,
        "metrics": metrics,
        "penalties": penalty_out,
        "triggered_rules": fired if next_action != "request_re_record" else [],
        "evidence": evidence,
        "model": {
            "label_id": SHOT_LABEL_ID,
            "forced": True,
            "shot_confidence": (
                None if confidence is None or confidence != confidence else float(confidence)
            ),
            "note": (
                "Four-class probe is a confidence prior only. "
                "This path never returns pass, dribble, or first touch."
            ),
        },
    }
