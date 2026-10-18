"""Load rules/rules.json and evaluate geometric (and model-signal) rules.

Feedback strings come from rules.json — never duplicated here.

M3: when a geometric rule fires and a per-frame metric series is supplied,
attach ``evidence_frame`` pointing at the frame of peak violation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULES_PATH = REPO_ROOT / "rules" / "rules.json"


def load_rules(path: Path | str | None = None) -> dict[str, Any]:
    rules_path = Path(path) if path else DEFAULT_RULES_PATH
    with rules_path.open(encoding="utf-8") as f:
        return json.load(f)


def _metric_matches(rule: dict[str, Any], metrics: dict[str, float]) -> bool:
    metric_name = rule.get("metric")
    if not metric_name or metric_name not in metrics:
        return False
    value = metrics[metric_name]
    if value is None or (isinstance(value, float) and value != value):  # NaN
        return False

    op = rule.get("op")
    if op == "outside":
        lo, hi = rule["range"]
        return value < lo or value > hi
    if op == "inside":
        lo, hi = rule["range"]
        return lo <= value <= hi
    if op == "gt":
        return value > rule["value"]
    if op == "gte":
        return value >= rule["value"]
    if op == "lt":
        return value < rule["value"]
    if op == "lte":
        return value <= rule["value"]
    if op == "eq":
        return value == rule["value"]
    raise ValueError(f"Unknown geometric op: {op!r} on rule {rule.get('id')}")


def _signal_matches(rule: dict[str, Any], signals: dict[str, float]) -> bool:
    signal_name = rule.get("signal")
    if not signal_name or signal_name not in signals:
        return False
    value = signals[signal_name]
    op = rule.get("op")
    if op == "lt":
        return value < rule["value"]
    if op == "gt":
        return value > rule["value"]
    if op == "lte":
        return value <= rule["value"]
    if op == "gte":
        return value >= rule["value"]
    raise ValueError(f"Unknown model op: {op!r} on rule {rule.get('id')}")


def _label_applies(rule_label: str, label_id: str) -> bool:
    return rule_label == "*" or rule_label == label_id


def _frame_deviation(rule: dict[str, Any], value: float) -> float:
    """How strongly ``value`` violates the geometric rule (0 = on boundary / ok)."""
    if value != value:  # NaN
        return float("-inf")
    op = rule.get("op")
    if op == "outside":
        lo, hi = rule["range"]
        mid = 0.5 * (lo + hi)
        # Prefer frames farthest from the safe band midpoint when outside
        if value < lo:
            return lo - value
        if value > hi:
            return value - hi
        return float("-inf")  # inside safe band — not a violation frame
    if op == "inside":
        lo, hi = rule["range"]
        if lo <= value <= hi:
            # Inside trigger: peak toward band center
            mid = 0.5 * (lo + hi)
            return -abs(value - mid)
        return float("-inf")
    if op in ("gt", "gte"):
        thr = float(rule["value"])
        return value - thr if value > thr or (op == "gte" and value >= thr) else float("-inf")
    if op in ("lt", "lte"):
        thr = float(rule["value"])
        return thr - value if value < thr or (op == "lte" and value <= thr) else float("-inf")
    if op == "eq":
        return -abs(value - float(rule["value"]))
    return float("-inf")


def _evidence_frame_for_rule(
    rule: dict[str, Any],
    series: np.ndarray | list[float] | None,
    *,
    fps: float = 30.0,
) -> dict[str, Any] | None:
    """Pick the frame index with max violation for a fired geometric rule."""
    if series is None:
        return None
    arr = np.asarray(series, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return None
    best_i = -1
    best_dev = float("-inf")
    best_val = float("nan")
    for i, v in enumerate(arr):
        if not np.isfinite(v):
            continue
        dev = _frame_deviation(rule, float(v))
        if dev > best_dev:
            best_dev = dev
            best_i = i
            best_val = float(v)
    if best_i < 0 or best_dev == float("-inf"):
        # Fallback: extreme value by absolute magnitude relative to metric median
        finite = np.isfinite(arr)
        if not finite.any():
            return None
        # For outside/gt prefer max abs distance from mid or threshold
        idx = int(np.nanargmax(np.abs(arr - np.nanmedian(arr))))
        best_i = idx
        best_val = float(arr[idx])
        best_dev = 0.0
    return {
        "frame_index": int(best_i),
        "timestamp_s": round(float(best_i) / float(fps) if fps else 0.0, 4),
        "metric": rule.get("metric"),
        "value": round(best_val, 4) if best_val == best_val else None,
        "deviation": round(float(best_dev), 4) if best_dev != float("-inf") else None,
        "rule_id": rule.get("id"),
    }


def evaluate_rules(
    label_id: str,
    metrics: dict[str, float] | None = None,
    *,
    signals: dict[str, float] | None = None,
    rules_doc: dict[str, Any] | None = None,
    rules_path: Path | str | None = None,
    track: str | None = None,
    metric_series_map: dict[str, Any] | None = None,
    fps: float = 30.0,
) -> list[dict[str, Any]]:
    """Return triggered rules with feedback copied from the rules document.

    Parameters
    ----------
    label_id :
        Technique id (e.g. ``pass.inside``).
    metrics :
        Geometric metric name → value (e.g. ``plant_foot_angle_deg``).
    signals :
        Model-track signals (e.g. ``confidence``). Optional for M1.
    track :
        If ``"geometric"`` or ``"model"``, only that track; else both.
    metric_series_map :
        Optional metric name → per-frame series (T,) for evidence_frame (M3).
    fps :
        Frames per second for evidence timestamps.
    """
    doc = rules_doc if rules_doc is not None else load_rules(rules_path)
    metrics = metrics or {}
    signals = signals or {}
    series_map = metric_series_map or {}
    fired: list[dict[str, Any]] = []

    for rule in doc.get("rules", []):
        if not _label_applies(rule.get("label_id", ""), label_id):
            continue
        src = rule.get("from")
        if track and src != track:
            continue
        matched = False
        if src == "geometric":
            matched = _metric_matches(rule, metrics)
        elif src == "model":
            matched = _signal_matches(rule, signals)
        if not matched:
            continue
        item: dict[str, Any] = {
            "id": rule["id"],
            "from": src,
            "severity": rule.get("severity"),
            "feedback_zh": rule.get("feedback_zh"),
            "feedback_en": rule.get("feedback_en"),
            "metric": rule.get("metric"),
            "signal": rule.get("signal"),
            "observed": (
                metrics.get(rule["metric"])
                if src == "geometric" and rule.get("metric")
                else signals.get(rule["signal"])
                if src == "model" and rule.get("signal")
                else None
            ),
        }
        if src == "geometric" and rule.get("metric"):
            series = series_map.get(rule["metric"])
            ev = _evidence_frame_for_rule(rule, series, fps=fps)
            if ev is not None:
                item["evidence_frame"] = ev
        fired.append(item)
    return fired
