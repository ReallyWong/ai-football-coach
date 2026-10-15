"""Load rules/rules.json and evaluate geometric (and model-signal) rules.

Feedback strings come from rules.json — never duplicated here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


def evaluate_rules(
    label_id: str,
    metrics: dict[str, float] | None = None,
    *,
    signals: dict[str, float] | None = None,
    rules_doc: dict[str, Any] | None = None,
    rules_path: Path | str | None = None,
    track: str | None = None,
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
        If ``\"geometric\"`` or ``\"model\"``, only that track; else both.
    """
    doc = rules_doc if rules_doc is not None else load_rules(rules_path)
    metrics = metrics or {}
    signals = signals or {}
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
        fired.append(
            {
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
        )
    return fired
