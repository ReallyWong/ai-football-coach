#!/usr/bin/env python3
"""M1 offline demo: pose → COCO-17 → geometric metrics → rules engine.

Default path uses synthetic bad plant-foot keypoints (or the bundled fixture)
so no MediaPipe install is required.

Examples
--------
  python scripts/run_offline_demo.py
  python scripts/run_offline_demo.py --pose-json fixtures/demo_pass_inside_pose.json
  python scripts/run_offline_demo.py --video clip.mp4   # needs mediapipe
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.adapters.mediapipe_to_coco17 import (  # noqa: E402
    MP_TO_COCO17,
    mediapipe_to_coco17,
)
from pipeline.metrics.geometric import compute_geometric_metrics  # noqa: E402
from pipeline.pose.load_mediapipe_json import load_mediapipe_json  # noqa: E402
from pipeline.rules_engine import evaluate_rules  # noqa: E402

# COCO indices for synthetic builders
L_SHO, R_SHO, L_HIP, R_HIP = 5, 6, 11, 12
L_KNEE, R_KNEE, L_ANK, R_ANK = 13, 14, 15, 16


def _empty_mp(t: int) -> np.ndarray:
    return np.zeros((t, 33, 3), dtype=np.float64)


def _set_coco_via_mp(mp: np.ndarray, coco_idx: int, x: float, y: float, conf: float = 1.0) -> None:
    mp_idx = MP_TO_COCO17[coco_idx]
    mp[:, mp_idx, 0] = x
    mp[:, mp_idx, 1] = y
    mp[:, mp_idx, 2] = conf


def make_synthetic_mp33(*, bad_plant_foot: bool, t: int = 8) -> np.ndarray:
    """Build a tiny MediaPipe-33 clip.

    Standing figure, y-down. Left leg is plant foot.
    bad: shin ~45° from +x (outside [70,110])
    good: shin ~90° (vertical, inside range)
    """
    mp = _empty_mp(t)
    # torso
    _set_coco_via_mp(mp, L_SHO, 90, 80)
    _set_coco_via_mp(mp, R_SHO, 130, 80)
    _set_coco_via_mp(mp, L_HIP, 95, 160)
    _set_coco_via_mp(mp, R_HIP, 125, 160)
    # right (non-plant) leg — roughly vertical
    _set_coco_via_mp(mp, R_KNEE, 125, 220)
    _set_coco_via_mp(mp, R_ANK, 125, 280)
    # left plant leg
    _set_coco_via_mp(mp, L_KNEE, 95, 220)
    if bad_plant_foot:
        # shin ~45°: ankle southeast of knee
        _set_coco_via_mp(mp, L_ANK, 95 + 56.6, 220 + 56.6)
    else:
        _set_coco_via_mp(mp, L_ANK, 95, 280)
    # nose etc. (optional)
    _set_coco_via_mp(mp, 0, 110, 40)
    return mp


def _try_video_to_mp33(video_path: Path) -> np.ndarray:
    try:
        import mediapipe as mp  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "MediaPipe is not installed. Install with:\n"
            "  pip install mediapipe\n"
            "Or run without --video using the default synthetic path / --pose-json."
        ) from exc

    # Prefer classic Pose solution if present; otherwise fail with clear message.
    if not hasattr(mp, "solutions"):
        raise SystemExit(
            "Installed mediapipe package has no mp.solutions.Pose API. "
            "Use --pose-json with a MotionDiff export, or the default synthetic demo."
        )

    import cv2  # mediapipe stacks usually pull opencv

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    frames = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            row = np.full((33, 3), np.nan, dtype=np.float64)
            if result.pose_landmarks:
                h, w = bgr.shape[:2]
                for i, lm in enumerate(result.pose_landmarks.landmark):
                    row[i, 0] = lm.x * w
                    row[i, 1] = lm.y * h
                    row[i, 2] = float(getattr(lm, "visibility", 1.0) or 1.0)
            frames.append(row)
    finally:
        cap.release()
        pose.close()

    if not frames:
        raise SystemExit(f"No frames decoded from {video_path}")
    return np.stack(frames, axis=0)


def to_coco17(keypoints: np.ndarray, topology: str) -> np.ndarray:
    if topology.startswith("coco") or keypoints.shape[1] == 17:
        return np.asarray(keypoints, dtype=np.float64)
    if keypoints.shape[1] == 33 or topology.startswith("mediapipe"):
        return mediapipe_to_coco17(keypoints)
    raise ValueError(f"Cannot map topology={topology!r} shape={keypoints.shape}")


def run_pipeline(
    coco17: np.ndarray,
    *,
    label_id: str = "pass.inside",
) -> tuple[dict[str, float], list[dict]]:
    metrics = compute_geometric_metrics(coco17)
    fired = evaluate_rules(label_id, metrics, track="geometric")
    return metrics, fired


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M1 offline geometric rules demo")
    parser.add_argument(
        "--pose-json",
        type=Path,
        help="MotionDiff-like or coach clip JSON (T×33 or T×17)",
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Optional video path (requires mediapipe)",
    )
    parser.add_argument(
        "--label-id",
        default="pass.inside",
        help="Technique label_id (default: pass.inside)",
    )
    parser.add_argument(
        "--good",
        action="store_true",
        help="Use synthetic GOOD plant foot (should not fire plant_foot rule)",
    )
    parser.add_argument(
        "--json-out",
        action="store_true",
        help="Also print a JSON summary line",
    )
    args = parser.parse_args(argv)

    topology = "mediapipe33"
    if args.video and args.pose_json:
        print("Pass only one of --video / --pose-json", file=sys.stderr)
        return 2

    if args.video:
        mp33 = _try_video_to_mp33(args.video)
        coco = to_coco17(mp33, "mediapipe33")
        source = f"video:{args.video}"
    elif args.pose_json:
        kps, meta = load_mediapipe_json(args.pose_json)
        coco = to_coco17(kps, meta.get("topology", "mediapipe33"))
        source = f"pose-json:{args.pose_json}"
        if meta.get("label_id") and args.label_id == "pass.inside":
            # honor fixture label when user left default
            pass
    else:
        default_fixture = REPO_ROOT / "fixtures" / "demo_pass_inside_pose.json"
        if default_fixture.is_file() and not args.good:
            kps, meta = load_mediapipe_json(default_fixture)
            coco = to_coco17(kps, meta.get("topology", "mediapipe33"))
            source = f"fixture:{default_fixture.name}"
        else:
            mp33 = make_synthetic_mp33(bad_plant_foot=not args.good)
            coco = mediapipe_to_coco17(mp33)
            source = "synthetic:" + ("good" if args.good else "bad_plant_foot")

    metrics, fired = run_pipeline(coco, label_id=args.label_id)

    print(f"source={source}")
    print(f"label_id={args.label_id}")
    print(f"T={coco.shape[0]} V={coco.shape[1]}")
    print("metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}" if v == v else f"  {k}: nan")
    print(f"rules_fired ({len(fired)}):")
    if not fired:
        print("  (none)")
    for rule in fired:
        print(f"  - {rule['id']} [{rule.get('severity')}]")
        print(f"    zh: {rule.get('feedback_zh')}")
        print(f"    en: {rule.get('feedback_en')}")
        if rule.get("observed") is not None:
            print(f"    observed: {rule['observed']:.2f}")

    if args.json_out:
        print(
            json.dumps(
                {
                    "label_id": args.label_id,
                    "metrics": metrics,
                    "rules_fired": [
                        {
                            "id": r["id"],
                            "from": r["from"],
                            "severity": r["severity"],
                            "feedback_zh": r["feedback_zh"],
                            "feedback_en": r["feedback_en"],
                        }
                        for r in fired
                    ],
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
