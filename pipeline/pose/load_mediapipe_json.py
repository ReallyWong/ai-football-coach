"""Load MotionDiff-like MediaPipe JSON (or coach clip arrays) into (T,V,C).

Supported shapes
----------------
1. Clip object with ``keypoints`` as nested list ``T×V×C`` (V=33 or 17).
2. Clip object with ``frames`` list — each frame has ``people[0].keypoints``
   as either a length-33 list of ``{x,y,visibility|confidence|...}`` dicts
   (MotionDiff mediapipe-pose-frame/1.0) or a flat ``[[x,y,c], ...]``.
3. Directory of per-frame ``*_keypoints.json`` files (MotionDiff export).
4. Bare nested list / ndarray dumped as JSON.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

MP_COUNT = 33
COCO_COUNT = 17


def _keypoint_to_xyz(kp: Any) -> list[float]:
    if isinstance(kp, (list, tuple)) and len(kp) >= 2:
        x, y = float(kp[0]), float(kp[1])
        conf = float(kp[2]) if len(kp) > 2 and kp[2] is not None else 1.0
        return [x, y, conf]
    if isinstance(kp, dict):
        # Prefer image-space x/y; fall back to normalized_*
        x = kp.get("x", kp.get("normalized_x"))
        y = kp.get("y", kp.get("normalized_y"))
        if x is None or y is None:
            return [float("nan"), float("nan"), 0.0]
        conf = kp.get("confidence")
        if conf is None:
            vis = kp.get("visibility")
            pre = kp.get("presence")
            vals = [v for v in (vis, pre) if v is not None]
            conf = float(min(vals)) if vals else 1.0
        return [float(x), float(y), float(conf)]
    raise TypeError(f"Unsupported keypoint entry: {type(kp)!r}")


def _frame_people_to_row(frame: dict[str, Any]) -> np.ndarray | None:
    people = frame.get("people") or []
    if not people:
        return None
    person = people[0]
    kps = person.get("keypoints") or []
    if not kps:
        return None
    # Named MotionDiff keypoints may be unordered — sort by index if present
    if isinstance(kps[0], dict) and "index" in kps[0]:
        kps = sorted(kps, key=lambda p: int(p["index"]))
    rows = [_keypoint_to_xyz(kp) for kp in kps]
    return np.asarray(rows, dtype=np.float64)


def _load_frame_file(path: Path) -> np.ndarray | None:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "people" in data:
        return _frame_people_to_row(data)
    if isinstance(data, dict) and "keypoints" in data and isinstance(
        data["keypoints"], list
    ):
        kps = data["keypoints"]
        if kps and isinstance(kps[0], (list, tuple, dict)):
            # single-frame keypoints list
            if not isinstance(kps[0], (list, tuple)) or (
                isinstance(kps[0], (list, tuple)) and not isinstance(kps[0][0], (list, tuple))
            ):
                return np.asarray([_keypoint_to_xyz(kp) for kp in kps], dtype=np.float64)
    return None


def load_mediapipe_json(path: Path | str) -> tuple[np.ndarray, dict[str, Any]]:
    """Load pose JSON from ``path``.

    Returns
    -------
    keypoints : (T, V, 3) float64
        V is 33 (MediaPipe) or 17 (already COCO) depending on source.
    meta : dict
        Includes ``topology`` (``mediapipe33`` | ``coco17``), ``fps``, etc.
    """
    path = Path(path)
    meta: dict[str, Any] = {"source_path": str(path)}

    if path.is_dir():
        files = sorted(path.glob("*_keypoints.json"))
        if not files:
            files = sorted(p for p in path.glob("*.json") if p.name != "run_summary.json")
        rows = []
        for fp in files:
            row = _load_frame_file(fp)
            if row is None:
                # keep timeline: NaN frame with 33 joints
                rows.append(np.full((MP_COUNT, 3), np.nan))
            else:
                rows.append(row)
        if not rows:
            raise FileNotFoundError(f"No keypoint JSON frames in {path}")
        arr = np.stack(rows, axis=0)
        v = arr.shape[1]
        meta["topology"] = "mediapipe33" if v == MP_COUNT else "coco17" if v == COCO_COUNT else f"v{v}"
        meta["T"] = int(arr.shape[0])
        meta["V"] = int(v)
        return arr, meta

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        meta["topology"] = (
            "mediapipe33" if arr.shape[1] == MP_COUNT
            else "coco17" if arr.shape[1] == COCO_COUNT
            else f"v{arr.shape[1]}"
        )
        meta["T"] = int(arr.shape[0])
        meta["V"] = int(arr.shape[1])
        return arr, meta

    if not isinstance(data, dict):
        raise TypeError(f"Unsupported JSON root type: {type(data)!r}")

    meta.update({k: data[k] for k in ("label_id", "fps", "view", "T", "V") if k in data})

    if "keypoints" in data and isinstance(data["keypoints"], list):
        arr = np.asarray(data["keypoints"], dtype=np.float64)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        if arr.ndim != 3:
            raise ValueError(f"keypoints must be T×V×C, got shape {arr.shape}")
        meta["topology"] = data.get(
            "topology",
            "mediapipe33" if arr.shape[1] == MP_COUNT
            else "coco17" if arr.shape[1] == COCO_COUNT
            else f"v{arr.shape[1]}",
        )
        meta["T"] = int(arr.shape[0])
        meta["V"] = int(arr.shape[1])
        return arr, meta

    if "frames" in data and isinstance(data["frames"], list):
        rows = []
        for fr in data["frames"]:
            row = _frame_people_to_row(fr) if isinstance(fr, dict) else None
            if row is None and isinstance(fr, dict) and "keypoints" in fr:
                kps = fr["keypoints"]
                row = np.asarray([_keypoint_to_xyz(kp) for kp in kps], dtype=np.float64)
            if row is None:
                rows.append(np.full((MP_COUNT, 3), np.nan))
            else:
                rows.append(row)
        arr = np.stack(rows, axis=0)
        meta["topology"] = (
            "mediapipe33" if arr.shape[1] == MP_COUNT
            else "coco17" if arr.shape[1] == COCO_COUNT
            else f"v{arr.shape[1]}"
        )
        meta["T"] = int(arr.shape[0])
        meta["V"] = int(arr.shape[1])
        return arr, meta

    # Single MotionDiff frame file
    row = _frame_people_to_row(data)
    if row is not None:
        arr = row[np.newaxis, ...]
        meta["topology"] = (
            "mediapipe33" if arr.shape[1] == MP_COUNT
            else "coco17" if arr.shape[1] == COCO_COUNT
            else f"v{arr.shape[1]}"
        )
        meta["T"] = 1
        meta["V"] = int(arr.shape[1])
        return arr, meta

    raise ValueError(
        f"Unrecognized pose JSON structure in {path}. "
        "Expected keypoints[], frames[], or MotionDiff people[]."
    )
