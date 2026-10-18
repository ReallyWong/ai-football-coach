"""Geometric metrics from COCO-17 sequences (T×V×C with C=(x,y,conf)).

Joint indices (docs/pose_contract.md):
  5/6  left/right_shoulder
  11/12 left/right_hip
  13/14 left/right_knee
  15/16 left/right_ankle

plant_foot_angle_deg
  2D angle of the plant shin (knee → ankle) measured from the image +x axis,
  so a vertical shin is ~90°. Uses the plant side with higher mean ankle
  confidence (default left when equal). Image y increases downward.
  Rule pass.inside.plant_foot: op=outside range=[70,110] → warn when tilted.

torso_lean_deg
  Absolute 2D angle between mid-shoulder→mid-hip and image vertical.
  0° = upright; larger = more lean. Rule shot.instep.lean: op=gt value=25.
"""
from __future__ import annotations

import numpy as np

# COCO-17
NOSE, L_SHO, R_SHO = 0, 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANK, R_ANK = 15, 16


def _xy(frame: np.ndarray, idx: int) -> np.ndarray:
    return frame[idx, :2].astype(np.float64)


def _conf(frame: np.ndarray, idx: int) -> float:
    if frame.shape[-1] < 3:
        return 1.0
    return float(frame[idx, 2])


def _shin_angle_from_horizontal_deg(knee_xy: np.ndarray, ankle_xy: np.ndarray) -> float:
    """Angle of shin vector (knee→ankle) from +x, degrees in (-180, 180].

    Vertical downward shin → ~90°. Vertical upward → ~-90° or 270.
    We fold to [0, 180] so upright (either way) sits near 90°.
    """
    shin = ankle_xy - knee_xy
    norm = np.linalg.norm(shin)
    if norm < 1e-8:
        return float("nan")
    # atan2(dy, dx): +y down → vertical down shin has angle +90°
    ang = float(np.degrees(np.arctan2(shin[1], shin[0])))
    # Fold to [0, 180]: |ang| with wrap so 90 stays 90, -90 → 90
    ang = abs(ang)
    if ang > 180:
        ang = 360 - ang
    # Map obtuse mirror: we want vertical ≈ 90, near-horizontal ≈ 0 or 180
    # abs(atan2) already gives [0,180]; vertical down/up both near 90.
    return ang


def plant_foot_angle_deg(
    coco17: np.ndarray,
    *,
    plant_side: str | None = None,
    aggregate: str = "median",
) -> float:
    """Median (or mean) plant-foot shin angle over frames, degrees.

    Parameters
    ----------
    coco17 : (T, 17, C) array
    plant_side : \"left\", \"right\", or None (auto by ankle confidence)
    aggregate : \"median\" or \"mean\"
    """
    arr = np.asarray(coco17, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != 17:
        raise ValueError(f"Expected (T,17,C), got {arr.shape}")

    if plant_side is None:
        left_c = float(np.nanmean([_conf(f, L_ANK) for f in arr]))
        right_c = float(np.nanmean([_conf(f, R_ANK) for f in arr]))
        plant_side = "left" if left_c >= right_c else "right"

    if plant_side == "left":
        knee_i, ank_i = L_KNEE, L_ANK
    elif plant_side == "right":
        knee_i, ank_i = R_KNEE, R_ANK
    else:
        raise ValueError("plant_side must be 'left', 'right', or None")

    angles = []
    for frame in arr:
        a = _shin_angle_from_horizontal_deg(_xy(frame, knee_i), _xy(frame, ank_i))
        if np.isfinite(a):
            angles.append(a)
    if not angles:
        return float("nan")
    vals = np.asarray(angles)
    if aggregate == "mean":
        return float(np.mean(vals))
    return float(np.median(vals))


def torso_lean_deg(coco17: np.ndarray, *, aggregate: str = "median") -> float:
    """Absolute lean of torso axis from vertical, degrees."""
    arr = np.asarray(coco17, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != 17:
        raise ValueError(f"Expected (T,17,C), got {arr.shape}")

    leans = []
    for frame in arr:
        mid_sho = (_xy(frame, L_SHO) + _xy(frame, R_SHO)) / 2.0
        mid_hip = (_xy(frame, L_HIP) + _xy(frame, R_HIP)) / 2.0
        torso = mid_hip - mid_sho  # shoulder → hip (downward when upright)
        norm = np.linalg.norm(torso)
        if norm < 1e-8:
            continue
        # Angle from vertical-down (0, 1)
        vertical = np.array([0.0, 1.0])
        cos_a = float(np.clip(np.dot(torso, vertical) / norm, -1.0, 1.0))
        leans.append(float(np.degrees(np.arccos(cos_a))))
    if not leans:
        return float("nan")
    vals = np.asarray(leans)
    if aggregate == "mean":
        return float(np.mean(vals))
    return float(np.median(vals))


def compute_geometric_metrics(coco17: np.ndarray, **kwargs) -> dict[str, float]:
    """Return the geometric metric dict expected by the rules engine."""
    return {
        "plant_foot_angle_deg": plant_foot_angle_deg(coco17, **kwargs),
        "torso_lean_deg": torso_lean_deg(coco17),
    }


def plant_foot_angle_series(
    coco17: np.ndarray,
    *,
    plant_side: str | None = None,
) -> np.ndarray:
    """Per-frame plant-foot shin angle, shape (T,). NaN where undefined."""
    arr = np.asarray(coco17, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != 17:
        raise ValueError(f"Expected (T,17,C), got {arr.shape}")

    if plant_side is None:
        left_c = float(np.nanmean([_conf(f, L_ANK) for f in arr]))
        right_c = float(np.nanmean([_conf(f, R_ANK) for f in arr]))
        plant_side = "left" if left_c >= right_c else "right"

    if plant_side == "left":
        knee_i, ank_i = L_KNEE, L_ANK
    elif plant_side == "right":
        knee_i, ank_i = R_KNEE, R_ANK
    else:
        raise ValueError("plant_side must be 'left', 'right', or None")

    out = np.full(arr.shape[0], np.nan, dtype=np.float64)
    for i, frame in enumerate(arr):
        a = _shin_angle_from_horizontal_deg(_xy(frame, knee_i), _xy(frame, ank_i))
        if np.isfinite(a):
            out[i] = a
    return out


def torso_lean_series(coco17: np.ndarray) -> np.ndarray:
    """Per-frame absolute torso lean from vertical, shape (T,). NaN where undefined."""
    arr = np.asarray(coco17, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != 17:
        raise ValueError(f"Expected (T,17,C), got {arr.shape}")

    out = np.full(arr.shape[0], np.nan, dtype=np.float64)
    for i, frame in enumerate(arr):
        mid_sho = (_xy(frame, L_SHO) + _xy(frame, R_SHO)) / 2.0
        mid_hip = (_xy(frame, L_HIP) + _xy(frame, R_HIP)) / 2.0
        torso = mid_hip - mid_sho
        norm = np.linalg.norm(torso)
        if norm < 1e-8:
            continue
        vertical = np.array([0.0, 1.0])
        cos_a = float(np.clip(np.dot(torso, vertical) / norm, -1.0, 1.0))
        out[i] = float(np.degrees(np.arccos(cos_a)))
    return out
