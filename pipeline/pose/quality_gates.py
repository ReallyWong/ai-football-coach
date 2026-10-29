"""Image and pose quality gates for the shooting video path.

OpenCV measures blur (Laplacian variance) and brightness. Pose keypoints
then add person size and joint-confidence coverage. A failed gate asks the
caller to re-record instead of publishing a score.

Thresholds are deterministic so the same clip always gets the same gate.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Image gate. Laplacian variance below this is too soft to judge a strike.
MIN_BLUR_VAR = 30.0
MIN_BRIGHTNESS = 25.0
MAX_BRIGHTNESS = 230.0

# Pose gate. The player has to be large enough, and enough joints visible.
MIN_PERSON_FRAC = 0.05
MIN_POSE_OK_FRAC = 0.4
MIN_FRAMES = 8
CONF_OK = 0.3
MIN_OK_JOINTS = 8

# COCO-17 core used to decide a frame is usable: both shoulders, both hips.
_CORE_JOINTS = (5, 6, 11, 12)


def measure_blur_brightness(frame: Any) -> tuple[float, float]:
    """Laplacian variance and mean brightness for one BGR frame.

    The gray image and Laplacian are temporaries. Callers that stream video
    should drop the frame after this returns.
    """
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        bright = float(np.mean(gray))
    finally:
        del gray
    return blur, bright


def quality_from_measurements(
    blurs: list[float],
    brights: list[float],
) -> dict[str, float | int]:
    """Median blur and brightness at the first, middle, and last samples.

    ``frames`` is the full sample count, matching :func:`quality_gates`.
    """
    n = len(blurs)
    if n == 0 or n != len(brights):
        return {"blur_var": 0.0, "brightness": 0.0, "frames": int(n)}
    sample_at = sorted({0, n // 2, n - 1})
    return {
        "blur_var": float(np.median([blurs[index] for index in sample_at])),
        "brightness": float(np.median([brights[index] for index in sample_at])),
        "frames": int(n),
    }


def quality_gates(frames_bgr: list[Any]) -> dict[str, float | int]:
    """Blur and brightness on a few sample frames. Does not import cv2 until called.

    ``pass`` is computed later by :func:`merge_quality` once pose stats exist.
    """
    if not frames_bgr:
        return {"blur_var": 0.0, "brightness": 0.0, "frames": 0}

    n = len(frames_bgr)
    blurs: list[float] = []
    brights: list[float] = []
    for index in sorted({0, n // 2, n - 1}):
        blur, bright = measure_blur_brightness(frames_bgr[index])
        blurs.append(blur)
        brights.append(bright)
    return {
        "blur_var": float(np.median(blurs)),
        "brightness": float(np.median(brights)),
        "frames": int(n),
    }


def _bbox_fraction(
    frame: np.ndarray,
    frame_wh: tuple[int, int] | None,
) -> float | None:
    """Visible-joint box as a fraction of the frame. None if the scale is unknown."""
    conf = frame[:, 2] if frame.shape[-1] >= 3 else np.ones(frame.shape[0])
    visible = (
        np.isfinite(conf)
        & (conf >= 0.2)
        & np.isfinite(frame[:, 0])
        & np.isfinite(frame[:, 1])
    )
    if int(visible.sum()) < 4:
        return 0.0
    xs = frame[visible, 0]
    ys = frame[visible, 1]
    width = float(xs.max() - xs.min())
    height = float(ys.max() - ys.min())
    area = max(0.0, width) * max(0.0, height)
    max_coord = float(max(np.max(np.abs(xs)), np.max(np.abs(ys))))
    # Normalized keypoints are already in frame units.
    if max_coord <= 1.5:
        return float(np.clip(area, 0.0, 1.0))
    if frame_wh is None:
        return None
    frame_w, frame_h = frame_wh
    if frame_w <= 0 or frame_h <= 0:
        return None
    return float(np.clip(area / float(frame_w * frame_h), 0.0, 1.0))


def pose_quality(
    coco17: np.ndarray,
    *,
    frame_wh: tuple[int, int] | None = None,
) -> dict[str, float | int | None]:
    """Joint coverage and person size from a COCO-17 clip (T, 17, 3)."""
    arr = np.asarray(coco17, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != 17:
        raise ValueError(f"Expected (T, 17, C), got {arr.shape}")
    ok_frames = 0
    fractions: list[float] = []
    for frame in arr:
        conf = frame[:, 2] if frame.shape[-1] >= 3 else np.ones(17)
        joints_ok = np.isfinite(conf) & (conf >= CONF_OK)
        core_ok = all(bool(joints_ok[i]) for i in _CORE_JOINTS)
        if int(joints_ok.sum()) >= MIN_OK_JOINTS and core_ok:
            ok_frames += 1
        frac = _bbox_fraction(frame, frame_wh)
        if frac is not None:
            fractions.append(frac)
    frames = int(arr.shape[0])
    return {
        "pose_ok_frac": (ok_frames / frames) if frames else 0.0,
        "person_frac": float(np.median(fractions)) if fractions else None,
        "frames": frames,
    }


def merge_quality(
    image_q: dict[str, Any] | None,
    pose_q: dict[str, Any] | None,
) -> dict[str, Any]:
    """Combine image and pose stats into one gate with ``pass`` and ``fail_reasons``."""
    merged: dict[str, Any] = {}
    for source in (image_q, pose_q):
        if not source:
            continue
        for key, value in source.items():
            if key in {"pass", "fail_reasons"}:
                continue
            merged[key] = value

    reasons: list[str] = []
    blur = merged.get("blur_var")
    if blur is not None and float(blur) < MIN_BLUR_VAR:
        reasons.append("blur")
    bright = merged.get("brightness")
    if bright is not None and not (MIN_BRIGHTNESS <= float(bright) <= MAX_BRIGHTNESS):
        reasons.append("brightness")
    frames = merged.get("frames")
    if frames is not None and int(frames) < MIN_FRAMES:
        reasons.append("too_few_frames")
    person = merged.get("person_frac")
    if person is not None and float(person) < MIN_PERSON_FRAC:
        reasons.append("person_small")
    pose_ok = merged.get("pose_ok_frac")
    if pose_ok is not None and float(pose_ok) < MIN_POSE_OK_FRAC:
        reasons.append("pose_low_conf")

    merged["fail_reasons"] = reasons
    merged["pass"] = len(reasons) == 0
    return merged
