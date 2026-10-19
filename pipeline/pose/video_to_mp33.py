"""Video → MediaPipe-33 landmarks (T×33×3)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def video_to_mp33(
    video_path: str | Path,
    *,
    max_frames: int = 180,
    stride: int = 1,
) -> tuple[np.ndarray, float]:
    """Return (T×33×3 float64, fps). Raises ImportError / ValueError with clear text."""
    try:
        import mediapipe as mp  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "MediaPipe is not installed. pip install mediapipe opencv-python-headless"
        ) from exc

    if not hasattr(mp, "solutions"):
        raise ImportError(
            "Installed mediapipe has no mp.solutions.Pose. "
            "Try: pip install 'mediapipe>=0.10,<0.10.14' opencv-python-headless"
        )

    import cv2

    path = Path(video_path)
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
    )
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0) or 30.0
    frames: list[np.ndarray] = []
    idx = 0
    try:
        while len(frames) < max_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            if stride > 1 and (idx % stride) != 0:
                idx += 1
                continue
            idx += 1
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
        raise ValueError("No frames decoded from video")
    # Drop all-nan frames at ends
    arr = np.stack(frames, axis=0)
    valid = ~np.isnan(arr[:, :, 2]).all(axis=1)
    if not valid.any():
        raise ValueError("No pose detected in video — try a clearer side/diagonal view")
    arr = arr[valid]
    return arr, fps
