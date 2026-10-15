"""MediaPipe Pose 33 → COCO-17 index remap.

Mapping follows docs/pose_contract.md. Never treat MP indices as COCO indices.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# COCO-17 index → MediaPipe landmark index (pose_contract.md)
MP_TO_COCO17: tuple[int, ...] = (
    0,   # 0 nose
    2,   # 1 left_eye
    5,   # 2 right_eye
    7,   # 3 left_ear
    8,   # 4 right_ear
    11,  # 5 left_shoulder
    12,  # 6 right_shoulder
    13,  # 7 left_elbow
    14,  # 8 right_elbow
    15,  # 9 left_wrist
    16,  # 10 right_wrist
    23,  # 11 left_hip
    24,  # 12 right_hip
    25,  # 13 left_knee
    26,  # 14 right_knee
    27,  # 15 left_ankle
    28,  # 16 right_ankle
)

MP_LANDMARK_COUNT = 33
COCO17_JOINT_COUNT = 17


def mediapipe_to_coco17(frames: np.ndarray | Sequence) -> np.ndarray:
    """Remap MediaPipe-33 keypoints to COCO-17.

    Parameters
    ----------
    frames :
        Either an array of shape ``(T, 33, C)`` with ``C >= 2`` (typically
        ``C=3`` for x, y, conf), or a sequence of length-T frames each shaped
        ``(33, C)``.

    Returns
    -------
    np.ndarray
        Array of shape ``(T, 17, C)`` with the same channel count as input.
        Confidence / extra channels are copied through unchanged for the
        selected joints.
    """
    arr = np.asarray(frames, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected (T,33,C) or (33,C), got shape {arr.shape}")
    t, v, c = arr.shape
    if v != MP_LANDMARK_COUNT:
        raise ValueError(
            f"Expected V={MP_LANDMARK_COUNT} MediaPipe landmarks, got V={v}"
        )
    if c < 2:
        raise ValueError(f"Expected at least 2 channels (x,y), got C={c}")

    out = np.empty((t, COCO17_JOINT_COUNT, c), dtype=np.float64)
    for coco_idx, mp_idx in enumerate(MP_TO_COCO17):
        out[:, coco_idx, :] = arr[:, mp_idx, :]
    return out
