"""Synthetic side-view shot skeletons for tests and the API sample.

These are geometric fixtures, not learned motion. ``lean='back'`` puts the
shoulders behind the hips so torso lean is about 32°. ``plant='off'`` tilts
the plant shin to about 23°, outside the 70–110° rule.
"""
from __future__ import annotations

import numpy as np


def make_shot_coco17(
    *,
    lean: str = "upright",
    plant: str = "good",
    t: int = 16,
    conf: float = 1.0,
) -> np.ndarray:
    """Return ``(T, 17, 3)`` COCO-17 keypoints for one instep-shot pose."""
    if t < 1:
        raise ValueError("t must be >= 1")
    if lean == "upright":
        left_shoulder = (90.0, 80.0)
        right_shoulder = (130.0, 80.0)
    elif lean == "back":
        left_shoulder = (40.0, 80.0)
        right_shoulder = (80.0, 80.0)
    else:
        raise ValueError("lean must be 'upright' or 'back'")
    if plant == "good":
        left_ankle = (95.0, 280.0)
    elif plant == "off":
        left_ankle = (165.0, 250.0)
    else:
        raise ValueError("plant must be 'good' or 'off'")

    left_hip = (95.0, 160.0)
    right_hip = (125.0, 160.0)
    left_knee = (95.0, 220.0)
    right_knee = (125.0, 220.0)
    right_ankle = (125.0, 280.0)
    nose_x = 0.5 * (left_shoulder[0] + right_shoulder[0])
    points = {
        0: (nose_x, 40.0),
        1: (nose_x - 4.0, 36.0),
        2: (nose_x + 4.0, 36.0),
        3: (nose_x - 8.0, 40.0),
        4: (nose_x + 8.0, 40.0),
        5: left_shoulder,
        6: right_shoulder,
        7: (left_shoulder[0] - 5.0, 120.0),
        8: (right_shoulder[0] + 5.0, 120.0),
        9: (left_shoulder[0] - 8.0, 150.0),
        10: (right_shoulder[0] + 8.0, 150.0),
        11: left_hip,
        12: right_hip,
        13: left_knee,
        14: right_knee,
        15: left_ankle,
        16: right_ankle,
    }
    arr = np.zeros((t, 17, 3), dtype=np.float64)
    for index, (x, y) in points.items():
        arr[:, index, 0] = x
        arr[:, index, 1] = y
        arr[:, index, 2] = conf
    return arr
