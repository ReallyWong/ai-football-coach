"""Caps for shot-video analysis on a 512Mi instance.

The competition service rejects clips that would make a decoded-frame spike
likely, and callers drop native allocations after each analysis. These checks
do not import OpenCV.
"""
from __future__ import annotations

import gc

# A 10-second 720p side view at a demo bitrate fits well under this.
MAX_SHOT_BYTES = 20 * 1024 * 1024
# Phone "10 second" clips often land a little over 10s.
MAX_SHOT_SECONDS = 12.0
# 720p, landscape or portrait.
MAX_SHOT_LONG_SIDE = 1280
MAX_SHOT_SHORT_SIDE = 720
# Container duration is often a fraction over the nominal length.
SHOT_DURATION_SLACK_SECONDS = 0.5


class ShotClipLimitError(Exception):
    """Clip exceeds the demo-safe size, duration, or resolution cap."""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


def shot_too_large_detail() -> str:
    mb = MAX_SHOT_BYTES // (1024 * 1024)
    return f"Shot clip is larger than {mb} MB."


def resolution_over_limit(width: int, height: int) -> bool:
    if width <= 0 or height <= 0:
        return False
    long_side = max(int(width), int(height))
    short_side = min(int(width), int(height))
    return long_side > MAX_SHOT_LONG_SIDE or short_side > MAX_SHOT_SHORT_SIDE


def resolution_limit_detail(width: int, height: int) -> str:
    return (
        f"Shot clip resolution {int(width)}x{int(height)} exceeds the 720p limit "
        f"(long side ≤ {MAX_SHOT_LONG_SIDE}, short side ≤ {MAX_SHOT_SHORT_SIDE})."
    )


def duration_over_limit(seconds: float) -> bool:
    return float(seconds) > MAX_SHOT_SECONDS + SHOT_DURATION_SLACK_SECONDS


def duration_limit_detail(seconds: float) -> str:
    return (
        f"Shot clip duration {float(seconds):.1f}s exceeds the "
        f"{MAX_SHOT_SECONDS:.0f}s limit."
    )


def release_shot_memory() -> None:
    """Free cyclic garbage and return glibc arenas to the OS.

    OpenCV frame buffers are freed in Python before this runs. Without
    ``malloc_trim``, RSS on the 512Mi service can stay at the high-water
    mark after the arrays are gone.
    """
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        return
