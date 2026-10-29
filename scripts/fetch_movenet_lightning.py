#!/usr/bin/env python3
"""Download MoveNet SinglePose Lightning for the contest image.

The weight is Apache-2.0 and is not committed. ``Dockerfile.comp`` runs this
after ``COPY``. The live ``Dockerfile`` does not. MediaPipe is not enabled.

Source: Xenova/movenet-singlepose-lightning ``onnx/model.onnx`` (pinned commit).
The graph input is int32 NHWC 192×192, pixels 0–255. The output squeezes to
``(17, 3)`` as ``(y, x, score)`` in COCO order. A sibling JSON tells
``prepare_input`` to feed int32, which is what OpenCV 5 accepts for this file.
"""
from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MOVENET_COMMIT = "ed0f314bb7356fd1dbf1e4f52c2d40791bf6534f"
MOVENET_URL = (
    "https://huggingface.co/Xenova/movenet-singlepose-lightning/resolve/"
    f"{MOVENET_COMMIT}/onnx/model.onnx"
)
DEFAULT_DEST = REPO_ROOT / "models" / "pose" / "movenet_lightning.onnx"
MIN_BYTES = 8_000_000

SPEC = {
    "family": "movenet",
    "input_size": 192,
    "layout": "nhwc",
    "scale": 1.0,
    "coord_order": "yx",
    "letterbox": False,
    "input_dtype": "int32",
    "source": MOVENET_URL,
    "license": "Apache-2.0",
    "note": (
        "Google MoveNet SinglePose Lightning via the Xenova ONNX export. "
        "Input is int32 NHWC 192, pixels 0-255. Output squeezes to (17, 3) "
        "as (y, x, score) in COCO order."
    ),
}


def fetch(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        MOVENET_URL,
        headers={"User-Agent": "ai-football-coach-contest-build"},
    )
    with urllib.request.urlopen(request, timeout=180) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out)
    size = dest.stat().st_size
    if size < MIN_BYTES:
        dest.unlink(missing_ok=True)
        raise SystemExit(f"MoveNet download is too small ({size} bytes): {MOVENET_URL}")
    sidecar = dest.with_suffix(".json")
    sidecar.write_text(json.dumps(SPEC, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {dest} ({size} bytes) and {sidecar}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()
    fetch(args.dest)


if __name__ == "__main__":
    main()
