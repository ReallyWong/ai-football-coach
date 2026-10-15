#!/usr/bin/env python3
"""Export models/youth_football_probe.pt → models/youth_football_probe.onnx."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.model.skeleton_probe import (  # noqa: E402
    COCO17_V,
    DEFAULT_T,
    NUM_CLASSES,
    SkeletonProbe,
)


def export_onnx(
    pt_path: Path | str,
    onnx_path: Path | str,
    *,
    t: int = DEFAULT_T,
    opset: int = 18,
) -> Path:
    pt_path = Path(pt_path)
    onnx_path = Path(onnx_path)
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    kwargs = ckpt.get("model_kwargs") or {
        "num_classes": NUM_CLASSES,
        "t": t,
        "v": COCO17_V,
    }
    model = SkeletonProbe(**kwargs)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    t_dim = int(kwargs.get("t", t))
    dummy = torch.zeros(1, 3, t_dim, COCO17_V, dtype=torch.float32)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["pose"],
        output_names=["logits"],
        dynamic_axes={"pose": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        dynamo=False,  # single-file ONNX (no .onnx.data)
    )
    return onnx_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export skeleton probe to ONNX")
    parser.add_argument(
        "--pt",
        type=Path,
        default=REPO_ROOT / "models" / "youth_football_probe.pt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "models" / "youth_football_probe.onnx",
    )
    parser.add_argument("--t", type=int, default=DEFAULT_T)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args(argv)

    if not args.pt.is_file():
        print(f"Missing checkpoint {args.pt}; run scripts/train_linear_probe.py first", file=sys.stderr)
        return 1

    path = export_onnx(args.pt, args.out, t=args.t, opset=args.opset)
    size_kb = path.stat().st_size / 1024
    print(f"wrote {path} ({size_kb:.1f} KiB)")

    # Smoke: onnxruntime round-trip
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"pose": np.zeros((1, 3, args.t, COCO17_V), dtype=np.float32)})[0]
    assert out.shape == (1, NUM_CLASSES), out.shape
    print(f"onnxruntime smoke ok  logits_shape={out.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
