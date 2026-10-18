#!/usr/bin/env python3
"""Generate a synthetic youth-football skeleton dataset (≥80 clips / class).

Writes under ``datasets/youth_football_synth/``:
  - ``manifest.jsonl`` — one JSON object per clip (committed)
  - ``clips/<label_id>/<id>.npy`` — float32 (T, 17, 3) COCO-17 (gitignored)
  - ``samples/`` — 1–2 tiny JSON clips for inspection (committed)
  - ``SPLIT`` note in README

Labels are synthetic by design — see datasets/README.md.
Reuses motion signatures from ``scripts/train_linear_probe.py``.
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

from pipeline.model.skeleton_probe import (  # noqa: E402
    CLASS_IDS,
    COCO17_V,
    DEFAULT_T,
    load_taxonomy_labels,
)
from scripts.train_linear_probe import SYNTH_FNS  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "datasets" / "youth_football_synth"
MIN_PER_CLASS = 80


def _write_json_sample(path: Path, clip: np.ndarray, label_id: str, class_id: int) -> None:
    """Tiny committed sample (T×V×C list) — keep T small."""
    # Subsample to ≤8 frames for a tiny committed artifact
    t = clip.shape[0]
    idx = np.linspace(0, t - 1, min(8, t)).astype(int)
    small = clip[idx]
    doc = {
        "label_id": label_id,
        "class_id": class_id,
        "topology": "coco17",
        "T": int(small.shape[0]),
        "V": COCO17_V,
        "C": 3,
        "fps": 30,
        "synthetic": True,
        "keypoints": np.round(small, 3).tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def generate(
    *,
    n_per_class: int = MIN_PER_CLASS,
    t: int = DEFAULT_T,
    seed: int = 42,
    out_dir: Path = DEFAULT_OUT,
    eval_frac: float = 0.2,
) -> dict:
    load_taxonomy_labels()
    if n_per_class < MIN_PER_CLASS:
        raise ValueError(f"n_per_class must be ≥ {MIN_PER_CLASS} (got {n_per_class})")

    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    clips_root = out_dir / "clips"
    samples_dir = out_dir / "samples"
    # Fresh clips dir contents for this run
    if clips_root.exists():
        for p in clips_root.rglob("*.npy"):
            p.unlink()
    clips_root.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.jsonl"
    rows: list[dict] = []
    counts = {lid: 0 for lid in CLASS_IDS}

    for class_id, (label_id, fn) in enumerate(zip(CLASS_IDS, SYNTH_FNS)):
        class_dir = clips_root / label_id.replace(".", "_")
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            clip = fn(t, rng).copy()
            clip[..., 0] += rng.uniform(-5, 5)
            clip[..., 1] += rng.uniform(-5, 5)
            clip = clip.astype(np.float32)
            clip_id = f"{label_id.replace('.', '_')}_{i:04d}"
            rel = f"clips/{label_id.replace('.', '_')}/{clip_id}.npy"
            abs_path = out_dir / rel
            np.save(abs_path, clip)
            # Deterministic split by index
            split = "eval" if i < int(round(n_per_class * eval_frac)) else "train"
            row = {
                "id": clip_id,
                "path": rel,
                "label_id": label_id,
                "class_id": class_id,
                "split": split,
                "T": int(clip.shape[0]),
                "V": COCO17_V,
                "C": 3,
                "synthetic": True,
                "seed": seed,
            }
            rows.append(row)
            counts[label_id] += 1
            if i == 0:
                _write_json_sample(
                    samples_dir / f"{label_id.replace('.', '_')}_sample.json",
                    clip,
                    label_id,
                    class_id,
                )

    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "version": "0.3.0",
        "synthetic": True,
        "n_per_class": n_per_class,
        "num_classes": len(CLASS_IDS),
        "class_ids": list(CLASS_IDS),
        "counts": counts,
        "T": t,
        "V": COCO17_V,
        "seed": seed,
        "eval_frac": eval_frac,
        "manifest": "manifest.jsonl",
        "note": "All clips are synthetic COCO-17 skeletons — not real youth video.",
    }
    (out_dir / "dataset_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-per-class", type=int, default=MIN_PER_CLASS)
    p.add_argument("--t", type=int, default=DEFAULT_T)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--eval-frac", type=float, default=0.2)
    args = p.parse_args(argv)
    meta = generate(
        n_per_class=args.n_per_class,
        t=args.t,
        seed=args.seed,
        out_dir=args.out_dir,
        eval_frac=args.eval_frac,
    )
    print(json.dumps(meta, indent=2))
    for lid, c in meta["counts"].items():
        assert c >= MIN_PER_CLASS, f"{lid} has only {c} clips"
    print(f"wrote {args.out_dir / 'manifest.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
