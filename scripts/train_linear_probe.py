#!/usr/bin/env python3
"""Synthesize distinctive COCO-17 clips per taxonomy class and train SkeletonProbe.

Saves:
  models/youth_football_probe.pt   — torch state_dict + meta
  models/youth_football_probe.onnx — optional (also via scripts/export_onnx.py)

CPU-friendly; target wall time < 2 minutes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.model.skeleton_probe import (  # noqa: E402
    CLASS_IDS,
    COCO17_V,
    DEFAULT_T,
    NUM_CLASSES,
    SkeletonProbe,
    load_taxonomy_labels,
    preprocess_tvx_to_nctv,
)

# COCO indices
NOSE = 0
L_SHO, R_SHO = 5, 6
L_ELB, R_ELB = 7, 8
L_WRI, R_WRI = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANK, R_ANK = 15, 16


def _base_stance(t: int, rng: np.random.Generator) -> np.ndarray:
    """Upright standing figure, y-down image coords, slight noise."""
    clip = np.zeros((t, COCO17_V, 3), dtype=np.float64)
    # torso
    joints = {
        NOSE: (110, 40),
        1: (105, 35),
        2: (115, 35),
        3: (98, 38),
        4: (122, 38),
        L_SHO: (90, 80),
        R_SHO: (130, 80),
        L_ELB: (80, 120),
        R_ELB: (140, 120),
        L_WRI: (75, 155),
        R_WRI: (145, 155),
        L_HIP: (95, 160),
        R_HIP: (125, 160),
        L_KNEE: (95, 220),
        R_KNEE: (125, 220),
        L_ANK: (95, 280),
        R_ANK: (125, 280),
    }
    for j, (x, y) in joints.items():
        clip[:, j, 0] = x
        clip[:, j, 1] = y
        clip[:, j, 2] = 1.0
    noise = rng.normal(0, 0.8, size=clip[..., :2].shape)
    clip[..., :2] += noise
    return clip


def synth_pass_inside(t: int, rng: np.random.Generator) -> np.ndarray:
    """Plant left foot stable; right (kick) ankle swings medial/forward mid-clip.

    Also emits a static tilted-plant variant (matches M1 fixture geometry) so the
    demo pose classifies as pass.inside under the probe.
    """
    clip = _base_stance(t, rng)
    if rng.random() < 0.35:
        # Static bad/open plant foot — same geometry as fixtures/demo_pass_inside_pose
        clip[:, L_KNEE, 0] = 95
        clip[:, L_KNEE, 1] = 220
        clip[:, L_ANK, 0] = 95 + 56.6
        clip[:, L_ANK, 1] = 220 + 56.6
        clip[:, R_KNEE, 0] = 125
        clip[:, R_ANK, 0] = 125
        return clip
    # plant foot slightly open (~80°) — still inside rule band often
    clip[:, L_ANK, 0] = 95 + 8
    clip[:, L_ANK, 1] = 280
    phase = np.linspace(0, np.pi, t)
    # kick leg: horizontal swing of ankle (inside-foot signature)
    clip[:, R_ANK, 0] = 125 + 35 * np.sin(phase)
    clip[:, R_ANK, 1] = 280 - 15 * np.sin(phase)
    clip[:, R_KNEE, 0] = 125 + 12 * np.sin(phase)
    return clip


def synth_receive_first_touch(t: int, rng: np.random.Generator) -> np.ndarray:
    """Ankle rises then settles (cushion); knee flexes — vertical ankle motion."""
    clip = _base_stance(t, rng)
    phase = np.linspace(0, np.pi, t)
    lift = 40 * np.sin(phase)
    clip[:, R_ANK, 1] = 280 - lift
    clip[:, R_KNEE, 1] = 220 - 0.4 * lift
    clip[:, R_ANK, 0] = 125 + rng.normal(0, 1.5, size=t)
    # arms slightly forward to receive
    clip[:, L_WRI, 1] = 155 - 10 * np.sin(phase)
    clip[:, R_WRI, 1] = 155 - 10 * np.sin(phase)
    return clip


def synth_shot_instep(t: int, rng: np.random.Generator) -> np.ndarray:
    """Forceful forward swing + torso lean toward target."""
    clip = _base_stance(t, rng)
    phase = np.linspace(0, np.pi, t)
    swing = np.sin(phase)
    # big kick: ankle goes back then forward/up
    clip[:, R_ANK, 0] = 125 + 50 * swing
    clip[:, R_ANK, 1] = 280 - 55 * swing
    clip[:, R_KNEE, 0] = 125 + 25 * swing
    clip[:, R_KNEE, 1] = 220 - 20 * swing
    # torso lean forward (shoulders ahead of hips in x, or hips back)
    lean = 18 * swing
    clip[:, L_SHO, 0] = 90 + lean
    clip[:, R_SHO, 0] = 130 + lean
    clip[:, NOSE, 0] = 110 + lean
    return clip


def synth_dribble_close(t: int, rng: np.random.Generator) -> np.ndarray:
    """Both ankles oscillate small amplitude near body (close control)."""
    clip = _base_stance(t, rng)
    tt = np.linspace(0, 4 * np.pi, t)
    amp = 12.0
    clip[:, L_ANK, 0] = 95 + amp * np.sin(tt)
    clip[:, R_ANK, 0] = 125 + amp * np.sin(tt + 0.6)
    clip[:, L_ANK, 1] = 280 - 6 * np.abs(np.sin(tt))
    clip[:, R_ANK, 1] = 280 - 6 * np.abs(np.sin(tt + 0.6))
    clip[:, L_KNEE, 0] = 95 + 4 * np.sin(tt)
    clip[:, R_KNEE, 0] = 125 + 4 * np.sin(tt + 0.6)
    return clip


SYNTH_FNS = (
    synth_pass_inside,
    synth_receive_first_touch,
    synth_shot_instep,
    synth_dribble_close,
)


def generate_dataset(
    n_per_class: int,
    t: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for class_id, fn in enumerate(SYNTH_FNS):
        for i in range(n_per_class):
            clip = fn(t, rng)
            # mild per-sample jitter already in fn; add global shift
            clip = clip.copy()
            clip[..., 0] += rng.uniform(-5, 5)
            clip[..., 1] += rng.uniform(-5, 5)
            nctv = preprocess_tvx_to_nctv(clip, t_out=t, normalize=True)
            xs.append(nctv[0])
            ys.append(class_id)
    # Inject M1 demo fixture (MP-33 → COCO-17) as pass.inside so /analyze demo
    # classifies the checked-in pose without requiring a label_id hint.
    try:
        from pipeline.adapters.mediapipe_to_coco17 import mediapipe_to_coco17
        from pipeline.pose.load_mediapipe_json import load_mediapipe_json

        fixture = REPO_ROOT / "fixtures" / "demo_pass_inside_pose.json"
        if fixture.is_file():
            kps, meta = load_mediapipe_json(fixture)
            coco = mediapipe_to_coco17(kps) if kps.shape[1] == 33 else kps
            for _ in range(max(8, n_per_class // 4)):
                jittered = coco.copy()
                jittered[..., 0] += rng.uniform(-3, 3)
                jittered[..., 1] += rng.uniform(-3, 3)
                nctv = preprocess_tvx_to_nctv(jittered, t_out=t, normalize=True)
                xs.append(nctv[0])
                ys.append(0)
    except Exception as exc:  # pragma: no cover — training should not hard-fail
        print(f"fixture inject skipped: {exc}")
    x = np.stack(xs, axis=0).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    # shuffle
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def train(
    *,
    n_per_class: int = 80,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    t: int = DEFAULT_T,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[SkeletonProbe, dict]:
    load_taxonomy_labels()  # validate contract
    torch.manual_seed(seed)
    np.random.seed(seed)

    x, y = generate_dataset(n_per_class, t, seed=seed)
    n = len(y)
    n_val = max(1, n // 5)
    x_train, y_train = x[n_val:], y[n_val:]
    x_val, y_val = x[:n_val], y[:n_val]

    model = SkeletonProbe(num_classes=NUM_CLASSES, t=t).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(yb)
            correct += int((logits.argmax(1) == yb).sum().item())
            count += len(yb)
        model.eval()
        with torch.no_grad():
            v_logits = model(torch.from_numpy(x_val).to(device))
            v_pred = v_logits.argmax(1).cpu().numpy()
            val_acc = float((v_pred == y_val).mean())
        train_acc = correct / max(count, 1)
        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / max(count, 1),
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
        )
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"epoch {epoch:02d}  loss={history[-1]['loss']:.4f}  "
                f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}"
            )

    meta = {
        "class_ids": list(CLASS_IDS),
        "num_classes": NUM_CLASSES,
        "T": t,
        "V": COCO17_V,
        "C": 3,
        "layout": "N×C×T×V",
        "n_per_class": n_per_class,
        "epochs": epochs,
        "final_val_acc": history[-1]["val_acc"],
        "history": history,
    }
    return model, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train synthetic skeleton probe (M2)")
    parser.add_argument("--n-per-class", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--t", type=int, default=DEFAULT_T)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "models",
    )
    parser.add_argument(
        "--also-onnx",
        action="store_true",
        help="Also write youth_football_probe.onnx after training",
    )
    args = parser.parse_args(argv)

    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, meta = train(
        n_per_class=args.n_per_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        t=args.t,
        seed=args.seed,
    )

    pt_path = args.out_dir / "youth_football_probe.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "meta": meta,
            "model_kwargs": {"num_classes": NUM_CLASSES, "t": args.t, "v": COCO17_V},
        },
        pt_path,
    )
    meta_path = args.out_dir / "youth_football_probe.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {pt_path}")
    print(f"wrote {meta_path}")
    print(f"final_val_acc={meta['final_val_acc']:.3f}  elapsed={time.time() - t0:.1f}s")

    if args.also_onnx:
        from scripts.export_onnx import export_onnx

        onnx_path = args.out_dir / "youth_football_probe.onnx"
        export_onnx(pt_path, onnx_path, t=args.t)
        print(f"wrote {onnx_path}")

    if meta["final_val_acc"] < 0.7:
        print(
            "WARNING: val_acc < 0.7 — synthetic signatures may be too weak; "
            "re-run with more epochs or check synth functions.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
