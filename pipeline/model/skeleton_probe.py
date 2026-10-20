"""Tiny temporal MLP probe on COCO-17 skeletons (M2).

Contract (docs/pose_contract.md, integrations/pyskl.md):
  disk / API clip : T × V × C   (C = x, y, conf)
  model input     : N × C × T × V
  model output    : N × num_classes logits  (taxonomy class_ids 0..3)

Preprocess matches pose_contract: root-center mid-hip, scale by shoulder width.
This head is a CPU-trainable stand-in; replace with ST-GCN++ from pyskl later
without changing the I/O shapes (see docs/m2_model.md).

torch is optional. Preprocess and ONNX inference import when torch is absent.
Constructing SkeletonProbe still requires torch (local train / ONNX export).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# requirements-deploy.txt ships ONNX Runtime, not torch. Importing this module
# must succeed on that image; training and ONNX export still require torch.
try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - deploy image and tests/test_deploy_import.py
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

_TORCH_REQUIRED = (
    "torch is required to train or export SkeletonProbe. "
    "The analyze API uses ONNX Runtime and does not need torch."
)


def _require_torch():
    if torch is None or nn is None:
        raise ImportError(_TORCH_REQUIRED)
    return torch, nn


REPO_ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = REPO_ROOT / "labels" / "taxonomy.json"

COCO17_V = 17
DEFAULT_T = 64
NUM_CLASSES = 4
# Channels used by the network (x, y); conf is kept in the tensor but not mixed in.
XY_CHANNELS = 2

# COCO-17 indices
L_SHO, R_SHO = 5, 6
L_HIP, R_HIP = 11, 12

CLASS_IDS: tuple[str, ...] = (
    "pass.inside",
    "receive.first_touch",
    "shot.instep",
    "dribble.close",
)


def load_taxonomy_labels(path: Path | str | None = None) -> list[dict[str, Any]]:
    tax_path = Path(path) if path else TAXONOMY_PATH
    with tax_path.open(encoding="utf-8") as f:
        doc = json.load(f)
    labels = sorted(doc["labels"], key=lambda x: int(x["class_id"]))
    if len(labels) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} labels, got {len(labels)}")
    for i, lab in enumerate(labels):
        if int(lab["class_id"]) != i:
            raise ValueError(f"class_id mismatch at {i}: {lab}")
        if lab["id"] != CLASS_IDS[i]:
            raise ValueError(f"label id mismatch at {i}: {lab['id']} vs {CLASS_IDS[i]}")
    return labels


def root_center_scale(coco_txv: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Root-center + shoulder-width scale on (T, V, C). conf channel unchanged."""
    arr = np.asarray(coco_txv, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != COCO17_V:
        raise ValueError(f"Expected (T,{COCO17_V},C), got {arr.shape}")
    out = arr.copy()
    xy = out[..., :2]
    mid_hip = 0.5 * (xy[:, L_HIP, :] + xy[:, R_HIP, :])  # (T, 2)
    xy = xy - mid_hip[:, None, :]
    shoulder = xy[:, L_SHO, :] - xy[:, R_SHO, :]
    widths = np.linalg.norm(shoulder, axis=-1)  # (T,)
    scale = np.maximum(widths, eps)[:, None, None]
    xy = xy / scale
    out[..., :2] = xy
    return out


def pad_or_resample_t(coco_txv: np.ndarray, t_out: int = DEFAULT_T) -> np.ndarray:
    """Pad (repeat last) or linearly resample time to ``t_out`` frames."""
    arr = np.asarray(coco_txv, dtype=np.float64)
    t = arr.shape[0]
    if t == t_out:
        return arr
    if t == 0:
        raise ValueError("Empty clip")
    if t == 1:
        return np.repeat(arr, t_out, axis=0)
    # Linear index remap
    src = np.linspace(0, t - 1, t_out)
    idx0 = np.floor(src).astype(int)
    idx1 = np.minimum(idx0 + 1, t - 1)
    w = (src - idx0).astype(np.float64)[:, None, None]
    return (1.0 - w) * arr[idx0] + w * arr[idx1]


def preprocess_tvx_to_nctv(
    coco_txv: np.ndarray | Sequence,
    *,
    t_out: int = DEFAULT_T,
    normalize: bool = True,
) -> np.ndarray:
    """(T,V,C) → (1,C,T,V) float32 ready for the probe / ONNX."""
    arr = np.asarray(coco_txv, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"Expected (T,V,C), got {arr.shape}")
    if arr.shape[1] != COCO17_V:
        raise ValueError(f"Expected V={COCO17_V}, got {arr.shape[1]}")
    if arr.shape[2] < 2:
        raise ValueError(f"Need at least x,y channels, got C={arr.shape[2]}")
    if arr.shape[2] == 2:
        conf = np.ones((*arr.shape[:2], 1), dtype=np.float64)
        arr = np.concatenate([arr, conf], axis=-1)
    arr = pad_or_resample_t(arr, t_out)
    if normalize:
        arr = root_center_scale(arr)
    # T×V×C → N×C×T×V
    nctv = np.transpose(arr[None, ...], (0, 3, 1, 2)).astype(np.float32)
    return nctv


class SkeletonProbe(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Temporal average-pool + 2-layer MLP on flattened joint xy.

    Intentionally tiny so CPU train + ONNX export stay under ~2 minutes.
    Subclasses ``nn.Module`` when torch is installed; constructing it without
    torch raises ImportError so the API process can still import this module.
    """

    def __init__(
        self,
        *,
        num_classes: int = NUM_CLASSES,
        t: int = DEFAULT_T,
        v: int = COCO17_V,
        hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        _require_torch()
        super().__init__()
        self.num_classes = num_classes
        self.t = t
        self.v = v
        # Per-frame joint feature: V * XY
        feat_dim = v * XY_CHANNELS
        self.temporal = nn.Sequential(
            nn.Conv1d(feat_dim, hidden, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C, T, V) — uses first two channels (x, y)."""
        if x.ndim != 4:
            raise ValueError(f"Expected NCHW-like N×C×T×V, got {tuple(x.shape)}")
        n, c, t, v = x.shape
        if v != self.v:
            raise ValueError(f"Expected V={self.v}, got {v}")
        xy = x[:, :XY_CHANNELS, :, :]  # N,2,T,V
        # N,2,T,V → N, T, V*2 → N, V*2, T for Conv1d
        feat = xy.permute(0, 2, 3, 1).contiguous().view(n, t, v * XY_CHANNELS)
        feat = feat.permute(0, 2, 1)  # N, F, T
        h = self.temporal(feat)  # N, hidden, 1
        return self.head(h)


def nctv_to_torch(nctv: np.ndarray, device: torch.device | str | None = None) -> torch.Tensor:
    _torch, _ = _require_torch()
    t = _torch.from_numpy(np.asarray(nctv, dtype=np.float32))
    if device is not None:
        t = t.to(device)
    return t
