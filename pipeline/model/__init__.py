"""Lightweight skeleton classifier head (M2 stand-in for ST-GCN++).

Same I/O contract as pyskl ST-GCN++: N×C×T×V → logits[num_classes].
Swap for real ST-GCN++ weights later — see docs/m2_model.md.
"""
from pipeline.model.skeleton_probe import (
    CLASS_IDS,
    COCO17_V,
    DEFAULT_T,
    NUM_CLASSES,
    SkeletonProbe,
    load_taxonomy_labels,
    preprocess_tvx_to_nctv,
    root_center_scale,
)

__all__ = [
    "CLASS_IDS",
    "COCO17_V",
    "DEFAULT_T",
    "NUM_CLASSES",
    "SkeletonProbe",
    "load_taxonomy_labels",
    "preprocess_tvx_to_nctv",
    "root_center_scale",
]
