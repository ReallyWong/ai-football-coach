# Pose contract (M0)

Canonical topology for training and inference: **COCO-17**.

## Sources

| Role | Source | Notes |
|------|--------|-------|
| Primary | MediaPipe Pose (33 landmarks) via MotionDiff export | Convert with the MP-33 → COCO-17 table below |
| Topology | COCO-17 | Do **not** treat MediaPipe indices as COCO indices |

RuView (WiFi DensePose) is **post-M3 optional** and is not on the v1 main path.

## Tensor layouts

| Stage | Layout | Channels |
|-------|--------|----------|
| On disk (clip file) | `T × V × C` | `C = (x, y, conf)` |
| Model input (ST-GCN++ / pyskl) | `N × C × T × V` | same channels |

Defaults:

- `fps = 30`
- `T = 64` frames (clip window)
- `V = 17` (COCO-17)

## Normalization

1. **Root-center**: subtract mid-hip `(left_hip + right_hip) / 2` from all joints.
2. **Scale**: divide by shoulder width `‖left_shoulder − right_shoulder‖` (with a small epsilon floor).

Apply after MP → COCO remap; keep `conf` unchanged by centering/scaling.

## COCO-17 joint index

| Index | Name |
|------:|------|
| 0 | nose |
| 1 | left_eye |
| 2 | right_eye |
| 3 | left_ear |
| 4 | right_ear |
| 5 | left_shoulder |
| 6 | right_shoulder |
| 7 | left_elbow |
| 8 | right_elbow |
| 9 | left_wrist |
| 10 | right_wrist |
| 11 | left_hip |
| 12 | right_hip |
| 13 | left_knee |
| 14 | right_knee |
| 15 | left_ankle |
| 16 | right_ankle |

## MediaPipe Pose 33 → COCO-17 mapping

MediaPipe landmark index → COCO-17 index. Unlisted MP landmarks are dropped (not in COCO-17).

| COCO-17 | Name | MediaPipe index | MP name |
|--------:|------|----------------:|---------|
| 0 | nose | 0 | nose |
| 1 | left_eye | 2 | left_eye |
| 2 | right_eye | 5 | right_eye |
| 3 | left_ear | 7 | left_ear |
| 4 | right_ear | 8 | right_ear |
| 5 | left_shoulder | 11 | left_shoulder |
| 6 | right_shoulder | 12 | right_shoulder |
| 7 | left_elbow | 13 | left_elbow |
| 8 | right_elbow | 14 | right_elbow |
| 9 | left_wrist | 15 | left_wrist |
| 10 | right_wrist | 16 | right_wrist |
| 11 | left_hip | 23 | left_hip |
| 12 | right_hip | 24 | right_hip |
| 13 | left_knee | 25 | left_knee |
| 14 | right_knee | 26 | right_knee |
| 15 | left_ankle | 27 | left_ankle |
| 16 | right_ankle | 28 | right_ankle |

**Forbidden:** using MediaPipe indices (0–32) as if they were COCO-17 indices (0–16). Always remap through this table (or the adapter in `pipeline/adapters/`).

## Clip annotation

See `pipeline/schemas/pose_clip.schema.json` for the on-disk clip metadata shape (`label_id`, `view`, `fps`, `T`, `V=17`, keypoints path, quality flags).
