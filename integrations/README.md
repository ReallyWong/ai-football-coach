# Integrations

How this owned product sits on top of forks / upstream. **Do not vendor entire forks here** — document adapters and pin versions; adapters stay in-repo under `pipeline/adapters/`.

| Role | Repo | Pin | In v1 main path? | Interface |
|------|------|-----|------------------|-----------|
| Pose front-end (primary) | [ReallyWong/MotionDiff](https://github.com/ReallyWong/MotionDiff) | TBD (commit/tag) | **Yes** | Export JSON / keypoints per `docs/pose_contract.md` (MP-33 → COCO-17) |
| ST-GCN++ train/infer | [ReallyWong/pyskl](https://github.com/ReallyWong/pyskl) *(intended fork; may not exist yet — create from [kennymckormick/pyskl](https://github.com/kennymckormick/pyskl))* | TBD (commit/tag) | **Yes** | ST-GCN++ with `graph=coco`, `num_classes` from `labels/taxonomy.json`; serve `POST /analyze` |
| Sports CV / ROI helpers | [ReallyWong/sports](https://github.com/ReallyWong/sports) | TBD | Optional | Detection / tracking helpers only — **not** the classifier |
| Annotation style | [ReallyWong/SoccerNetv2-DevKit](https://github.com/ReallyWong/SoccerNetv2-DevKit) | TBD | Optional | Label / eval patterns only |
| WiFi DensePose | [ReallyWong/RuView](https://github.com/ReallyWong/RuView) | TBD | **No — post-M3 optional** | Optional non-camera pose; dashed path in architecture |

## Main path (v1)

```
Camera / video → MediaPipe (MotionDiff export) → COCO-17 adapter → ST-GCN++ (pyskl) → rules dual-track → coach-ui
```

See also: [`motiondiff.md`](motiondiff.md), [`pyskl.md`](pyskl.md), `docs/pose_contract.md`.
