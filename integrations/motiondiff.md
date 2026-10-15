# MotionDiff (pose front-end)

Repo: https://github.com/ReallyWong/MotionDiff  
Pin: **TBD** (commit/tag when locked)

## Reuse

- **Export tool** path that emits MediaPipe Pose landmarks (or already-remapped COCO-17) as clip JSON / arrays.
- **POSE_DATA_CONTRACT ideas** — tensor layout, fps, confidence channel — align with `docs/pose_contract.md` in this repo (COCO-17 is authoritative for train/infer).

## Do not import

- Gradio squat (or other fitness-demo) product UI
- Whole MotionDiff app tree as a vendored submodule of the coach product
- Treating MotionDiff’s MediaPipe indices as COCO indices

## Interface

MotionDiff → this repo:

1. Export keypoints + clip metadata matching `pipeline/schemas/pose_clip.schema.json`
2. If export is MP-33, remap via `pipeline/adapters` using the table in `docs/pose_contract.md`
3. Downstream: COCO-17 → pyskl ST-GCN++

MotionDiff is on the **v1 main path** as the primary pose front-end.
