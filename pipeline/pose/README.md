# pipeline/pose

Pose ingestion for the offline geometric path (M1).

| Module | Role |
|--------|------|
| `load_mediapipe_json.py` | Load MotionDiff-like per-frame / clip JSON → `(T,V,3)` |

Supported inputs:

- Coach clip JSON with `keypoints` as `T×33×3` or `T×17×3`
- Clip with `frames[].people[].keypoints` (MotionDiff `mediapipe-pose-frame/1.0` style)
- Directory of `*_keypoints.json` frame files

Remap MP-33 → COCO-17 with `pipeline.adapters.mediapipe_to_coco17` (see `docs/pose_contract.md`).
Video decoding via MediaPipe is optional in `scripts/run_offline_demo.py` (`--video`).
