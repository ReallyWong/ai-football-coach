# pipeline/pose

Pose ingestion for the offline geometric path (M1).

| Module | Role |
|--------|------|
| `load_mediapipe_json.py` | Load MotionDiff-like per-frame / clip JSON → `(T,V,3)` |
| `quality_gates.py` | Blur, brightness, person size, joint coverage for a shot clip |
| `opencv_dnn_pose.py` | OpenCV 5 decode + MoveNet/YOLO-pose ONNX → COCO-17. Not imported by the Render image until a video request |

Supported inputs:

- Coach clip JSON with `keypoints` as `T×33×3` or `T×17×3`
- Clip with `frames[].people[].keypoints` (MotionDiff `mediapipe-pose-frame/1.0` style)
- Directory of `*_keypoints.json` frame files

Remap MP-33 → COCO-17 with `pipeline.adapters.mediapipe_to_coco17` (see `docs/pose_contract.md`).
Video decoding via MediaPipe is optional in `scripts/run_offline_demo.py` (`--video`).

The competition shot path decodes with OpenCV 5 and runs DNN pose. MediaPipe on that path is opt-in (`AIFC_ALLOW_MP_FALLBACK=1`) and only if the ONNX file is missing. Weights: `models/pose/README.md`. Dependency pin: `requirements-comp.txt`, not `requirements-deploy.txt`.
