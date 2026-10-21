# models/pose

OpenCV 5 DNN weights for the shooting video path (`POST /analyze/shot/video`).

These files are **not vendored**. MoveNet and YOLO-pose checkpoints are large and carry their own licenses. The scoring API still runs from keypoints without them. The live Render image does not load this directory.

| File | Family | Input |
|------|--------|--------|
| `movenet_lightning.onnx` | MoveNet single-pose | 192, NHWC, pixels 0–255, keypoints `(y, x, score)` |
| `movenet_thunder.onnx` | MoveNet single-pose | 256, same layout |
| `yolo_pose_n.onnx` | YOLO-pose n | 640, NCHW, RGB 0–1, letterboxed |

Set `AIFC_POSE_ONNX` to an absolute path if the file lives somewhere else.

A sibling JSON file overrides the filename guess:

```json
{
  "family": "movenet",
  "input_size": 192,
  "layout": "nhwc",
  "scale": 1.0,
  "coord_order": "yx",
  "letterbox": false
}
```

Export notes:

- MoveNet Lightning / Thunder: TensorFlow Hub single-pose, then ONNX. Output must squeeze to `(17, 3)` in COCO order.
- YOLO-pose: `yolo export model=yolo11n-pose.pt format=onnx`. Output row is `4 box + 1 score + 17×3`.

`cv2.dnn.readNetFromONNX` prefers `ENGINE_NEW` on OpenCV 5.0.0.93 and falls through to other engines if that graph cannot load the file. The response records the engine that actually loaded.

`tests/fixtures/constant_movenet.onnx` is a tiny constant graph so tests can load `ENGINE_NEW` without downloading weights. It is not a pose estimator.

MediaPipe is not a substitute. It runs only when `AIFC_ALLOW_MP_FALLBACK=1` and this ONNX file is missing. Do not describe that fallback as the competition pose backend.

TODO(PR-B): the skeleton overlay from `render_skeleton_bgr` is uploaded next to the clip in S3. It is not written to the Render disk.
