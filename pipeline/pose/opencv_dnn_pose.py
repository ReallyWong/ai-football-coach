"""OpenCV 5 video decode, DNN pose, and skeleton overlay for shooting clips.

Primary pose backend is an ONNX keypoint model run with ``cv2.dnn``:

- MoveNet single-pose (Lightning 192 or Thunder 256), output ``(1, 1, 17, 3)``
- YOLO-pose (v8 / 11), output channels ``4 box + 1 score + 17×3``

Both decoders emit COCO-17 ``(T, 17, 3)`` as ``(x, y, conf)`` in pixel space
so the existing geometric rules and ONNX shot probe can stay unchanged.

MediaPipe is not imported unless ``AIFC_ALLOW_MP_FALLBACK=1`` and the ONNX
file is missing. The Render deploy image does not install OpenCV; callers
get a clear ImportError instead of a silent fallback.

Weights are not committed. ``Dockerfile.comp`` downloads MoveNet Lightning.
See ``models/pose/README.md``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from pipeline.pose.quality_gates import (
    measure_blur_brightness,
    merge_quality,
    pose_quality,
    quality_from_measurements,
    quality_gates,
)
from pipeline.pose.shot_clip_limits import (
    MAX_SHOT_BYTES,
    MAX_SHOT_SECONDS,
    SHOT_DURATION_SLACK_SECONDS,
    ShotClipLimitError,
    duration_limit_detail,
    duration_over_limit,
    release_shot_memory,
    resolution_limit_detail,
    resolution_over_limit,
    shot_too_large_detail,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
POSE_DIR = REPO_ROOT / "models" / "pose"
OPENCV_PIN = "5.0.0.93"

# Prefer the OpenCV 5 graph. Later names cover 5.0.0 vs newer 5.x renames.
_ENGINE_PREFERENCE = ("ENGINE_NEW", "ENGINE_OPENCV", "ENGINE_CLASSIC", "ENGINE_AUTO")

_KNOWN_NAMES = (
    "movenet_lightning.onnx",
    "movenet_thunder.onnx",
    "yolo_pose_n.onnx",
    "yolo11n-pose.onnx",
    "yolov8n-pose.onnx",
)

COCO_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


class PoseModelMissingError(FileNotFoundError):
    """Raised when the OpenCV DNN weights are absent and fallback is off."""


@dataclass(frozen=True)
class PoseModelSpec:
    family: str  # movenet | yolo_pose
    input_size: int
    layout: str  # nchw | nhwc
    scale: float
    coord_order: str  # yx (MoveNet) | xy (YOLO)
    letterbox: bool
    input_dtype: str = "float32"  # float32 | int32 (MoveNet Lightning ONNX)


def opencv_runtime_info() -> dict[str, Any]:
    """Version metadata. Imports cv2 only when called."""
    import cv2

    version = str(cv2.__version__)
    major = int(version.split(".", 1)[0])
    return {
        "version": version,
        "major": major,
        "compliant": major >= 5,
        "pinned": OPENCV_PIN,
    }


def find_pose_model(explicit: str | Path | None = None) -> Path | None:
    """Resolve a pose ONNX path. Does not import OpenCV."""
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None
    env = os.environ.get("AIFC_POSE_ONNX", "").strip()
    if env:
        path = Path(env)
        return path if path.is_file() else None
    for name in _KNOWN_NAMES:
        path = POSE_DIR / name
        if path.is_file():
            return path
    return None


def spec_for_path(path: Path) -> PoseModelSpec:
    """Read a sibling ``.json`` spec, otherwise infer from the file name."""
    sidecar = path.with_suffix(".json")
    if sidecar.is_file():
        import json

        data = json.loads(sidecar.read_text(encoding="utf-8"))
        family = str(data.get("family", "movenet"))
        return PoseModelSpec(
            family=family,
            input_size=int(data.get("input_size", 256 if "thunder" in path.name else 192)),
            layout=str(data.get("layout", "nchw" if family == "yolo_pose" else "nhwc")),
            scale=float(data.get("scale", 1.0 / 255.0 if family == "yolo_pose" else 1.0)),
            coord_order=str(data.get("coord_order", "xy" if family == "yolo_pose" else "yx")),
            letterbox=bool(data.get("letterbox", family == "yolo_pose")),
            input_dtype=str(data.get("input_dtype", "float32")),
        )
    name = path.name.lower()
    if "yolo" in name:
        return PoseModelSpec(
            family="yolo_pose",
            input_size=640,
            layout="nchw",
            scale=1.0 / 255.0,
            coord_order="xy",
            letterbox=True,
        )
    size = 256 if "thunder" in name else 192
    return PoseModelSpec(
        family="movenet",
        input_size=size,
        layout="nhwc",
        scale=1.0,
        coord_order="yx",
        letterbox=False,
    )


def preferred_engine_name(cv2_module: Any) -> str | None:
    """First OpenCV 5 DNN engine constant this build actually exposes."""
    dnn = getattr(cv2_module, "dnn", None)
    if dnn is None:
        return None
    for name in _ENGINE_PREFERENCE:
        if getattr(dnn, name, None) is not None:
            return name
    return None


def _read_net(path: Path) -> tuple[Any, str]:
    import cv2

    last_error: Exception | None = None
    tried: list[str] = []
    for name in _ENGINE_PREFERENCE:
        engine = getattr(cv2.dnn, name, None)
        if engine is None:
            continue
        tried.append(name)
        try:
            return cv2.dnn.readNetFromONNX(str(path), engine=engine), name
        except TypeError:
            try:
                return cv2.dnn.readNetFromONNX(str(path), engine), name
            except Exception as exc:  # noqa: BLE001 — try the next engine
                last_error = exc
        except Exception as exc:  # noqa: BLE001 — model may need a different engine
            last_error = exc
    try:
        return cv2.dnn.readNetFromONNX(str(path)), "default"
    except Exception as exc:
        last_error = exc
    raise RuntimeError(
        f"OpenCV {cv2.__version__} could not load ONNX pose model {path}. "
        f"Tried engines {tried}. Last error: {last_error}"
    )


def load_pose_net(model_path: str | Path) -> tuple[Any, PoseModelSpec, str]:
    """Load ``model_path`` with the OpenCV 5 DNN engine. Returns (net, spec, engine)."""
    path = Path(model_path)
    if not path.is_file():
        raise PoseModelMissingError(_missing_model_message(path))
    spec = spec_for_path(path)
    net, engine = _read_net(path)
    return net, spec, engine


def _missing_model_message(path: Path | None = None) -> str:
    where = f" ({path})" if path else ""
    return (
        "OpenCV DNN pose model not found"
        f"{where}. Place MoveNet or YOLO-pose ONNX at "
        "models/pose/movenet_lightning.onnx or set AIFC_POSE_ONNX. "
        "See models/pose/README.md. MediaPipe is only used when "
        "AIFC_ALLOW_MP_FALLBACK=1."
    )


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is not installed on this service. Video shot analysis needs "
            "requirements-comp.txt (opencv-python-headless=="
            f"{OPENCV_PIN}). The Render deploy image intentionally does not "
            "install it yet. POST /analyze/shot with keypoints still works."
        ) from exc
    return cv2


def decode_movenet_output(
    raw: np.ndarray,
    frame_w: int,
    frame_h: int,
    *,
    coord_order: str = "yx",
) -> np.ndarray:
    """MoveNet ``(1, 1, 17, 3)`` → COCO-17 pixels. Default order is ``(y, x, score)``."""
    arr = np.asarray(raw, dtype=np.float64)
    arr = np.squeeze(arr)
    if arr.shape == (3, 17):
        arr = arr.T
    if arr.shape != (17, 3):
        raise ValueError(f"MoveNet output must squeeze to (17, 3), got {np.asarray(raw).shape}")
    if coord_order == "yx":
        y_norm, x_norm, conf = arr[:, 0], arr[:, 1], arr[:, 2]
    elif coord_order == "xy":
        x_norm, y_norm, conf = arr[:, 0], arr[:, 1], arr[:, 2]
    else:
        raise ValueError(f"coord_order must be 'yx' or 'xy', got {coord_order!r}")
    out = np.zeros((17, 3), dtype=np.float64)
    out[:, 0] = x_norm * float(frame_w)
    out[:, 1] = y_norm * float(frame_h)
    out[:, 2] = conf
    return out


def decode_yolo_pose_output(
    raw: np.ndarray,
    frame_w: int,
    frame_h: int,
    input_size: int,
    geom: dict[str, Any] | None = None,
) -> np.ndarray:
    """Best YOLO-pose person → COCO-17 pixels. Empty detection returns conf 0."""
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"YOLO-pose output must be 2D after squeeze, got {np.asarray(raw).shape}")
    if arr.shape[0] in {51, 56, 57} or (arr.shape[0] <= 64 and arr.shape[1] >= arr.shape[0]):
        arr = arr.T
    if arr.shape[1] < 56:
        raise ValueError(f"YOLO-pose row width {arr.shape[1]} is below 56")
    scores = arr[:, 4]
    if not np.isfinite(scores).any() or float(np.nanmax(scores)) <= 0.0:
        return np.zeros((17, 3), dtype=np.float64)
    person = arr[int(np.nanargmax(scores))]
    kpts = person[5:56].reshape(17, 3).astype(np.float64, copy=True)
    kpts[:, :2] = _unmap_xy(kpts[:, :2], frame_w, frame_h, input_size, geom)
    return kpts


def _unmap_xy(
    xy: np.ndarray,
    frame_w: int,
    frame_h: int,
    input_size: int,
    geom: dict[str, Any] | None,
) -> np.ndarray:
    mode = (geom or {}).get("mode", "stretch")
    if mode == "letterbox":
        scale = float(geom["scale"]) if geom and geom.get("scale") else 1.0
        pad_x = float(geom.get("pad_x", 0.0)) if geom else 0.0
        pad_y = float(geom.get("pad_y", 0.0)) if geom else 0.0
        if scale == 0.0:
            return xy
        out = xy.copy()
        out[:, 0] = (xy[:, 0] - pad_x) / scale
        out[:, 1] = (xy[:, 1] - pad_y) / scale
        return out
    out = xy.copy()
    out[:, 0] = xy[:, 0] / float(input_size) * float(frame_w)
    out[:, 1] = xy[:, 1] / float(input_size) * float(frame_h)
    return out


def _letterbox(frame_bgr: np.ndarray, size: int) -> tuple[np.ndarray, dict[str, Any]]:
    import cv2

    height, width = frame_bgr.shape[:2]
    scale = min(size / max(height, 1), size / max(width, 1))
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    resized = cv2.resize(frame_bgr, (resized_w, resized_h))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - resized_w) // 2
    pad_y = (size - resized_h) // 2
    canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized
    return canvas, {"mode": "letterbox", "scale": scale, "pad_x": pad_x, "pad_y": pad_y}


def prepare_input(frame_bgr: np.ndarray, spec: PoseModelSpec) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a blob and the geometry needed to map keypoints back to the frame."""
    import cv2

    size = spec.input_size
    if spec.letterbox:
        canvas, geom = _letterbox(frame_bgr, size)
    else:
        canvas = cv2.resize(frame_bgr, (size, size))
        geom = {"mode": "stretch"}
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    if spec.input_dtype == "int32":
        # Xenova/TF Hub MoveNet Lightning declares an int32 image. OpenCV
        # rejects a float blob for that graph. Pixels stay 0–255.
        rgb = np.clip(rgb, 0, 255).astype(np.int32)
    elif spec.input_dtype == "float32":
        rgb = rgb.astype(np.float32)
        if spec.scale != 1.0:
            rgb *= np.float32(spec.scale)
    else:
        raise ValueError(f"input_dtype must be 'float32' or 'int32', got {spec.input_dtype!r}")
    if spec.layout == "nhwc":
        blob = rgb.reshape(1, size, size, 3)
    elif spec.layout == "nchw":
        blob = np.transpose(rgb, (2, 0, 1))[None, ...]
    else:
        raise ValueError(f"layout must be 'nchw' or 'nhwc', got {spec.layout!r}")
    return blob, geom


def infer_frame(net: Any, spec: PoseModelSpec, frame_bgr: np.ndarray) -> np.ndarray:
    """One BGR frame → ``(17, 3)`` pixel keypoints."""
    blob, geom = prepare_input(frame_bgr, spec)
    net.setInput(blob)
    raw = net.forward()
    height, width = frame_bgr.shape[:2]
    if spec.family == "movenet":
        return decode_movenet_output(raw, width, height, coord_order=spec.coord_order)
    if spec.family == "yolo_pose":
        return decode_yolo_pose_output(raw, width, height, spec.input_size, geom)
    raise ValueError(f"Unknown pose family {spec.family!r}")


def render_skeleton_bgr(frame_bgr: np.ndarray, keypoints_17x3: np.ndarray) -> np.ndarray:
    """Draw a COCO-17 skeleton. Returns a copy. Used later for the S3 overlay."""
    import cv2

    canvas = np.array(frame_bgr, copy=True)
    kps = np.asarray(keypoints_17x3, dtype=np.float64)
    if kps.shape[0] != 17:
        raise ValueError(f"Expected 17 joints, got {kps.shape}")
    for a, b in COCO_EDGES:
        if kps[a, 2] < 0.2 or kps[b, 2] < 0.2:
            continue
        if not np.isfinite(kps[a, 0]) or not np.isfinite(kps[b, 0]):
            continue
        p1 = (int(round(kps[a, 0])), int(round(kps[a, 1])))
        p2 = (int(round(kps[b, 0])), int(round(kps[b, 1])))
        cv2.line(canvas, p1, p2, (56, 189, 248), 2, cv2.LINE_AA)
    for joint in kps:
        if joint[2] < 0.2 or not np.isfinite(joint[0]):
            continue
        center = (int(round(joint[0])), int(round(joint[1])))
        cv2.circle(canvas, center, 3, (250, 204, 21), -1, cv2.LINE_AA)
    return canvas


def _capture_fps(cap: Any, cv2_module: Any) -> float:
    fps = float(cap.get(cv2_module.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(fps) or fps < 1.0:
        return 30.0
    return fps


def _metadata_duration_seconds(cap: Any, cv2_module: Any, fps: float) -> float | None:
    frame_count = float(cap.get(cv2_module.CAP_PROP_FRAME_COUNT) or 0.0)
    if not np.isfinite(frame_count) or frame_count <= 0 or fps <= 0:
        return None
    return frame_count / fps


def _frame_wh(frame: np.ndarray) -> tuple[int, int]:
    height, width = int(frame.shape[0]), int(frame.shape[1])
    if resolution_over_limit(width, height):
        raise ShotClipLimitError(resolution_limit_detail(width, height))
    return width, height


def _reject_if_too_long(seen_frames: int, fps: float) -> None:
    seconds = seen_frames / fps if fps > 0 else float(seen_frames)
    if duration_over_limit(seconds):
        raise ShotClipLimitError(duration_limit_detail(seconds))


def _scan_unknown_duration(
    cap: Any,
    cv2_module: Any,
    path: Path,
    fps: float,
    width: int,
    height: int,
) -> tuple[Any, int, int]:
    """Read-and-drop until the duration cap. Reopen so pose can start at frame 0.

    Used only when the container does not report a frame count. One frame is
    live at a time.
    """
    limit_frames = max(
        1,
        int(np.floor((MAX_SHOT_SECONDS + SHOT_DURATION_SLACK_SECONDS) * fps)),
    )
    seen = 0
    while seen <= limit_frames:
        ok, frame = cap.read()
        if not ok:
            break
        try:
            if width <= 0 or height <= 0:
                width, height = _frame_wh(frame)
        finally:
            del frame
        seen += 1
    else:
        # One frame past the cap is enough to reject. Do not read the rest.
        _reject_if_too_long(seen, fps)
    # Pose must start at frame 0. Reopen so a codec that cannot seek still works.
    # On failure the original capture stays open and the caller releases it.
    reopened = cv2_module.VideoCapture(str(path))
    if not reopened.isOpened():
        reopened.release()
        raise ValueError(f"Could not open video: {path}")
    cap.release()
    return reopened, width, height


def _open_limited_capture(path: Path) -> tuple[Any, float, int, int]:
    """Open ``path`` or raise ``ShotClipLimitError`` / ``ValueError``.

    Returns the capture positioned at the first frame, plus fps and frame size.
    Frame size may be ``(0, 0)`` when the container does not advertise it; the
    first decoded frame is checked later.
    """
    cv2 = _require_cv2()
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    if path.is_file() and path.stat().st_size > MAX_SHOT_BYTES:
        raise ShotClipLimitError(shot_too_large_detail())
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not open video: {path}")
    try:
        fps = _capture_fps(cap, cv2)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if resolution_over_limit(width, height):
            raise ShotClipLimitError(resolution_limit_detail(width, height))
        duration = _metadata_duration_seconds(cap, cv2, fps)
        if duration is None:
            cap, width, height = _scan_unknown_duration(cap, cv2, path, fps, width, height)
        elif duration_over_limit(duration):
            raise ShotClipLimitError(duration_limit_detail(duration))
        return cap, fps, width, height
    except Exception:
        cap.release()
        raise


def _sample_plan(fps: float, max_frames: int, max_seconds: float) -> tuple[int, int]:
    """Same stride as the previous full-buffer decoder. The file duration cap is separate."""
    max_to_consider = max(1, int(float(max_seconds) * fps))
    stride = max(1, int(np.ceil(max_to_consider / max(1, max_frames))))
    return max_to_consider, stride


def decode_video_frames(
    video_path: str | Path,
    *,
    max_frames: int = 120,
    max_seconds: float = 10.0,
) -> tuple[list[np.ndarray], float]:
    """Read up to ``max_seconds`` of video, subsampled to ``max_frames``.

    Materializes the sampled frames. The shot API does not use this; it poses
    one frame at a time. Kept for tests and callers that explicitly want a list,
    and it still refuses clips over the 512Mi caps.
    """
    path = Path(video_path)
    cap, fps, _width, _height = _open_limited_capture(path)
    max_to_consider, stride = _sample_plan(fps, max_frames, max_seconds)
    frames: list[np.ndarray] = []
    index = 0
    try:
        while index < max_to_consider and len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride == 0:
                _frame_wh(frame)
                frames.append(frame)
            else:
                del frame
            index += 1
    finally:
        cap.release()
    if not frames:
        raise ValueError("No frames decoded")
    return frames, fps


def _stream_pose_rows(
    cap: Any,
    *,
    fps: float,
    width: int,
    height: int,
    max_frames: int,
    max_seconds: float,
    forward: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, list[float], list[float], int, int]:
    """Pose and quality one sampled frame at a time. Does not keep the images."""
    max_to_consider, stride = _sample_plan(fps, max_frames, max_seconds)
    rows: list[np.ndarray] = []
    blurs: list[float] = []
    brights: list[float] = []
    index = 0
    while index < max_to_consider and len(rows) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        try:
            if index % stride == 0:
                width, height = _frame_wh(frame)
                row = np.asarray(forward(frame), dtype=np.float64)
                if row.shape != (17, 3):
                    raise ValueError(f"Pose forward must return (17, 3), got {row.shape}")
                blur, bright = measure_blur_brightness(frame)
                rows.append(row)
                blurs.append(blur)
                brights.append(bright)
        finally:
            del frame
        index += 1
    if not rows:
        raise ValueError("No frames decoded")
    return np.stack(rows, axis=0), blurs, brights, width, height


def analyze_bgr_frames(
    frames_bgr: list[np.ndarray],
    *,
    forward_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    model_path: str | Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Quality-gate frames and run DNN pose (or ``forward_fn`` in tests).

    Returns COCO-17 ``(T, 17, 3)`` and metadata. Does not score the shot.
    """
    if not frames_bgr:
        raise ValueError("No frames decoded")
    image_q = quality_gates(frames_bgr)
    engine_name = None
    backend = "injected" if forward_fn is not None else "dnn_pending"
    if forward_fn is None:
        resolved = find_pose_model(model_path)
        if resolved is None:
            raise PoseModelMissingError(_missing_model_message(Path(model_path) if model_path else None))
        net, spec, engine_name = load_pose_net(resolved)
        backend = f"dnn_{spec.family}"

        def forward_fn(frame: np.ndarray, _net=net, _spec=spec) -> np.ndarray:
            return infer_frame(_net, _spec, frame)

    rows: list[np.ndarray] = []
    for frame in frames_bgr:
        row = np.asarray(forward_fn(frame), dtype=np.float64)
        if row.shape != (17, 3):
            raise ValueError(f"Pose forward must return (17, 3), got {row.shape}")
        rows.append(row)
    coco = np.stack(rows, axis=0)
    height, width = frames_bgr[0].shape[:2]
    pose_q = pose_quality(coco, frame_wh=(width, height))
    info = opencv_runtime_info()
    meta: dict[str, Any] = {
        "opencv_version": info["version"],
        "opencv_compliant": info["compliant"],
        "quality": merge_quality(image_q, pose_q),
        "frames": int(coco.shape[0]),
        "pose_backend": backend,
        "dnn_engine": engine_name,
        "frame_wh": [int(width), int(height)],
    }
    return coco, meta


def _meta_for_pose(
    coco: np.ndarray,
    image_q: dict[str, Any],
    *,
    width: int,
    height: int,
    backend: str,
    engine_name: str | None,
    fps: float,
    max_seconds: float,
) -> dict[str, Any]:
    pose_q = pose_quality(coco, frame_wh=(width, height))
    info = opencv_runtime_info()
    return {
        "opencv_version": info["version"],
        "opencv_compliant": info["compliant"],
        "quality": merge_quality(image_q, pose_q),
        "frames": int(coco.shape[0]),
        "pose_backend": backend,
        "dnn_engine": engine_name,
        "frame_wh": [int(width), int(height)],
        "fps": fps,
        "max_seconds": float(max_seconds),
    }


def video_to_coco17_opencv(
    video_path: str | Path,
    *,
    model_path: str | Path | None = None,
    max_frames: int = 120,
    max_seconds: float = 10.0,
    forward_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    allow_mediapipe_fallback: bool | None = None,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Decode with OpenCV and estimate COCO-17 pose.

    Frames are posed one at a time and dropped before the next read, so a
    720p clip does not keep the whole sample list in RAM. Fallback to
    MediaPipe runs only when explicitly enabled and the ONNX file is missing.
    """
    if allow_mediapipe_fallback is None:
        allow_mediapipe_fallback = os.environ.get("AIFC_ALLOW_MP_FALLBACK", "") == "1"

    path = Path(video_path)
    cap = None
    net = None
    forward = forward_fn
    try:
        cap, fps, width, height = _open_limited_capture(path)
        engine_name = None
        if forward is None:
            resolved = find_pose_model(model_path)
            if resolved is None:
                cap.release()
                cap = None
                if not allow_mediapipe_fallback:
                    raise PoseModelMissingError(
                        _missing_model_message(Path(model_path) if model_path else None)
                    )
                return _mediapipe_fallback(
                    path,
                    max_frames=max_frames,
                    max_seconds=max_seconds,
                    frame_wh=(width, height),
                )
            net, spec, engine_name = load_pose_net(resolved)
            backend = f"dnn_{spec.family}"

            def forward(frame: np.ndarray, _net=net, _spec=spec) -> np.ndarray:
                return infer_frame(_net, _spec, frame)
        else:
            backend = "injected"

        coco, blurs, brights, width, height = _stream_pose_rows(
            cap,
            fps=fps,
            width=width,
            height=height,
            max_frames=max_frames,
            max_seconds=max_seconds,
            forward=forward,
        )
        meta = _meta_for_pose(
            coco,
            quality_from_measurements(blurs, brights),
            width=width,
            height=height,
            backend=backend,
            engine_name=engine_name,
            fps=fps,
            max_seconds=max_seconds,
        )
        del blurs, brights
        return coco, fps, meta
    finally:
        if cap is not None:
            cap.release()
        # Drop the closure that owns the DNN net before collection.
        forward = None
        net = None
        release_shot_memory()


def _stream_image_quality(
    video_path: str | Path,
    *,
    max_frames: int,
    max_seconds: float,
) -> tuple[dict[str, float | int], tuple[int, int]]:
    """Blur and brightness without retaining frames. One frame is live at a time."""
    cap, _fps, width, height = _open_limited_capture(Path(video_path))
    _max_to_consider, stride = _sample_plan(_fps, max_frames, max_seconds)
    blurs: list[float] = []
    brights: list[float] = []
    index = 0
    try:
        while index < _max_to_consider and len(blurs) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            try:
                if index % stride == 0:
                    width, height = _frame_wh(frame)
                    blur, bright = measure_blur_brightness(frame)
                    blurs.append(blur)
                    brights.append(bright)
            finally:
                del frame
            index += 1
    finally:
        cap.release()
    if not blurs:
        raise ValueError("No frames decoded")
    return quality_from_measurements(blurs, brights), (width, height)


def _mediapipe_fallback(
    video_path: str | Path,
    *,
    max_frames: int,
    max_seconds: float,
    frame_wh: tuple[int, int],
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Last-resort pose. MediaPipe reads the file again; OpenCV does not keep frames."""
    from pipeline.adapters.mediapipe_to_coco17 import mediapipe_to_coco17
    from pipeline.pose.video_to_mp33 import video_to_mp33

    mp33, fps = video_to_mp33(video_path, max_frames=max_frames)
    coco = mediapipe_to_coco17(mp33)
    image_q, measured_wh = _stream_image_quality(
        video_path,
        max_frames=max_frames,
        max_seconds=max_seconds,
    )
    width, height = measured_wh if measured_wh[0] and measured_wh[1] else frame_wh
    meta = _meta_for_pose(
        coco,
        image_q,
        width=int(width),
        height=int(height),
        backend="mediapipe_fallback",
        engine_name=None,
        fps=float(fps),
        max_seconds=max_seconds,
    )
    return coco, float(fps), meta
