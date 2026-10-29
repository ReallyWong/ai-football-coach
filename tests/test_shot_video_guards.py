"""Single-flight and clip caps for POST /analyze/shot/video on a 512Mi instance.

No live Render and no network. OpenCV cases skip when requirements-comp.txt
is not installed.
"""
from __future__ import annotations

import gc
import sys
import threading
import weakref
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.pose.shot_clip_limits import (  # noqa: E402
    MAX_SHOT_BYTES,
    MAX_SHOT_LONG_SIDE,
    MAX_SHOT_SECONDS,
    MAX_SHOT_SHORT_SIDE,
    ShotClipLimitError,
    duration_over_limit,
    release_shot_memory,
    resolution_over_limit,
)
from pipeline.shot.synthetic import make_shot_coco17  # noqa: E402
from services.api.app import SHOT_VIDEO_BUSY_DETAIL, _shot_video_lock  # noqa: E402
from services.api.shot_s3 import ShotStorageError, download_shot_object  # noqa: E402

BUCKET = "nati-shot-coach-660008586940"


def test_demo_caps_allow_a_short_720p_clip_and_reject_the_rest():
    assert MAX_SHOT_BYTES == 20 * 1024 * 1024
    assert MAX_SHOT_SECONDS == 12.0
    assert MAX_SHOT_LONG_SIDE == 1280
    assert MAX_SHOT_SHORT_SIDE == 720
    assert resolution_over_limit(1280, 720) is False
    assert resolution_over_limit(720, 1280) is False
    assert resolution_over_limit(1920, 1080) is True
    assert resolution_over_limit(1080, 1920) is True
    assert resolution_over_limit(1280, 721) is True
    assert duration_over_limit(12.0) is False
    assert duration_over_limit(12.5) is False
    assert duration_over_limit(12.6) is True
    release_shot_memory()


def test_upload_over_the_byte_cap_is_413_and_does_not_decode(monkeypatch):
    monkeypatch.setattr("services.api.app.MAX_SHOT_BYTES", 32)

    def _boom(*_args, **_kwargs):
        raise AssertionError("over-cap upload must not be decoded")

    monkeypatch.setattr("pipeline.pose.opencv_dnn_pose.video_to_coco17_opencv", _boom)

    from fastapi.testclient import TestClient
    from services.api.app import app

    client = TestClient(app)
    response = client.post(
        "/analyze/shot/video",
        files={"file": ("clip.mp4", b"x" * 64, "video/mp4")},
        data={"max_seconds": "10"},
    )
    assert response.status_code == 413
    assert response.json()["detail"] == "Shot clip is larger than 20 MB."


def test_empty_upload_is_400(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise AssertionError("empty upload must not be decoded")

    monkeypatch.setattr("pipeline.pose.opencv_dnn_pose.video_to_coco17_opencv", _boom)

    from fastapi.testclient import TestClient
    from services.api.app import app

    client = TestClient(app)
    response = client.post(
        "/analyze/shot/video",
        files={"file": ("clip.mp4", b"", "video/mp4")},
        data={"max_seconds": "10"},
    )
    assert response.status_code == 400


def test_clip_limit_error_maps_to_413(monkeypatch):
    def _limited(*_args, **_kwargs):
        raise ShotClipLimitError(
            "Shot clip resolution 1920x1080 exceeds the 720p limit "
            "(long side ≤ 1280, short side ≤ 720)."
        )

    monkeypatch.setattr("pipeline.pose.opencv_dnn_pose.video_to_coco17_opencv", _limited)

    from fastapi.testclient import TestClient
    from services.api.app import app

    client = TestClient(app)
    response = client.post(
        "/analyze/shot/video",
        files={"file": ("clip.mp4", b"fake-video", "video/mp4")},
        data={"max_seconds": "10"},
    )
    assert response.status_code == 413
    body = response.json()
    assert "1920x1080" in body["detail"]
    assert "720p" in body["detail"]


def test_second_shot_video_is_429_while_one_is_running(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    coco = make_shot_coco17(lean="upright", plant="good")

    def _block(_path, **_kwargs):
        started.set()
        assert release.wait(5)
        return coco, 30.0, {
            "frame_wh": [320, 240],
            "opencv_version": "5.0.0",
            "pose_backend": "dnn_movenet",
            "frames": int(coco.shape[0]),
            "opencv_compliant": True,
            "dnn_engine": "ENGINE_NEW",
            "quality": {
                "blur_var": 80.0,
                "brightness": 120.0,
                "frames": int(coco.shape[0]),
            },
        }

    monkeypatch.setattr("pipeline.pose.opencv_dnn_pose.video_to_coco17_opencv", _block)

    from fastapi.testclient import TestClient
    from services.api.app import app

    results: list[tuple[str, int, dict]] = []

    def _post(tag: str) -> None:
        client = TestClient(app)
        response = client.post(
            "/analyze/shot/video",
            files={"file": ("clip.mp4", b"fake-video", "video/mp4")},
            data={"max_seconds": "10"},
        )
        results.append((tag, response.status_code, response.json()))

    first = threading.Thread(target=_post, args=("first",))
    second = threading.Thread(target=_post, args=("second",))
    first.start()
    assert started.wait(5), "first analysis did not start"
    second.start()
    second.join(3)
    try:
        assert not second.is_alive(), "second caller blocked instead of returning 429"
    finally:
        release.set()
        first.join(5)
        second.join(5)

    by_tag = {tag: (status, body) for tag, status, body in results}
    assert by_tag["second"][0] == 429
    assert by_tag["second"][1]["detail"] == SHOT_VIDEO_BUSY_DETAIL
    assert by_tag["first"][0] == 200, by_tag["first"][1]
    assert by_tag["first"][1]["agent"]["next_action"] == "accept_and_coach"


def test_busy_slot_rejects_s3_video_and_still_scores_keypoints(monkeypatch):
    monkeypatch.setenv("SHOT_BUCKET", BUCKET)

    def _boom(*_args, **_kwargs):
        raise AssertionError("busy slot must not download or decode")

    monkeypatch.setattr("services.api.app.download_shot_object", _boom)
    monkeypatch.setattr("pipeline.pose.opencv_dnn_pose.video_to_coco17_opencv", _boom)

    from fastapi.testclient import TestClient
    from services.api.app import app

    client = TestClient(app)
    assert _shot_video_lock.acquire(blocking=False)
    try:
        busy = client.post("/analyze/shot", json={"s3_key": "uploads/demo.mp4"})
        assert busy.status_code == 429
        assert busy.json()["detail"] == SHOT_VIDEO_BUSY_DETAIL

        upright = make_shot_coco17(lean="upright", plant="good")
        keyed = client.post(
            "/analyze/shot",
            json={"keypoints": upright.tolist(), "topology": "coco17", "fps": 30},
        )
        assert keyed.status_code == 200, keyed.text
        assert keyed.json()["agent"]["next_action"] == "accept_and_coach"
    finally:
        _shot_video_lock.release()


def test_s3_download_rejects_objects_over_the_demo_cap(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SHOT_BUCKET", BUCKET)

    class FakeClient:
        def head_object(self, Bucket, Key):
            assert Key == "uploads/demo.mp4"
            return {"ContentLength": MAX_SHOT_BYTES + 1}

        def download_file(self, Bucket, Key, Filename):
            raise AssertionError("over-cap object should not be downloaded")

    monkeypatch.setattr("services.api.shot_s3._s3_client", lambda region: FakeClient())
    with pytest.raises(ShotStorageError) as caught:
        download_shot_object("uploads/demo.mp4", tmp_path / "clip.mp4")
    assert caught.value.status_code == 413
    assert caught.value.detail == "Shot clip is larger than 20 MB."


def _opencv_installed() -> bool:
    try:
        import cv2  # noqa: F401
    except ImportError:
        return False
    return True


def _write_clip(path: Path, frames: list[np.ndarray], fps: float = 30.0) -> Path:
    import cv2

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


def _forward(frame: np.ndarray) -> np.ndarray:
    return make_shot_coco17(lean="back", plant="good", t=1)[0]


@pytest.mark.skipif(not _opencv_installed(), reason="opencv-python-headless is not installed")
def test_streaming_pose_drops_frames_and_matches_sampled_quality(tmp_path: Path, monkeypatch):
    from pipeline.pose import opencv_dnn_pose as pose
    from pipeline.pose.quality_gates import quality_gates

    real_decode = pose.decode_video_frames

    def _buffer(*_args, **_kwargs):
        raise AssertionError("shot scoring must not materialize every frame")

    monkeypatch.setattr(pose, "decode_video_frames", _buffer)

    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8) for _ in range(8)]
    path = _write_clip(tmp_path / "shot.avi", frames)
    held: list[weakref.ReferenceType[np.ndarray]] = []

    def _watch(frame: np.ndarray) -> np.ndarray:
        held.append(weakref.ref(frame))
        return _forward(frame)

    coco, _fps, meta = pose.video_to_coco17_opencv(
        path,
        forward_fn=_watch,
        max_frames=8,
        max_seconds=8 / 30,
        allow_mediapipe_fallback=False,
    )
    assert coco.shape[0] >= 1
    assert meta["pose_backend"] == "injected"
    assert meta["quality"]["frames"] == coco.shape[0]
    gc.collect()
    assert held, "forward was not called"
    assert all(ref() is None for ref in held)

    decoded, _decoded_fps = real_decode(path, max_frames=8, max_seconds=8 / 30)
    batched = quality_gates(decoded)
    assert meta["quality"]["blur_var"] == pytest.approx(batched["blur_var"])
    assert meta["quality"]["brightness"] == pytest.approx(batched["brightness"])
    del decoded
    gc.collect()


@pytest.mark.skipif(not _opencv_installed(), reason="opencv-python-headless is not installed")
def test_over_cap_resolution_duration_and_bytes_skip_pose(tmp_path: Path, monkeypatch):
    from pipeline.pose.opencv_dnn_pose import video_to_coco17_opencv

    calls = {"n": 0}

    def _count(frame: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        return _forward(frame)

    hd = np.zeros((1080, 1920, 3), dtype=np.uint8)
    hd_path = _write_clip(tmp_path / "hd.avi", [hd])
    with pytest.raises(ShotClipLimitError, match="1920x1080"):
        video_to_coco17_opencv(hd_path, forward_fn=_count, allow_mediapipe_fallback=False)
    assert calls["n"] == 0

    small = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(4)]
    short = _write_clip(tmp_path / "short.avi", small)
    monkeypatch.setattr(
        "pipeline.pose.opencv_dnn_pose._metadata_duration_seconds",
        lambda _cap, _cv2, _fps: 30.0,
    )
    with pytest.raises(ShotClipLimitError, match="30.0s"):
        video_to_coco17_opencv(short, forward_fn=_count, allow_mediapipe_fallback=False)
    assert calls["n"] == 0

    monkeypatch.setattr(
        "pipeline.pose.opencv_dnn_pose._metadata_duration_seconds",
        lambda _cap, _cv2, _fps: None,
    )
    # The scan bound is read from the pose module; the predicate reads the limits module.
    for module in (
        "pipeline.pose.opencv_dnn_pose",
        "pipeline.pose.shot_clip_limits",
    ):
        monkeypatch.setattr(f"{module}.MAX_SHOT_SECONDS", 0.05)
        monkeypatch.setattr(f"{module}.SHOT_DURATION_SLACK_SECONDS", 0.0)
    with pytest.raises(ShotClipLimitError, match="exceeds the 0s limit"):
        video_to_coco17_opencv(short, forward_fn=_count, allow_mediapipe_fallback=False)
    assert calls["n"] == 0

    monkeypatch.undo()
    monkeypatch.setattr("pipeline.pose.opencv_dnn_pose.MAX_SHOT_BYTES", 8)
    with pytest.raises(ShotClipLimitError, match="20 MB"):
        video_to_coco17_opencv(short, forward_fn=_count, allow_mediapipe_fallback=False)
    assert calls["n"] == 0


@pytest.mark.skipif(not _opencv_installed(), reason="opencv-python-headless is not installed")
def test_720p_and_unknown_duration_still_pose_one_frame_at_a_time(tmp_path: Path, monkeypatch):
    """A legal 720p clip is scored, including when the container has no duration."""
    from pipeline.pose.opencv_dnn_pose import video_to_coco17_opencv

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[100:400, 200:600] = (20, 180, 40)
    path = _write_clip(tmp_path / "720p.avi", [frame, frame])
    calls = {"n": 0}

    def _count(image: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        assert image.shape == (720, 1280, 3)
        return _forward(image)

    coco, _fps, meta = video_to_coco17_opencv(
        path,
        forward_fn=_count,
        max_frames=8,
        max_seconds=1,
        allow_mediapipe_fallback=False,
    )
    assert calls["n"] == coco.shape[0] >= 1
    assert meta["frame_wh"] == [1280, 720]

    calls["n"] = 0
    monkeypatch.setattr(
        "pipeline.pose.opencv_dnn_pose._metadata_duration_seconds",
        lambda _cap, _cv2, _fps: None,
    )
    coco, _fps, meta = video_to_coco17_opencv(
        path,
        forward_fn=_count,
        max_frames=8,
        max_seconds=1,
        allow_mediapipe_fallback=False,
    )
    assert calls["n"] == coco.shape[0] >= 1
    assert meta["frame_wh"] == [1280, 720]
