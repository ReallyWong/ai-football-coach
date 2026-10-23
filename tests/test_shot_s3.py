"""S3 key checks and the shot download path. No network and no real AWS keys."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.shot.synthetic import make_shot_coco17  # noqa: E402
from services.api.shot_s3 import (  # noqa: E402
    ShotStorageError,
    download_shot_object,
    validate_shot_s3_key,
)

PRESIGN_URL = "https://s14hcf6sc0.execute-api.ap-southeast-1.amazonaws.com/presign"
BUCKET = "nati-shot-coach-660008586940"


@pytest.mark.parametrize(
    "key",
    [
        "",
        "traces/demo.mp4",
        "uploads/../secrets.mp4",
        "/uploads/demo.mp4",
        "uploads/demo.mp4?X-Amz-Signature=abc",
        "https://s3.example/uploads/demo.mp4",
        "uploads/demo.txt",
        "uploads/nested/../../demo.mp4",
    ],
)
def test_validate_shot_s3_key_rejects_escape_and_non_video(key: str):
    with pytest.raises(ShotStorageError) as caught:
        validate_shot_s3_key(key)
    assert caught.value.status_code == 400


def test_validate_shot_s3_key_accepts_upload_object():
    assert validate_shot_s3_key(" uploads/abc.MP4 ") == "uploads/abc.MP4"


def test_download_requires_bucket_before_boto3(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("SHOT_BUCKET", raising=False)

    def _boom(_region: str):
        raise AssertionError("boto3 client should not be built without SHOT_BUCKET")

    monkeypatch.setattr("services.api.shot_s3._s3_client", _boom)
    with pytest.raises(ShotStorageError) as caught:
        download_shot_object("uploads/demo.mp4", tmp_path / "clip.mp4")
    assert caught.value.status_code == 503
    assert "SHOT_BUCKET" in caught.value.detail


def test_download_reports_missing_boto3(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SHOT_BUCKET", BUCKET)

    def _missing(_region: str):
        raise ShotStorageError(503, "boto3 is not installed on this service")

    monkeypatch.setattr("services.api.shot_s3._s3_client", _missing)
    with pytest.raises(ShotStorageError) as caught:
        download_shot_object("uploads/demo.mp4", tmp_path / "clip.mp4")
    assert "boto3" in caught.value.detail


def test_download_maps_missing_object_and_writes_bytes(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SHOT_BUCKET", BUCKET)
    monkeypatch.setenv("AWS_REGION", "ap-southeast-1")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
    dest = tmp_path / "clip.mp4"

    class FakeClient:
        def head_object(self, Bucket, Key):
            assert Bucket == BUCKET
            assert Key == "uploads/demo.webm"
            return {"ContentLength": 4}

        def download_file(self, Bucket, Key, Filename):
            Path(Filename).write_bytes(b"clip")

    monkeypatch.setattr("services.api.shot_s3._s3_client", lambda region: FakeClient())
    assert download_shot_object("uploads/demo.webm", dest) == "uploads/demo.webm"
    assert dest.read_bytes() == b"clip"

    class Missing:
        def head_object(self, Bucket, Key):
            raise FakeNotFound()

    class FakeNotFound(Exception):
        response = {"Error": {"Code": "NoSuchKey"}, "ResponseMetadata": {"HTTPStatusCode": 404}}

    monkeypatch.setattr("services.api.shot_s3._s3_client", lambda region: Missing())
    with pytest.raises(ShotStorageError) as caught:
        download_shot_object("uploads/missing.mp4", tmp_path / "missing.mp4")
    assert caught.value.status_code == 404
    assert "EXAMPLEKEY" not in caught.value.detail


def test_analyze_shot_s3_scores_downloaded_clip(monkeypatch):
    monkeypatch.setenv("SHOT_BUCKET", BUCKET)
    monkeypatch.setenv("AWS_REGION", "ap-southeast-1")
    # Public AWS docs examples, not a real account. They must not echo in the response.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
    seen: dict[str, object] = {}

    def fake_download(s3_key: str, dest: Path) -> str:
        seen["key"] = s3_key
        dest.write_bytes(b"not-a-real-video")
        return s3_key

    coco = make_shot_coco17(lean="back", plant="good")

    def fake_video(path, **kwargs):
        assert Path(path).read_bytes() == b"not-a-real-video"
        assert kwargs["max_seconds"] == 10.0
        return coco, 30.0, {
            "frame_wh": [320, 360],
            "opencv_version": "5.0.0",
            "pose_backend": "dnn_movenet",
            "frames": int(coco.shape[0]),
            "opencv_compliant": True,
            "dnn_engine": "ENGINE_NEW",
            "quality": {"blur_var": 80.0, "brightness": 120.0, "frames": int(coco.shape[0])},
        }

    monkeypatch.setattr("services.api.app.download_shot_object", fake_download)
    monkeypatch.setattr("pipeline.pose.opencv_dnn_pose.video_to_coco17_opencv", fake_video)

    from fastapi.testclient import TestClient
    from services.api.app import app

    client = TestClient(app)
    response = client.post(
        "/analyze/shot",
        json={"s3_key": "uploads/demo.mp4", "keypoints": [[[0, 0, 0]]], "max_seconds": 10},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert seen["key"] == "uploads/demo.mp4"
    assert body["label_id"] == "shot.instep"
    assert body["source"]["type"] == "s3"
    assert body["source"]["s3_key"] == "uploads/demo.mp4"
    assert body["source"]["bucket"] == BUCKET
    assert body["agent"]["next_action"] == "assign_drill"
    assert body["agent"]["drill_id"] == "drill.lean_wall"
    assert body["coaching_zh"]
    assert "AKIA" not in json.dumps(body)
    assert "EXAMPLEKEY" not in json.dumps(body)

    health = client.get("/health")
    assert health.status_code == 200
    s3 = health.json()["shot"]["s3"]
    assert s3["bucket_configured"] is True
    assert s3["bucket"] == BUCKET
    assert s3["region"] == "ap-southeast-1"
    assert s3["credentials_configured"] is True
    assert "EXAMPLEKEY" not in json.dumps(s3)


def test_analyze_shot_s3_opencv_missing_is_503(monkeypatch):
    monkeypatch.delenv("SHOT_BUCKET", raising=False)
    monkeypatch.setenv("SHOT_BUCKET", BUCKET)

    def fake_download(s3_key: str, dest: Path) -> str:
        dest.write_bytes(b"clip")
        return s3_key

    def fake_video(path, **kwargs):
        raise ImportError(
            "OpenCV is not installed on this service. Video shot analysis needs requirements-comp.txt"
        )

    monkeypatch.setattr("services.api.app.download_shot_object", fake_download)
    monkeypatch.setattr("pipeline.pose.opencv_dnn_pose.video_to_coco17_opencv", fake_video)

    from fastapi.testclient import TestClient
    from services.api.app import app

    client = TestClient(app)
    response = client.post("/analyze/shot", json={"s3_key": "uploads/demo.mp4"})
    assert response.status_code == 503
    assert "OpenCV" in response.json()["detail"]

    fundamentals = client.post(
        "/analyze",
        json={"pose_path": "fixtures/demo_pass_inside_pose.json", "label_id": "pass.inside"},
    )
    assert fundamentals.status_code == 200, fundamentals.text
    assert fundamentals.json()["label_id"] == "pass.inside"


def test_coach_ui_shot_upload_uses_live_presign():
    html = (REPO_ROOT / "coach-ui" / "index.html").read_text(encoding="utf-8")
    assert PRESIGN_URL in html
    assert "s3_key" in html
    assert "request_re_record" in html
    assert "/analyze/shot" in html
    assert "AKIA" not in html
    assert "AWS_SECRET" not in html
    docs = (REPO_ROOT / "docs" / "deploy_aws.md").read_text(encoding="utf-8")
    assert PRESIGN_URL in docs
    assert BUCKET in docs
    assert "SHOT_BUCKET" in docs
    assert "AWS_ACCESS_KEY_ID" in docs
    assert "Bedrock" in docs
