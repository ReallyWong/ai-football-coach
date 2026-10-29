"""Fetch a browser-uploaded shot clip from S3.

boto3 is imported only when ``POST /analyze/shot`` carries ``s3_key``.
The live Render image (``requirements-deploy.txt``) does not install boto3
or OpenCV. Set ``SHOT_BUCKET`` on a contest service that installs
``requirements-comp.txt``. See ``docs/deploy_aws.md``.
"""
from __future__ import annotations

import os
from pathlib import Path

from pipeline.pose.shot_clip_limits import MAX_SHOT_BYTES, shot_too_large_detail
_PREFIX = "uploads/"
_ALLOWED_SUFFIXES = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
DEFAULT_REGION = "ap-southeast-1"


class ShotStorageError(Exception):
    """Caller-facing S3 failure. ``detail`` must not include credentials."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def shot_bucket() -> str:
    return os.environ.get("SHOT_BUCKET", "").strip()


def shot_region() -> str:
    region = (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or DEFAULT_REGION
    )
    return region.strip() or DEFAULT_REGION


def shot_s3_status() -> dict[str, object]:
    """Health hint. Booleans and the bucket name only — never key material."""
    try:
        import boto3  # noqa: F401
        boto3_present = True
    except ImportError:
        boto3_present = False
    bucket = shot_bucket()
    access_key = bool(os.environ.get("AWS_ACCESS_KEY_ID", "").strip())
    secret_key = bool(os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip())
    return {
        "bucket_configured": bool(bucket),
        "bucket": bucket or None,
        "region": shot_region(),
        "boto3_present": boto3_present,
        "credentials_configured": access_key and secret_key,
    }


def validate_shot_s3_key(s3_key: str) -> str:
    """Accept only an ``uploads/<file>`` object key, not a URL or a path escape."""
    key = (s3_key or "").strip()
    if not key or len(key) > 1024:
        raise ShotStorageError(400, "s3_key must be a non-empty uploads/ object key.")
    if "://" in key or "?" in key or "\\" in key or "\x00" in key:
        raise ShotStorageError(
            400,
            "s3_key must be an object key such as uploads/<id>.mp4, not a URL.",
        )
    if key.startswith("/"):
        raise ShotStorageError(400, "s3_key must be a relative uploads/ key.")
    parts = key.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ShotStorageError(400, "s3_key must not contain empty or '..' segments.")
    if not key.startswith(_PREFIX) or key.endswith("/"):
        raise ShotStorageError(400, "s3_key must name an object under uploads/.")
    suffix = Path(parts[-1]).suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        allowed = ", ".join(sorted(_ALLOWED_SUFFIXES))
        raise ShotStorageError(400, f"s3_key must end with one of: {allowed}.")
    return key


def _s3_client(region: str):
    try:
        import boto3
    except ImportError as exc:
        raise ShotStorageError(
            503,
            "boto3 is not installed on this service. S3 shot fetch belongs on a "
            "contest service that installs requirements-comp.txt. The live Render "
            "image (requirements-deploy.txt) intentionally omits boto3 and OpenCV. "
            "Keypoint POST /analyze/shot still works.",
        ) from exc
    return boto3.client("s3", region_name=region)


def _map_s3_exception(exc: Exception) -> tuple[int, str] | None:
    name = type(exc).__name__
    if name in {"NoCredentialsError", "PartialCredentialsError"}:
        return (
            503,
            "AWS credentials are not set. On the contest service set "
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (and AWS_SESSION_TOKEN "
            "when the key is temporary). Do not commit those values.",
        )
    if name == "EndpointConnectionError":
        return (502, "Could not reach S3. Check AWS_REGION (ap-southeast-1).")
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return None
    error = response.get("Error") or {}
    code = str(error.get("Code") or "")
    status = (response.get("ResponseMetadata") or {}).get("HTTPStatusCode")
    if code in {"404", "NoSuchKey", "NotFound"} or status == 404:
        return (404, "No shot object at that s3_key.")
    if code in {"403", "AccessDenied"} or status == 403:
        return (
            503,
            "S3 GetObject was denied. The contest credentials need s3:GetObject "
            "on SHOT_BUCKET/uploads/*.",
        )
    if code in {"InvalidAccessKeyId", "SignatureDoesNotMatch", "ExpiredToken", "InvalidToken"}:
        return (
            503,
            "AWS credentials were rejected. Check AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY on the contest service.",
        )
    if code:
        return (502, f"S3 request failed ({code}).")
    return None


def download_shot_object(s3_key: str, dest: Path) -> str:
    """HeadObject then GetObject into ``dest``. Returns the validated key.

    Uses the default boto3 credential chain (env vars, then a role if one exists).
    """
    key = validate_shot_s3_key(s3_key)
    bucket = shot_bucket()
    if not bucket:
        raise ShotStorageError(
            503,
            "SHOT_BUCKET is not set. On the contest Render service set "
            "SHOT_BUCKET to the shot bucket and AWS_REGION=ap-southeast-1. "
            "Leave the live fundamentals service unset until it installs "
            "requirements-comp.txt.",
        )
    client = _s3_client(shot_region())
    try:
        head = client.head_object(Bucket=bucket, Key=key)
        size = int(head.get("ContentLength") or 0)
        if size <= 0:
            raise ShotStorageError(400, "The S3 object is empty.")
        if size > MAX_SHOT_BYTES:
            raise ShotStorageError(413, shot_too_large_detail())
        client.download_file(bucket, key, str(dest))
    except ShotStorageError:
        raise
    except Exception as exc:
        mapped = _map_s3_exception(exc)
        if mapped is not None:
            raise ShotStorageError(*mapped) from exc
        raise ShotStorageError(502, "S3 GetObject failed.") from exc
    if not dest.is_file() or dest.stat().st_size <= 0:
        raise ShotStorageError(502, "S3 download produced an empty file.")
    if dest.stat().st_size > MAX_SHOT_BYTES:
        dest.unlink(missing_ok=True)
        raise ShotStorageError(413, shot_too_large_detail())
    return key
