"""Render image installs requirements-deploy.txt and does not include torch or OpenCV.

Importing the API must succeed in that environment. Train/export still require
torch and fail with ImportError only when SkeletonProbe is constructed.
Video shot pose imports cv2 only inside the video route.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_CHECK = r"""
import builtins

_real_import = builtins.__import__

def _guard(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".", 1)[0]
    if root in {"torch", "cv2"}:
        raise ModuleNotFoundError(f"No module named '{root}'")
    return _real_import(name, globals, locals, fromlist, level)

builtins.__import__ = _guard

from services.api.app import app
from pipeline.model.skeleton_probe import SkeletonProbe, preprocess_tvx_to_nctv
import numpy as np

paths = {getattr(route, "path", "") for route in app.routes}
needed = {"/health", "/analyze", "/modes", "/signature-moves"}
missing = needed - paths
if missing:
    raise SystemExit(f"missing routes: {sorted(missing)}")
if not any(path == "/coach-ui" or path.startswith("/coach-ui/") for path in paths):
    raise SystemExit(f"coach-ui mount missing from {sorted(paths)}")

clip = np.zeros((4, 17, 3), dtype=np.float64)
clip[:, 11, :2] = (95, 160)
clip[:, 12, :2] = (125, 160)
clip[:, 5, :2] = (90, 80)
clip[:, 6, :2] = (130, 80)
clip[..., 2] = 1
nctv = preprocess_tvx_to_nctv(clip)
if nctv.shape != (1, 3, 64, 17):
    raise SystemExit(f"unexpected preprocess shape {nctv.shape}")

try:
    SkeletonProbe()
except ImportError as exc:
    if "torch" not in str(exc).lower():
        raise SystemExit(f"expected torch ImportError, got: {exc}") from exc
else:
    raise SystemExit("SkeletonProbe() should fail when torch is missing")

print("ok")
"""


def test_api_imports_without_torch():
    proc = subprocess.run(
        [sys.executable, "-c", _CHECK],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
