#!/usr/bin/env bash
# Start the M2 analyze API (ONNX + geometric rules).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
ONNX="${AIFC_ONNX_PATH:-$ROOT/models/youth_football_probe.onnx}"
if [[ ! -f "$ONNX" ]]; then
  echo "ONNX missing at $ONNX — training + exporting first..."
  python scripts/train_linear_probe.py
  python scripts/export_onnx.py
fi
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
exec python -m uvicorn services.api.app:app --host "$HOST" --port "$PORT"
