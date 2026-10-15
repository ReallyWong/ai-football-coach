# models/

Runtime artifacts for M2 analyze:

| File | Produced by | Used by |
|------|-------------|---------|
| `youth_football_probe.pt` | `scripts/train_linear_probe.py` | `scripts/export_onnx.py` |
| `youth_football_probe.onnx` | `scripts/export_onnx.py` | `services/api/app.py` |
| `youth_football_probe.meta.json` | train script | debugging |

The `.onnx` file is committed so `POST /analyze` works out of the box.
Re-train anytime on CPU; see `docs/m2_model.md`.
