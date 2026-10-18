# scripts/

## Offline demo (M1)

Geometric track only.

```bash
pip install -r requirements-m1.txt
python scripts/run_offline_demo.py
```

Expected: prints `pass.inside.plant_foot` with Chinese feedback containing **支撑脚**.

| Flag | Effect |
|------|--------|
| *(default)* | Load `fixtures/demo_pass_inside_pose.json` (bad plant foot) |
| `--good` | Synthetic upright plant foot — rule should **not** fire |
| `--pose-json PATH` | MotionDiff-like or coach clip JSON (`T×33` or `T×17`) |
| `--video PATH` | Requires optional `mediapipe` |
| `--json-out` | Extra JSON summary line |

## Train / export probe (M2)

```bash
pip install -r requirements-m2.txt
python scripts/train_linear_probe.py
python scripts/export_onnx.py
# or: python scripts/train_linear_probe.py --also-onnx
```

Writes `models/youth_football_probe.{pt,onnx,meta.json}`.

## Analyze API (M2)

```bash
bash scripts/run_analyze_api.sh
# GET  /health
# POST /analyze  {"pose_path":"fixtures/demo_pass_inside_pose.json","label_id":"pass.inside"}
```

```bash
python -m pytest tests/ -q
```

## Synthetic dataset + confusion eval (M3)

```bash
pip install -r requirements-m2.txt
pip install matplotlib   # PNG only
python scripts/generate_synthetic_dataset.py --n-per-class 80
python scripts/eval_confusion.py
```

Writes `datasets/youth_football_synth/manifest.jsonl` (+ gitignored `clips/**/*.npy`),
`reports/confusion_matrix.{json,png}`, `reports/classification_report.md`, and `docs/eval.md`.

Coach UI evidence jump:

```bash
python -m http.server 8765
# open http://127.0.0.1:8765/coach-ui/
```
