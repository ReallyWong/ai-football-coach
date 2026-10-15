# scripts/

## Offline demo (M1)

Geometric track only — no ST-GCN++ (that is M2).

```bash
# from repo root (numpy required)
pip install -r requirements-m1.txt
python scripts/run_offline_demo.py
```

Expected: prints `pass.inside.plant_foot` with Chinese feedback containing **支撑脚**.

Useful flags:

| Flag | Effect |
|------|--------|
| *(default)* | Load `fixtures/demo_pass_inside_pose.json` (bad plant foot) |
| `--good` | Synthetic upright plant foot — rule should **not** fire |
| `--pose-json PATH` | MotionDiff-like or coach clip JSON (`T×33` or `T×17`) |
| `--video PATH` | Requires optional `mediapipe`; fails clearly if missing |
| `--json-out` | Extra JSON summary line for UI binding checks |

```bash
python scripts/run_offline_demo.py --pose-json fixtures/demo_pass_inside_pose.json
python -m pytest tests/test_m1_plant_foot.py -q
```
