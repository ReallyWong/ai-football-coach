# AI Football Coach

> Youth technical correction — camera → pose → ST-GCN++ scoring → coach feedback. No physical robot in v1.

**One-liner (EN)** — Camera/video → MediaPipe/MotionDiff → COCO-17 → ST-GCN++/pyskl → dual-track rules → coach-ui.  
**一句话定位（中）** — 摄像头/视频 → MediaPipe/MotionDiff → COCO-17 → ST-GCN++/pyskl → 双轨纠错规则 → 教练 UI。

## Product link

```
Camera / training video
        ↓
MediaPipe Pose (MotionDiff export)     [RuView = post-M3 optional]
        ↓
COCO-17 adapter  (docs/pose_contract.md)
        ↓
ST-GCN++ action scoring (pyskl)  — M2: lightweight ONNX probe stand-in
        ↓
Rules dual-track (geometric + model) + Coach UI  ← this repo (owned IP)
```

## Repo layout

| Path | What |
|------|------|
| `labels/` | Football technique taxonomy (`taxonomy.json` v0.2) |
| `rules/` | Dual-track correction rules (`rules.json` v0.2) |
| `pipeline/` | Pose loaders, MP→COCO-17 adapter, geometric metrics, rules engine, **model probe** |
| `models/` | ONNX / torch probe artifacts (`youth_football_probe.onnx`) |
| `services/api/` | FastAPI `POST /analyze` + `GET /health` |
| `scripts/` | Offline demo, train/export probe, run API |
| `docs/pose_contract.md` | COCO-17 topology, MP→COCO map, tensor layouts |
| `docs/m2_model.md` | Probe ↔ ST-GCN++ swap guide |
| `coach-ui/` | Coach feedback UI (scaffold) |
| `fixtures/` | Sample analyze payloads + synthetic pose JSON |
| `integrations/` | Fork wiring: MotionDiff, pyskl, sports, SoccerNet, RuView |
| `docs/` | Positioning, architecture |

## Related forks (not owned here)

- [MotionDiff](https://github.com/ReallyWong/MotionDiff) — **v1 pose front-end**
- [pyskl](https://github.com/ReallyWong/pyskl) — fork of [kennymckormick/pyskl](https://github.com/kennymckormick/pyskl) (pinned in `integrations/pyskl.md`)
- [sports](https://github.com/ReallyWong/sports) ← roboflow/sports (optional ROI)
- [SoccerNetv2-DevKit](https://github.com/ReallyWong/SoccerNetv2-DevKit) (annotation style)
- [RuView](https://github.com/ReallyWong/RuView) — **post-M3 optional**, not v1 main path

Do not vendor entire forks — pin versions; keep adapters in-repo.

## Status

| Milestone | Status |
|-----------|--------|
| **M0** contracts | Locked (`taxonomy.json`, `rules.json`, pose contract) |
| **M1** offline geometric pipeline | Merged — `python scripts/run_offline_demo.py` |
| **M2** ONNX 4-class probe + `POST /analyze` | **This branch** — synthetic train → ONNX → FastAPI |

### M2 quickstart

```bash
pip install -r requirements-m2.txt
# CPU torch if needed: pip install torch --index-url https://download.pytorch.org/whl/cpu
python scripts/train_linear_probe.py
python scripts/export_onnx.py
bash scripts/run_analyze_api.sh
```

```bash
curl -s http://127.0.0.1:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"pose_path":"fixtures/demo_pass_inside_pose.json","label_id":"pass.inside"}'
```

See `docs/m2_model.md` for swapping the probe for real ST-GCN++ weights from pyskl.

## License

MIT
