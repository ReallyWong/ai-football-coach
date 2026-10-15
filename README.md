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
ST-GCN++ action scoring (pyskl)
        ↓
Rules dual-track (geometric + model) + Coach UI  ← this repo (owned IP)
```

## Repo layout

| Path | What |
|------|------|
| `labels/` | Football technique taxonomy (`taxonomy.json` v0.2) |
| `rules/` | Dual-track correction rules (`rules.json` v0.2) |
| `pipeline/` | Pose schemas, adapter/metric stubs |
| `docs/pose_contract.md` | COCO-17 topology, MP→COCO map, tensor layouts |
| `coach-ui/` | Coach feedback UI (scaffold) |
| `fixtures/` | Sample `POST /analyze` payloads for UI binding |
| `integrations/` | Fork wiring: MotionDiff, pyskl, sports, SoccerNet, RuView |
| `docs/` | Positioning, architecture |

## Related forks (not owned here)

- [MotionDiff](https://github.com/ReallyWong/MotionDiff) — **v1 pose front-end**
- [pyskl](https://github.com/ReallyWong/pyskl) — intended fork of [kennymckormick/pyskl](https://github.com/kennymckormick/pyskl) (create if missing)
- [sports](https://github.com/ReallyWong/sports) ← roboflow/sports (optional ROI)
- [SoccerNetv2-DevKit](https://github.com/ReallyWong/SoccerNetv2-DevKit) (annotation style)
- [RuView](https://github.com/ReallyWong/RuView) — **post-M3 optional**, not v1 main path

Do not vendor entire forks — pin versions; keep adapters in-repo.

## Status

M0 contracts locked: pose, labels, rules, fork wiring. Implementation of exporters / train loops comes later.

## License

MIT
