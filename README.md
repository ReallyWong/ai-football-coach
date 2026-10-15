# AI Football Coach

> Youth technical correction — camera → pose → ST-GCN++ scoring → coach feedback. No physical robot in v1.

**One-liner (EN)** — _TBD by Grok Bot_  
**一句话定位（中）** — _待秘书填写_

## Product link

```
Camera / training video
        ↓
Pose front-end (RuView / sports CV)
        ↓
ST-GCN++ action scoring (pyskl)
        ↓
Correction rules + Coach UI  ← this repo (owned IP)
```

## Repo layout

| Path | What |
|------|------|
| `labels/` | Football technique action label taxonomy |
| `rules/` | Correction / coaching rules mapped to labels |
| `coach-ui/` | Coach feedback UI (scaffold) |
| `integrations/` | How we wire forks: sports, SoccerNet, MotionDiff, RuView, pyskl |
| `docs/` | Positioning, elevator pitch, architecture notes |

## Related forks (not owned here)

- [sports](https://github.com/ReallyWong/sports) ← roboflow/sports
- [SoccerNetv2-DevKit](https://github.com/ReallyWong/SoccerNetv2-DevKit)
- [MotionDiff](https://github.com/ReallyWong/MotionDiff)
- [RuView](https://github.com/ReallyWong/RuView)
- External: [pyskl / ST-GCN++](https://github.com/kennymckormick/pyskl)

## Status

Scaffold — labels, rules, and UI to be filled. Play online targets stay on classic Founder's Journey pages; this product is separate.

## License

MIT
