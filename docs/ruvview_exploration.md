# RuView exploration path (docs-only)

**Status:** post-camera optional. **Not** the v1 main path.

[RuView](https://github.com/ReallyWong/RuView) explores WiFi DensePose-style sensing. For AI Football Coach:

| Question | Answer |
|----------|--------|
| Is RuView required for M0–M3? | **No** |
| Is it on the v1 product path? | **No** — camera / training video → MediaPipe (MotionDiff) → COCO-17 |
| When might it matter? | Optional post-M3 research: non-camera pose when a gym has WiFi sensing infrastructure |
| Code in this repo? | **None** — do not vendor RuView; no DensePose runtime here |

## v1 main path (reminder)

```
Camera / video
    → MediaPipe Pose (MotionDiff export)
    → COCO-17 adapter
    → ST-GCN++ / ONNX probe
    → dual-track rules
    → coach-ui
```

RuView would sit as a **dashed** alternate pose front-end after the camera path is solid — same COCO-17 contract if adapted, never a blocker for coach feedback.

## Decision

Keep RuView as an exploration note under `integrations/` and this doc. Ship M3 (dataset, eval, evidence UI) without any RuView dependency.
