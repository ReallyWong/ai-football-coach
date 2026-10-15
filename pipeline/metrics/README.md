# pipeline/metrics

Geometric metric extractors referenced by `rules/rules.json` (`from: geometric`).
Consume **COCO-17** sequences (`T×V×C`), not MediaPipe indices.

| Metric | Joints used | Meaning |
|--------|-------------|---------|
| `plant_foot_angle_deg` | plant knee→ankle (auto left/right by ankle conf) | 2D shin angle from +x; vertical ≈ **90°**. Rule `pass.inside.plant_foot`: `outside [70,110]` → warn |
| `torso_lean_deg` | mid-shoulder → mid-hip vs vertical | Absolute lean degrees. Rule `shot.instep.lean`: `gt 25` → tip |

```python
from pipeline.metrics.geometric import compute_geometric_metrics
metrics = compute_geometric_metrics(coco17)  # dict of floats
```

Model confidence / class scores live on dual-track `from: model` rules — not here.
