# pipeline/metrics

Geometric metric extractors referenced by `rules/rules.json` (`from: geometric`).

Examples (planned):

- `plant_foot_angle_deg`
- `torso_lean_deg`

These consume COCO-17 sequences (`T×V×C`), not MediaPipe indices.
Model confidence / class scores live on the dual-track `from: model` rules — not here.
