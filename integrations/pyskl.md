# pyskl / ST-GCN++

Upstream: https://github.com/kennymckormick/pyskl

Plan:

1. Fork or submodule for experiments (not required for product README).
2. Export skeleton sequences in COCO / NTU-style keypoints.
3. Fine-tune or linear-probe on youth football label set in `labels/`.
4. Serve top-k label + confidence into `rules/` then `coach-ui/`.
