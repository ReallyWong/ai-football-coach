# Integrations

How this owned product sits on top of forks / upstream:

| Component | Repo | Role in pipeline |
|-----------|------|------------------|
| Pose / presence | ReallyWong/RuView | Optional non-camera pose front-end |
| Sports CV utils | ReallyWong/sports | Detection / tracking helpers |
| Dataset tooling | ReallyWong/SoccerNetv2-DevKit | Labels / evaluation patterns |
| Motion demo gen | ReallyWong/MotionDiff | Optional demonstration clips |
| ST-GCN++ | kenny/mckormick pyskl | Action recognition backbone |

Do not vendor entire forks here — document adapters and pin versions.
