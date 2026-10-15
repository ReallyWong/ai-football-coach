# Architecture notes

v1 scope: youth technical correction, no embodied robot.

Main pipeline (M0 contract):

**Camera / video → MediaPipe (MotionDiff export) → COCO-17 adapter → ST-GCN++ (pyskl) → rules dual-track → coach-ui**

Pose topology and layouts: [`pose_contract.md`](pose_contract.md).

```mermaid
flowchart LR
  V[Camera / video] --> MD[MotionDiff / MediaPipe]
  MD --> A[COCO-17 adapter]
  A --> S[ST-GCN++ pyskl]
  A --> G[Geometric metrics]
  S --> R{Rules dual-track}
  G --> R
  R --> U[Coach UI]
  RV-. optional post-M3 .-> A
  RV[RuView WiFi DensePose]
```

- Solid path = v1 main.
- RuView is **dashed / optional (post-M3)** — not required for M0–M3.
- Dual-track rules: `from: geometric` (angles from COCO-17) vs `from: model` (ST-GCN++ confidence / scores). They are not interchangeable — see `rules/README.md`.

Owned IP in this repo: labels, rules, Coach UI, pose/rules contracts, in-repo adapters. Forks stay external; pin versions in `integrations/`.
