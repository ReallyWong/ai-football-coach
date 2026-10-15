# Architecture notes

v1 scope: youth technical correction, no embodied robot.

```mermaid
flowchart LR
  V[Video] --> P[Pose]
  P --> S[ST-GCN++]
  S --> R[Rules]
  R --> U[Coach UI]
```

Owned IP in this repo: labels, rules, Coach UI, narrative.
