# pipeline/adapters

In-repo adapters between forks and this product. **Do not vendor whole forks.**

| Adapter | Role | Status |
|---------|------|--------|
| `mediapipe_to_coco17.py` | MediaPipe-33 → COCO-17 index remap (`docs/pose_contract.md`) | **M1** |
| `pyskl_input` | Pack disk `T×V×C` → model `N×C×T×V`, graph=`coco` | planned (M2) |
| `analyze_response` | Map ST-GCN++ scores + rules hits → coach-ui payload | planned |

```python
from pipeline.adapters.mediapipe_to_coco17 import mediapipe_to_coco17
coco17 = mediapipe_to_coco17(mp33)  # (T,33,C) → (T,17,C)
```

Pin upstream commits in `integrations/`; keep adapter code here.
