# pipeline/adapters

In-repo adapters that sit between forks and this product. **Do not vendor whole forks.**

Planned modules:

| Adapter | Role |
|---------|------|
| `mp_to_coco17` | MediaPipe-33 → COCO-17 index remap + layout |
| `pyskl_input` | Pack disk `T×V×C` → model `N×C×T×V`, graph=`coco` |
| `analyze_response` | Map ST-GCN++ scores + rules hits → coach-ui payload |

Pin upstream commits in `integrations/`; keep adapter code here.
