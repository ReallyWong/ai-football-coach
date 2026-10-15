# pyskl / ST-GCN++

Upstream: https://github.com/kennymckormick/pyskl  
Intended fork: https://github.com/ReallyWong/pyskl *(create if missing — do not vendor the whole tree into this repo)*

## Pin policy

- Pin a **commit SHA or release tag** in this file when the fork is cut.
- Current pin: **TBD**
- Bump pins deliberately; adapters in `pipeline/adapters/` absorb API drift.

## Expected adapters (in this repo)

| Path (planned) | Role |
|----------------|------|
| `pipeline/adapters/mp_to_coco17` | Pose remap before training/infer |
| `pipeline/adapters/pyskl_input` | Disk `T×V×C` → `N×C×T×V`, `graph=coco` |
| `pipeline/adapters/analyze_response` | Scores + rules → coach-ui JSON |

## Training / graph contract

- Keypoint topology: **COCO-17** (`docs/pose_contract.md`)
- `graph = coco` (pyskl graph config)
- `num_classes` = `labels/taxonomy.json` → `num_classes` (currently **4**, class_id 0..3)
- Do not invent extra classes without taxonomy + data

## Serve contract: `POST /analyze`

Sketch of the JSON response coach-ui (and `fixtures/demo_pass_inside.json`) bind to:

```json
{
  "label_id": "pass.inside",
  "class_id": 0,
  "confidence": 0.86,
  "scores": {
    "form": 84,
    "timing": 80,
    "balance": 82,
    "overall": 82
  },
  "topk": [
    {"label_id": "pass.inside", "class_id": 0, "confidence": 0.86},
    {"label_id": "receive.first_touch", "class_id": 1, "confidence": 0.08}
  ],
  "rules_fired": [
    {
      "id": "pass.inside.plant_foot",
      "from": "geometric",
      "severity": "warn",
      "feedback_zh": "…",
      "feedback_en": "…"
    }
  ],
  "pose": {
    "topology": "coco17",
    "T": 64,
    "V": 17,
    "fps": 30
  }
}
```

Model `confidence` feeds `from: model` rules; geometric metrics feed `from: geometric` rules — see `rules/README.md`.
