# pyskl / ST-GCN++

Upstream: https://github.com/kennymckormick/pyskl  
Fork: https://github.com/ReallyWong/pyskl ✅ created via `gh repo fork`

## Pin policy

- Pin a **commit SHA or release tag** in this file when the fork is cut.
- **Current pin:** `f2bf3a6b08e2e8dec744692d64efdb187fd6719a` (fork default `main` at create time; upstream “[Fix] Changed np.Inf to np.inf (#265)”)
- Bump pins deliberately; adapters in this repo absorb API drift.
- Do **not** vendor the full pyskl tree into ai-football-coach.

## M2 stand-in (this repo)

Until ST-GCN++ weights are trained on youth data, M2 uses an in-repo lightweight
probe with the **same tensor contract**:

- Code: `pipeline/model/skeleton_probe.py`
- Train: `scripts/train_linear_probe.py` (synthetic COCO-17 clips)
- ONNX: `models/youth_football_probe.onnx`
- Docs: `docs/m2_model.md` (how to swap heads)

## Expected adapters (in this repo)

| Path | Role |
|------|------|
| `pipeline/adapters/mediapipe_to_coco17.py` | Pose remap before training/infer |
| `pipeline/model/skeleton_probe.py` | `T×V×C` → `N×C×T×V` preprocess + probe |
| `services/api/app.py` | Scores + rules → coach-ui JSON (`POST /analyze`) |

## Training / graph contract

- Keypoint topology: **COCO-17** (`docs/pose_contract.md`)
- `graph = coco` (pyskl graph config)
- `num_classes` = `labels/taxonomy.json` → `num_classes` (currently **4**, class_id 0..3)
- Do not invent extra classes without taxonomy + data

## Serve contract: `POST /analyze`

Response shape (compatible with `fixtures/demo_pass_inside.json`), plus M2 fields
`metrics` and `triggered_rules` (alias of `rules_fired`):

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
  "metrics": {
    "plant_foot_angle_deg": 45.0,
    "torso_lean_deg": 5.0
  },
  "triggered_rules": [
    {
      "id": "pass.inside.plant_foot",
      "from": "geometric",
      "severity": "warn",
      "feedback_zh": "支撑脚朝向偏了，尽量指向传球目标方向。",
      "feedback_en": "Plant foot is off-line — point it toward the pass target."
    }
  ],
  "rules_fired": [],
  "pose": {
    "topology": "coco17",
    "T": 64,
    "V": 17,
    "fps": 30
  }
}
```

Model `confidence` feeds `from: model` rules; geometric metrics feed `from: geometric` rules — see `rules/README.md`.
