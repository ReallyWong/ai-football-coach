# Correction rules (dual-track)

Maps pose geometry **and** ST-GCN++ signals → coach feedback.

**Canonical file:** [`rules.json`](rules.json) **v0.2.0**.  
`rules.draft.json` is superseded.

## Dual-track (`from`)

| `from` | Meaning | Fields |
|--------|---------|--------|
| `geometric` | Angles / distances from COCO-17 joints (`pipeline/metrics`) | `metric`, `op`, `value` or `range` |
| `model` | Classifier confidence / class-score signals from ST-GCN++ | `signal`, `op`, `value` or `range` |

**ST-GCN++ class scores ≠ geometric metrics.** Do not treat a softmax score as a plant-foot angle. Rules declare which track they use.

## Rule shape

- `id`, `label_id` (or `"*"` for any label)
- `from`: `geometric` | `model`
- `metric` (geometric) **or** `signal` (model)
- `op`: e.g. `gt`, `lt`, `outside`, `inside`
- `value` and/or `range`
- `severity`: `tip` | `warn` | `critical`
- `feedback_zh` / `feedback_en`

## Examples in v0.2

- `pass.inside.plant_foot` — geometric plant-foot angle
- `shot.instep.lean` — geometric torso lean
- `any.low_confidence` — model confidence tip (dual-track concrete example)
