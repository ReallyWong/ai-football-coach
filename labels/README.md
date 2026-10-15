# Action labels

Self-owned football technique taxonomy for youth coaching.

**Canonical file:** [`taxonomy.json`](taxonomy.json) **v0.2.0** (M0).  
`taxonomy.draft.json` is superseded — do not extend it.

## v0.2 classes (`class_id` 0..3)

| class_id | id | EN | 中文 |
|--------:|----|----|------|
| 0 | `pass.inside` | Inside-foot pass | 脚内侧传球 |
| 1 | `receive.first_touch` | First touch | 停球/第一脚 |
| 2 | `shot.instep` | Instep shot | 正脚背射门 |
| 3 | `dribble.close` | Close control dribble | 近身盘带 |

Each label has: `id`, `class_id`, `name_zh` / `name_en`, semantic `joints`, `joint_indices_coco17`, `score_dimensions` `[form, timing, balance]`, optional `metrics`, and `clip_hint`.

Joint indices follow COCO-17 — see `docs/pose_contract.md`. Never use MediaPipe indices here.

## Room to grow (not in v0.2)

- Passing: driven / lofted (future `class_id` ≥ 4)
- Receiving: cushion
- Shooting: side-foot / volley
- Dribbling: change of direction
- Defending: jockey / tackle timing (optional)

Do not add fake labels until clip data exists. `num_classes` for pyskl must match `taxonomy.json`.
