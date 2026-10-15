# M2 model: lightweight skeleton probe → ST-GCN++ swap

## Why a probe?

Full pyskl **ST-GCN++** fine-tune on real youth clips needs GPU + labeled data.
M2 ships a **CPU-trainable temporal MLP probe** with the **same I/O contract** so
the analyze API, rules wiring, and coach-ui payload stay stable.

## I/O contract (do not change when swapping)

| Side | Layout | Notes |
|------|--------|-------|
| Input | `N × C × T × V` float32 | `C=3` (x,y,conf), `T=64`, `V=17` COCO |
| Output | `N × 4` logits | class_id order from `labels/taxonomy.json` |
| Preprocess | root-center mid-hip + shoulder-width scale | `pipeline/model/skeleton_probe.py` |

Class order:

| class_id | label_id |
|---------:|----------|
| 0 | `pass.inside` |
| 1 | `receive.first_touch` |
| 2 | `shot.instep` |
| 3 | `dribble.close` |

## Artifacts

| Path | Role |
|------|------|
| `models/youth_football_probe.pt` | Torch checkpoint (train output) |
| `models/youth_football_probe.onnx` | Runtime artifact loaded by `services/api/app.py` |
| `scripts/train_linear_probe.py` | Synthetic data + train |
| `scripts/export_onnx.py` | `.pt` → `.onnx` |

## Train / export (CPU, &lt; 2 min)

```bash
python scripts/train_linear_probe.py
python scripts/export_onnx.py
```

Or train and export in one step:

```bash
python scripts/train_linear_probe.py --also-onnx
```

## Serve

```bash
bash scripts/run_analyze_api.sh
# curl example:
curl -s http://127.0.0.1:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"pose_path":"fixtures/demo_pass_inside_pose.json","label_id":"pass.inside"}'
```

## Swap to real ST-GCN++ (pyskl)

1. Fine-tune ST-GCN++ in [ReallyWong/pyskl](https://github.com/ReallyWong/pyskl) with `graph=coco`, `num_classes=4`.
2. Export ONNX (or TorchScript) with input name `pose` shaped `N×C×T×V` and output `logits` shaped `N×4`.
3. Point `AIFC_ONNX_PATH` at the new file (or replace `models/youth_football_probe.onnx`).
4. Keep preprocess identical (`preprocess_tvx_to_nctv`) so geometric rules still see the raw COCO-17 clip.

No API response schema change is required — only the logit source.

## Synthetic training note

`train_linear_probe.py` generates per-class clips with distinctive motion
signatures (inside-foot swing, cushion lift, instep drive + lean, bilateral
ankle oscillation). This is a **demo prior**, not a substitute for real youth
labeled data.
