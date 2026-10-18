# Datasets

## `youth_football_synth/` (M3)

**Synthetic** COCO-17 skeleton clips for the 4 taxonomy classes. Not real youth video.

| Artifact | Committed? | Notes |
|----------|------------|-------|
| `manifest.jsonl` | **Yes** | One line per clip: id, path, label_id, class_id, split |
| `dataset_meta.json` | **Yes** | Counts, seed, disclaimer |
| `samples/*.json` | **Yes** | 1 tiny JSON sample per class (≤8 frames) |
| `clips/**/*.npy` | **No** (gitignored) | float32 `(T, 17, 3)` — regenerate locally |

### Layout

```
datasets/youth_football_synth/
  manifest.jsonl
  dataset_meta.json
  samples/
    pass_inside_sample.json
    ...
  clips/                    # gitignored *.npy
    pass_inside/
    receive_first_touch/
    shot_instep/
    dribble_close/
```

### Regenerate (≥80 clips / class)

```bash
python scripts/generate_synthetic_dataset.py --n-per-class 80
```

Then evaluate:

```bash
python scripts/eval_confusion.py
```

### Gitignore policy

Large binaries (`*.npy` under `datasets/`) stay out of git. Keep the manifest and tiny JSON samples so CI / docs know the expected layout without downloading gigabytes.
