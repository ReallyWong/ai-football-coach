# pipeline/pose

Where pose ingestion and clip windowing will live.

Intended contents (not implemented in M0):

- Load MotionDiff / MediaPipe export JSON
- Remap MP-33 → COCO-17 per `docs/pose_contract.md`
- Root-center + shoulder-width scale
- Write `T×V×C` arrays + `pose_clip` metadata

No runnable exporter yet — contracts only.
