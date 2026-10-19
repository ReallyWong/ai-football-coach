# Coach UI

Youth feedback screen: scores, correction tips, and **evidence-frame** jump (M3).

## Quick view

From the repo root:

```bash
python -m http.server 8765
# open http://127.0.0.1:8765/coach-ui/
```

Loads `fixtures/demo_analyze_response.json` by default (analyze-shaped payload with
`evidence.frame_index` + optional `keypoints` for the stick figure). Use **Jump to evidence**
or the scrubber. You can also paste a live `POST /analyze` JSON response into the textarea.

v1: static HTML+JS (no robot hardware).
