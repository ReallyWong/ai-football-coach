# OpenCV AI Competition 2026 — shooting slice

**Product:** NATI 哪踢 / ScoutMind, youth football.  
**This repo:** AI Football Coach, the technique-correction slice.  
**This competition branch:** instep shooting only (`shot.instep` / 射门).

Deadline used by the team: **26 Oct 2026, 11:45pm PDT**.

One clip in, two things out:

- a score from **0 to 100**
- **one** coaching sentence

Passing, dribbling, and first touch stay on `POST /analyze`. They are not part of this slice.

## What runs today (PR-A)

| Piece | Where | Status |
|-------|--------|--------|
| OpenCV **5.0.0.93** pin | `requirements-comp.txt` | Local / competition env only |
| Decode, blur, brightness, person size, joint coverage | `pipeline/pose/quality_gates.py` | Done |
| MoveNet / YOLO-pose via `cv2.dnn` (`ENGINE_NEW`) | `pipeline/pose/opencv_dnn_pose.py` | Done; weights not vendored |
| MediaPipe | `video_to_mp33.py` | Fallback only if `AIFC_ALLOW_MP_FALLBACK=1` |
| Score + one sentence + `next_action` | `pipeline/shot/scoring.py` | Done, in process |
| `POST /analyze/shot` | keypoints or `pose_path` | Works on the current Render image |
| `POST /analyze/shot/video` | multipart file | Needs the comp env (OpenCV 5) |
| `s3_key` | request field | **501** until PR-B |

The deterministic policy is the agent brain for now:

```
if not quality.pass: request_re_record     # no score
elif geometric fault: assign_drill         # one sentence, one drill
else: accept_and_coach                      # score + one reinforcement
```

Bedrock does not choose the action. When PR-B lands, Bedrock may rewrite the sentence from the same observation.

## What must not move yet

`Dockerfile` installs `requirements-deploy.txt` only. That file has **no torch and no OpenCV**. A recent production failure was `ModuleNotFoundError: torch`. This branch does not put torch or OpenCV on that image.

`POST /analyze/shot` with keypoints does not import `cv2`. `POST /analyze/shot/video` on the current Render service returns **503** with an install hint. `GET /health` and `POST /analyze` keep their previous fields; health only gains a `shot` object.

Do not point `render.yaml` at `requirements-comp.txt` until a **second** Render service exists.

## Install and try

```bash
pip install -r requirements-comp.txt
# optional weights: models/pose/README.md
bash scripts/run_analyze_api.sh
```

```bash
curl -s http://127.0.0.1:8000/analyze/shot \
  -H 'Content-Type: application/json' \
  -d '{"pose_path":"fixtures/demo_shot_instep_pose.json"}'
```

User-facing fields are `score` and `coaching_zh`. `triggered_rules` is an audit list for the geometric faults; the product shows one sentence. `agent.trace_s3_key` is `null` until PR-B.

## Later PRs

### PR-B — AWS (required for the competition)

- [ ] S3 bucket for clips, overlays, and traces
- [ ] API Gateway + Lambda: presign upload
- [ ] Lambda: receive the observation, call `decide_next_action`, ask Bedrock for **one** sentence, write `trace.json`
- [ ] `POST /analyze/shot` accepts `s3_key` and reads the object (`SHOT_BUCKET`, `AWS_REGION`, `AGENT_API_URL`)
- [ ] `docs/deploy_aws.md` with least-privilege IAM
- [ ] Do not replace the Render service with this stack; add AWS beside it

### PR-C — coach-ui shooting mode

- [ ] One flow: upload or record a shot, show score + one sentence + `next_action`
- [ ] `request_re_record` hides the score
- [ ] `assign_drill` shows `drill_id` (`drill.lean_wall`, `drill.plant_target`)

### PR-D — report and submit

- [ ] `docs/opencv26_report.md` from `report-outline.md`, then PDF
- [ ] Architecture diagram exported from `architecture.md` (OpenCV 5 and AWS both visible)
- [ ] ≤5 min video, evaluation traces, tag `opencv26-submit`
- [ ] Checklist in `submission-checklist.md`

COOL / Graviton is optional and does not block the submission.
