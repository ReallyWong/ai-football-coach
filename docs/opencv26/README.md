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
| MoveNet / YOLO-pose via `cv2.dnn` (`ENGINE_NEW`) | `pipeline/pose/opencv_dnn_pose.py` | Done. Contest image downloads Lightning at build; the file is not in git |
| MediaPipe | `video_to_mp33.py` | Fallback only if `AIFC_ALLOW_MP_FALLBACK=1` |
| Score + one sentence + `next_action` | `pipeline/shot/scoring.py` | Done, in process |
| `POST /analyze/shot` | keypoints or `pose_path` | Works on the current Render image |
| `POST /analyze/shot/video` | multipart file | Needs the comp env (OpenCV 5) |
| `s3_key` | `uploads/…` object in `SHOT_BUCKET` | GetObject, then the video path. Needs the comp env (boto3 + OpenCV). **503** on the live image |
| coach-ui **射门上传** | presign → S3 PUT → `/analyze/shot` | Wired to the live presign API. See [docs/deploy_aws.md](../deploy_aws.md) |

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

User-facing fields are `score` and `coaching_zh`. `triggered_rules` is an audit list for the geometric faults; the product shows one sentence. `agent.trace_s3_key` stays `null` until a trace is stored. Bedrock is not in this path.

S3 clip fetch and the coach-ui upload flow are in [docs/deploy_aws.md](../deploy_aws.md). The live Render image still does not install OpenCV or boto3.

## Submission package

| Piece | Path |
|-------|------|
| Technical report (PDF) | [opencv26_technical_report.pdf](opencv26_technical_report.pdf) |
| Report source | [opencv26_report.md](opencv26_report.md) |
| As-deployed diagram | [architecture.png](architecture.png) |
| Perceive → decide → act | [agent-workflow.png](agent-workflow.png) |

Bedrock and the agent Lambda stay deferred in that report. The ≤5 minute video and the `opencv26-submit` tag are still outstanding.

## Later PRs

### PR-B — AWS

- [x] S3 bucket + API Gateway presign (live; no Bedrock)
- [x] `POST /analyze/shot` reads `s3_key` via `SHOT_BUCKET`, `AWS_REGION`, and the standard AWS credential env vars
- [x] `docs/deploy_aws.md` with least-privilege IAM and the contest Render env
- [x] Live Render service unchanged (`Dockerfile` / `requirements-deploy.txt`). Contest image is `Dockerfile.comp`
- [ ] Agent Lambda, Bedrock sentence rewrite, `trace.json` (`AGENT_API_URL` still unused)

### PR-C — coach-ui shooting mode

- [x] One flow: choose or record a shot, presign, S3 PUT, show score + one sentence + `next_action`
- [x] `request_re_record` hides the score
- [x] `assign_drill` shows `drill_id` (`drill.lean_wall`, `drill.plant_target`)

### PR-D — report and submit

- [x] [opencv26_report.md](opencv26_report.md) and [opencv26_technical_report.pdf](opencv26_technical_report.pdf)
- [x] [architecture.png](architecture.png) and [agent-workflow.png](agent-workflow.png)
- [ ] ≤5 min video
- [ ] Tag `opencv26-submit` (not created)
- [x] Checklist in [submission-checklist.md](submission-checklist.md)

COOL / Graviton is optional and does not block the submission.
