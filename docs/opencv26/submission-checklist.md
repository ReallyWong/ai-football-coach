# OpenCV AI Competition 2026 checklist

Deadline the team is using: **26 Oct 2026, 11:45pm PDT**.

Project name to submit: **ScoutMind Shot Coach / NATI 射门 Agent**.

One sentence: a 10-second youth shot goes through OpenCV 5 pose and quality checks, an agent chooses re-record or a drill, and the player gets one sentence they can use.

Built with so far: OpenCV 5, AWS S3, API Gateway, Lambda (presign only), ONNX Runtime, FastAPI, Render. Bedrock is still deferred.

## Required

- [x] Technical report: [opencv26_technical_report.pdf](opencv26_technical_report.pdf) and [opencv26_report.md](opencv26_report.md)
- [ ] Repo the judges can open (this branch, later tag `opencv26-submit` — tag not created)
- [ ] Pinned dependencies plus build / deploy / test steps (`requirements-comp.txt`, this folder, `models/pose/README.md`)
- [x] Architecture diagram that labels OpenCV 5 and AWS: [architecture.png](architecture.png) (notes in [architecture.md](architecture.md))
- [ ] Working endpoint or an agreed live screen-share
- [ ] Video, 5 minutes or less (team, the app working, architecture, main result) — still outstanding
- [ ] Evaluation evidence, including failures

## Agentic Vision

- [x] Workflow diagram (perceive → decide → act): [agent-workflow.png](agent-workflow.png) (notes in [agent-workflow.md](agent-workflow.md))
- [ ] Trace or demo where an OpenCV 5 measurement changes the next action
- [ ] Success, failure, observability, and a human check (coach still reviews the sentence)

## COOL (optional)

- [ ] COOL version and Graviton instance
- [ ] Repeatable measurement against a baseline
- [ ] Evidence that COOL ran the claimed workload

Do not block the shooting path on COOL.

## Still open after PR-A

- [x] AWS minimal path: S3, presign Lambda, `POST /analyze/shot` `s3_key` (`docs/deploy_aws.md`)
- [ ] Agent Lambda, Bedrock sentence, traces
- [x] Contest image downloads MoveNet Lightning at build (`scripts/fetch_movenet_lightning.py`). The weight stays out of git
- [x] coach-ui shooting upload (presign → S3 → score). Contest service still needs `Dockerfile.comp`
- [x] PR-D report PDF and diagrams in this folder (`opencv26_technical_report.pdf`, `architecture.png`, `agent-workflow.png`)
- [ ] ≤5 min video (still outstanding) and tag `opencv26-submit`
- [ ] Second Render service before `requirements-comp.txt` is allowed onto a deploy image
