# OpenCV AI Competition 2026 checklist

Deadline the team is using: **26 Oct 2026, 11:45pm PDT**.

Project name to submit: **ScoutMind Shot Coach / NATI 射门 Agent**.

One sentence: a 10-second youth shot goes through OpenCV 5 pose and quality checks, an agent chooses re-record or a drill, and the player gets one sentence they can use.

Built with so far: OpenCV 5, AWS S3, API Gateway, Lambda (presign only), ONNX Runtime, FastAPI, Render. Bedrock is still deferred.

## Required

- [ ] Technical report (problem, users, architecture, OpenCV 5, AWS, evaluation, limits, responsible use)
- [ ] Repo the judges can open (this branch, later tag `opencv26-submit`)
- [ ] Pinned dependencies plus build / deploy / test steps (`requirements-comp.txt`, this folder, `models/pose/README.md`)
- [ ] Architecture diagram that labels OpenCV 5 and AWS (`architecture.md`)
- [ ] Working endpoint or an agreed live screen-share
- [ ] Video, 5 minutes or less (team, the app working, architecture, main result)
- [ ] Evaluation evidence, including failures

## Agentic Vision

- [ ] Workflow diagram (perceive → decide → act) — draft in `agent-workflow.md`
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
- [ ] Real MoveNet or YOLO-pose ONNX checked in or published as a release asset
- [x] coach-ui shooting upload (presign → S3 → score). Contest service still needs `Dockerfile.comp`
- [ ] PR-D report PDF and the 5-minute video
- [ ] Second Render service before `requirements-comp.txt` is allowed onto a deploy image
