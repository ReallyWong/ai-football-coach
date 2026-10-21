# Technical report outline

Target file: `docs/opencv26_report.md`, then PDF. About 6–10 pages. This outline is the PR-D skeleton. It is not the report.

1. **Title / team** — Ruoli Wang (ReallyWong). NATI 哪踢 / ScoutMind. Track: Agentic Vision.
2. **Problem and users** — Youth shooting feedback is slow and expensive. Parents and youth coaches. This slice is shooting only.
3. **Related product** — AI Football Coach on Render. The competition build is the compliant slice plus an agent, not a replacement of the whole coach.
4. **System overview** — Diagram from `architecture.md` with OpenCV 5, AWS, and Render all labeled.
5. **OpenCV 5** — `opencv-python-headless==5.0.0.93`. Decode, quality gates, DNN pose, skeleton overlay. Say why this is analysis rather than a single image read. MediaPipe is fallback only.
6. **AWS** — S3, API Gateway, Lambda, Bedrock. Presigned upload, no long-lived upload keys in the client. Mark anything not deployed yet as not deployed.
7. **Agent loop** — Observation → `next_action` → one sentence. Contrast with a chatbot that only explains a score. Point at traces.
8. **Score and coaching** — 0–100 from plant foot and torso lean, with a small ONNX confidence prior. One sentence. The sentence for a fault is copied from `rules.json`.
9. **Evaluation** — Clips, quality-gate behavior, agreement of `next_action` with the policy, failure cases.
10. **Limits and responsible use** — Draft below.
11. **Reproducibility** — `requirements-comp.txt`, commands in `docs/opencv26/README.md`, IAM notes from PR-B.
12. **Appendix** — Traces, sample overlay, confusion context for the probe (the probe is a stand-in, not ST-GCN++).

## Limits draft

- 2D pose has no ball-speed or goal truth. The score is a technique heuristic, not a scout rating.
- Side light and occlusion push many clips to `request_re_record`.
- A rewritten sentence can still sound wrong. A coach should review it before it is treated as instruction.
- The synthetic ONNX probe is not a large labeled shooting dataset.
- The Render deploy image does not run OpenCV until a second service exists. Keypoint scoring on the current service is not the video claim.
