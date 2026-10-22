# AWS shot clips (minimal stack)

Region `ap-southeast-1`. This path stores the clip in S3 and scores it with the existing OpenCV pipeline on a contest API. **Bedrock is not deployed.** `decide_next_action` still runs inside `pipeline/shot/scoring.py`. `agent.trace_s3_key` stays `null`.

Do not commit access keys. The names below are the only values that belong in git.

## Live resources

| Resource | Value |
|----------|--------|
| Bucket | `nati-shot-coach-660008586940` |
| `POST /presign` | `https://s14hcf6sc0.execute-api.ap-southeast-1.amazonaws.com/presign` |
| Object prefix | `uploads/<id>.mp4` (browser clips). `traces/` and `overlays/` are reserved and are not fetched by `/analyze/shot`. |

Presign contract:

```bash
curl -X POST 'https://s14hcf6sc0.execute-api.ap-southeast-1.amazonaws.com/presign' \
  -H 'Content-Type: application/json' \
  -d '{"filename":"clip.mp4","content_type":"video/mp4"}'
```

Response fields the client uses: `upload_url`, `s3_key`, `headers`. PUT the file bytes to `upload_url` with that `Content-Type`. The signed URL expires in about 15 minutes.

## Browser flow (coach-ui)

1. Open **射门上传 / Shot upload**.
2. The page POSTs `{filename, content_type}` to `PRESIGN_URL`.
3. It PUTs the video to `upload_url`.
4. It POSTs `{s3_key, max_seconds: 10}` to `POST /analyze/shot`.

Default presign URL is the live endpoint above (`DEFAULT_PRESIGN_URL` in `coach-ui/index.html`). Override it with `?presign=` or `localStorage.aifc_presign_url`.

Analysis defaults to same-origin `/analyze/shot`. On GitHub Pages set `?api=https://<contest-service>` (or `localStorage.aifc_shot_api`). A `501` body means that server is still the old stub. A `503` body names the missing piece (`SHOT_BUCKET`, boto3, or OpenCV).

`request_re_record` hides the score. `assign_drill` shows `drill_id`.

## Contest Render service

The live `ai-football-coach` service stays on `Dockerfile` + `requirements-deploy.txt` (no torch, no OpenCV, no boto3). `GET /health` and `POST /analyze` do not call AWS. Keypoint `POST /analyze/shot` still works there. `s3_key` on that image returns **503** until the bucket is set, and video decode still needs OpenCV.

Use a **second** service (or a separate env you are willing to put the comp image on):

- Docker file: `Dockerfile.comp` (installs `requirements-comp.txt`)
- Do not point `render.yaml` at `Dockerfile.comp`. That blueprint is the live fundamentals service on `main`.

Environment:

| Variable | Value |
|----------|--------|
| `SHOT_BUCKET` | `nati-shot-coach-660008586940` |
| `AWS_REGION` | `ap-southeast-1` |
| `AWS_ACCESS_KEY_ID` | IAM access key. Dashboard only. |
| `AWS_SECRET_ACCESS_KEY` | IAM secret. Dashboard only. |
| `AWS_SESSION_TOKEN` | Only when the key is temporary. |

Render: open the contest service → **Environment** → add the variables → save, which redeploys. `GET /health` then shows `shot.s3.bucket_configured`, `shot.s3.region`, `shot.s3.boto3_present`, and `shot.s3.credentials_configured`. The health payload never includes the secret.

Least privilege for that IAM user (GetObject covers HeadObject; ListBucket is not required):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadShotUploads",
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": "arn:aws:s3:::nati-shot-coach-660008586940/uploads/*"
    }
  ]
}
```

`POST /analyze/shot` with `s3_key`:

1. Rejects anything that is not `uploads/<file>` with a video suffix.
2. If `SHOT_BUCKET` is unset, **503** before any AWS call.
3. Imports boto3 and downloads the object (80 MB cap) with the default credential chain.
4. Runs the same OpenCV 5 decode, quality gates, and DNN pose path as `POST /analyze/shot/video`.
5. Returns `score`, `coaching_zh`, and `agent.next_action`. `source.type` is `s3`.

Missing boto3 or OpenCV is **503** with an install hint. A missing object is **404**. The keypoint body is unchanged when `s3_key` is omitted.

## Still deferred

- Bedrock rewriting the one sentence
- Agent Lambda (`AGENT_API_URL`)
- Writing `traces/` and setting `agent.trace_s3_key`
