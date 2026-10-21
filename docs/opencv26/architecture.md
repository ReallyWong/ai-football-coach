# Shot agent architecture

NATI 哪踢 / ScoutMind youth football. AI Football Coach is the technique-correction slice. The competition demo is **instep shooting only**.

OpenCV 5 does the video work. AWS is the agent and the clip store. Render keeps the low-latency API it already serves. PR-A implements the Render shooting path and the decision policy in process. The AWS boxes below are the PR-B contract, not a claim that they are deployed.

```mermaid
flowchart TB
  subgraph Client
    Phone[Phone: 10s shot]
    UI[coach-ui shot mode]
  end

  subgraph AWS["AWS — PR-B"]
    APIGW[API Gateway]
    L1[Lambda: presign]
    L2[Lambda: agent_decide]
    S3[(S3: clips, overlays, traces)]
    BR[Bedrock: one sentence]
  end

  subgraph Render["Render FastAPI — PR-A in process"]
    API[POST /analyze/shot]
    CV[OpenCV 5.0.0.93 decode, quality, DNN pose]
    ONNX[ONNX shot probe + geometric rules]
  end

  Phone --> UI
  UI -->|1. POST /presign| APIGW --> L1
  L1 -->|2. PUT url| S3
  UI -->|3. upload| S3
  UI -->|4. POST /analyze/shot s3_key| API
  API -->|GetObject| S3
  API --> CV --> ONNX
  ONNX -->|observation| L2
  L2 --> BR
  L2 -->|5. next_action + one sentence| UI
  L2 -->|trace.json| S3
```

Until PR-B, step 4 can also be keypoints or `POST /analyze/shot/video`, and step 5 is computed inside the API by `decide_next_action`. `s3_key` returns HTTP 501 so a client cannot mistake a local score for an S3 round-trip.

## Who does what

| Layer | Job |
|-------|-----|
| OpenCV 5 | `VideoCapture`, Laplacian blur, brightness, DNN pose (MoveNet or YOLO-pose ONNX), skeleton overlay |
| Geometry + ONNX probe | Force `shot.instep`. Plant foot and torso lean become 0–100. The probe only nudges the score |
| Agent | `request_re_record`, `assign_drill`, or `accept_and_coach` from those measurements |
| Bedrock (PR-B) | Rewrite the one sentence. It does not pick `next_action` |
| S3 (PR-B) | Clip, overlay, trace for the judges |
| Render | Public API. Deploy image stays on `requirements-deploy.txt` until a second service exists |

## Response the UI should read

```json
{
  "label_id": "shot.instep",
  "score": 72,
  "coaching_zh": "上身后仰过多，容易打飞；微微前倾压球。",
  "agent": {
    "next_action": "assign_drill",
    "drill_id": "drill.lean_wall",
    "reason_codes": ["quality_ok", "torso_lean_high"],
    "policy": "deterministic_v1",
    "trace_s3_key": null
  },
  "opencv": {
    "version": "5.0.0",
    "pose_backend": "dnn_movenet",
    "frames_used": 87,
    "compliant": true,
    "pinned": "5.0.0.93"
  }
}
```

`score` is `null` when `next_action` is `request_re_record`.

## Out of scope

Full ST-GCN++ replacement, COOL/Graviton benchmarks, multi-skill UI, robots, RuView.
