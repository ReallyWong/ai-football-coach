# 30-second demo script (EN) — for X / README

Spoken copy for a screen recording of the M2 path. Pair with Chinese in `docs/demo_script.zh.md` (secretary).

## One breath (≈30s, X / LinkedIn)

Youth football doesn’t need another match replay — it needs “what was wrong with that touch?”

Watch: we load a pose clip, score the skill, and fire a coaching tip.

“Plant foot is off-line — point it toward the pass target.”

That’s AI Football Coach: camera → pose → scoring → correction. No robot in v1. Demo link in bio / README.

## Slightly longer cut (≈45s, README / pitch)

1. **Hook** — Youth players don’t need more film study; they need the next touch fixed.
2. **Show** — `POST /analyze` on a pass clip returns label + confidence + geometric rules.
3. **Proof** — Triggered tip: *Plant foot is off-line — point it toward the pass target.*
4. **Close** — Owned layer is labels, rules, and Coach UI; ST-GCN++ weights can swap later. Global story: Football Match Intelligence / AI football coach.

## What to show on screen

| Beat | Show |
|------|------|
| 0–5s | README one-liner or coach-ui card |
| 5–15s | Terminal: start API + `curl` `/analyze` |
| 15–25s | Highlight `triggered_rules[].feedback_en` |
| 25–30s | Repo link + “no robot in v1” |

## Commands (copy-paste)

```bash
bash scripts/run_analyze_api.sh
curl -s http://127.0.0.1:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"pose_path":"fixtures/demo_pass_inside_pose.json","label_id":"pass.inside"}'
```

Offline (no API):

```bash
python scripts/run_offline_demo.py
```

## Soft CTA for X

Quote or reply with a short clip + soft ask: *Building in public — youth coaches, what tip would you want next after plant-foot?*

Do **not** hard-pitch “DM me to invest” in the first demo posts.
