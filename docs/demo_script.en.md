# 60–90s demo script (EN) — coach-ui Demo mode

Spoken beats for a screen recording. Pair with Chinese in `docs/demo_script.zh.md`.

The hero path is **Demo mode inside coach-ui**. No terminal, no MediaPipe. It works on GitHub Pages and on Render.

## One breath (≈30s, open the longer cut with this)

Youth football doesn’t need another match replay — it needs “what was wrong with that touch?”

Watch: Demo mode loads an inside-foot pass, scores the skill, and fires a coaching tip.

“Plant foot is off-line — point it toward the pass target.”

Jump to the problem frame and you can see where it went wrong. That’s AI Football Coach: pose → scoring → correction. No robot in v1.

## What to show (60–90s, click path)

Open [coach-ui](https://reallywong.github.io/ai-football-coach/coach-ui/) (or Render `/coach-ui/`). Full screen. Keep the terminal out of the shot.

| Time | On screen | Do |
|------|-----------|----|
| 0–8s | Headline: “What was wrong with that touch?” | Pause so the line reads |
| 8–18s | Simulated clip looping; Inside-foot pass is selected | Click **Demo mode** |
| 18–22s | Clip holds; feedback area appears | Click **Analyze this clip** |
| 22–28s | “Analyzing…” progress | Stay on it |
| 28–45s | Tip “Plant foot is off-line…”, scores, confidence | Point at the tip |
| 45–58s | Stick figure jumps to the highlighted problem frame; angle and deviation sit under the scrubber | Click **Jump to problem frame** |
| 58–68s | This touch / After correction cards (after is a placeholder) | Glance |
| 68–80s | Optional: switch to Instep shot and analyze again (back lean) | Optional |
| 80–90s | Signature Move catalog — Marseille turn and others, each marked Preseed | Click **Browse Signature Moves**, then Marseille turn |

Close: the owned layer is labels, rules, and Coach UI. No robot in v1.

## Before you record

- Use a private window so you don’t land in Signature Moves from a previous visit.
- Switch the page to English.
- Zoom about 110%. The page is already dark.
- Do **not** open `/docs` — that is API reference, not the product.
- Do **not** expect a video upload to score anything on Pages or Render. There is no MediaPipe there. The amber note on the upload box says so.
- A static server is enough locally. The API can stay off:

```bash
python -m http.server 8765
# open http://127.0.0.1:8765/coach-ui/
```

## Soft CTA

Quote or reply with the clip and a soft ask: *Building in public — youth coaches, what tip would you want next after plant-foot?*

Do **not** hard-pitch “DM me to invest” in the first demo posts.

## Optional: real video on a local API (not the hero path)

`POST /analyze/video` runs only when this machine has MediaPipe and the API is up. Demo mode does not call it.

```bash
bash scripts/run_analyze_api.sh
```
