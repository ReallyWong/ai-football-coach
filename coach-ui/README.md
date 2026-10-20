# Coach UI

Youth coach screen: a **Demo mode** walkthrough, scores, correction tips, evidence-frame jump, and a preseed Signature Move catalog.

Demo mode does not call `http://127.0.0.1:8000`. It loads relative fixtures:

| Page | Fixture URL the button uses |
|------|-----------------------------|
| `https://reallywong.github.io/ai-football-coach/coach-ui/` | `https://reallywong.github.io/ai-football-coach/fixtures/…` |
| Render `/coach-ui/` | `/fixtures/…` (mounted next to the UI) |
| `python -m http.server` from the repo root, then `/coach-ui/` | `/fixtures/…` |

Clips: `fixtures/demo_analyze_response.json` (inside-foot pass, plant-foot rule) and `fixtures/demo_shot_instep_response.json` (instep shot, torso-lean rule). The Signature Move cards read `fixtures/signature_catalog.json` (same shape as `GET /signature-moves`, status **preseed**).

## Open

GitHub Pages or Render, or from the repo root:

```bash
python -m http.server 8765
# http://127.0.0.1:8765/coach-ui/
```

Click **演示模式 / Demo mode**. The first scenario is 脚内侧传球. **正脚背射门 / Instep shot** is the second chip. **分析这段 / Analyze this clip** holds on 「正在分析…」 for about 1.2s, then shows the tip, scores, confidence, stick-figure scrub, and **跳到问题帧 / Jump to problem frame**. **看看招牌动作** opens 马赛回旋 and the rest of the preseed catalog.

Video upload still posts to `/analyze/video` when a local API with MediaPipe is running (same origin, or `127.0.0.1:8000` if you opened the UI on localhost). Pages and Render have no MediaPipe — the upload note says to use Demo mode. `/docs` is API reference, not this screen.

## Record the walkthrough (60–90s)

Follow [docs/demo_script.zh.md](../docs/demo_script.zh.md) or [docs/demo_script.en.md](../docs/demo_script.en.md). No terminal in the hero shot.

1. Private window, language set, zoom about 110%, bookmarks bar hidden.
2. Open coach-ui and start recording on the hook 「这一脚哪里不对？」 / “What was wrong with that touch?”.
3. Click **Demo mode** → **Analyze this clip** → wait out 「正在分析…」 → **Jump to problem frame**.
4. Optional: switch the chip to instep shot and analyze once more.
5. Click **Browse Signature Moves** and open 马赛回旋 / Marseille turn (badge: 预置 / Preseed).
6. Stop around 60–90s. Do not open `/docs` or a curl session.

v1: static HTML+JS for the demo path. No robot.
