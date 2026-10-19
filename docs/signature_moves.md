# Signature Move mode · 球星招牌动作

## Why / 为什么

Kids (and many adults) fall in love with football by **copying stars** — Marseille turn, elastico, step-overs, rainbow — not by memorizing “inside-foot pass” first.

Product bet: ship a **Signature Move** mode beside youth fundamentals so practice feels like fandom + skill, not homework.

小孩接触足球，常常是从模仿球星开始。我们把「招牌动作」做成独立模式，和基础技术纠错并行——练得更上瘾，也更好做全球内容（X / LinkedIn demo）。

## Architecture / 架构

| Layer | Fundamentals (v1 live) | Signature Move (preseed) |
|-------|------------------------|---------------------------|
| Labels | `labels/taxonomy.json` class_id 0–3 | `labels/signature_moves.json` ids `sig.*` |
| Rules | geometric + model dual-track | cue stubs in `rules/signature_rules.json` |
| Model | ONNX 4-class probe | **not** mixed into class_id 0–3 yet |
| UI | Coach UI mode toggle | catalog + cue cards |

Do **not** expand the live ONNX head until we have labeled signature clips. Catalog + cues ship first.

## Starter pack

马赛回旋 · 牛尾巴 · 踩单车 · 油炸丸子 · 克鲁伊夫转身 · 肩部假动作 · 拉波纳 · 帽子戏法挑过

## Next engineering steps

1. Collect short clips per move (side/diagonal).
2. Attach cue rules → optional geometric metrics.
3. Train a separate head or retrieval matcher; keep fundamentals model stable.
4. Demo content: “Learn Zidane’s turn in 30s” for global GTM.
