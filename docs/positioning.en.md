# Positioning (EN)

## One-liner

**AI Football Coach**: film a skill with a phone or training camera, get automatic pose scoring and actionable technical corrections for youth players — no physical robot in v1.

## Elevator pitch (~30 seconds)

Youth football doesn’t need another match replay. It needs a clear answer to “what was wrong with that touch, and how do I fix it next time?” We turn training video into skeleton sequences, use ST-GCN++ to recognize technique actions (pass, first touch, shot, and more), then map them through coaching rules into concrete correction feedback. Pose and data plumbing reuse proven open-source building blocks; the owned layer is action labels, correction rules, and the Coach experience. The public story matches Football Match Intelligence / AI football coach: ship a globally demoable coaching assistant first, then consider embodied robots later.

## Boundaries (what we are not)

- **In scope**: individual / small-group technique recognition, scoring, correction tips, and review assets for training.
- **Out of scope (v1)**: humanoid robot sparring, full-field tactical decisions, or replacing a licensed coach’s full duty of care.
- **Not another CV toolkit**: this repo is the product layer (labels + rules + Coach UI), not a fork dump of models.

## Target users

- Youth coaches: review fundamentals at scale, cut repetitive verbal cues.
- Parents / players: see “what drifted” right after a session.
- Global GTM: a demo and content asset under an English-first Football Match Intelligence narrative.
