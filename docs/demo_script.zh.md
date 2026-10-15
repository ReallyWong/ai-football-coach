# 30 秒 Demo 口播（中文）— 用于录屏 / README / 国内渠道

对应英文版：`docs/demo_script.en.md`（Grok Bot）。按 M2 路径录屏即可。

## 一口气版（≈30 秒，短视频 / 朋友圈）

青少年踢球缺的不是再看一遍比赛回放，而是：「这一脚哪里不对？」

看：我们载入一段姿态，打出技术评分，立刻给出教练提示——

「支撑脚朝向偏了，请指向传球目标。」

这就是 AI Football Coach：摄像头 → 姿态 → 评分 → 纠错。v1 不上机器人。Demo 见 README。

## 稍长版（≈45 秒，README / 路演）

1. **钩子** — 青训要的是改下一次触球，不是堆更多录像。
2. **演示** — `POST /analyze` 对传球片段返回动作类别、置信度，以及几何纠错规则。
3. **证据** — 触发提示：*支撑脚朝向偏了，请指向传球目标。*
4. **收束** — 自研层是标签、规则和 Coach UI；ST-GCN++ 权重可后换。对外叙事：Football Match Intelligence / AI football coach。

## 画面节拍

| 节拍 | 画面 |
|------|------|
| 0–5s | README 一句话或 coach-ui 卡片 |
| 5–15s | 终端：启动 API + `curl` `/analyze` |
| 15–25s | 高亮 `triggered_rules[].feedback_zh`（或同款支撑脚提示） |
| 25–30s | 仓库链接 + 「v1 不上机器人」 |

## 命令（可复制）

```bash
bash scripts/run_analyze_api.sh
curl -s http://127.0.0.1:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"pose_path":"fixtures/demo_pass_inside_pose.json","label_id":"pass.inside"}'
```

离线（不启 API）：

```bash
python scripts/run_offline_demo.py
```

## 短视频轻 CTA

配短片软提问：*公开搭建中——青训教练，支撑脚之后你最想先要哪条提示？*

首发 demo **不要**硬推「私信谈投资」。
