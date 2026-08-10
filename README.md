# EasyWar

即时策略占地游戏（Rust + Bevy）。玩法唯一事实来源见 `docs/GDD.md`，数值见 `docs/BALANCE.md`，计划见 `docs/ROADMAP.md`。

- `crates/logic`：纯逻辑层，不依赖渲染，可无头运行
- `crates/app`：Bevy 表现层
- AI：简单/中等/困难三档规则 AI，以及开发中的“神经模型 V5”（当前仅双线梯形、编织双环启用）。
