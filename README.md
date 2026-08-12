# EasyWar

即时策略占地游戏（Rust + Bevy）。玩法唯一事实来源见 `docs/GDD.md`，数值见 `docs/BALANCE.md`，计划见 `docs/ROADMAP.md`。

- `crates/logic`：纯逻辑层，不依赖渲染，可无头运行
- `crates/app`：Bevy 表现层
- AI：简单/中等/困难三档规则 AI，以及可并排试玩的“神经模型 V5～V11”。V5～V10 在双线梯形、编织双环启用；V11 额外支持留出的外环横梁。V11 从 V10 初始化，只使用冻结历史模型和终局胜负进行自博弈，并在 Rust 中采用固定种子的温度 0.5 采样；快捷键 `4`～`9`、`0` 可直接对比各代模型。

真人试玩埋点默认开启，直接运行 `cargo run -p easywar-app` 即会为每局 1v1 对局在 `training/telemetry/` 生成一份 JSONL。设置 `EASYWAR_TELEMETRY=0` 可关闭，也可将变量值设为其他输出目录。数据只会落盘，不会自动混入训练。
