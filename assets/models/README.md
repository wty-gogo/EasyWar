# 游戏内模型

`neural_v5.ewnn` 是菜单“神经模型 V5”的内嵌推理权重：

- 来源：`training/checkpoints/multistrategy-course-v5-best.pt`
- 使用策略：`strategy_id = 0`（基准策略）
- 格式：`EWNNv1`，小端 `f32`，只包含两层卷积和动作头
- SHA-256：`e175adcecb8ed09a437670e6a352ecbd5ac8095f36f3b8ef4997977540016efc`
- 支持地图：`dual_ladder_1v1.toml`、`braided_rings_1v1.toml`

重新导出后必须运行 `cargo test -p easywar-app neural_ai`，确认 Rust 与 PyTorch 的固定轨迹动作一致。
