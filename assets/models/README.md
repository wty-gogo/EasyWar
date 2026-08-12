# 游戏内模型

`neural_v6.ewnn` 是菜单“神经模型 V6”的内嵌推理权重：

- 来源：`training/checkpoints/unified-v6-focused2-best.pt`
- 使用策略：`strategy_id = 0`（基准策略）
- 格式：`EWNNv1`，小端 `f32`，只包含两层卷积和动作头
- SHA-256：`af6499172f08b21d7f95e924944869e3ae747063898cf66cb65ed15c25580a1f`
- 支持地图：`dual_ladder_1v1.toml`、`braided_rings_1v1.toml`

`neural_v5.ewnn` 保留为同种子训练和泛化对照基线，并由菜单“神经模型 V5”独立加载：

- 来源：`training/checkpoints/multistrategy-course-v5-best.pt` 的 `strategy_id = 0`；
- SHA-256：`e175adcecb8ed09a437670e6a352ecbd5ac8095f36f3b8ef4997977540016efc`；
- 支持地图与 V6 相同。

`neural_v7.ewnn` 是菜单“神经模型 V7·战术实验”的内嵌推理权重：

- 来源：`training/checkpoints/tactical-v7-replay-full.pt`；
- 格式：`EWNNv2`，在 v1 动作头上增加源点/目标的全局上下文投影；
- 观察：22 通道，并启用动态击穿成本的战术动作边界；
- SHA-256：`7a6b0896038007cc269cdb4f60d454f3d820fb258f859e5cfc4933cc2af12c70`；
- 支持地图与 V5/V6 相同，但尚未通过 hard 发布门禁，只作为真人比较实验项。

`neural_v8.ewnn` 是菜单“神经模型 V8·强化实验”的内嵌推理权重：

- 来源：`training/checkpoints/tactical-v8-ppo-best.pt`，锚定 V7 后对 normal/hard 执行 30 次低学习率 PPO，最佳点为 update 25；
- 格式与观察：`EWNNv2`、22 通道、动态成本战术动作边界；
- SHA-256：`433339a975e58452e6168ec9dd40dd6658606408806829b08d2477f069a8c5e2`；
- 独立同种子 96 局相对 V7 从 46 胜提升到 52 胜，easy 均为 32/32，normal 从 14/32 提升到 20/32，hard 均为 0/32；
- 仍未通过 hard 发布门禁，只作为强化效果对比实验项。

`neural_v9.ewnn` 是菜单“神经模型 V9·蓄兵实验”的内嵌推理权重：

- 来源：`training/checkpoints/tactical-v9-logistics-bc.pt`，从 V8 初始化，以 hard 老师和 `normal:hard=1:2` 状态分布完成 600 次 DAgger；
- 动作边界：活动进攻源必须先停流再换线，仅允许紧急转防；后方可向低于 80% 上限、且确为通往非己方目标首个中转点的安全前线据点补给；
- SHA-256：`8c91c69561cb4998ecf70c0a88053caa1afdd46d29f7c093f1de1748e6b45ce4`；
- 原 V9 边界下独立同种子 96 局为 easy 32/32、normal 16/32、hard 0/32；V10 扩展可承担长程动作后，冻结 V9 在当前共享边界下复测为 32/32、13/32、0/32；
- 后续停流奖励 PPO 提高了停流率和正常完赛率，但降低总胜率，因此没有覆盖该模仿检查点。hard 仍未通过，只作为真人比较实验项。

`neural_v10.ewnn` 是菜单“神经模型 V10·长程实验”的内嵌推理权重：

- 来源：`training/checkpoints/tactical-v10-blend-50.pt`，对 V9 与修复长程动作契约后的平衡 DAgger 候选做 50% 同源权重插值；
- 格式与观察：`EWNNv2`、22 通道、动态成本战术动作边界；
- SHA-256：`3133e3db68fa6111805a00b5562f4f244c5847374f4e07db56600121a87556a4`；
- 同种子 96 局为 easy 32/32、normal 28/32、hard 23/32；另组未参与筛选的 48 局为 16/16、16/16、11/16；
- 自动胜率只批准它作为更强的真人试玩对比项，不构成地图平衡结论。

`neural_v11.ewnn` 是菜单“神经模型 V11·自博弈”的内嵌推理权重：

- 来源：`training/checkpoints/selfplay-league-g001.pt`，从 V10 初始化，不使用规则老师动作，以 6 个冻结历史对手和纯终局胜负完成 196,608 次训练决策、556 局终局；
- 推理：`EWNNv2`、22 通道、动态成本战术动作边界；Rust 使用固定种子的温度 0.5 采样，V5～V10 的确定性行为不变；
- SHA-256：`98a701ead455deba5ca068ab1406b2afa4bd4ddc548f3bb416320c13bafc35c4`；
- 对 V10 的晋级赛为 35 胜、17 负、12 次循环；独立 144 局规则对照为 136 胜、8 负、无未完赛，V10 同种子为 130 胜、8 负、5 次循环、1 次预算终止；
- 三张独立图的 hard 为 40/48，其中完全留出的外环横梁为 15/16，因此 V11 额外在该图启用；自动结果仍需真人试玩确认手感。

快捷键 `4`、`5`、`6`、`7`、`8`、`9`、`0` 分别选择 V5、V6、V7、V8、V9、V10、V11。

重新导出后必须运行 `cargo test -p easywar-app neural_ai`，确认 Rust 与 PyTorch 的固定轨迹动作一致；采样模型还要校验相同温度、相同随机数下的动作序列。
