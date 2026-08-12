# EasyWar 1v1 强化学习

Rust 提供权威无头环境，Python/PyTorch 负责 GPU 训练。H 图只保留为不产生梯度的接口夹具；当前训练只使用双线梯形和编织双环。双层三角 1v1 投影暂停训练，只保留地图资产与静态回归；外环加横梁保持为完全留出的泛化考试图。

## 环境

```bash
cd training
uv sync --python 3.12
```

`uv` 会创建项目本地 `.venv`，构建 PyO3 扩展并安装 PyTorch/TorchRL。Apple Silicon 默认优先使用 MPS，NVIDIA 环境默认优先使用 CUDA。

## 真人试玩数据采集

从仓库根目录正常启动游戏即可自动埋点：

```bash
cargo run -p easywar-app
```

默认输出到 `training/telemetry/`；设置 `EASYWAR_TELEMETRY=0` 可关闭，也可将变量值直接设为其他目录。当前只采集双人地图，写入失败不会中断游戏。每局一个 JSONL，包含：

- `session_started`：地图、难度、输入方式和观察/动作空间版本；
- `decision`：玩家实际命令、AI 实际命令，以及每秒一次的玩家等待样本；所有动作都绑定决策前的 22 通道观察、玩家级合法动作和战术候选动作；
- `session_ended` / `session_aborted`：胜负、终局占领数，或中途退出。

观察使用稀疏的 `[index, value]` 对保存，缺省位置均为 0，可无损还原为 `RL_OBSERVATION_CHANNELS × 17 × 13` 的稠密数组。`actor_role=player` 才能作为真人模仿标签；AI 记录只用于对照诊断，不能混作真人老师。采集文件已被 Git 忽略，且在建立去重、质量筛选、训练/验证隔离之前不会自动进入训练。

一局结束后可先复验原始双方指令是否在权威逻辑中得到相同胜者与终局时间：

```bash
cargo run -p easywar-app -- \
  --verify-telemetry training/telemetry/<对局文件>.jsonl
```

随后检查样本并训练“玩家影子”。玩家影子不是最终 AI，而是把多局真人的“局面 → 操作”变成会随局势响应的冻结对手：

```bash
cd training
uv run python human_replay.py telemetry
uv run python train_human_shadow.py telemetry \
  --initialize-from checkpoints/selfplay-league-g001.pt \
  --checkpoint checkpoints/human-shadow.pt
```

数据按整局拆分训练集和验证集，同一局的相邻状态不会跨集合泄漏；每秒等待会相对真实指令下采样。默认使用玩家级完整合法动作，而不是旧战术掩码，因为真人可能做出手工边界从未开放的有效操作。得到玩家影子后，将其与 V10、V11 一起加入冻结对手池，再使用纯终局胜负训练候选；候选仍需通过 V11 冠军赛、真人影子留出局和 easy/normal/hard 回归，不能只针对某一局晋级。

## 最小正交冒烟训练

```bash
uv run python train_ppo.py \
  --phase main \
  --rule-opponents easy normal hard \
  --num-envs 24 \
  --imitation-updates 2 \
  --rollout-steps 8 \
  --updates 1 \
  --epochs 1
```

## 正式课程

```bash
uv run python train_ppo.py \
  --phase main --rule-opponents easy normal hard --teacher normal \
  --initialize-from checkpoints/main-stage2-low-lr-best.pt \
  --anchor-checkpoint checkpoints/main-stage2-low-lr-best.pt \
  --anchor-kl-coef 0.05 \
  --num-envs 24 --imitation-updates 0 --updates 20 --learning-rate 3e-5 \
  --validation-every 5 --validation-episodes 16 \
  --validation-opponents easy normal hard \
  --minimum-validation-completion 0.8 --minimum-validation-win-rate 0.5 \
  --checkpoint checkpoints/main.pt \
  --best-checkpoint checkpoints/main-best.pt \
  --report runs/main.jsonl
```

模仿阶段只使用与玩家相同的观察和合法动作，不读取隐藏状态；其作用是让策略先学会规则 AI 的基本动作，再由 PPO 优化胜负目标。

`--tactical-actions` 启用动态成本动作边界：从空闲己方据点出发，只要完整路径成本可承担且不会经过中间据点，非相邻目标也可以直接成为合法动作；重复兵流、活动源直接换线和会被己方枢纽截留的长线仍被排除。成本包含目标恢复/生产、行军期间恢复、沿途防御和在途双方兵力。当前观察为 22 通道，新增目标恢复、双方对目标的在途投入以及归一化坐标；动作头还用全图池化上下文修正源点与目标评分。在线 DAgger 使用固定容量蓄水池混合复习整局阶段，避免只学习当前时刻后遗忘开局。

未收束终局必须比正常失败罚得更重。V8 课程前，`Stalemate/CycleDetected/BudgetExceeded=-0.5`、`Lost=-1` 会诱导模型拖满预算；现统一把未收束终局设为 `-1.1`。V8 从 V7 初始化并以 V7 为 KL 锚点，使用 `1e-5` 学习率、`0.3` 采样温度和 30 次 normal/hard PPO，训练内最佳位于 update 25。

V9 修正了 V7/V8 可通过“活动源每秒改派不同目标”绕过重复兵流过滤的问题：普通进攻与战略补给只能从空闲据点发起，活动源必须先停流再换线，只有受攻击且低于 40% 上限的己方据点允许紧急转防；后方到安全前线的补给要求目标确为通往非己方目标的首个中转据点、低于 80% 上限，且源驻军不少于 30。完成任务的己方补给流若继续等待会收到即时负奖励。正式 V9 使用 V8 初始化、hard 老师、`normal:hard=1:2` 状态采样、48 环境、600 次 DAgger、`3e-5` 学习率、`0.2` 模型采样概率与指令权重 8。后续 PPO 虽提高停流率和正常完赛率，但总胜率下降，因此导出模仿检查点而非 PPO 终点。

V10 先修复老师与动作边界的契约：完整成本估算已证明可承担且无中间据点时，不再额外要求目标紧邻己方领土；受威胁据点的增援优先由能直达的最近前线发起，后方无法越过己方枢纽假装直达。修复后 hard 规则老师经战术掩码对 normal/hard 的 8 组四向控制从 `3/8、0/8` 恢复到 `8/8、4/8`，与未加战术掩码一致。基于新老师完成 hard DAgger，再将 V9 与平衡微调候选按浮点权重 `50%:50%` 插值；同种子 96 局为 easy 32/32、normal 28/32、hard 23/32，另组留出 48 局为 16/16、16/16、11/16。时间差分和加深卷积候选均未超过该模型，因此未导出。

## 无老师自博弈联盟

V11 从 V10 初始化，但训练过程不调用 `expert_actions`，也不使用规则老师动作标签。候选与 V6/V8/V9/V10 及近期冻结候选组成的历史池对局，只保留胜 `+1`、负 `-1`、循环或超限 `-1.1` 的终局奖励。冻结对手按概率采样动作，避免确定性策略反复落入同一条循环。

```bash
uv run python train_selfplay_league.py \
  --generations 1 --updates 24 \
  --num-envs 32 --rollout-steps 256 \
  --direct-episodes 64 --rule-episodes 8
```

每代候选必须对当前冠军达到至少 52% 的总样本胜率和 75% 的完赛率，而且任一规则基准的胜率或完赛率下降不得超过 12.5%；未通过者只进入历史对手库，不替换冠军。第一代正式训练完成 196,608 次决策、556 局终局，晋级赛为 35 胜、17 负、12 次循环。独立 144 局中 V11 为 136 胜、8 负且全部完赛；Rust 只对 V11 使用固定种子的温度 0.5 采样，V5～V10 仍保持确定性推理。

规则老师的大部分时间会等待积累兵力，因此模仿损失默认把非等待指令按 `--command-weight 6` 加权，避免模型仅靠预测等待获得虚高准确率。日志会同时显示总动作准确率与指令准确率。采样默认以 `--dagger-model-prob 0.5` 混入模型自己的动作，再让老师标注由错误动作到达的状态，以降低纯行为克隆的分布偏移。

`--initialize-from` 只加载权重，用于切换课程或对手；`--resume` 恢复模型、优化器、累计更新数和下一段环境种子，只允许相同阶段、对手和老师。更新数表示累计目标，例如已完成 20 次 PPO 后以 `--updates 30` 恢复，只会继续 21～30：

```bash
uv run python train_ppo.py \
  --phase main --rule-opponents easy normal hard --teacher normal \
  --resume checkpoints/main.pt \
  --imitation-updates 1500 --updates 30
```

训练内验证只使用当前训练阶段的地图和独立种子。每个“训练图 × 规则难度”因子都必须同时达到 `--minimum-validation-completion` 和 `--minimum-validation-win-rate`；这里的胜率以全部样本为分母，因此僵局、循环和预算终止不会被隐藏。通过后才按“总样本取胜率优先、正常完赛率打破平局”选择最佳模型。`--anchor-checkpoint` 冻结旧模型，以 `KL(旧策略 || 当前策略)` 约束模仿和 PPO；锚点路径与系数写入检查点，恢复时不得改变。`--early-stop-patience` 可在连续多次无改进后提前停止。

评测和训练内验证还会从玩家可见观察与动作中记录逐局行为：首次指令与首次进攻决策、按席位自同构归一的首攻目标、同源据点改派目标的换线代理、对失守格的反攻、进攻目标覆盖、重复指令，以及按“地图 × 对手 × 席位 × 提交序”和终局分层的汇总。代表回放会复验这些行为指标。行为指标当前只用于诊断，不参与最佳模型硬门禁；尤其不能把跨地图或跨席位聚合后的首攻目标数误报成同一条件下的策略多样性。

## 可控多策略实验

模型支持把策略编号作为显式输入。四策略课程用 96 个环境完整正交覆盖“2 张地图 × 3 档难度 × 2 个席位 × 2 种提交顺序 × 4 个策略”。策略 0 保留为基准；策略 1～3 可分别训练为主动压制、中立扩张和关联铺路。`--strategy-adapter-only` 会冻结主体网络及基准策略，只更新新增策略向量：

```bash
uv run python train_ppo.py \
  --strategy-count 4 --num-envs 96 \
  --strategy-adapter-only \
  --strategy-diversity-coef 0.1 \
  --strategy-specialization-coef 0.2 \
  --initialize-from checkpoints/anchored-multidifficulty-course-best.pt \
  --anchor-checkpoint checkpoints/anchored-multidifficulty-course-best.pt \
  --anchor-kl-coef 0.5 \
  --validation-every 4 --validation-episodes 4
```

独立评测后可生成不要求读者理解 JSON 字段的中文报告：

```bash
uv run python behavior_report.py \
  runs/multistrategy-easy.json \
  runs/multistrategy-normal.json \
  runs/multistrategy-hard.json \
  --output runs/multistrategy-readable.md
```

策略向量不同或 JS 散度上升，只能证明控制输入影响了策略分布。稳定分化必须在同地图、同难度、同席位、同提交序下比较；首攻最高票平票时不得按记录顺序挑出“主开局”，训练意图命中率也必须报告相对同条件基准的增减。

## 导出游戏内神经模型

游戏不依赖 Python 或 ONNX 运行库。导出器只保留策略 0 的两层卷积和动作头，生成约 216 KiB 的小端权重文件；Rust 与 PyTorch 必须在固定真实观察上选择相同动作：

```bash
uv run python export_game_model.py \
  checkpoints/unified-v6-focused2-best.pt \
  --strategy-id 0 \
  --output ../assets/models/neural_v6.ewnn

cargo test -p easywar-app neural_ai
```

## 历史模型对手池

```bash
uv run python train_ppo.py \
  --phase main --rule-opponents normal \
  --historical-opponent checkpoints/main-stage1-best.pt checkpoints/main-stage2-low-lr-best.pt \
  --initialize-from checkpoints/main-stage2-low-lr-best.pt \
  --anchor-checkpoint checkpoints/main-stage2-low-lr-best.pt --anchor-kl-coef 0.05 \
  --num-envs 24 --updates 20 --learning-rate 3e-5 \
  --validation-every 5 --validation-opponents easy normal hard \
  --checkpoint checkpoints/historical-pool.pt
```

历史模型全部冻结并稳定轮换。批量环境按地图、席位和双方提交顺序正交分配；规则对手课程按地图、难度和席位正交分配。对手池清单写入检查点，恢复训练时不得改变。规则 AI 验证仍用于识别对旧能力的遗忘。

## 留出考试

```bash
uv run python evaluate.py checkpoints/main-best.pt \
  --map ring_chord_1v1.toml \
  --opponent normal \
  --episodes 100 \
  --json runs/heldout.json \
  --replays runs/heldout-replays.json

uv run python replay.py runs/heldout-replays.json
```

考试图不得参与梯度更新或模型挑选。模型胜率用于验证 AI 泛化，不作为地图平衡或趣味性的单独证据。
