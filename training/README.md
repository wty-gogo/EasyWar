# EasyWar 1v1 强化学习

Rust 提供权威无头环境，Python/PyTorch 负责 GPU 训练。H 图只保留为不产生梯度的接口夹具；当前训练只使用双线梯形和编织双环。双层三角 1v1 投影暂停训练，只保留地图资产与静态回归；外环加横梁保持为完全留出的泛化考试图。

## 环境

```bash
cd training
uv sync --python 3.12
```

`uv` 会创建项目本地 `.venv`，构建 PyO3 扩展并安装 PyTorch/TorchRL。Apple Silicon 默认优先使用 MPS，NVIDIA 环境默认优先使用 CUDA。

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

## 导出游戏内神经模型 V5

游戏不依赖 Python 或 ONNX 运行库。导出器只保留策略 0 的两层卷积和动作头，生成约 216 KiB 的小端权重文件；Rust 与 PyTorch 必须在固定真实观察上选择相同动作：

```bash
uv run python export_game_model.py \
  checkpoints/multistrategy-course-v5-best.pt \
  --strategy-id 0 \
  --output ../assets/models/neural_v5.ewnn

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
