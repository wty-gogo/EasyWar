# EasyWar 架构决策记录（ARCHITECTURE）v1

> 状态：经拷问式逐条确认（2026-08-05）。
> 本文档是**技术架构的唯一事实来源**；玩法见 `docs/GDD.md`，数值见 `docs/BALANCE.md`，计划见 `docs/ROADMAP.md`。

## 0. 驱动力

- app 层失控：`main.rs` 膨胀至 1014 行单文件，菜单/对局/渲染同步/输入/HUD 全在一起。
- M4（3~6 阵营混战、WASM、RL 训练管线）要求干净的无头逻辑层与可替换表现层。
- 旧架构"逻辑结构体 + 每帧镜像进 ECS"的 sync 层（`sync_cells`/`sync_squads`）是最丑、最易腐的部分。

## 1. 决策记录

| # | 决策点 | 结论 |
|---|---|---|
| 1 | crate 划分 | 保留 `crates/logic` / `crates/app` 两 crate。价值在编译期强制力：logic 依赖谁由 Cargo.toml 把关 |
| 2 | 统一范式 | 全 ECS。**logic 依赖白名单 = `bevy_ecs` + `bevy_app` + `serde` + `toml`**；`bevy_render`/`bevy_winit` 等一律禁入 |
| 3 | 宿主形态 | logic 暴露 `GamePlugin`。app = DefaultPlugins + GamePlugin + 表现层；无头/RL = MinimalPlugins + GamePlugin。无头冒烟入口收进 logic（`--headless` 二进制），让"逻辑可独立运行"成为 CI 天天验证的事实 |
| 4 | tick 模型 | 自定义 `SimTick` schedule，宿主显式驱动（app 侧 driver 累积 dt；无头/RL 直接 `run_schedule`）。tick 内**单线程、显式链式排序**，迭代顺序即确定性 |
| 5 | 存储形态 | 全实体化：格子/小队/兵流都是实体；`GridLookup` 资源（CellIdx→Entity）服务寻路。禁止"格子住资源、精灵住实体"的混合形态——那会把镜像层请回来 |
| 6 | 写入口 | 意图事件：玩家输入与 AI 决策都发 `Intent`；`apply_intents` 位于 SimTick 链首统一应用。**渲染层只读逻辑组件，永不写**；意图管道天然解锁回放/RL 轨迹 |
| 7 | 模块布局 | logic 按模拟步骤切；app 按界面职责切（main.rs 收敛为 <100 行组合根） |
| 8 | 迁移路径 | 五步 staging（见 §3），重写期**行为冻结**，第 1~3 步时间盒 5 个工作日 |

## 2. 目标结构

```
crates/logic/src/
  lib.rs        # re-export
  plugin.rs     # GamePlugin + SimTick schedule 定义（链序在此一处声明）
  components.rs # 纯数据组件，无逻辑
  intents.rs    # Intent 枚举 + IntentQueue + apply_intents（写在最前）
  map.rs        # 地图/据点生成 + GridLookup（TOML 加载）
  economy.rs    # 产兵 + 回防
  streams.rs    # 兵流建立/改道/终止/出兵节奏
  movement.rs   # 小队行军 + 寻路
  combat.rs     # 双向抵扣 ×3 + 占领
  victory.rs    # 胜负判定
  ai.rs         # AI 决策系统（发 Intent，与玩家同管道）
  rl.rs         # 固定观察/动作契约 + 外部策略控制器
  bin/headless.rs

crates/app/src/
  main.rs    # 组合根：插件装配、State 注册（<100 行）
  menu.rs / input.rs / neural_ai.rs / render.rs / overlay.rs / hud.rs / ending.rs / driver.rs
```

`SimTick` 链序（与旧循环严格同构，黄金快照验证过 parity）：
`apply_intents → economy → streams → movement → combat → victory → ai_decide → policy_decide`
（AI 在链尾决策，看到本 tick 最新状态；意图在下一 tick 链首统一生效——与旧「update → AI → 立即生效」循环等价）

## 3. 迁移路径（staging）

- [x] **第 0 步**：git 初始化；黄金快照（`logic/tests/golden.{rs,snap}`：三个确定性场景，每 100 tick 采样粗粒度不变量；比对规则：胜者必须一致、兵力/小队数 ±2、结束 tick ±10%）。
- [x] **第 1 步**：logic 原地 ECS 重写（不碰 app）。（2026-08-05 完成）
- [x] **第 2 步**：无头先行——复活全部测试 + `headless` 二进制对黄金快照。（14+1 测试全绿；修复 AI 计时器初值 parity bug）
- [x] **第 3 步**：port app 表现层。（2026-08-05 完成：1014 行 main.rs 拆为组合根 + common/driver/render/input/overlay/hud/menu/ending 八模块；逻辑实体与渲染实体共存同一 World，渲染只读组件；`--auto` 冒烟 60s 零 panic）
- [ ] **第 4 步**：真人试玩，恢复 M2/M3 节奏调优。

## 4. 铁律（违反即架构腐化）

1. logic 的 Cargo.toml 依赖白名单外一律禁入。
2. 组件的 `&mut` 只出现在 `SimTick` 系统里；app 拿不到逻辑组件的可变引用。
3. 玩家与 AI 共用同一条意图管道——AI 作弊在结构上不可能。
4. 重写期行为冻结：不顺手改玩法、不调数值，否则黄金快照失去对照意义。
