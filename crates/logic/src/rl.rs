//! 强化学习环境：以玩家级派兵意图驱动权威逻辑，并输出视角归一的固定尺寸观察。
//!
//! 第一阶段仍以 ECS 调度作为参考后端。接口刻意不暴露 ECS，后续可在保持逐步
//! 一致的前提下替换为批量纯数据内核。

use crate::board::Board;
use crate::world_ext::load_squads;
use crate::*;
use bevy_app::App;
use bevy_ecs::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::thread;

pub const RL_MAX_WIDTH: usize = 17;
pub const RL_MAX_HEIGHT: usize = 13;
pub const RL_MAX_CELLS: usize = RL_MAX_WIDTH * RL_MAX_HEIGHT;
pub const RL_MAX_BASES: usize = 16;
pub const RL_OBSERVATION_CHANNELS: usize = 22;
pub const RL_OBSERVATION_LEN: usize = RL_OBSERVATION_CHANNELS * RL_MAX_WIDTH * RL_MAX_HEIGHT;
pub const RL_ACTION_COUNT: usize = 1 + RL_MAX_BASES * RL_MAX_CELLS + RL_MAX_BASES;

const NO_OP_ACTION: usize = 0;
const SET_STREAM_START: usize = 1;
const STOP_STREAM_START: usize = SET_STREAM_START + RL_MAX_BASES * RL_MAX_CELLS;

#[derive(Clone, Debug)]
pub struct RlConfig {
    pub map_path: PathBuf,
    pub subjects_dir: PathBuf,
    pub seed: u64,
    pub learner_faction: FactionId,
    pub opponent_faction: FactionId,
    pub opponent_params: AiParams,
    /// 为真时由调用方同时提交双方动作，不创建规则 AI 控制器。
    pub external_opponent: bool,
    /// 使用共享战术估算收窄动作空间；旧模型与玩家级合法动作保持关闭。
    pub tactical_actions: bool,
    pub submit_order: SubmitOrder,
    pub seat_transform: SeatTransform,
    pub decision_interval_seconds: f32,
    /// 无领地变化、无新兵流目标变化的最大持续时间。
    pub stagnation_seconds: f32,
    /// 工程保险预算，不代表玩法时间上限；触发后单独报告。
    pub max_decisions: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeatTransform {
    Identity,
    Vertical,
    Rotational,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubmitOrder {
    LearnerFirst,
    OpponentFirst,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpisodeEnd {
    Ongoing,
    Won,
    Lost,
    Stalemate,
    CycleDetected,
    BudgetExceeded,
}

impl EpisodeEnd {
    pub fn is_terminal(self) -> bool {
        self != Self::Ongoing
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RlObservation {
    pub values: Vec<f32>,
    pub action_mask: Vec<bool>,
    /// 据点槽位对应的固定观察网格下标；`-1` 表示未使用槽位。
    pub base_cells: Vec<i32>,
    pub width: usize,
    pub height: usize,
    pub time: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RlStep {
    pub observation: RlObservation,
    pub reward: f32,
    pub end: EpisodeEnd,
    pub action_applied: bool,
    pub opponent_action_applied: bool,
    pub decision: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RlAction {
    NoOp,
    SetStream {
        base_slot: usize,
        target_grid: usize,
    },
    StopStream {
        base_slot: usize,
    },
}

/// 可替换决策器的最小契约。规则策略、神经网络和回放策略共用此接口。
pub trait Policy: Send + Sync {
    fn select_action(&mut self, observation: &RlObservation) -> usize;
}

pub struct PolicyController {
    pub faction: FactionId,
    pub opponent: FactionId,
    decision_interval: f32,
    timer: f32,
    tactical_actions: bool,
    policy: Box<dyn Policy>,
}

impl PolicyController {
    pub fn new(
        faction: FactionId,
        opponent: FactionId,
        decision_interval: f32,
        policy: Box<dyn Policy>,
    ) -> Self {
        Self {
            faction,
            opponent,
            decision_interval,
            timer: decision_interval,
            tactical_actions: false,
            policy,
        }
    }

    pub fn new_tactical(
        faction: FactionId,
        opponent: FactionId,
        decision_interval: f32,
        policy: Box<dyn Policy>,
    ) -> Self {
        Self {
            faction,
            opponent,
            decision_interval,
            timer: decision_interval,
            tactical_actions: true,
            policy,
        }
    }
}

#[derive(Resource, Default)]
pub struct PolicyControllers(pub Vec<PolicyController>);

/// 为实时游戏中的外部策略生成与训练完全相同的玩家视角观察。
pub fn observe_world(
    world: &mut World,
    learner: FactionId,
    opponent: FactionId,
) -> Result<RlObservation, String> {
    let base_cells = sorted_base_cells(world)?;
    observation(world, &base_cells, learner, opponent, false)
}

pub fn observe_world_tactical(
    world: &mut World,
    learner: FactionId,
    opponent: FactionId,
) -> Result<RlObservation, String> {
    let base_cells = sorted_base_cells(world)?;
    observation(world, &base_cells, learner, opponent, true)
}

/// 把固定动作空间编号还原为玩家意图；等待动作返回 `None`。
pub fn world_action_to_intent(
    world: &World,
    learner: FactionId,
    action_id: usize,
) -> Result<Option<Intent>, String> {
    let base_cells = sorted_base_cells(world)?;
    match decode_action(action_id)? {
        RlAction::NoOp => Ok(None),
        RlAction::SetStream {
            base_slot,
            target_grid,
        } => base_cells
            .get(base_slot)
            .copied()
            .zip(cell_from_fixed_grid(
                world.resource::<GridLookup>(),
                target_grid,
            ))
            .map(|(source, target)| Intent::SetStream {
                faction: learner,
                source,
                target,
            })
            .ok_or_else(|| "神经网络派兵动作无法映射到当前地图".to_string())
            .map(Some),
        RlAction::StopStream { base_slot } => base_cells
            .get(base_slot)
            .copied()
            .map(|source| Intent::StopStream {
                faction: learner,
                source,
            })
            .ok_or_else(|| "神经网络停流动作无法映射到当前据点".to_string())
            .map(Some),
    }
}

/// 把玩家意图编码为与训练一致的固定动作编号，供回放和真人埋点复用。
pub fn world_intent_to_action(world: &World, intent: Intent) -> Result<usize, String> {
    let base_cells = sorted_base_cells(world)?;
    intent_to_action(world, intent, &base_cells)
}

/// SimTick 链尾：外部策略按训练决策间隔读取观察，并把合法动作写入统一意图队列。
pub fn policy_decide(world: &mut World) {
    if world.resource::<Winner>().0.is_some() {
        return;
    }
    let Some(mut controllers) = world.remove_resource::<PolicyControllers>() else {
        return;
    };
    let mut intents = Vec::new();
    controllers.0.iter_mut().for_each(|controller| {
        controller.timer -= SIM_DT;
        if controller.timer > 0.0 {
            return;
        }
        controller.timer = controller.decision_interval;
        let observed = if controller.tactical_actions {
            observe_world_tactical(world, controller.faction, controller.opponent)
        } else {
            observe_world(world, controller.faction, controller.opponent)
        };
        let Ok(observation) = observed else {
            return;
        };
        let action = controller.policy.select_action(&observation);
        let legal = observation
            .action_mask
            .get(action)
            .copied()
            .unwrap_or(false);
        if legal {
            if let Ok(Some(intent)) = world_action_to_intent(world, controller.faction, action) {
                intents.push(intent);
            }
        }
    });
    world.insert_resource(controllers);
    let mut queue = world.resource_mut::<IntentQueue>();
    intents.into_iter().for_each(|intent| queue.push(intent));
}

pub struct RlEnv {
    config: RlConfig,
    world: World,
    base_cells: Vec<CellIdx>,
    previous_potential: f32,
    last_progress_signature: u64,
    last_progress_time: f32,
    seen_states: HashMap<u64, usize>,
    decisions: usize,
    end: EpisodeEnd,
}

impl RlEnv {
    pub fn new(config: RlConfig) -> Result<Self, String> {
        validate_config(&config)?;
        let mut world = build_world(&config)?;
        let base_cells = sorted_base_cells(&world)?;
        if base_cells.len() > RL_MAX_BASES {
            return Err(format!(
                "强化学习环境最多编码 {RL_MAX_BASES} 个据点，地图实际为 {} 个",
                base_cells.len()
            ));
        }
        let previous_potential =
            strategic_potential(&mut world, config.learner_faction, config.opponent_faction);
        let last_progress_signature = progress_signature(&mut world);
        let mut environment = Self {
            config,
            world,
            base_cells,
            previous_potential,
            last_progress_signature,
            last_progress_time: 0.0,
            seen_states: HashMap::new(),
            decisions: 0,
            end: EpisodeEnd::Ongoing,
        };
        let initial_state = dynamic_state_signature(&mut environment.world);
        environment.seen_states.insert(initial_state, 1);
        Ok(environment)
    }

    pub fn reset(&mut self, seed: u64) -> Result<RlObservation, String> {
        self.config.seed = seed;
        *self = Self::new(self.config.clone())?;
        self.observe()
    }

    pub fn observe(&mut self) -> Result<RlObservation, String> {
        observation(
            &mut self.world,
            &self.base_cells,
            self.config.learner_faction,
            self.config.opponent_faction,
            self.config.tactical_actions,
        )
    }

    /// 以对手为己方生成同构观察，供外部模型对手决策。
    pub fn observe_opponent(&mut self) -> Result<RlObservation, String> {
        if !self.config.external_opponent {
            return Err("只有外部对手环境可以读取对手观察".into());
        }
        observation(
            &mut self.world,
            &self.base_cells,
            self.config.opponent_faction,
            self.config.learner_faction,
            self.config.tactical_actions,
        )
    }

    /// 返回规则 AI 在同一玩家视角下会选择的合法动作，用于行为克隆热身。
    pub fn expert_action(&mut self, params: AiParams) -> Result<usize, String> {
        let controller =
            AiController::seeded(self.config.learner_faction, params, self.config.seed);
        let intent = crate::ai::decide_now(&mut self.world, &controller);
        let action = intent
            .map(|intent| intent_to_action(&self.world, intent, &self.base_cells))
            .transpose()
            .map(|action| action.unwrap_or(NO_OP_ACTION))?;
        let observation = self.observe()?;
        Ok(observation
            .action_mask
            .get(action)
            .copied()
            .unwrap_or(false)
            .then_some(action)
            .unwrap_or(NO_OP_ACTION))
    }

    pub fn step(&mut self, action_id: usize) -> Result<RlStep, String> {
        if self.config.external_opponent {
            return Err("外部对手环境必须使用 step_external 同时提交双方动作".into());
        }
        self.advance(action_id, None)
    }

    pub fn step_external(
        &mut self,
        learner_action_id: usize,
        opponent_action_id: usize,
    ) -> Result<RlStep, String> {
        if !self.config.external_opponent {
            return Err("规则对手环境不能使用 step_external".into());
        }
        self.advance(learner_action_id, Some(opponent_action_id))
    }

    fn advance(
        &mut self,
        learner_action_id: usize,
        opponent_action_id: Option<usize>,
    ) -> Result<RlStep, String> {
        if self.end.is_terminal() {
            return Err("回合已经结束，请先 reset".into());
        }
        let learner_action = decode_action(learner_action_id)?;
        let opponent_action = opponent_action_id.map(decode_action).transpose()?;
        let tactical_penalty = tactical_action_penalty(
            &mut self.world,
            &self.base_cells,
            self.config.learner_faction,
            learner_action,
        );
        let apply_learner = |world: &mut World| {
            apply_action(
                world,
                &self.base_cells,
                self.config.learner_faction,
                learner_action,
            )
        };
        let apply_opponent = |world: &mut World, action| {
            apply_action(
                world,
                &self.base_cells,
                self.config.opponent_faction,
                action,
            )
        };
        let (action_applied, opponent_action_applied) =
            match (self.config.submit_order, opponent_action) {
                (SubmitOrder::LearnerFirst, Some(action)) => (
                    apply_learner(&mut self.world),
                    apply_opponent(&mut self.world, action),
                ),
                (SubmitOrder::OpponentFirst, Some(action)) => {
                    let opponent_applied = apply_opponent(&mut self.world, action);
                    (apply_learner(&mut self.world), opponent_applied)
                }
                (_, None) => (apply_learner(&mut self.world), false),
            };
        let ticks = (self.config.decision_interval_seconds / SIM_DT)
            .round()
            .max(1.0) as usize;
        for _ in 0..ticks {
            self.world
                .try_run_schedule(SimTick)
                .map_err(|error| format!("SimTick 运行失败: {error}"))?;
            if self.world.resource::<Winner>().0.is_some() {
                break;
            }
        }
        self.decisions += 1;
        self.end = self.classify_end();

        let current_potential = strategic_potential(
            &mut self.world,
            self.config.learner_faction,
            self.config.opponent_faction,
        );
        let shaping = (current_potential - self.previous_potential) * 0.02;
        self.previous_potential = current_potential;
        let reward = match self.end {
            EpisodeEnd::Won => 1.0,
            EpisodeEnd::Lost => -1.0,
            // 未收束曾只罚 -0.5，模型因此会把拖满预算当成比失败更好的策略。
            // 现在让僵局、循环和工程预算超限明确劣于正常失败，推动策略主动收官。
            EpisodeEnd::Stalemate | EpisodeEnd::CycleDetected | EpisodeEnd::BudgetExceeded => -1.1,
            EpisodeEnd::Ongoing => shaping + tactical_penalty,
        };
        Ok(RlStep {
            observation: self.observe()?,
            reward,
            end: self.end,
            action_applied,
            opponent_action_applied,
            decision: self.decisions,
        })
    }

    pub fn end(&self) -> EpisodeEnd {
        self.end
    }

    pub fn decisions(&self) -> usize {
        self.decisions
    }

    pub fn game_time(&self) -> f32 {
        self.world.resource::<GameClock>().time
    }

    fn classify_end(&mut self) -> EpisodeEnd {
        if let Some(winner) = self.world.resource::<Winner>().0 {
            return if winner == self.config.learner_faction {
                EpisodeEnd::Won
            } else {
                EpisodeEnd::Lost
            };
        }
        if self.decisions >= self.config.max_decisions {
            return EpisodeEnd::BudgetExceeded;
        }

        let time = self.world.resource::<GameClock>().time;
        let progress = progress_signature(&mut self.world);
        if progress != self.last_progress_signature {
            self.last_progress_signature = progress;
            self.last_progress_time = time;
        } else if time - self.last_progress_time >= self.config.stagnation_seconds {
            return EpisodeEnd::Stalemate;
        }

        let state = dynamic_state_signature(&mut self.world);
        let repeated = {
            let count = self.seen_states.entry(state).or_default();
            *count += 1;
            *count >= 3
        };
        if repeated && has_active_motion(&mut self.world) {
            EpisodeEnd::CycleDetected
        } else {
            EpisodeEnd::Ongoing
        }
    }
}

pub fn encode_set_stream_action(base_slot: usize, target_grid: usize) -> Result<usize, String> {
    if base_slot >= RL_MAX_BASES || target_grid >= RL_MAX_CELLS {
        return Err("派兵动作超出固定动作空间".into());
    }
    Ok(SET_STREAM_START + base_slot * RL_MAX_CELLS + target_grid)
}

pub fn encode_stop_stream_action(base_slot: usize) -> Result<usize, String> {
    if base_slot >= RL_MAX_BASES {
        return Err("停流动作超出固定动作空间".into());
    }
    Ok(STOP_STREAM_START + base_slot)
}

pub fn decode_action(action_id: usize) -> Result<RlAction, String> {
    match action_id {
        NO_OP_ACTION => Ok(RlAction::NoOp),
        action if action < STOP_STREAM_START => {
            let encoded = action - SET_STREAM_START;
            Ok(RlAction::SetStream {
                base_slot: encoded / RL_MAX_CELLS,
                target_grid: encoded % RL_MAX_CELLS,
            })
        }
        action if action < RL_ACTION_COUNT => Ok(RlAction::StopStream {
            base_slot: action - STOP_STREAM_START,
        }),
        _ => Err(format!("动作 {action_id} 超出 0..{RL_ACTION_COUNT}")),
    }
}

/// 多环境同步推进；线程只影响执行顺序，不影响逐环境结果顺序。
pub fn step_batch(
    environments: &mut [RlEnv],
    actions: &[usize],
    thread_count: usize,
) -> Vec<Result<RlStep, String>> {
    if environments.len() != actions.len() {
        return vec![Err("环境数量与动作数量不一致".into())];
    }
    if environments.is_empty() {
        return Vec::new();
    }
    let workers = thread_count.max(1).min(environments.len());
    let chunk_size = environments.len().div_ceil(workers);
    thread::scope(|scope| {
        let handles = environments
            .chunks_mut(chunk_size)
            .zip(actions.chunks(chunk_size))
            .map(|(environment_chunk, action_chunk)| {
                scope.spawn(move || {
                    environment_chunk
                        .iter_mut()
                        .zip(action_chunk)
                        .map(|(environment, &action)| environment.step(action))
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("强化学习工作线程不应 panic"))
            .collect()
    })
}

/// 外部双方策略的多环境同步推进；环境、学习者动作和对手动作保持稳定对应。
pub fn step_batch_external(
    environments: &mut [RlEnv],
    learner_actions: &[usize],
    opponent_actions: &[usize],
    thread_count: usize,
) -> Vec<Result<RlStep, String>> {
    if environments.len() != learner_actions.len() || environments.len() != opponent_actions.len() {
        return vec![Err("环境数量与双方动作数量不一致".into())];
    }
    if environments.is_empty() {
        return Vec::new();
    }
    let workers = thread_count.max(1).min(environments.len());
    let chunk_size = environments.len().div_ceil(workers);
    thread::scope(|scope| {
        environments
            .chunks_mut(chunk_size)
            .zip(learner_actions.chunks(chunk_size))
            .zip(opponent_actions.chunks(chunk_size))
            .map(|((environment_chunk, learner_chunk), opponent_chunk)| {
                scope.spawn(move || {
                    environment_chunk
                        .iter_mut()
                        .zip(learner_chunk)
                        .zip(opponent_chunk)
                        .map(|((environment, &learner), &opponent)| {
                            environment.step_external(learner, opponent)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .flat_map(|handle| handle.join().expect("强化学习工作线程不应 panic"))
            .collect()
    })
}

fn validate_config(config: &RlConfig) -> Result<(), String> {
    if config.learner_faction == NEUTRAL
        || config.opponent_faction == NEUTRAL
        || config.learner_faction == config.opponent_faction
    {
        return Err("学习者与对手必须是两个不同的非中立阵营".into());
    }
    if !config.decision_interval_seconds.is_finite() || config.decision_interval_seconds <= 0.0 {
        return Err("决策间隔必须大于 0".into());
    }
    if !config.stagnation_seconds.is_finite()
        || config.stagnation_seconds < config.decision_interval_seconds
    {
        return Err("停滞检测窗口不得小于决策间隔".into());
    }
    if config.max_decisions == 0 {
        return Err("工程保险决策预算必须大于 0".into());
    }
    Ok(())
}

fn build_world(config: &RlConfig) -> Result<World, String> {
    let mut app = App::new();
    app.add_plugins(GamePlugin);
    spawn_map_seeded(
        app.world_mut(),
        &config.map_path,
        &config.subjects_dir,
        config.seed,
    )?;
    apply_seat_transform(&mut app, config.seat_transform)?;
    let factions = app.world().resource::<Factions>();
    let ids = factions
        .0
        .iter()
        .map(|faction| faction.id)
        .collect::<HashSet<_>>();
    if ids != HashSet::from([config.learner_faction, config.opponent_faction]) {
        return Err("第一版强化学习环境只支持恰好两个参战阵营的 1v1 地图".into());
    }
    let controllers = (!config.external_opponent)
        .then(|| {
            vec![AiController::seeded(
                config.opponent_faction,
                config.opponent_params,
                config.seed,
            )]
        })
        .unwrap_or_default();
    app.world_mut().insert_resource(AiControllers(controllers));
    Ok(std::mem::take(app.world_mut()))
}

fn apply_seat_transform(app: &mut App, transform: SeatTransform) -> Result<(), String> {
    if transform == SeatTransform::Identity {
        return Ok(());
    }
    let lookup = app.world().resource::<GridLookup>().clone();
    let bases = app.world().resource::<BaseList>().clone();
    bases.0.iter().for_each(|&entity| {
        let mut owner = app
            .world_mut()
            .get_mut::<Owner>(entity)
            .expect("据点缺少 Owner");
        owner.0 = match owner.0 {
            1 => 2,
            2 => 1,
            other => other,
        };
    });
    let transformed = bases
        .0
        .iter()
        .map(|entity| {
            let cell = lookup
                .cells
                .iter()
                .position(|candidate| candidate == entity)
                .ok_or_else(|| "据点不在 GridLookup 中".to_string())?;
            let (x, y) = lookup.xy(cell);
            let (target_x, target_y) = match transform {
                SeatTransform::Identity => (x, y),
                SeatTransform::Vertical => (lookup.width - 1 - x, y),
                SeatTransform::Rotational => (lookup.width - 1 - x, lookup.height - 1 - y),
            };
            let target = lookup.entity(lookup.idx(target_x, target_y));
            app.world()
                .get::<Base>(target)
                .is_some()
                .then_some(target)
                .ok_or_else(|| format!("席位自同构目标 ({target_x}, {target_y}) 不是据点"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    app.world_mut().insert_resource(BaseList(transformed));
    Ok(())
}

fn sorted_base_cells(world: &World) -> Result<Vec<CellIdx>, String> {
    let lookup = world.resource::<GridLookup>();
    let bases = world.resource::<BaseList>();
    let base_entities = bases.0.iter().copied().collect::<HashSet<_>>();
    Ok(lookup
        .cells
        .iter()
        .enumerate()
        .filter_map(|(cell, entity)| base_entities.contains(entity).then_some(cell))
        .collect())
}

fn observation(
    world: &mut World,
    base_cells: &[CellIdx],
    learner: FactionId,
    opponent: FactionId,
    tactical_actions: bool,
) -> Result<RlObservation, String> {
    let lookup = world.resource::<GridLookup>().clone();
    if lookup.width > RL_MAX_WIDTH || lookup.height > RL_MAX_HEIGHT {
        return Err(format!(
            "地图 {}×{} 超出强化学习观察上限 {}×{}",
            lookup.width, lookup.height, RL_MAX_WIDTH, RL_MAX_HEIGHT
        ));
    }
    let board = Board::load(world);
    let mut values = vec![0.0; RL_OBSERVATION_LEN];
    let channel_index = |channel: usize, cell: CellIdx| {
        let (x, y) = lookup.xy(cell);
        channel * RL_MAX_CELLS + y * RL_MAX_WIDTH + x
    };
    let mut associated_owner = vec![NEUTRAL; lookup.cells.len()];
    for &base_cell in base_cells {
        let entity = lookup.entity(base_cell);
        let owner = world.get::<Owner>(entity).expect("据点缺少 Owner").0;
        let base = world.get::<Base>(entity).expect("据点缺少 Base");
        base.linked
            .iter()
            .for_each(|&cell| associated_owner[cell] = owner);
    }
    for (cell, &entity) in lookup.cells.iter().enumerate() {
        let kind = *world.get::<CellKind>(entity).expect("格子缺少 CellKind");
        let owner = world.get::<Owner>(entity).expect("格子缺少 Owner").0;
        let garrison = world.get::<Garrison>(entity).expect("格子缺少 Garrison");
        values[channel_index(0, cell)] = f32::from(kind.enterable());
        values[channel_index(1, cell)] = f32::from(kind == CellKind::LinkedTile);
        values[channel_index(2, cell)] = f32::from(kind == CellKind::Base);
        values[channel_index(3, cell)] = f32::from(owner == learner);
        values[channel_index(4, cell)] = f32::from(owner == opponent);
        values[channel_index(5, cell)] = f32::from(owner == NEUTRAL && kind.enterable());
        values[channel_index(6, cell)] = (garrison.cur / 200.0).clamp(0.0, 5.0);
        values[channel_index(7, cell)] = if garrison.max.is_finite() {
            (garrison.max / 200.0).clamp(0.0, 5.0)
        } else {
            1.0
        };
        values[channel_index(8, cell)] = f32::from(associated_owner[cell] == learner);
        values[channel_index(9, cell)] = f32::from(associated_owner[cell] == opponent);
        values[channel_index(10, cell)] =
            f32::from(associated_owner[cell] == NEUTRAL && kind == CellKind::LinkedTile);
        values[channel_index(17, cell)] = (board.cell_recovery_per_sec(cell) / 5.0).clamp(0.0, 1.0);
        let (x, y) = lookup.xy(cell);
        values[channel_index(20, cell)] = x as f32 / (RL_MAX_WIDTH - 1) as f32;
        values[channel_index(21, cell)] = y as f32 / (RL_MAX_HEIGHT - 1) as f32;
    }

    let mut squad_query = world.query::<&Squad>();
    squad_query.iter(world).for_each(|squad| {
        let channel = if squad.faction == learner { 11 } else { 12 };
        let cell = squad.current_cell();
        let progress = squad.t.clamp(0.0, 1.0);
        values[channel_index(channel, cell)] += squad.troops / 200.0 * (1.0 - progress);
        if let Some(&next) = squad.path.get(squad.seg + 1) {
            values[channel_index(channel, next)] += squad.troops / 200.0 * progress;
        }
        if squad.mode == SquadMode::ToTarget {
            if let Some(&target) = squad.path.last() {
                let commitment_channel = if squad.faction == learner { 18 } else { 19 };
                values[channel_index(commitment_channel, target)] += squad.troops / 200.0;
            }
        }
    });
    let mut stream_query = world.query::<&Stream>();
    stream_query
        .iter(world)
        .filter(|stream| stream.active)
        .for_each(|stream| {
            if stream.faction == learner {
                values[channel_index(13, stream.source)] = 1.0;
                values[channel_index(14, stream.target)] = 1.0;
            } else {
                values[channel_index(16, stream.source)] = 1.0;
                values[channel_index(15, stream.target)] = 1.0;
            }
        });

    let action_mask = action_mask(world, base_cells, learner, tactical_actions);
    let mut encoded_base_cells = vec![-1; RL_MAX_BASES];
    base_cells.iter().enumerate().for_each(|(slot, &cell)| {
        let (x, y) = lookup.xy(cell);
        encoded_base_cells[slot] = (y * RL_MAX_WIDTH + x) as i32;
    });
    Ok(RlObservation {
        values,
        action_mask,
        base_cells: encoded_base_cells,
        width: lookup.width,
        height: lookup.height,
        time: world.resource::<GameClock>().time,
    })
}

fn action_mask(
    world: &mut World,
    base_cells: &[CellIdx],
    learner: FactionId,
    tactical_actions: bool,
) -> Vec<bool> {
    let lookup = world.resource::<GridLookup>().clone();
    let board = Board::load(world);
    let squads = load_squads(world);
    let streams = crate::world_ext::load_streams(world);
    let threatened = |target: CellIdx| {
        squads.iter().any(|(_, squad)| {
            squad.faction != learner
                && squad.mode == SquadMode::ToTarget
                && squad.path.last() == Some(&target)
        })
    };
    let strategic_staging = tactical_actions
        .then(|| strategic_staging_pairs(&board, learner))
        .unwrap_or_default();
    let mut mask = vec![false; RL_ACTION_COUNT];
    mask[NO_OP_ACTION] = true;
    for (slot, &source) in base_cells.iter().enumerate() {
        let source_entity = lookup.entity(source);
        let owned = world
            .get::<Owner>(source_entity)
            .is_some_and(|owner| owner.0 == learner);
        if !owned {
            continue;
        }
        let source_is_active = streams.iter().any(|(_, stream)| {
            stream.active && stream.faction == learner && stream.source == source
        });
        for target in 0..lookup.cells.len() {
            let enterable = world
                .get::<CellKind>(lookup.entity(target))
                .is_some_and(CellKind::enterable);
            let tactically_allowed = !tactical_actions
                || if board.owner[target] != learner {
                    !source_is_active
                        && crate::tactics::estimate_attack_board(
                            &board, &squads, &streams, learner, source, target,
                        )
                        .is_some_and(|estimate| {
                            !estimate.repeats_active_stream
                                && estimate.can_launch(board.garrison[source], 1.0)
                        })
                } else if board.kind[target] == CellKind::Base {
                    let target_base = board.bases.iter().find(|base| base.cell == target);
                    let needs_defense = threatened(target)
                        && target_base.is_some_and(|base| {
                            board.garrison[target] < 0.4 * board.base_garrison_cap(base)
                        });
                    let needs_staging = strategic_staging.contains(&(source, target));
                    (needs_defense || (!source_is_active && needs_staging))
                        && crate::tactics::estimate_attack_board(
                            &board, &squads, &streams, learner, source, target,
                        )
                        .is_some_and(|estimate| {
                            estimate.intermediate_base.is_none() && !estimate.repeats_active_stream
                        })
                } else {
                    false
                };
            if enterable && target != source && tactically_allowed {
                let target_grid = fixed_grid_index(&lookup, target);
                mask[encode_set_stream_action(slot, target_grid).expect("地图尺寸已验证")] = true;
            }
        }
        if stream_from(world, learner, source).is_some() {
            mask[encode_stop_stream_action(slot).expect("据点数量已验证")] = true;
        }
    }
    mask
}

/// 找出“后方据点 → 前线中转据点”的战略蓄兵动作。
///
/// 只有当某个己方据点是从后方源通往非己方目标时遇到的第一个中间据点，且其驻军
/// 尚未达到上限的 80%，才开放补给。这样既能复现困难规则 AI 的分段推进，也不会把
/// 任意安全据点之间的来回倒兵暴露给策略网络。
fn strategic_staging_pairs(board: &Board, learner: FactionId) -> HashSet<(CellIdx, CellIdx)> {
    let objectives = board
        .kind
        .iter()
        .enumerate()
        .filter_map(|(cell, kind)| {
            (kind.enterable() && board.owner[cell] != learner).then_some(cell)
        })
        .collect::<Vec<_>>();

    board
        .bases
        .iter()
        .filter(|source| board.owner[source.cell] == learner && board.garrison[source.cell] >= 30.0)
        .flat_map(|source| {
            objectives.iter().filter_map(move |&objective| {
                let path = board.find_path(source.cell, objective, learner)?;
                let staging = path
                    .iter()
                    .copied()
                    .skip(1)
                    .take(path.len().saturating_sub(2))
                    .find(|&cell| board.kind[cell] == CellKind::Base)?;
                let staging_base = board.bases.iter().find(|base| base.cell == staging)?;
                (board.owner[staging] == learner
                    && board.garrison[staging] < 0.8 * board.base_garrison_cap(staging_base))
                .then_some((source.cell, staging))
            })
        })
        .collect()
}

fn apply_action(
    world: &mut World,
    base_cells: &[CellIdx],
    learner: FactionId,
    action: RlAction,
) -> bool {
    match action {
        RlAction::NoOp => true,
        RlAction::SetStream {
            base_slot,
            target_grid,
        } => base_cells
            .get(base_slot)
            .copied()
            .zip(cell_from_fixed_grid(
                world.resource::<GridLookup>(),
                target_grid,
            ))
            .is_some_and(|(source, target)| {
                dispatch_intent(
                    world,
                    Intent::SetStream {
                        faction: learner,
                        source,
                        target,
                    },
                )
            }),
        RlAction::StopStream { base_slot } => {
            base_cells.get(base_slot).copied().is_some_and(|source| {
                let active = stream_from(world, learner, source).is_some();
                active
                    && dispatch_intent(
                        world,
                        Intent::StopStream {
                            faction: learner,
                            source,
                        },
                    )
            })
        }
    }
}

fn tactical_action_penalty(
    world: &mut World,
    base_cells: &[CellIdx],
    learner: FactionId,
    action: RlAction,
) -> f32 {
    let completed_reinforcements = completed_reinforcement_sources(world, learner);
    let acted_source = match action {
        RlAction::SetStream { base_slot, .. } | RlAction::StopStream { base_slot } => {
            base_cells.get(base_slot).copied()
        }
        RlAction::NoOp => None,
    };
    let maintenance_penalty = (!completed_reinforcements.is_empty()
        && !acted_source.is_some_and(|source| completed_reinforcements.contains(&source)))
    .then_some(-0.02)
    .unwrap_or(0.0);
    let RlAction::SetStream {
        base_slot,
        target_grid,
    } = action
    else {
        return maintenance_penalty;
    };
    let Some((source, target)) = base_cells.get(base_slot).copied().zip(cell_from_fixed_grid(
        world.resource::<GridLookup>(),
        target_grid,
    )) else {
        return maintenance_penalty - 0.02;
    };
    let target_entity = world.resource::<GridLookup>().entity(target);
    let target_kind = *world
        .get::<CellKind>(target_entity)
        .expect("目标格缺少 CellKind");
    let target_owner = world
        .get::<Owner>(target_entity)
        .expect("目标格缺少 Owner")
        .0;
    if target_owner == learner && target_kind != CellKind::Base {
        return maintenance_penalty - 0.01;
    }
    let source_garrison = world
        .get::<Garrison>(world.resource::<GridLookup>().entity(source))
        .map(|garrison| garrison.cur)
        .unwrap_or(0.0);
    let Some(estimate) = crate::tactics::estimate_attack(world, learner, source, target) else {
        return maintenance_penalty - 0.02;
    };
    if estimate.repeats_active_stream {
        return maintenance_penalty - 0.01;
    }
    if estimate.intermediate_base.is_some() || !estimate.required_source_garrison.is_finite() {
        return maintenance_penalty - 0.02;
    }
    if target_owner == learner {
        return maintenance_penalty;
    }
    let shortfall = (estimate.required_source_garrison - source_garrison).max(0.0);
    if estimate.required_source_garrison <= 0.0 {
        maintenance_penalty
    } else {
        maintenance_penalty - 0.02 * (shortfall / estimate.required_source_garrison).clamp(0.0, 1.0)
    }
}

/// 已经完成任务却仍在持续抽兵的己方补给流。
fn completed_reinforcement_sources(world: &mut World, learner: FactionId) -> HashSet<CellIdx> {
    let board = Board::load(world);
    let squads = load_squads(world);
    let streams = crate::world_ext::load_streams(world);
    streams
        .iter()
        .filter_map(|(_, stream)| {
            if !stream.active
                || stream.faction != learner
                || board.kind[stream.target] != CellKind::Base
                || board.owner[stream.target] != learner
            {
                return None;
            }
            let target_base = board.bases.iter().find(|base| base.cell == stream.target)?;
            let threatened = squads.iter().any(|(_, squad)| {
                squad.faction != learner
                    && squad.mode == SquadMode::ToTarget
                    && squad.path.last() == Some(&stream.target)
            });
            let target_ratio = if threatened { 0.4 } else { 0.8 };
            (board.garrison[stream.target] >= target_ratio * board.base_garrison_cap(target_base))
                .then_some(stream.source)
        })
        .collect()
}

fn intent_to_action(
    world: &World,
    intent: Intent,
    base_cells: &[CellIdx],
) -> Result<usize, String> {
    match intent {
        Intent::SetStream { source, target, .. } => {
            let target_grid = fixed_grid_index(world.resource::<GridLookup>(), target);
            base_cells
                .iter()
                .position(|&cell| cell == source)
                .ok_or_else(|| "规则 AI 的派兵源不在据点槽位中".to_string())
                .and_then(|slot| encode_set_stream_action(slot, target_grid))
        }
        Intent::StopStream { source, .. } => base_cells
            .iter()
            .position(|&cell| cell == source)
            .ok_or_else(|| "规则 AI 的停流源不在据点槽位中".to_string())
            .and_then(encode_stop_stream_action),
    }
}

fn fixed_grid_index(lookup: &GridLookup, cell: CellIdx) -> usize {
    let (x, y) = lookup.xy(cell);
    y * RL_MAX_WIDTH + x
}

fn cell_from_fixed_grid(lookup: &GridLookup, target_grid: usize) -> Option<CellIdx> {
    let x = target_grid % RL_MAX_WIDTH;
    let y = target_grid / RL_MAX_WIDTH;
    (x < lookup.width && y < lookup.height).then(|| lookup.idx(x, y))
}

fn strategic_potential(world: &mut World, learner: FactionId, opponent: FactionId) -> f32 {
    let board = Board::load(world);
    let squads = load_squads(world);
    let mut learner_bases = 0.0;
    let mut opponent_bases = 0.0;
    let mut learner_linked = 0.0;
    let mut opponent_linked = 0.0;
    let mut linked_total: f32 = 0.0;
    for (&kind, &owner) in board.kind.iter().zip(&board.owner) {
        if kind == CellKind::LinkedTile {
            linked_total += 1.0;
            learner_linked += f32::from(owner == learner);
            opponent_linked += f32::from(owner == opponent);
        }
    }
    board.bases.iter().for_each(|base| {
        let owner = board.owner[base.cell];
        learner_bases += f32::from(owner == learner);
        opponent_bases += f32::from(owner == opponent);
    });
    let base_total = board.bases.len().max(1) as f32;
    let territory = (learner_linked - opponent_linked) / linked_total.max(1.0);
    let base_advantage = (learner_bases - opponent_bases) / base_total;
    let learner_mobile = crate::ai::total_troops_board(&board, &squads, learner);
    let opponent_mobile = crate::ai::total_troops_board(&board, &squads, opponent);
    let troop_total = learner_mobile + opponent_mobile;
    let troop_advantage = if troop_total > 0.1 {
        (learner_mobile - opponent_mobile) / troop_total
    } else {
        0.0
    };
    0.25 * territory + 0.5 * base_advantage + 0.25 * troop_advantage
}

fn progress_signature(world: &mut World) -> u64 {
    let lookup = world.resource::<GridLookup>().clone();
    let mut hash = Hash64::new();
    lookup.cells.iter().for_each(|&entity| {
        hash.add_u8(world.get::<Owner>(entity).expect("格子缺少 Owner").0);
    });
    let mut streams = world
        .query::<&Stream>()
        .iter(world)
        .filter(|stream| stream.active)
        .map(|stream| (stream.faction, stream.source, stream.target))
        .collect::<Vec<_>>();
    streams.sort_unstable();
    streams.into_iter().for_each(|(faction, source, target)| {
        hash.add_u8(faction);
        hash.add_usize(source);
        hash.add_usize(target);
    });
    hash.finish()
}

fn dynamic_state_signature(world: &mut World) -> u64 {
    let lookup = world.resource::<GridLookup>().clone();
    let mut hash = Hash64::new();
    lookup.cells.iter().for_each(|&entity| {
        hash.add_u8(world.get::<Owner>(entity).expect("格子缺少 Owner").0);
        let garrison = world
            .get::<Garrison>(entity)
            .expect("格子缺少 Garrison")
            .cur;
        hash.add_u32((garrison * 2.0).round().max(0.0) as u32);
    });
    let mut streams = world
        .query::<&Stream>()
        .iter(world)
        .filter(|stream| stream.active)
        .map(|stream| (stream.faction, stream.source, stream.target))
        .collect::<Vec<_>>();
    streams.sort_unstable();
    streams.into_iter().for_each(|(faction, source, target)| {
        hash.add_u8(faction);
        hash.add_usize(source);
        hash.add_usize(target);
    });
    let mut squads = world
        .query::<&Squad>()
        .iter(world)
        .map(|squad| {
            (
                squad.faction,
                squad.current_cell(),
                (squad.troops * 2.0).round().max(0.0) as u32,
            )
        })
        .collect::<Vec<_>>();
    squads.sort_unstable();
    squads.into_iter().for_each(|(faction, cell, troops)| {
        hash.add_u8(faction);
        hash.add_usize(cell);
        hash.add_u32(troops);
    });
    hash.finish()
}

fn has_active_motion(world: &mut World) -> bool {
    let has_stream = world
        .query::<&Stream>()
        .iter(world)
        .any(|stream| stream.active);
    let has_squad = world.query::<&Squad>().iter(world).next().is_some();
    has_stream || has_squad
}

struct Hash64(u64);

impl Hash64 {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn add_u8(&mut self, value: u8) {
        self.0 ^= value as u64;
        self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
    }

    fn add_u32(&mut self, value: u32) {
        value
            .to_le_bytes()
            .into_iter()
            .for_each(|byte| self.add_u8(byte));
    }

    fn add_usize(&mut self, value: usize) {
        (value as u64)
            .to_le_bytes()
            .into_iter()
            .for_each(|byte| self.add_u8(byte));
    }

    fn finish(self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_frontline_above_defense_threshold_can_receive_staging_troops() {
        let assets = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets");
        let mut environment = RlEnv::new(RlConfig {
            map_path: assets.join("maps/dual_ladder_1v1.toml"),
            subjects_dir: assets.join("subjects"),
            seed: 19,
            learner_faction: 1,
            opponent_faction: 2,
            opponent_params: AiParams::hard(),
            external_opponent: true,
            tactical_actions: true,
            submit_order: SubmitOrder::LearnerFirst,
            seat_transform: SeatTransform::Identity,
            decision_interval_seconds: 1.0,
            stagnation_seconds: 300.0,
            max_decisions: 1_200,
        })
        .expect("创建前线蓄兵测试环境失败");

        let lookup = environment.world.resource::<GridLookup>().clone();
        let rear = lookup.idx(1, 6);
        let frontline = lookup.idx(4, 3);
        environment
            .world
            .get_mut::<Owner>(lookup.entity(frontline))
            .expect("前线据点缺少归属")
            .0 = environment.config.learner_faction;
        environment
            .world
            .get_mut::<Garrison>(lookup.entity(rear))
            .expect("后方据点缺少驻军")
            .cur = 50.0;
        environment
            .world
            .get_mut::<Garrison>(lookup.entity(frontline))
            .expect("前线据点缺少驻军")
            .cur = 50.0;

        let observation = environment.observe().expect("读取前线蓄兵观察失败");
        let source_slot = environment
            .base_cells
            .iter()
            .position(|&cell| cell == rear)
            .expect("后方据点应有动作槽位");
        let action = encode_set_stream_action(source_slot, fixed_grid_index(&lookup, frontline))
            .expect("前线蓄兵动作应可编码");

        assert!(
            observation.action_mask[action],
            "安全前线已有 50 兵，高于 40% 防守线但低于 80% 蓄兵线时仍应允许后方补给"
        );
    }

    #[test]
    fn affordable_long_attack_without_intermediate_base_is_allowed() {
        let assets = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets");
        let mut environment = RlEnv::new(RlConfig {
            map_path: assets.join("maps/dual_ladder_1v1.toml"),
            subjects_dir: assets.join("subjects"),
            seed: 31,
            learner_faction: 1,
            opponent_faction: 2,
            opponent_params: AiParams::hard(),
            external_opponent: true,
            tactical_actions: true,
            submit_order: SubmitOrder::LearnerFirst,
            seat_transform: SeatTransform::Identity,
            decision_interval_seconds: 1.0,
            stagnation_seconds: 300.0,
            max_decisions: 1_200,
        })
        .expect("创建长程进攻测试环境失败");

        let lookup = environment.world.resource::<GridLookup>().clone();
        let source = lookup.idx(4, 3);
        let target = lookup.idx(8, 3);
        environment
            .world
            .get_mut::<Owner>(lookup.entity(source))
            .expect("前线据点缺少归属")
            .0 = environment.config.learner_faction;
        environment
            .world
            .get_mut::<Garrison>(lookup.entity(source))
            .expect("前线据点缺少驻军")
            .cur = 240.0;

        let estimate = crate::tactics::estimate_attack(
            &mut environment.world,
            environment.config.learner_faction,
            source,
            target,
        )
        .expect("长程目标应可估算");
        assert!(estimate.intermediate_base.is_none());
        assert!(estimate.can_launch(240.0, 1.0));

        let observation = environment.observe().expect("读取长程进攻观察失败");
        let source_slot = environment
            .base_cells
            .iter()
            .position(|&cell| cell == source)
            .expect("前线据点应有动作槽位");
        let action = encode_set_stream_action(source_slot, fixed_grid_index(&lookup, target))
            .expect("长程进攻动作应可编码");

        assert!(
            observation.action_mask[action],
            "完整成本估算已经证明可承担且路径没有中间据点时，不应再要求目标必须紧邻己方领土"
        );
    }

    #[test]
    fn hard_expert_stages_defense_through_the_first_friendly_base() {
        let assets = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets");
        let mut environment = RlEnv::new(RlConfig {
            map_path: assets.join("maps/dual_ladder_1v1.toml"),
            subjects_dir: assets.join("subjects"),
            seed: 37,
            learner_faction: 1,
            opponent_faction: 2,
            opponent_params: AiParams::hard(),
            external_opponent: true,
            tactical_actions: true,
            submit_order: SubmitOrder::LearnerFirst,
            seat_transform: SeatTransform::Identity,
            decision_interval_seconds: 1.0,
            stagnation_seconds: 300.0,
            max_decisions: 1_200,
        })
        .expect("创建分段防守测试环境失败");

        let lookup = environment.world.resource::<GridLookup>().clone();
        let rear = lookup.idx(1, 6);
        let staging = lookup.idx(4, 3);
        let threatened = lookup.idx(8, 3);
        let enemy_source = lookup.idx(15, 6);
        [(staging, 20.0), (threatened, 10.0), (rear, 160.0)]
            .into_iter()
            .for_each(|(cell, garrison)| {
                environment
                    .world
                    .get_mut::<Owner>(lookup.entity(cell))
                    .expect("测试据点缺少归属")
                    .0 = environment.config.learner_faction;
                environment
                    .world
                    .get_mut::<Garrison>(lookup.entity(cell))
                    .expect("测试据点缺少驻军")
                    .cur = garrison;
            });
        environment.world.spawn(Squad {
            faction: environment.config.opponent_faction,
            troops: 13.0,
            path: vec![enemy_source, threatened],
            seg: 0,
            t: 0.0,
            mode: SquadMode::ToTarget,
            stream: Entity::PLACEHOLDER,
            return_after_target: false,
            seq: 0,
        });

        let action = environment
            .expert_action(AiParams::hard())
            .expect("读取困难专家分段防守动作失败");
        let expected_slot = environment
            .base_cells
            .iter()
            .position(|&cell| cell == staging)
            .expect("前线据点应有动作槽位");
        let expected =
            encode_set_stream_action(expected_slot, fixed_grid_index(&lookup, threatened))
                .expect("前线防守动作应可编码");

        assert_eq!(
            action, expected,
            "后方据点无法越过己方枢纽直达受威胁据点，应改由能直达目标的最近前线增援"
        );
    }

    #[test]
    fn active_offensive_stream_must_stop_before_changing_targets() {
        let assets = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets");
        let mut environment = RlEnv::new(RlConfig {
            map_path: assets.join("maps/dual_ladder_1v1.toml"),
            subjects_dir: assets.join("subjects"),
            seed: 23,
            learner_faction: 1,
            opponent_faction: 2,
            opponent_params: AiParams::hard(),
            external_opponent: true,
            tactical_actions: true,
            submit_order: SubmitOrder::LearnerFirst,
            seat_transform: SeatTransform::Identity,
            decision_interval_seconds: 1.0,
            stagnation_seconds: 300.0,
            max_decisions: 1_200,
        })
        .expect("创建换线约束测试环境失败");

        let lookup = environment.world.resource::<GridLookup>().clone();
        let source = lookup.idx(1, 6);
        let first_target = lookup.idx(2, 6);
        environment
            .world
            .get_mut::<Garrison>(lookup.entity(source))
            .expect("源据点缺少驻军")
            .cur = 80.0;
        assert!(dispatch_intent(
            &mut environment.world,
            Intent::SetStream {
                faction: environment.config.learner_faction,
                source,
                target: first_target,
            },
        ));

        let observation = environment.observe().expect("读取换线约束观察失败");
        let source_slot = environment
            .base_cells
            .iter()
            .position(|&cell| cell == source)
            .expect("源据点应有动作槽位");
        let set_stream_start = SET_STREAM_START + source_slot * RL_MAX_CELLS;
        let set_stream_end = set_stream_start + RL_MAX_CELLS;
        let stop = encode_stop_stream_action(source_slot).expect("停流动作应可编码");

        assert!(
            observation.action_mask[set_stream_start..set_stream_end]
                .iter()
                .all(|&allowed| !allowed),
            "普通进攻兵流未停止前不得每秒改派目标"
        );
        assert!(observation.action_mask[stop], "活动兵流必须保留停流动作");
    }

    #[test]
    fn completed_reinforcement_penalizes_waiting_but_not_stopping() {
        let assets = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets");
        let mut environment = RlEnv::new(RlConfig {
            map_path: assets.join("maps/dual_ladder_1v1.toml"),
            subjects_dir: assets.join("subjects"),
            seed: 29,
            learner_faction: 1,
            opponent_faction: 2,
            opponent_params: AiParams::hard(),
            external_opponent: true,
            tactical_actions: true,
            submit_order: SubmitOrder::LearnerFirst,
            seat_transform: SeatTransform::Identity,
            decision_interval_seconds: 1.0,
            stagnation_seconds: 300.0,
            max_decisions: 1_200,
        })
        .expect("创建补给停流奖励测试环境失败");

        let lookup = environment.world.resource::<GridLookup>().clone();
        let source = lookup.idx(1, 6);
        let target = lookup.idx(4, 3);
        environment
            .world
            .get_mut::<Owner>(lookup.entity(target))
            .expect("补给目标缺少归属")
            .0 = environment.config.learner_faction;
        environment
            .world
            .get_mut::<Garrison>(lookup.entity(target))
            .expect("补给目标缺少驻军")
            .cur = 64.0;
        assert!(dispatch_intent(
            &mut environment.world,
            Intent::SetStream {
                faction: environment.config.learner_faction,
                source,
                target,
            },
        ));
        let source_slot = environment
            .base_cells
            .iter()
            .position(|&cell| cell == source)
            .expect("补给源应有动作槽位");

        let waiting = tactical_action_penalty(
            &mut environment.world,
            &environment.base_cells,
            environment.config.learner_faction,
            RlAction::NoOp,
        );
        let stopping = tactical_action_penalty(
            &mut environment.world,
            &environment.base_cells,
            environment.config.learner_faction,
            RlAction::StopStream {
                base_slot: source_slot,
            },
        );

        assert!(waiting < stopping - 0.01, "完成补给后继续等待必须劣于停流");
        assert_eq!(stopping, 0.0);
    }
}
