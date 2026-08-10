//! GamePlugin 与 SimTick schedule 定义。链序在此一处声明。

use bevy_app::{App, Plugin};
use bevy_ecs::prelude::*;
use bevy_ecs::schedule::ScheduleLabel;

/// 每个 SimTick 的游戏时长（秒）。宿主每 run 一次 SimTick = 推进一个固定步长。
pub const SIM_DT: f32 = 1.0 / 64.0;

/// 逻辑 tick schedule：宿主显式驱动（app 的 driver / 无头 runner / RL）。
/// 单线程、显式链式排序——迭代顺序即确定性。
#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SimTick;

pub struct GamePlugin;

impl Plugin for GamePlugin {
    fn build(&self, app: &mut App) {
        app.add_schedule(Schedule::new(SimTick));
        app.init_resource::<crate::components::SeqCounter>()
            .init_resource::<crate::components::GameClock>()
            .init_resource::<crate::components::Winner>()
            .init_resource::<crate::components::Factions>()
            .init_resource::<crate::intents::IntentQueue>()
            .init_resource::<crate::ai::AiControllers>()
            .init_resource::<crate::rl::PolicyControllers>();
        // 链序与旧循环严格同构：
        // apply_intents（旧：AI 指令在 update 后立即生效）
        // → economy → streams → movement → combat → victory（旧 update 的五个阶段）
        // → ai_decide（旧：AI 在 update 之后看到最新状态；意图下一 tick 链首生效）
        app.add_systems(
            SimTick,
            (
                crate::intents::apply_intents,
                crate::economy::economy,
                crate::streams::streams,
                crate::movement::movement,
                crate::combat::combat,
                crate::victory::victory,
                crate::ai::ai_decide,
                crate::rl::policy_decide,
            )
                .chain(),
        );
    }
}
