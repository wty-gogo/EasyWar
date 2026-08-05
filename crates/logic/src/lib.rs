//! EasyWar 逻辑层：纯 bevy_ecs，不依赖任何渲染/窗口 crate。
//! 对外暴露 GamePlugin + SimTick，可无头运行、可变速 tick ——
//! 服务于游戏本体、AI 模拟与未来的 RL 训练。
//!
//! 依赖白名单（ARCHITECTURE.md §4）：bevy_ecs + bevy_app + serde + toml。

pub mod ai;
pub mod board;
pub mod combat;
pub mod components;
pub mod economy;
pub mod intents;
pub mod map;
pub mod movement;
pub mod plugin;
pub mod streams;
pub mod victory;
pub mod world_ext;

pub use ai::{AiControllers, AiController, AiParams};
pub use bevy_ecs::entity::Entity;
pub use components::*;
pub use intents::{Intent, IntentQueue};
pub use map::{load_subjects, parse_hex_color, spawn_map, spawn_map_custom, SubjectDef};
pub use plugin::{GamePlugin, SimTick, SIM_DT};

use bevy_ecs::prelude::*;

// ---------- 只读检查辅助（app HUD / 测试用；永不写） ----------

/// 阵营总兵力 = 据点驻军 + 在途小队
pub fn total_troops(world: &mut World, faction: FactionId) -> f32 {
    let board = board::Board::load(world);
    let squads = world_ext::load_squads(world);
    ai::total_troops_board(&board, &squads, faction)
}

/// 仍拥有至少一个据点的非中立阵营
pub fn alive_factions(world: &mut World) -> Vec<FactionId> {
    let board = board::Board::load(world);
    victory::alive_factions_board(&board)
}

/// 指定阵营当前拥有的据点数
pub fn base_count(world: &mut World, faction: FactionId) -> usize {
    let board = board::Board::load(world);
    board.bases.iter().filter(|b| board.owner[b.cell] == faction).count()
}

/// 某据点当前是否有该阵营的活跃兵流
pub fn stream_from(world: &mut World, faction: FactionId, source: CellIdx) -> Option<Stream> {
    world_ext::load_streams(world)
        .into_iter()
        .find(|(_, s)| s.active && s.faction == faction && s.source == source)
        .map(|(_, s)| s)
}

/// 纯最短路径（测试 / 回放 / 调试用；SimTick 内的寻路走 Board 内部）
pub fn find_path(world: &mut World, from: CellIdx, to: CellIdx, faction: FactionId) -> Option<Vec<CellIdx>> {
    board::Board::load(world).find_path(from, to, faction)
}

/// 据点实时产能（base_cell 为据点格子下标）
pub fn base_production(world: &mut World, base_cell: CellIdx) -> f32 {
    let board = board::Board::load(world);
    let b = board.bases.iter().find(|b| b.cell == base_cell).expect("不是据点格子");
    board.base_production(b)
}

/// 据点驻军上限
pub fn base_garrison_cap(world: &mut World, base_cell: CellIdx) -> f32 {
    let board = board::Board::load(world);
    let b = board.bases.iter().find(|b| b.cell == base_cell).expect("不是据点格子");
    board.base_garrison_cap(b)
}

/// 立即应用一个意图（绕过队列；测试/调试用，游戏运行走 IntentQueue）
pub fn dispatch_intent(world: &mut World, intent: Intent) -> bool {
    match intent {
        Intent::SetStream { faction, source, target } => {
            intents::set_stream(world, faction, source, target)
        }
        Intent::StopStream { faction, source } => {
            intents::stop_stream(world, faction, source);
            true
        }
    }
}
