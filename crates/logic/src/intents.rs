//! 意图管道：玩家输入与 AI 决策的唯一写入口。
//! `SimTick` 链首的 `apply_intents` 统一排干队列并应用。

use crate::board::Board;
use crate::components::*;
use crate::world_ext::{load_squads, load_streams, write_squad, write_stream};
use bevy_ecs::prelude::*;

/// 对游戏状态的全部修改意图。玩家输入与 AI 决策共用。
#[derive(Clone, Copy, Debug)]
pub enum Intent {
    /// 建立/改道兵流（同一源据点只维持一条）
    SetStream { faction: FactionId, source: CellIdx, target: CellIdx },
    /// 停止兵流，途中兵到目标后回家
    StopStream { faction: FactionId, source: CellIdx },
}

/// 意图队列：宿主（app 输入系统 / 无头 runner）在 tick 之间 push，
/// `apply_intents` 在 tick 内排干。
#[derive(Resource, Default, Debug)]
pub struct IntentQueue(pub Vec<Intent>);

impl IntentQueue {
    pub fn push(&mut self, intent: Intent) {
        self.0.push(intent);
    }
}

/// SimTick 链首：排干意图队列并应用。除此之外没有任何系统能改游戏状态。
pub fn apply_intents(world: &mut World) {
    if world.resource::<Winner>().0.is_some() {
        world.resource_mut::<IntentQueue>().0.clear();
        return;
    }
    let intents = std::mem::take(&mut world.resource_mut::<IntentQueue>().0);
    if intents.is_empty() {
        return;
    }
    for intent in intents {
        match intent {
            Intent::SetStream { faction, source, target } => {
                set_stream(world, faction, source, target);
            }
            Intent::StopStream { faction, source } => stop_stream(world, faction, source),
        }
    }
}

/// 建立或改道兵流。逐行移植自旧 model.rs 的 GameState::set_stream。
pub(crate) fn set_stream(world: &mut World, faction: FactionId, source: CellIdx, target: CellIdx) -> bool {
    let board = Board::load(world);
    if board.kind[source] != CellKind::Base || board.owner[source] != faction {
        return false;
    }
    if !board.kind[target].enterable() || source == target {
        return false;
    }
    let Some(path) = board.find_path(source, target, faction) else {
        return false;
    };

    // 找出要替换的旧兵流，为其在途小队算好改道路径
    let mut streams = load_streams(world);
    let mut squads = load_squads(world);
    let replaced: Vec<usize> = streams
        .iter()
        .enumerate()
        .filter(|(_, pair)| pair.1.active && pair.1.faction == faction && pair.1.source == source)
        .map(|(i, _)| i)
        .collect();
    for &si in &replaced {
        let old_entity = streams[si].0;
        let mut new_paths: Vec<(usize, Vec<CellIdx>)> = Vec::new();
        for (qi, pair) in squads.iter().enumerate() {
            let sq = &pair.1;
            if sq.stream == old_entity && sq.mode == SquadMode::ToTarget {
                if let Some(p) = board.find_path(sq.current_cell(), target, faction) {
                    new_paths.push((qi, p));
                }
            }
        }
        for (qi, p) in new_paths {
            let sq = &mut squads[qi].1;
            sq.path = p;
            sq.seg = 0;
            sq.t = 0.0;
            sq.return_after_target = false; // 改道即新任务，取消"到点后回家"
        }
        streams[si].1.active = false;
    }
    for (e, sq) in &squads {
        write_squad(world, *e, sq);
    }
    for (e, s) in &streams {
        write_stream(world, *e, s);
    }

    let seq = world.resource_mut::<SeqCounter>().next();
    world.spawn(Stream { faction, source, target, path, spawn_accum: 0.0, active: true, seq });
    true
}

/// 停止兵流，途中兵回家
pub(crate) fn stop_stream(world: &mut World, faction: FactionId, source: CellIdx) {
    let mut streams = load_streams(world);
    let mut squads = load_squads(world);
    let targets: Vec<usize> = streams
        .iter()
        .enumerate()
        .filter(|(_, pair)| pair.1.active && pair.1.faction == faction && pair.1.source == source)
        .map(|(i, _)| i)
        .collect();
    for si in targets {
        recall_stream(&mut streams, &mut squads, si);
    }
    for (e, sq) in &squads {
        write_squad(world, *e, sq);
    }
    for (e, s) in &streams {
        write_stream(world, *e, s);
    }
}

/// 停用兵流：在途小队**继续飞向目标**，到达后幸存的再返回源据点。
/// `si` 为按 seq 排序后的下标。
pub(crate) fn recall_stream(streams: &mut [(Entity, Stream)], squads: &mut [(Entity, Squad)], si: usize) {
    let entity = streams[si].0;
    streams[si].1.active = false;
    for (_, sq) in squads.iter_mut() {
        if sq.stream == entity && sq.mode == SquadMode::ToTarget {
            sq.return_after_target = true;
        }
    }
}
