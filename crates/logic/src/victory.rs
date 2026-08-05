//! 阶段 5：胜负判定——只剩一个非中立阵营拥有据点 → 该阵营获胜。

use crate::board::Board;
use crate::components::*;
use bevy_ecs::prelude::*;

pub fn victory(world: &mut World) {
    if world.resource::<Winner>().0.is_some() {
        return;
    }
    let board = Board::load(world);
    let alive = alive_factions_board(&board);
    if alive.len() == 1 {
        world.resource_mut::<Winner>().0 = Some(alive[0]);
    }
}

/// 仍拥有至少一个据点的非中立阵营
pub(crate) fn alive_factions_board(board: &Board) -> Vec<FactionId> {
    let mut alive: Vec<FactionId> = Vec::new();
    for b in &board.bases {
        let owner = board.owner[b.cell];
        if owner != NEUTRAL && !alive.contains(&owner) {
            alive.push(owner);
        }
    }
    alive
}
