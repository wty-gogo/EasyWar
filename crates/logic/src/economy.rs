//! 阶段 1：据点产兵 + 地块/中立据点回防。推进游戏时钟。

use crate::board::Board;
use crate::components::*;
use crate::plugin::SIM_DT;
use bevy_ecs::prelude::*;

pub fn economy(world: &mut World) {
    if world.resource::<Winner>().0.is_some() {
        return;
    }
    world.resource_mut::<GameClock>().time += SIM_DT;

    let mut board = Board::load(world);

    // 据点产兵（封顶只阻止增长，不删除已有驻军：丢失地块导致上限下降时存量保留）
    for i in 0..board.bases.len() {
        let prod = board.base_production(&board.bases[i]);
        if prod > 0.0 {
            let cap = board.base_garrison_cap(&board.bases[i]);
            let cell = board.bases[i].cell;
            let g = board.garrison[cell];
            if g < cap {
                board.garrison[cell] = (g + prod * SIM_DT).min(cap);
                board.touch(cell);
            }
        }
    }
    // 回防：中立格子（含中立要塞）+ 任何阵营的地块（非据点）
    for i in 0..board.kind.len() {
        let regens = board.owner[i] == NEUTRAL || board.kind[i] != CellKind::Base;
        if regens && board.kind[i].enterable() && board.garrison[i] < board.garrison_max[i] {
            board.garrison[i] =
                (board.garrison[i] + board.rules.regen_per_sec * SIM_DT).min(board.garrison_max[i]);
            board.touch(i);
        }
    }

    board.flush(world);
}
