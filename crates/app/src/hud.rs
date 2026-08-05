//! HUD 与调试信息。

use crate::common::*;
use bevy::prelude::*;
use easywar_logic::*;

pub fn update_hud(
    drag: Res<DragState>,
    difficulty: Res<DifficultyName>,
    hud: Res<DebugHud>,
    streams: Query<&Stream>,
    squads: Query<&Squad>,
    mut q: Query<&mut Text2d, With<HudText>>,
) {
    let Ok(mut text) = q.get_single_mut() else {
        return; // 棋盘渲染实体尚未生成
    };
    let active = streams.iter().filter(|s| s.active).count();
    text.0 = format!(
        "难度[{}](1/2/3切换) · 兵流 {} 条 · 小队 {} · 选中 {} 个据点 · {}",
        difficulty.0,
        active,
        squads.iter().count(),
        drag.selected.len(),
        hud.last_event
    );
}
