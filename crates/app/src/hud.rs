//! HUD 与调试信息。

use crate::common::*;
use crate::telemetry::TelemetryRecorder;
use bevy::prelude::*;
use easywar_logic::*;

pub fn update_hud(
    drag: Res<DragState>,
    difficulty: Res<DifficultyName>,
    hud: Res<DebugHud>,
    telemetry: Res<TelemetryRecorder>,
    streams: Query<&Stream>,
    squads: Query<&Squad>,
    mut q: Query<&mut Text2d, With<HudText>>,
) {
    let Ok(mut text) = q.single_mut() else {
        return; // 棋盘渲染实体尚未生成
    };
    let active = streams.iter().filter(|s| s.active).count();
    let telemetry_status = telemetry
        .is_active()
        .then_some(" · 埋点开启")
        .unwrap_or_default();
    text.0 = format!(
        "难度[{}](1～9/0切换) · 兵流 {} 条 · 小队 {} · 选中 {} 个据点{} · {}",
        difficulty.0,
        active,
        squads.iter().count(),
        drag.selected.len(),
        telemetry_status,
        hud.last_event
    );
}
