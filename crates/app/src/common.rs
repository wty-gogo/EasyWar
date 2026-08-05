//! 共享常量、状态机、资源与标记组件。

use bevy::prelude::*;
use easywar_logic::*;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

pub const CELL: f32 = 44.0;
pub const BORDER: f32 = 4.0;
pub const STEP: f32 = 48.0;
pub const PLAYER: FactionId = 1;

pub const DIFFICULTIES: [(&str, fn() -> AiParams); 3] = [
    ("简单", AiParams::easy),
    ("中等", AiParams::normal),
    ("困难", AiParams::hard),
];

// ---------- 状态机 ----------

#[derive(States, Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum AppState {
    #[default]
    Menu,
    Playing,
    Ended,
}

// ---------- 资源 ----------

#[derive(Resource)]
pub struct SubjectList(pub Vec<SubjectDef>);

#[derive(Resource)]
pub struct MenuSelection {
    pub subject: usize,
    pub difficulty: usize,
}

#[derive(Resource)]
pub struct DifficultyName(pub &'static str);

/// 关联地块 → 归属学科颜色（中立地块的淡染色用）
#[derive(Resource, Default)]
pub struct LinkedTint(pub HashMap<CellIdx, [f32; 4]>);

#[derive(Resource, Default)]
pub struct DragState {
    pub dragging: Option<CellIdx>,
    pub selected: HashSet<CellIdx>,
    pub press_pos: Option<Vec2>,
}

#[derive(Resource, Default)]
pub struct DebugHud {
    pub last_event: String,
}

#[derive(Resource)]
pub struct EndInfo {
    pub winner: FactionId,
    pub player_bases: usize,
    pub player_tiles: usize,
    pub enemy_bases: usize,
    pub enemy_tiles: usize,
}

/// 棋盘实体已生成标记
#[derive(Resource)]
pub struct BoardSpawned;

/// SimTick 驱动的时间累积器
#[derive(Resource, Default)]
pub struct SimAccum(pub f32);

// ---------- 标记组件 ----------

#[derive(Component)]
pub struct MenuEntity;
#[derive(Component)]
pub struct BoardEntity;
#[derive(Component)]
pub struct EndEntity;

#[derive(Component)]
pub struct MenuButton {
    pub action: MenuAction,
    pub center: Vec2,
    pub half: Vec2,
}

#[derive(Clone, Copy)]
pub enum MenuAction {
    Subject(usize),
    Difficulty(usize),
    Start,
}

#[derive(Component)]
pub struct EndButton {
    pub restart: bool,
    pub center: Vec2,
    pub half: Vec2,
}

/// 渲染实体 → 逻辑格子实体
#[derive(Component)]
pub struct CellBorder(pub Entity);
#[derive(Component)]
pub struct CellFill(pub Entity);
#[derive(Component)]
pub struct CellLabel(pub Entity, pub String);
#[derive(Component)]
pub struct SquadDot;
#[derive(Component)]
pub struct HudText;

// ---------- 工具函数 ----------

pub fn workspace_assets() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets")
}

pub fn cleanup<T: Component>(mut commands: Commands, q: Query<Entity, With<T>>) {
    for e in q.iter() {
        commands.entity(e).despawn();
    }
}

pub fn grid_origin(lookup: &GridLookup) -> Vec2 {
    Vec2::new(
        -(lookup.width as f32 - 1.0) * STEP / 2.0,
        (lookup.height as f32 - 1.0) * STEP / 2.0,
    )
}

pub fn cell_pos(lookup: &GridLookup, origin: Vec2, i: CellIdx) -> Vec2 {
    let (x, y) = lookup.xy(i);
    Vec2::new(origin.x + x as f32 * STEP, origin.y - y as f32 * STEP)
}

pub fn fmt_num(v: f32) -> String {
    if v >= 1000.0 {
        format!("{:.1}k", v / 1000.0)
    } else {
        format!("{}", v.floor() as i32)
    }
}

pub fn faction_color(factions: &Factions, owner: FactionId) -> [f32; 4] {
    factions
        .0
        .iter()
        .find(|f| f.id == owner)
        .map(|f| f.color)
        .unwrap_or([0.5, 0.5, 0.5, 1.0])
}
