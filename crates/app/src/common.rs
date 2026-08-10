//! 共享常量、状态机、资源与标记组件。

use bevy::prelude::*;
use easywar_logic::*;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

pub const CELL: f32 = 44.0;
pub const BORDER: f32 = 4.0;
pub const STEP: f32 = 48.0;
pub const PLAYER: FactionId = 1;

#[derive(Clone, Copy)]
pub enum DifficultyKind {
    Rule(fn() -> AiParams),
    NeuralV5,
}

#[derive(Clone, Copy)]
pub struct DifficultyChoice {
    pub name: &'static str,
    pub kind: DifficultyKind,
}

pub const DIFFICULTIES: [DifficultyChoice; 4] = [
    DifficultyChoice {
        name: "简单",
        kind: DifficultyKind::Rule(AiParams::easy),
    },
    DifficultyChoice {
        name: "中等",
        kind: DifficultyKind::Rule(AiParams::normal),
    },
    DifficultyChoice {
        name: "困难",
        kind: DifficultyKind::Rule(AiParams::hard),
    },
    DifficultyChoice {
        name: "神经模型 V5",
        kind: DifficultyKind::NeuralV5,
    },
];

pub fn configured_difficulty() -> usize {
    match std::env::var("EASYWAR_DIFFICULTY").as_deref() {
        Ok("easy" | "0") => 0,
        Ok("normal" | "1") => 1,
        Ok("hard" | "2") => 2,
        Ok("neural-v5" | "3") => 3,
        _ => 1,
    }
}

#[derive(Clone, Copy)]
pub struct MapChoice {
    pub name: &'static str,
    pub file: &'static str,
}

pub const MAPS: [MapChoice; 7] = [
    MapChoice {
        name: "经典 H",
        file: "h_1v1.toml",
    },
    MapChoice {
        name: "双线梯形",
        file: "dual_ladder_1v1.toml",
    },
    MapChoice {
        name: "编织双环",
        file: "braided_rings_1v1.toml",
    },
    MapChoice {
        name: "外环横梁·实验",
        file: "ring_chord_1v1.toml",
    },
    MapChoice {
        name: "三足环·3人混战",
        file: "tripod_ring_3ffa.toml",
    },
    MapChoice {
        name: "双层三角·3人候选",
        file: "layered_triangle_3ffa.toml",
    },
    MapChoice {
        name: "三叶风车·3人候选",
        file: "three_leaf_windmill_3ffa.toml",
    },
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
    pub map: usize,
}

#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputMode {
    Desktop,
    Touch,
}

impl InputMode {
    pub fn from_environment() -> Self {
        match std::env::var("EASYWAR_INPUT").as_deref() {
            Ok("touch") => Self::Touch,
            _ => Self::Desktop,
        }
    }
}

#[derive(Resource)]
pub struct DifficultyName(pub &'static str);

#[derive(Resource)]
pub struct CurrentMapFile(pub String);

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
    pub winner_name: String,
    pub player_bases: usize,
    pub player_tiles: usize,
    pub rival_bases: usize,
    pub rival_tiles: usize,
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
    Map(usize),
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
