//! 对局生命周期与 SimTick 驱动。
//!
//! - `enter_playing`：把地图生成进**同一个 World**（逻辑实体与渲染实体共存）
//! - `drive_sim`：按墙钟累积，显式驱动 SimTick（倍速/暂停的天然挂点）
//! - `check_end`：Winner 出现 → 结算数据 → 切到 Ended

use crate::common::*;
use crate::neural_ai::{configured_controllers, NeuralModelResource};
use crate::telemetry::{PendingPlayerCommands, TelemetryRecorder};
use bevy::prelude::*;
use easywar_logic::*;
use std::path::{Path, PathBuf};

/// 每帧最多补跑的 tick 数（防止卡顿后的死亡螺旋）
const MAX_CATCH_UP: usize = 8;

fn selected_map_path(root: &Path, selected: usize) -> PathBuf {
    let configured = std::env::var("EASYWAR_MAP")
        .unwrap_or_else(|_| MAPS.get(selected).unwrap_or(&MAPS[0]).file.to_string());
    let path = PathBuf::from(configured);
    if path.is_absolute() {
        path
    } else {
        root.join("maps").join(path)
    }
}

pub fn enter_playing(world: &mut World) {
    let selection = world.resource::<MenuSelection>();
    let subjects = world.resource::<SubjectList>();
    let player_subject = subjects.0[selection.subject].id.clone();
    let ai_subject = subjects
        .0
        .iter()
        .find(|s| s.id != player_subject)
        .map(|s| s.id.clone())
        .unwrap();
    let difficulty = selection.difficulty;
    let selected_map = selection.map;

    let root = workspace_assets();
    let map_path = selected_map_path(&root, selected_map);
    let map_file = map_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("未知地图")
        .to_string();
    spawn_map_custom(
        world,
        &map_path,
        &root.join("subjects"),
        Some(&player_subject),
        Some(&ai_subject),
    )
    .expect("地图加载失败");

    // 据点和关联地块共用学科色边框，让中立区域也能一眼看出归属关系。
    let tint = {
        let subjects = &world.resource::<SubjectList>().0;
        let bases = world.resource::<BaseList>();
        let lookup = world.resource::<GridLookup>();
        RegionTint(
            bases
                .0
                .iter()
                .filter_map(|&entity| {
                    let base = world.get::<Base>(entity)?;
                    let color = subjects
                        .iter()
                        .find(|subject| subject.id == base.subject_id)
                        .map(|subject| parse_hex_color(&subject.color))?;
                    let base_cell = lookup.cells.iter().position(|&cell| cell == entity)?;
                    Some((base_cell, base.linked.clone(), color))
                })
                .flat_map(|(base_cell, linked, color)| {
                    std::iter::once(base_cell)
                        .chain(linked)
                        .map(move |cell| (cell, color))
                })
                .collect(),
        )
    };

    let (controllers, policy_controllers, diff_name) = configured_controllers(
        difficulty,
        &map_file,
        world.resource::<Factions>(),
        world.resource::<NeuralModelResource>(),
    );

    world.insert_resource(tint);
    world.insert_resource(controllers);
    world.insert_resource(policy_controllers);
    world.insert_resource(DifficultyName(diff_name));
    world.insert_resource(CurrentMapFile(map_file));
    world.insert_resource(DragState::default());
    world.insert_resource(DebugHud::default());
    // 重置对局状态资源（GamePlugin 在 App 启动时 init 过，重开一局要归零）
    world.insert_resource(GameClock::default());
    world.insert_resource(Winner::default());
    world.insert_resource(IntentQueue::default());
    world.insert_resource(PendingPlayerCommands::default());
    world.insert_resource(SeqCounter::default());
    world.insert_resource(SimAccum::default());
    let input_mode = *world.resource::<InputMode>();
    if let Some(mut recorder) = world.remove_resource::<TelemetryRecorder>() {
        let status = recorder.start_session(world, input_mode);
        world.insert_resource(recorder);
        if let Some(status) = status {
            world.resource_mut::<DebugHud>().last_event = status;
        }
    }
    // 棋盘渲染实体由 render::spawn_board_system 在下一帧生成（等资源就绪）
}

pub fn exit_playing(world: &mut World) {
    if let Some(mut recorder) = world.remove_resource::<TelemetryRecorder>() {
        recorder.close_session(world);
        world.insert_resource(recorder);
    }
    // 逻辑实体：格子（含虚空格）+ 小队 + 兵流
    let lookup = world.resource::<GridLookup>().clone();
    for e in lookup.cells {
        world.despawn(e);
    }
    let mut q_squad = world.query_filtered::<Entity, With<Squad>>();
    let squads: Vec<Entity> = q_squad.iter(world).collect();
    for e in squads {
        world.despawn(e);
    }
    let mut q_stream = world.query_filtered::<Entity, With<Stream>>();
    let streams: Vec<Entity> = q_stream.iter(world).collect();
    for e in streams {
        world.despawn(e);
    }
    // 渲染实体
    let mut q_board = world.query_filtered::<Entity, With<BoardEntity>>();
    let visuals: Vec<Entity> = q_board.iter(world).collect();
    for e in visuals {
        world.despawn(e);
    }
    world.remove_resource::<GridLookup>();
    world.remove_resource::<BaseList>();
    world.remove_resource::<Rules>();
    world.remove_resource::<Factions>();
    world.remove_resource::<RegionTint>();
    world.remove_resource::<DifficultyName>();
    world.remove_resource::<CurrentMapFile>();
    world.remove_resource::<BoardSpawned>();
    world.remove_resource::<AiControllers>();
    world.remove_resource::<PolicyControllers>();
}

/// 按墙钟累积并显式驱动 SimTick。暂停/倍速/快进只改这里。
pub fn drive_sim(world: &mut World) {
    // 地图未生成（OnEnter 当帧）或已分胜负时不推进
    if world.get_resource::<GridLookup>().is_none() {
        return;
    }
    let dt = world.resource::<Time>().delta_secs().min(0.25);
    let mut acc = world.resource_mut::<SimAccum>().0 + dt;
    let mut steps = 0;
    while acc >= SIM_DT && steps < MAX_CATCH_UP {
        if world.resource::<Winner>().0.is_some() {
            break;
        }
        world.try_run_schedule(SimTick).expect("SimTick 未注册");
        acc -= SIM_DT;
        steps += 1;
    }
    world.resource_mut::<SimAccum>().0 = acc;
}

pub fn check_end(
    mut commands: Commands,
    winner: Res<Winner>,
    factions: Res<Factions>,
    cells: Query<(&CellKind, &Owner)>,
    mut next: ResMut<NextState<AppState>>,
) {
    if let Some(w) = winner.0 {
        let count = |f: FactionId, kind: CellKind| {
            cells
                .iter()
                .filter(|(k, o)| o.0 == f && **k == kind)
                .count()
        };
        commands.insert_resource(EndInfo {
            winner: w,
            winner_name: factions
                .0
                .iter()
                .find(|faction| faction.id == w)
                .map(|faction| faction.name.clone())
                .unwrap_or_else(|| format!("阵营 {w}")),
            player_bases: count(PLAYER, CellKind::Base),
            player_tiles: count(PLAYER, CellKind::LinkedTile),
            rival_bases: cells
                .iter()
                .filter(|(kind, owner)| {
                    owner.0 != NEUTRAL && owner.0 != PLAYER && **kind == CellKind::Base
                })
                .count(),
            rival_tiles: cells
                .iter()
                .filter(|(kind, owner)| {
                    owner.0 != NEUTRAL && owner.0 != PLAYER && **kind == CellKind::LinkedTile
                })
                .count(),
        });
        next.set(AppState::Ended);
    }
}
