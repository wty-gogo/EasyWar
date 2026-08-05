//! 对局生命周期与 SimTick 驱动。
//!
//! - `enter_playing`：把地图生成进**同一个 World**（逻辑实体与渲染实体共存）
//! - `drive_sim`：按墙钟累积，显式驱动 SimTick（倍速/暂停的天然挂点）
//! - `check_end`：Winner 出现 → 结算数据 → 切到 Ended

use crate::common::*;
use bevy::prelude::*;
use easywar_logic::*;

/// 每帧最多补跑的 tick 数（防止卡顿后的死亡螺旋）
const MAX_CATCH_UP: usize = 8;

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

    let root = workspace_assets();
    spawn_map_custom(
        world,
        &root.join("maps/h_1v1.toml"),
        &root.join("subjects"),
        Some(&player_subject),
        Some(&ai_subject),
    )
    .expect("地图加载失败");

    // 关联地块淡染色表
    let subject_list = world.resource::<SubjectList>().0.clone();
    let mut tint = LinkedTint::default();
    let bases = world.resource::<BaseList>().clone();
    for &e in bases.0.iter() {
        let b = world.get::<Base>(e).unwrap();
        if let Some(s) = subject_list.iter().find(|s| s.id == b.subject_id) {
            let c = parse_hex_color(&s.color);
            for &t in &b.linked {
                tint.0.insert(t, c);
            }
        }
    }

    let (diff_name, diff_params) = DIFFICULTIES[difficulty];
    let factions = world.resource::<Factions>().0.clone();
    let controllers = factions
        .iter()
        .filter(|f| !f.is_player)
        .map(|f| AiController::new(f.id, diff_params()))
        .collect();

    world.insert_resource(tint);
    world.insert_resource(AiControllers(controllers));
    world.insert_resource(DifficultyName(diff_name));
    world.insert_resource(DragState::default());
    world.insert_resource(DebugHud::default());
    // 重置对局状态资源（GamePlugin 在 App 启动时 init 过，重开一局要归零）
    world.insert_resource(GameClock::default());
    world.insert_resource(Winner::default());
    world.insert_resource(IntentQueue::default());
    world.insert_resource(SeqCounter::default());
    world.insert_resource(SimAccum::default());
    // 棋盘渲染实体由 render::spawn_board_system 在下一帧生成（等资源就绪）
}

pub fn exit_playing(world: &mut World) {
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
    world.remove_resource::<LinkedTint>();
    world.remove_resource::<DifficultyName>();
    world.remove_resource::<BoardSpawned>();
    world.remove_resource::<AiControllers>();
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
    cells: Query<(&CellKind, &Owner)>,
    mut next: ResMut<NextState<AppState>>,
) {
    if let Some(w) = winner.0 {
        let count = |f: FactionId, kind: CellKind| {
            cells.iter().filter(|(k, o)| o.0 == f && **k == kind).count()
        };
        commands.insert_resource(EndInfo {
            winner: w,
            player_bases: count(PLAYER, CellKind::Base),
            player_tiles: count(PLAYER, CellKind::LinkedTile),
            enemy_bases: count(2, CellKind::Base),
            enemy_tiles: count(2, CellKind::LinkedTile),
        });
        next.set(AppState::Ended);
    }
}
