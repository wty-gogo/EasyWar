//! 逻辑层单元测试（ECS 版移植，与迁移前 14 个测试一一对应）。

use bevy_app::App;
use easywar_logic::*;
use std::path::PathBuf;

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets")
}

fn build() -> App {
    let mut app = App::new();
    app.add_plugins(GamePlugin);
    spawn_map(
        app.world_mut(),
        &assets_dir().join("maps/h_1v1.toml"),
        &assets_dir().join("subjects"),
    )
    .expect("地图加载失败");
    app
}

fn tick(app: &mut App, n: usize) {
    for _ in 0..n {
        app.world_mut().try_run_schedule(SimTick).unwrap();
    }
}

fn tick_secs(app: &mut App, secs: f32) {
    tick(app, (secs / SIM_DT) as usize);
}

// ---------- 格子读写辅助 ----------

fn owner(app: &mut App, idx: CellIdx) -> FactionId {
    let e = app.world_mut().resource::<GridLookup>().entity(idx);
    app.world_mut().get::<Owner>(e).unwrap().0
}

fn set_owner(app: &mut App, idx: CellIdx, f: FactionId) {
    let e = app.world_mut().resource::<GridLookup>().entity(idx);
    app.world_mut().get_mut::<Owner>(e).unwrap().0 = f;
}

fn garrison(app: &mut App, idx: CellIdx) -> f32 {
    let e = app.world_mut().resource::<GridLookup>().entity(idx);
    app.world_mut().get::<Garrison>(e).unwrap().cur
}

fn set_garrison(app: &mut App, idx: CellIdx, v: f32) {
    let e = app.world_mut().resource::<GridLookup>().entity(idx);
    app.world_mut().get_mut::<Garrison>(e).unwrap().cur = v;
}

fn cell_kind(app: &mut App, idx: CellIdx) -> CellKind {
    let e = app.world_mut().resource::<GridLookup>().entity(idx);
    *app.world_mut().get::<CellKind>(e).unwrap()
}

fn cell_label(app: &mut App, idx: CellIdx) -> Option<String> {
    let e = app.world_mut().resource::<GridLookup>().entity(idx);
    app.world_mut().get::<Label>(e).unwrap().0.clone()
}

/// 按学科 id 找据点格子下标
fn base_cell(app: &mut App, subject_id: &str) -> CellIdx {
    let (lookup, bases): (GridLookup, BaseList) = {
        let w = app.world_mut();
        (w.resource::<GridLookup>().clone(), w.resource::<BaseList>().clone())
    };
    for (i, &e) in lookup.cells.iter().enumerate() {
        if bases.0.contains(&e) {
            let b = app.world_mut().get::<Base>(e).unwrap();
            if b.subject_id == subject_id {
                return i;
            }
        }
    }
    panic!("找不到据点: {subject_id}");
}

fn base_linked(app: &mut App, subject_id: &str) -> Vec<CellIdx> {
    let cell = base_cell(app, subject_id);
    let e = app.world_mut().resource::<GridLookup>().entity(cell);
    app.world_mut().get::<Base>(e).unwrap().linked.clone()
}

fn squad_count(app: &mut App) -> usize {
    let mut q = app.world_mut().query::<&Squad>();
    q.iter(app.world_mut()).count()
}

fn set_stream(app: &mut App, faction: FactionId, source: CellIdx, target: CellIdx) -> bool {
    dispatch_intent(app.world_mut(), Intent::SetStream { faction, source, target })
}

// ---------- 地图与生产 ----------

#[test]
fn map_loads() {
    let mut app = build();
    let w = app.world_mut();
    let lookup = w.resource::<GridLookup>().clone();
    assert_eq!(lookup.width, 14);
    assert_eq!(lookup.height, 13);
    assert_eq!(w.resource::<BaseList>().0.len(), 6, "2 对阵据点 + 4 中立要塞");
    assert_eq!(w.resource::<Factions>().0.len(), 2, "player + ai");
    let mut linked = 0;
    let mut plain = 0;
    for i in 0..lookup.cells.len() {
        match cell_kind(&mut app, i) {
            CellKind::LinkedTile => {
                linked += 1;
                assert_eq!(owner(&mut app, i), NEUTRAL);
                assert!((21.0..=24.0).contains(&garrison(&mut app, i)), "防御值应在 21~24");
            }
            CellKind::Plain => plain += 1,
            _ => {}
        }
        if let Some(l) = cell_label(&mut app, i) {
            assert!(l.chars().count() <= 2 || cell_kind(&mut app, i) == CellKind::Base);
        }
    }
    assert_eq!(linked, 24, "关联地块 4×6 = 24");
    assert_eq!(plain, 0, "v2 地图没有普通地块");
}

#[test]
fn production_scales_with_linked_tiles() {
    let mut app = build();
    let player_base = base_cell(&mut app, "chinese");
    assert_eq!(base_production(app.world_mut(), player_base), 2.5, "初始无关联地块，仅基础产能");
    // 占领一块关联地块 → +0.2
    let tile = base_linked(&mut app, "chinese")[0];
    set_owner(&mut app, tile, 1);
    assert_eq!(base_production(app.world_mut(), player_base), 2.7);
    // 驻军上限随地块提升：80 + 10×1
    assert_eq!(base_garrison_cap(app.world_mut(), player_base), 90.0);
    // 敌方抢走 → 敌方不获益，我方失去加成
    set_owner(&mut app, tile, 2);
    assert_eq!(base_production(app.world_mut(), player_base), 2.5);
}

#[test]
fn shortest_path_exists_and_is_short() {
    let mut app = build();
    let pb = base_cell(&mut app, "chinese");
    let ab = base_cell(&mut app, "math");
    let path = find_path(app.world_mut(), pb, ab, 1).expect("应当有路径");
    // 语文(2,11) → 数学(11,1)：竖链 5 + 横梁 9 + 竖链 5 = 19 步，路径长度 20
    assert_eq!(path.len(), 20, "纯最短路径");
}

// ---------- 兵流与战斗 ----------

#[test]
fn stream_captures_tile_and_recalls() {
    let mut app = build();
    let source = base_cell(&mut app, "chinese");
    let target = base_linked(&mut app, "chinese")[0]; // 相邻的关联地块 (2,10)
    let def = garrison(&mut app, target);
    // 给足兵力确保一波拿下
    set_garrison(&mut app, source, 100.0);
    assert!(set_stream(&mut app, 1, source, target));

    // 模拟 5 秒：地块应被占领，兵流应停止
    tick_secs(&mut app, 5.0);
    assert_eq!(owner(&mut app, target), 1, "地块应被占领");
    assert!(
        stream_from(app.world_mut(), 1, source).is_none(),
        "地块被占领后兵流应自动停止"
    );
    assert!(garrison(&mut app, target) <= def, "防御被消耗后才占领");

    // 再模拟 10 秒：途中兵应全部回家，不再有该兵流的小队
    tick_secs(&mut app, 10.0);
    let mut q = app.world_mut().query::<&Squad>();
    let stray = q
        .iter(app.world_mut())
        .filter(|s| s.faction == 1 && s.mode == SquadMode::Return)
        .count();
    assert_eq!(stray, 0, "回家的小队应已并入据点");
}

#[test]
fn base_capture_turns_stream_into_reinforcement() {
    let mut app = build();
    let enemy = base_cell(&mut app, "math");
    // 枢纽规则下远征会被己方据点截留，进攻要逐段推进：
    // 第一段：物理（玩家已持有）→ 地理（中立要塞，必经之路上）
    let fort = base_cell(&mut app, "physics");
    let geo = base_cell(&mut app, "geography");
    set_owner(&mut app, fort, 1);
    set_garrison(&mut app, fort, 800.0);
    set_garrison(&mut app, geo, 20.0);
    set_garrison(&mut app, enemy, 10.0);
    assert!(set_stream(&mut app, 1, fort, enemy));

    let mut geo_taken = false;
    for _ in 0..(60.0 / SIM_DT) as usize {
        tick(&mut app, 1);
        if !geo_taken && owner(&mut app, geo) == 1 {
            geo_taken = true;
            // 第二段：从前线据点继续进攻
            set_garrison(&mut app, geo, 800.0);
            assert!(set_stream(&mut app, 1, geo, enemy));
        }
        if owner(&mut app, enemy) == 1 {
            break;
        }
    }
    assert!(geo_taken, "第一段应拿下中途要塞");
    assert_eq!(owner(&mut app, enemy), 1, "第二段应攻下敌方据点");
    // 占领后兵流继续（增援流），直到驻军归零
    assert!(stream_from(app.world_mut(), 1, geo).is_some(), "据点被占领后兵流应继续输送");
    assert_eq!(app.world_mut().resource::<Winner>().0, Some(1));
}

#[test]
fn opposing_squads_annihilate() {
    let mut app = build();
    let pb = base_cell(&mut app, "chinese");
    let ab = base_cell(&mut app, "math");
    // 直接构造两支小队放在同一格
    let cell = find_path(app.world_mut(), pb, ab, 1).unwrap()[5];
    for (faction, troops) in [(1, 10.0), (2, 7.0)] {
        let seq = app.world_mut().resource_mut::<SeqCounter>().next();
        app.world_mut().spawn(Squad {
            faction,
            troops,
            path: vec![cell],
            seg: 0,
            t: 0.0,
            mode: SquadMode::ToTarget,
            stream: Entity::PLACEHOLDER,
            return_after_target: false,
            seq,
        });
    }
    tick(&mut app, 1); // 只触发相遇结算
    let mut q = app.world_mut().query::<&Squad>();
    let squads: Vec<&Squad> = q.iter(app.world_mut()).collect();
    let remaining: f32 = squads.iter().map(|s| s.troops).sum();
    assert_eq!(remaining, 3.0, "10 vs 7 互抵后应剩 3");
    assert_eq!(squads[0].faction, 1, "剩者属于兵力大的一方");
}

#[test]
fn win_condition() {
    let mut app = build();
    // 玩家占领全部据点
    let cells: Vec<CellIdx> = {
        let (lookup, bases) = {
            let w = app.world_mut();
            (w.resource::<GridLookup>().clone(), w.resource::<BaseList>().clone())
        };
        bases.0.iter().map(|e| lookup.cells.iter().position(|c| c == e).unwrap()).collect()
    };
    for c in cells {
        set_owner(&mut app, c, 1);
    }
    tick(&mut app, 1);
    assert_eq!(app.world_mut().resource::<Winner>().0, Some(1));
}

#[test]
fn stream_stops_when_garrison_hits_zero() {
    let mut app = build();
    let source = base_cell(&mut app, "chinese");
    let target = base_cell(&mut app, "math"); // 远端敌方据点，短时间打不下来
    set_garrison(&mut app, source, 5.0); // 只够出 2 队（3+2）
    assert!(set_stream(&mut app, 1, source, target));
    // 模拟 3 秒：驻军应被抽干，兵流自动终止
    tick_secs(&mut app, 3.0);
    assert!(stream_from(app.world_mut(), 1, source).is_none(), "驻军归零后兵流应立即停止");
}

// ---------- M2: AI 测试 ----------

#[test]
fn ai_captures_linked_tiles() {
    let mut app = build();
    app.world_mut()
        .insert_resource(AiControllers(vec![AiController::new(2, AiParams::normal())]));
    tick_secs(&mut app, 120.0);
    // AI（数学）应在 120 秒内吃掉自己至少一块关联地块
    let linked = base_linked(&mut app, "math");
    let owned = linked.iter().filter(|&&c| owner(&mut app, c) == 2).count();
    assert!(owned >= 1, "AI 应该吃掉至少一块关联地块，实际 {owned}");
}

#[test]
fn ai_launches_total_attack_when_dominant() {
    let mut app = build();
    // 给 AI 压倒性优势，且扫清所有低优先级动作（扩张/吃地）的目标：
    // 全部要塞与全部关联地块都归 AI，只剩玩家据点可打
    let all_bases: Vec<(CellIdx, Vec<CellIdx>, String)> = {
        let (lookup, bases) = {
            let w = app.world_mut();
            (w.resource::<GridLookup>().clone(), w.resource::<BaseList>().clone())
        };
        bases
            .0
            .iter()
            .map(|&e| {
                let i = lookup.cells.iter().position(|c| *c == e).unwrap();
                let b = app.world_mut().get::<Base>(e).unwrap();
                (i, b.linked.clone(), b.subject_id.clone())
            })
            .collect()
    };
    for (cell, linked, subject) in &all_bases {
        if subject != "chinese" {
            set_owner(&mut app, *cell, 2);
            for &t in linked {
                set_owner(&mut app, t, 2);
            }
        }
    }
    let math = base_cell(&mut app, "math");
    set_garrison(&mut app, math, 120.0);
    app.world_mut()
        .insert_resource(AiControllers(vec![AiController::new(2, AiParams::normal())]));

    // 30 秒内 AI 应建立目标为玩家格子的兵流
    let mut attacked_player = false;
    for _ in 0..(30.0 / SIM_DT) as usize {
        tick(&mut app, 1);
        let targets: Vec<CellIdx> = {
            let mut q = app.world_mut().query::<&Stream>();
            q.iter(app.world_mut())
                .filter(|s| s.active && s.faction == 2)
                .map(|s| s.target)
                .collect()
        };
        if targets.iter().any(|&t| owner(&mut app, t) == 1) {
            attacked_player = true;
            break;
        }
    }
    assert!(attacked_player, "优势 AI 应对玩家发起进攻");
}

#[test]
fn ai_eventually_beats_passive_player() {
    let mut app = build();
    app.world_mut()
        .insert_resource(AiControllers(vec![AiController::new(2, AiParams::hard())]));
    tick_secs(&mut app, 600.0);
    assert_eq!(app.world_mut().resource::<Winner>().0, Some(2), "困难 AI 应能在 10 分钟内击败挂机玩家");
}

#[test]
fn stopped_stream_squads_reach_target_before_returning() {
    let mut app = build();
    let source = base_cell(&mut app, "chinese");
    let target = base_linked(&mut app, "chinese")[0]; // 相邻关联地块 (2,10)
    // 兵力刚好：能打掉地块但驻军会被抽干 → 触发"归零停兵"
    let def = garrison(&mut app, target);
    set_garrison(&mut app, source, def + 6.0);
    assert!(set_stream(&mut app, 1, source, target));

    // 跑到驻军归零、兵流终止
    let mut stopped_at = None;
    for step in 0..(30.0 / SIM_DT) as usize {
        tick(&mut app, 1);
        if stopped_at.is_none() && stream_from(app.world_mut(), 1, source).is_none() {
            stopped_at = Some(step);
        }
    }
    assert!(stopped_at.is_some(), "驻军归零后兵流应停止");

    // 最终所有小队都应消亡（回家并入或战死），地块被占领
    tick_secs(&mut app, 20.0);
    assert_eq!(owner(&mut app, target), 1, "地块最终应被占领");
    assert_eq!(squad_count(&mut app), 0, "所有小队最终都应抵达并了结");
    // 回家的兵应并入源据点：驻军 > 0
    assert!(garrison(&mut app, source) > 0.0);
}

#[test]
fn reinforcement_stream_stops_at_zero_too() {
    let mut app = build();
    let source = base_cell(&mut app, "chinese");
    // 直接让玩家拥有第二座据点（物理要塞），建立增援流
    let fort = base_cell(&mut app, "physics");
    set_owner(&mut app, fort, 1);
    set_garrison(&mut app, fort, 0.0);
    set_garrison(&mut app, source, 5.0); // 只够出 2 队就抽干
    assert!(set_stream(&mut app, 1, source, fort));

    // 模拟 10 秒：驻军抽到 0，增援流同样应终止（唯一停止条件 = 兵力归零）
    tick_secs(&mut app, 10.0);
    assert!(stream_from(app.world_mut(), 1, source).is_none(), "增援流在驻军归零后也应终止");
    // 但已派出的兵应抵达并入驻军
    let in_fort = garrison(&mut app, fort) > 0.0;
    let in_transit = {
        let mut q = app.world_mut().query::<&Squad>();
        q.iter(app.world_mut()).any(|s| s.faction == 1)
    };
    assert!(in_fort || in_transit, "已派出的增援兵应流向目标据点");
}

#[test]
fn squads_enter_friendly_base_on_collision() {
    let mut app = build();
    let source = base_cell(&mut app, "chinese"); // 语文 (2,11)
    // 玩家拿下物理要塞 (2,6) 作为中途枢纽
    let fort = base_cell(&mut app, "physics");
    set_owner(&mut app, fort, 1);
    set_garrison(&mut app, fort, 0.0);
    // 从语文出兵打化学要塞 (2,1)：最短路径必经物理据点
    let chem = base_cell(&mut app, "chemistry");
    set_garrison(&mut app, source, 1000.0);
    assert!(set_stream(&mut app, 1, source, chem));

    tick_secs(&mut app, 30.0);
    // 兵应在物理据点被"截留"：物理驻军 > 0，而化学要塞未被攻陷
    assert!(garrison(&mut app, fort) > 0.0, "途经小队应并入己方据点");
    assert_ne!(owner(&mut app, chem), 1, "兵被枢纽截留，化学不应被攻陷");
}
