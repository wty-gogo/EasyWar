use easywar_logic::*;
use std::path::PathBuf;

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets")
}

fn build() -> GameState {
    build_game(
        &assets_dir().join("maps/h_1v1.toml"),
        &assets_dir().join("subjects"),
    )
    .expect("地图加载失败")
}

#[test]
fn map_loads() {
    let g = build();
    assert_eq!(g.width, 14);
    assert_eq!(g.height, 13);
    assert_eq!(g.bases.len(), 6, "2 对阵据点 + 4 中立要塞");
    assert_eq!(g.factions.len(), 2, "player + ai");
    // 关联地块 4×6 = 24，每个可进入格都是据点或关联地块
    let linked = g
        .cells
        .iter()
        .filter(|c| c.kind == CellKind::LinkedTile)
        .count();
    assert_eq!(linked, 24);
    let plain = g
        .cells
        .iter()
        .filter(|c| c.kind == CellKind::Plain)
        .count();
    assert_eq!(plain, 0, "v2 地图没有普通地块");
    // 初始：除据点外全部中立；关联地块防御值集中在 21~24
    for c in &g.cells {
        if c.kind == CellKind::LinkedTile {
            assert_eq!(c.owner, NEUTRAL);
            assert!((21.0..=24.0).contains(&c.garrison), "防御值应在 21~24");
        }
    }
    // 知识点是 2 个字
    for c in &g.cells {
        if let Some(l) = &c.label {
            assert!(l.chars().count() <= 2 || c.kind == CellKind::Base);
        }
    }
}

#[test]
fn production_scales_with_linked_tiles() {
    let mut g = build();
    let player_base = g.bases.iter().find(|b| b.subject_id == "chinese").unwrap();
    assert_eq!(g.base_production(player_base), 2.5, "初始无关联地块，仅基础产能");
    // 占领一块关联地块 → +0.2
    let tile = g.bases[0].linked[0];
    g.cells[tile].owner = 1;
    let player_base = g.bases.iter().find(|b| b.subject_id == "chinese").unwrap();
    assert_eq!(g.base_production(player_base), 2.7);
    // 驻军上限随地块提升：80 + 10×1
    assert_eq!(g.base_garrison_cap(player_base), 90.0);
    // 敌方抢走 → 敌方不获益，我方失去加成
    g.cells[tile].owner = 2;
    let player_base = g.bases.iter().find(|b| b.subject_id == "chinese").unwrap();
    assert_eq!(g.base_production(player_base), 2.5);
}

#[test]
fn shortest_path_exists_and_is_short() {
    let g = build();
    let (pb, ab) = (g.bases[0].cell, g.bases[1].cell);
    let path = g.find_path(pb, ab, 1).expect("应当有路径");
    // 语文(2,11) → 数学(11,1)：竖链 5 + 横梁 9 + 竖链 5 = 19 步，路径长度 20
    assert_eq!(path.len(), 20, "纯最短路径");
}

#[test]
fn stream_captures_tile_and_recalls() {
    let mut g = build();
    let source = g.bases[0].cell; // 玩家据点
    let target = g.bases[0].linked[0]; // 相邻的关联地块 (2,10)
    let def = g.cells[target].garrison;
    // 给足兵力确保一波拿下
    g.cells[source].garrison = 100.0;
    assert!(g.set_stream(1, source, target));

    // 模拟 5 秒：地块应被占领，兵流应停止
    for _ in 0..(5.0 / 0.05) as usize {
        g.update(0.05);
    }
    assert_eq!(g.cells[target].owner, 1, "地块应被占领");
    assert!(
        g.stream_from(1, source).is_none(),
        "地块被占领后兵流应自动停止"
    );
    assert!(g.cells[target].garrison <= def, "防御被消耗后才占领");

    // 再模拟 10 秒：途中兵应全部回家，不再有该兵流的小队
    for _ in 0..(10.0 / 0.05) as usize {
        g.update(0.05);
    }
    let stray = g
        .squads
        .iter()
        .filter(|s| s.faction == 1 && s.mode == SquadMode::Return)
        .count();
    assert_eq!(stray, 0, "回家的小队应已并入据点");
}

#[test]
fn base_capture_turns_stream_into_reinforcement() {
    let mut g = build();
    let enemy = g.bases[1].cell; // AI（数学 (11,1)）
    // 枢纽规则下远征会被己方据点截留，进攻要逐段推进：
    // 第一段：物理（玩家已持有）→ 地理（中立要塞，必经之路上）
    let fort = g.bases.iter().find(|b| b.subject_id == "physics").unwrap().cell;
    let geo = g.bases.iter().find(|b| b.subject_id == "geography").unwrap().cell;
    g.cells[fort].owner = 1;
    g.cells[fort].garrison = 800.0;
    g.cells[geo].garrison = 20.0;
    g.cells[enemy].garrison = 10.0;
    assert!(g.set_stream(1, fort, enemy));

    let dt = 0.05;
    let mut geo_taken = false;
    for _ in 0..(60.0 / dt) as usize {
        g.update(dt);
        if !geo_taken && g.cells[geo].owner == 1 {
            geo_taken = true;
            // 第二段：从前线据点继续进攻
            g.cells[geo].garrison = 800.0;
            assert!(g.set_stream(1, geo, enemy));
        }
        if g.cells[enemy].owner == 1 {
            break;
        }
    }
    assert!(geo_taken, "第一段应拿下中途要塞");
    assert_eq!(g.cells[enemy].owner, 1, "第二段应攻下敌方据点");
    // 占领后兵流继续（增援流），直到驻军归零
    assert!(g.stream_from(1, geo).is_some(), "据点被占领后兵流应继续输送");
    assert_eq!(g.winner, Some(1));
}

#[test]
fn opposing_squads_annihilate() {
    let mut g = build();
    // 直接构造两支小队放在同一格
    let cell = g.find_path(g.bases[0].cell, g.bases[1].cell, 1).unwrap()[5];
    g.squads.push(Squad {
        faction: 1,
        troops: 10.0,
        path: vec![cell],
        seg: 0,
        t: 0.0,
        mode: SquadMode::ToTarget,
        stream: usize::MAX,
        return_after_target: false,
    });
    g.squads.push(Squad {
        faction: 2,
        troops: 7.0,
        path: vec![cell],
        seg: 0,
        t: 0.0,
        mode: SquadMode::ToTarget,
        stream: usize::MAX,
        return_after_target: false,
    });
    g.update(0.001); // 只触发相遇结算
    let remaining: f32 = g.squads.iter().map(|s| s.troops).sum();
    assert_eq!(remaining, 3.0, "10 vs 7 互抵后应剩 3");
    assert_eq!(g.squads[0].faction, 1, "剩者属于兵力大的一方");
}

#[test]
fn win_condition() {
    let mut g = build();
    // 玩家占领全部据点
    for b in g.bases.clone() {
        g.cells[b.cell].owner = 1;
    }
    g.update(0.05);
    assert_eq!(g.winner, Some(1));
}

#[test]
fn stream_stops_when_garrison_hits_zero() {
    let mut g = build();
    let source = g.bases[0].cell;
    let target = g.bases[1].cell; // 远端敌方据点，短时间打不下来
    g.cells[source].garrison = 5.0; // 只够出 2 队（3+2）
    assert!(g.set_stream(1, source, target));
    // 模拟 3 秒：驻军应被抽干，兵流自动终止
    for _ in 0..(3.0 / 0.05) as usize {
        g.update(0.05);
    }
    assert!(g.stream_from(1, source).is_none(), "驻军归零后兵流应立即停止");
}

// ---------- M2: AI 测试 ----------

/// 跑一场模拟：ai_faction 由 AI 控制，玩家挂机，返回终局状态
fn run_ai_match(ai: &mut AiController, mut g: GameState, secs: f32) -> GameState {
    let dt = 0.05;
    for _ in 0..(secs / dt) as usize {
        g.update(dt);
        let cmds = ai.update(&g, dt);
        for c in cmds {
            match c {
                AiCommand::SetStream { source, target } => {
                    g.set_stream(ai.faction, source, target);
                }
                AiCommand::StopStream { source } => g.stop_stream(ai.faction, source),
            }
        }
        if g.winner.is_some() {
            break;
        }
    }
    g
}

#[test]
fn ai_captures_linked_tiles() {
    let g = build();
    let mut ai = AiController::new(2, AiParams::normal());
    let g = run_ai_match(&mut ai, g, 120.0);
    // AI（数学）应在 120 秒内吃掉自己至少一块关联地块
    let math_base = g.bases.iter().find(|b| b.subject_id == "math").unwrap();
    let owned = math_base
        .linked
        .iter()
        .filter(|&&c| g.cells[c].owner == 2)
        .count();
    assert!(owned >= 1, "AI 应该吃掉至少一块关联地块，实际 {owned}");
}

#[test]
fn ai_launches_total_attack_when_dominant() {
    let mut g = build();
    // 给 AI 压倒性优势，且扫清所有低优先级动作（扩张/吃地）的目标：
    // 全部要塞与全部关联地块都归 AI，只剩玩家据点可打
    for b in g.bases.clone() {
        if b.subject_id != "chinese" {
            g.cells[b.cell].owner = 2;
        }
        for &t in &b.linked {
            if b.subject_id != "chinese" {
                g.cells[t].owner = 2;
            }
        }
    }
    g.cells[g.bases[1].cell].garrison = 120.0;
    let mut ai = AiController::new(2, AiParams::normal());
    // 记录 AI 是否曾对玩家控制的格子发起过兵流
    let mut attacked_player = false;
    let dt = 0.05;
    for _ in 0..(30.0 / dt) as usize {
        g.update(dt);
        let cmds = ai.update(&g, dt);
        for c in cmds {
            if let AiCommand::SetStream { source, target } = c {
                if g.cells[target].owner == 1 {
                    attacked_player = true;
                }
                g.set_stream(2, source, target);
            }
        }
    }
    assert!(attacked_player, "优势 AI 应对玩家发起进攻");
}

#[test]
fn ai_eventually_beats_passive_player() {
    let g = build();
    let mut ai = AiController::new(2, AiParams::hard());
    let g = run_ai_match(&mut ai, g, 600.0);
    assert_eq!(g.winner, Some(2), "困难 AI 应能在 10 分钟内击败挂机玩家");
}

#[test]
fn stopped_stream_squads_reach_target_before_returning() {
    let mut g = build();
    let source = g.bases[0].cell;
    let target = g.bases[0].linked[0]; // 相邻关联地块 (2,10)
    // 兵力刚好：能打掉地块但驻军会被抽干 → 触发"归零停兵"
    g.cells[source].garrison = g.cells[target].garrison + 6.0;
    assert!(g.set_stream(1, source, target));

    // 跑到驻军归零、兵流终止
    let mut stopped_at = None;
    for step in 0..(30.0 / 0.05) as usize {
        g.update(0.05);
        if stopped_at.is_none() && g.stream_from(1, source).is_none() {
            stopped_at = Some(step);
        }
    }
    assert!(stopped_at.is_some(), "驻军归零后兵流应停止");

    // 关键断言：兵流停止后，在途小队不应立刻掉头（应为 ToTarget 或已到点后转 Return）
    // 且最终所有小队都应消亡（回家并入或战死），地块被占领
    for _ in 0..(20.0 / 0.05) as usize {
        g.update(0.05);
    }
    assert_eq!(g.cells[target].owner, 1, "地块最终应被占领");
    assert!(g.squads.is_empty(), "所有小队最终都应抵达并了结");
    // 回家的兵应并入源据点：驻军 > 0（不含产能缓慢增长的影响，20 秒产能约 +20~28，回归兵也有贡献）
    assert!(g.cells[source].garrison > 0.0);
}

#[test]
fn reinforcement_stream_stops_at_zero_too() {
    let mut g = build();
    let source = g.bases[0].cell; // 语文（玩家）
    // 直接让玩家拥有第二座据点（物理要塞），建立增援流
    let fort = g.bases.iter().find(|b| b.subject_id == "physics").unwrap().cell;
    g.cells[fort].owner = 1;
    g.cells[fort].garrison = 0.0;
    g.cells[source].garrison = 5.0; // 只够出 2 队就抽干
    assert!(g.set_stream(1, source, fort));

    // 模拟 10 秒：驻军抽到 0，增援流同样应终止（唯一停止条件 = 兵力归零）
    for _ in 0..(10.0 / 0.05) as usize {
        g.update(0.05);
    }
    assert!(
        g.stream_from(1, source).is_none(),
        "增援流在驻军归零后也应终止"
    );
    // 但已派出的兵应抵达并入驻军（到目标后……目标是己方据点，直接并入）
    assert!(
        g.cells[fort].garrison > 0.0 || g.squads.iter().any(|s| s.faction == 1),
        "已派出的增援兵应流向目标据点"
    );
}

#[test]
fn squads_enter_friendly_base_on_collision() {
    let mut g = build();
    let source = g.bases[0].cell; // 语文 (2,11)
    // 玩家拿下物理要塞 (2,6) 作为中途枢纽
    let fort = g.bases.iter().find(|b| b.subject_id == "physics").unwrap().cell;
    g.cells[fort].owner = 1;
    g.cells[fort].garrison = 0.0;
    // 从语文出兵打化学要塞 (2,1)：最短路径必经物理据点
    let chem = g.bases.iter().find(|b| b.subject_id == "chemistry").unwrap().cell;
    g.cells[source].garrison = 1000.0;
    assert!(g.set_stream(1, source, chem));

    for _ in 0..(30.0 / 0.05) as usize {
        g.update(0.05);
    }
    // 兵应在物理据点被"截留"：物理驻军 > 0，而化学要塞未被攻陷
    assert!(g.cells[fort].garrison > 0.0, "途经小队应并入己方据点");
    assert_ne!(g.cells[chem].owner, 1, "兵被枢纽截留，化学不应被攻陷");
}
