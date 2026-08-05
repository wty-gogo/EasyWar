//! 黄金快照测试：架构迁移的行为对照组。
//!
//! 录制：`GOLDEN_RECORD=1 cargo test -p easywar-logic --test golden`
//! 比对：`cargo test -p easywar-logic --test golden`（默认）
//!
//! 快照内容为粗粒度不变量（每 100 tick 采样）：各阵营总兵力（整数）、
//! 小队数、各阵营据点数；结局记录胜者与结束 tick。
//! 比对规则：胜者必须一致；兵力/小队数容差 ±2；结束 tick 容差 ±10%。

use easywar_logic::*;
use std::fmt::Write as _;
use std::path::PathBuf;

const DT: f32 = 1.0 / 64.0;
const MAX_SECS: f32 = 900.0;
const SAMPLE_EVERY: usize = 100;

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets")
}

fn snap_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden.snap")
}

struct Scenario {
    name: &'static str,
    ai1: Option<AiParams>,
    ai2: Option<AiParams>,
}

fn scenarios() -> Vec<Scenario> {
    vec![
        Scenario { name: "normal_mirror", ai1: Some(AiParams::normal()), ai2: Some(AiParams::normal()) },
        Scenario { name: "hard_vs_idle", ai1: None, ai2: Some(AiParams::hard()) },
        Scenario { name: "easy_vs_hard", ai1: Some(AiParams::easy()), ai2: Some(AiParams::hard()) },
    ]
}

/// 运行一个场景，产出文本快照（行格式：`tick t1 t2 squads bases1 bases2`，结尾 `end winner end_tick`）
fn run_scenario(s: &Scenario) -> String {
    let mut game = build_game(
        &assets_dir().join("maps/h_1v1.toml"),
        &assets_dir().join("subjects"),
    )
    .expect("地图加载失败");

    let mut ais: Vec<AiController> = Vec::new();
    if let Some(p) = &s.ai1 {
        ais.push(AiController::new(1, p.clone()));
    }
    if let Some(p) = &s.ai2 {
        ais.push(AiController::new(2, p.clone()));
    }

    let mut out = String::new();
    let steps = (MAX_SECS / DT) as usize;
    let mut end_tick = steps;
    for i in 0..steps {
        game.update(DT);
        let mut cmds = Vec::new();
        for ai in ais.iter_mut() {
            cmds.extend(ai.update(&game, DT).into_iter().map(|c| (ai.faction, c)));
        }
        for (f, c) in cmds {
            match c {
                AiCommand::SetStream { source, target } => {
                    game.set_stream(f, source, target);
                }
                AiCommand::StopStream { source } => game.stop_stream(f, source),
            }
        }
        if i % SAMPLE_EVERY == 0 {
            let bases1 = game.bases.iter().filter(|b| game.cells[b.cell].owner == 1).count();
            let bases2 = game.bases.iter().filter(|b| game.cells[b.cell].owner == 2).count();
            writeln!(
                out,
                "{} {} {} {} {} {}",
                i,
                game.total_troops(1).round() as i64,
                game.total_troops(2).round() as i64,
                game.squads.len(),
                bases1,
                bases2
            )
            .unwrap();
        }
        if game.winner.is_some() {
            end_tick = i;
            break;
        }
    }
    writeln!(out, "end {:?} {}", game.winner, end_tick).unwrap();
    out
}

fn run_all() -> String {
    let mut out = String::new();
    for s in scenarios() {
        writeln!(out, "=== {}", s.name).unwrap();
        out.push_str(&run_scenario(&s));
    }
    out
}

/// 逐行比对：数值字段容差 ±2，`end` 行胜者一致、end_tick 容差 ±10%
fn assert_close(scenario: &str, expected: &str, actual: &str) {
    let mut line_no = 0;
    for (e, a) in expected.lines().zip(actual.lines()) {
        line_no += 1;
        let ev: Vec<&str> = e.split_whitespace().collect();
        let av: Vec<&str> = a.split_whitespace().collect();
        assert_eq!(ev.len(), av.len(), "[{scenario}] 第{line_no}行字段数不同: `{e}` vs `{a}`");
        if ev[0] == "end" {
            assert_eq!(ev[1], av[1], "[{scenario}] 胜者不同: `{e}` vs `{a}`");
            let et: f64 = ev[2].parse().unwrap();
            let at: f64 = av[2].parse().unwrap();
            let tol = (et * 0.10).max(64.0);
            assert!((et - at).abs() <= tol, "[{scenario}] 结束 tick 偏差过大: `{e}` vs `{a}`");
        } else {
            for (x, y) in ev.iter().zip(av.iter()) {
                let xi: i64 = x.parse().unwrap();
                let yi: i64 = y.parse().unwrap();
                assert!((xi - yi).abs() <= 2, "[{scenario}] 第{line_no}行数值偏差过大: `{e}` vs `{a}`");
            }
        }
    }
    assert_eq!(
        expected.lines().count(),
        actual.lines().count(),
        "[{scenario}] 行数不同（对局节奏发生变化）"
    );
}

#[test]
fn golden_snapshot() {
    let actual = run_all();
    if std::env::var("GOLDEN_RECORD").is_ok() {
        std::fs::write(snap_path(), &actual).expect("写入快照失败");
        println!("已录制黄金快照 -> {:?}", snap_path());
        return;
    }
    let expected = std::fs::read_to_string(snap_path()).expect(
        "快照文件不存在，先录制：GOLDEN_RECORD=1 cargo test -p easywar-logic --test golden",
    );
    // 每个 section 首行是场景名，跳过之，只比对数据行
    let section_body = |sec: &str| sec.lines().skip(1).collect::<Vec<_>>().join("\n");
    let mut exp_sections = expected.split("=== ").filter(|s| !s.is_empty());
    let mut act_sections = actual.split("=== ").filter(|s| !s.is_empty());
    for s in scenarios() {
        let e = exp_sections.next().expect("快照缺少场景");
        let a = act_sections.next().expect("实际运行缺少场景");
        assert_close(s.name, &section_body(e), &section_body(a));
    }
}
