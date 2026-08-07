//! 候选地图批量自博弈：同一种子交叉交换出生位与意图顺序，统计地图和模拟顺序偏差。
//!
//! 用法：`cargo run -p easywar-logic --bin selfplay -- [种子数] [最大秒数] [地图文件...]`

use bevy_app::App;
use easywar_logic::*;
use std::path::{Path, PathBuf};

const DEFAULT_MAPS: [&str; 3] = [
    "dual_ladder_1v1.toml",
    "braided_rings_1v1.toml",
    "ring_chord_1v1.toml",
];

#[derive(Clone, Copy, Debug)]
struct Outcome {
    winner: Option<FactionId>,
    time: f32,
}

#[derive(Default, Debug)]
struct Summary {
    left_wins: usize,
    right_wins: usize,
    faction_1_wins: usize,
    faction_2_wins: usize,
    first_wins: usize,
    second_wins: usize,
    timeouts: usize,
    completed_time: f32,
    completed: usize,
}

impl Summary {
    fn record(&mut self, outcome: Outcome, swapped: bool, reversed_order: bool) {
        match outcome.winner {
            Some(winner) => {
                self.completed += 1;
                self.completed_time += outcome.time;
                self.faction_1_wins += usize::from(winner == 1);
                self.faction_2_wins += usize::from(winner == 2);
                let left_faction = if swapped { 2 } else { 1 };
                let first_faction = if reversed_order { 2 } else { 1 };
                self.left_wins += usize::from(winner == left_faction);
                self.right_wins += usize::from(winner != left_faction);
                self.first_wins += usize::from(winner == first_faction);
                self.second_wins += usize::from(winner != first_faction);
            }
            None => self.timeouts += 1,
        }
    }

    fn average_time(&self) -> f32 {
        if self.completed == 0 {
            0.0
        } else {
            self.completed_time / self.completed as f32
        }
    }
}

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets")
}

fn resolve_map(name: &str) -> PathBuf {
    let path = Path::new(name);
    if path.is_absolute() || path.components().count() > 1 {
        path.to_path_buf()
    } else {
        assets_dir().join("maps").join(path)
    }
}

fn swap_starting_factions(app: &mut App) {
    let bases = app.world().resource::<BaseList>().clone();
    for entity in bases.0 {
        let mut owner = app
            .world_mut()
            .get_mut::<Owner>(entity)
            .expect("据点缺少 Owner");
        owner.0 = match owner.0 {
            1 => 2,
            2 => 1,
            other => other,
        };
    }
}

/// 纵向镜像地图时同步镜像据点遍历顺序，避免 TOML 声明顺序伪装成座位优势。
fn mirror_base_scan_order(app: &mut App) -> Result<(), String> {
    let lookup = app.world().resource::<GridLookup>().clone();
    let bases = app.world().resource::<BaseList>().clone();
    let mirrored = bases
        .0
        .iter()
        .map(|entity| {
            let cell = lookup
                .cells
                .iter()
                .position(|candidate| candidate == entity)
                .ok_or_else(|| "据点不在 GridLookup 中".to_string())?;
            let (x, y) = lookup.xy(cell);
            let mirror = lookup.idx(lookup.width - 1 - x, y);
            let mirrored_entity = lookup.entity(mirror);
            app.world()
                .get::<Base>(mirrored_entity)
                .ok_or_else(|| format!("镜像格 ({}, {}) 不是据点", lookup.width - 1 - x, y))?;
            Ok(mirrored_entity)
        })
        .collect::<Result<Vec<_>, String>>()?;
    app.world_mut().insert_resource(BaseList(mirrored));
    Ok(())
}

fn run_match(
    map_path: &Path,
    seed: u64,
    swapped: bool,
    reversed_order: bool,
    max_seconds: f32,
) -> Result<Outcome, String> {
    let mut app = App::new();
    app.add_plugins(GamePlugin);
    spawn_map_seeded(
        app.world_mut(),
        map_path,
        &assets_dir().join("subjects"),
        seed,
    )?;
    if swapped {
        swap_starting_factions(&mut app);
        mirror_base_scan_order(&mut app)?;
    }
    let controller_order = if reversed_order { [2, 1] } else { [1, 2] };
    app.world_mut().insert_resource(AiControllers(
        controller_order
            .map(|faction| AiController::seeded(faction, AiParams::normal(), seed))
            .to_vec(),
    ));

    let steps = (max_seconds / SIM_DT) as usize;
    for _ in 0..steps {
        app.world_mut()
            .try_run_schedule(SimTick)
            .map_err(|error| format!("SimTick 运行失败: {error}"))?;
        if app.world().resource::<Winner>().0.is_some() {
            break;
        }
    }
    Ok(Outcome {
        winner: app.world().resource::<Winner>().0,
        time: app.world().resource::<GameClock>().time,
    })
}

fn evaluate_map(map_path: &Path, seeds: usize, max_seconds: f32) -> Result<Summary, String> {
    (1..=seeds).try_fold(Summary::default(), |mut summary, seed| {
        for swapped in [false, true] {
            for reversed_order in [false, true] {
                let outcome =
                    run_match(map_path, seed as u64, swapped, reversed_order, max_seconds)?;
                summary.record(outcome, swapped, reversed_order);
            }
        }
        Ok(summary)
    })
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let seeds = args
        .first()
        .and_then(|value| value.parse().ok())
        .unwrap_or(10);
    let max_seconds = args
        .get(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(600.0);
    let map_names: Vec<&str> = if args.len() > 2 {
        args[2..].iter().map(String::as_str).collect()
    } else {
        DEFAULT_MAPS.to_vec()
    };

    println!(
        "[selfplay] 每张地图 {seeds} 组四向比赛（出生位 × 意图顺序），单局上限 {max_seconds:.0} 秒"
    );
    for name in map_names {
        let path = resolve_map(name);
        let summary = evaluate_map(&path, seeds, max_seconds)?;
        let total = seeds * 4;
        println!(
            "[selfplay] {} | 左胜 {} 右胜 {} | 阵营1胜 {} 阵营2胜 {} | 先提交胜 {} 后提交胜 {} | 超时 {} / {} | 完赛均时 {:.1}s",
            path.file_name().and_then(|value| value.to_str()).unwrap_or(name),
            summary.left_wins,
            summary.right_wins,
            summary.faction_1_wins,
            summary.faction_2_wins,
            summary.first_wins,
            summary.second_wins,
            summary.timeouts,
            total,
            summary.average_time(),
        );
    }
    Ok(())
}
