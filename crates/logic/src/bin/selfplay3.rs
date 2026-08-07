//! 三人候选地图无头自博弈：独立交叉出生席位、同帧提交顺序与据点扫描顺序。
//!
//! 用法：`cargo run -q -p easywar-logic --bin selfplay3 -- [种子数] [最大秒数] [地图文件...]`

use bevy_app::App;
use easywar_logic::*;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    mpsc, Arc,
};
use std::thread;

const DEFAULT_MAPS: [&str; 2] = [
    "layered_triangle_3ffa.toml",
    "three_leaf_windmill_3ffa.toml",
];
const PERMUTATIONS: [[FactionId; 3]; 6] = [
    [1, 2, 3],
    [1, 3, 2],
    [2, 1, 3],
    [2, 3, 1],
    [3, 1, 2],
    [3, 2, 1],
];

#[derive(Debug)]
struct Outcome {
    winner: Option<FactionId>,
    winner_seat: Option<usize>,
    winner_submit_position: Option<usize>,
    time: f32,
    first_tile_time: Option<f32>,
    first_neutral_base_time: Option<f32>,
    first_elimination_time: Option<f32>,
    first_eliminated_seat: Option<usize>,
    leader_changes: usize,
    max_pre_elimination_lead_ratio: f32,
    winner_was_trailing: bool,
}

#[derive(Default, Debug)]
struct TimedMetric {
    total: f32,
    count: usize,
}

impl TimedMetric {
    fn record(&mut self, value: Option<f32>) {
        if let Some(value) = value {
            self.total += value;
            self.count += 1;
        }
    }

    fn average(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            self.total / self.count as f32
        }
    }
}

#[derive(Default, Debug)]
struct Summary {
    seat_wins: [usize; 3],
    faction_wins: [usize; 3],
    submit_position_wins: [usize; 3],
    first_eliminated_seats: [usize; 3],
    completed: usize,
    timeouts: usize,
    completed_time: f32,
    first_tile: TimedMetric,
    first_neutral_base: TimedMetric,
    first_elimination: TimedMetric,
    leader_changes: usize,
    max_pre_elimination_lead_ratio: f32,
    comeback_wins: usize,
}

impl Summary {
    fn record(&mut self, outcome: Outcome) {
        self.first_tile.record(outcome.first_tile_time);
        self.first_neutral_base
            .record(outcome.first_neutral_base_time);
        self.first_elimination
            .record(outcome.first_elimination_time);
        self.leader_changes += outcome.leader_changes;
        self.max_pre_elimination_lead_ratio = self
            .max_pre_elimination_lead_ratio
            .max(outcome.max_pre_elimination_lead_ratio);
        if let Some(seat) = outcome.first_eliminated_seat {
            self.first_eliminated_seats[seat] += 1;
        }

        match outcome.winner {
            Some(winner) => {
                self.completed += 1;
                self.completed_time += outcome.time;
                self.faction_wins[(winner - 1) as usize] += 1;
                self.seat_wins[outcome.winner_seat.expect("胜者必须属于一个出生席位")] += 1;
                self.submit_position_wins[outcome
                    .winner_submit_position
                    .expect("胜者必须属于一个提交顺位")] += 1;
                self.comeback_wins += usize::from(outcome.winner_was_trailing);
            }
            None => self.timeouts += 1,
        }
    }

    fn average_completed_time(&self) -> f32 {
        if self.completed == 0 {
            0.0
        } else {
            self.completed_time / self.completed as f32
        }
    }

    fn average_leader_changes(&self, total: usize) -> f32 {
        if total == 0 {
            0.0
        } else {
            self.leader_changes as f32 / total as f32
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

fn base_cell(app: &App, entity: bevy_ecs::entity::Entity) -> Result<CellIdx, String> {
    app.world()
        .resource::<GridLookup>()
        .cells
        .iter()
        .position(|candidate| *candidate == entity)
        .ok_or_else(|| "据点不在 GridLookup 中".to_string())
}

fn starting_seats(app: &App) -> Result<Vec<CellIdx>, String> {
    let bases = app.world().resource::<BaseList>();
    let mut seats = bases
        .0
        .iter()
        .filter_map(|&entity| {
            app.world()
                .get::<Owner>(entity)
                .filter(|owner| owner.0 != NEUTRAL)
                .map(|owner| (owner.0, entity))
        })
        .map(|(owner, entity)| base_cell(app, entity).map(|cell| (owner, cell)))
        .collect::<Result<Vec<_>, _>>()?;
    seats.sort_by_key(|(owner, _)| *owner);
    if seats.len() != 3 || seats.iter().map(|(owner, _)| *owner).collect::<Vec<_>>() != [1, 2, 3] {
        return Err("三人评估器要求地图恰有阵营 1、2、3 的三个出生据点".into());
    }
    Ok(seats.into_iter().map(|(_, cell)| cell).collect())
}

fn assign_starting_factions(app: &mut App, seats: &[CellIdx], seat_factions: [FactionId; 3]) {
    let lookup = app.world().resource::<GridLookup>().clone();
    for (&cell, faction) in seats.iter().zip(seat_factions) {
        app.world_mut()
            .get_mut::<Owner>(lookup.entity(cell))
            .expect("出生据点缺少 Owner")
            .0 = faction;
    }
}

/// 让据点扫描顺序随三方据点图自同构一起变换，避免 TOML 声明顺序伪装成席位优势。
///
/// 两张候选图都按“3 出生 + 3 共享边节点 + 3 本地节点”声明；共享节点 0/1/2
/// 分别连接出生边 (0,1)/(1,2)/(2,0)。
fn permute_base_scan_order(app: &mut App, scan_permutation: [FactionId; 3]) -> Result<(), String> {
    let bases = app.world().resource::<BaseList>().clone();
    if bases.0.len() != 9 {
        return Err("三人评估器的扫描对齐要求 9 个据点".into());
    }
    let vertex_map = scan_permutation.map(|vertex| (vertex - 1) as usize);
    let edge_endpoints = [(0usize, 1usize), (1, 2), (2, 0)];
    let edge_map: [usize; 3] = std::array::from_fn(|edge_index| {
        let (left, right) = edge_endpoints[edge_index];
        let mapped = [vertex_map[left], vertex_map[right]];
        edge_endpoints
            .iter()
            .position(|&(candidate_left, candidate_right)| {
                mapped.contains(&candidate_left) && mapped.contains(&candidate_right)
            })
            .expect("出生置换必须诱导一条共享边")
    });
    let order = vertex_map
        .into_iter()
        .chain(edge_map.into_iter().map(|index| index + 3))
        .chain(vertex_map.into_iter().map(|index| index + 6))
        .map(|index| bases.0[index])
        .collect::<Vec<_>>();
    app.world_mut().insert_resource(BaseList(order));
    Ok(())
}

fn alive_factions(app: &App) -> [bool; 3] {
    let bases = app.world().resource::<BaseList>();
    std::array::from_fn(|index| {
        let faction = index as FactionId + 1;
        bases.0.iter().any(|entity| {
            app.world()
                .get::<Owner>(*entity)
                .is_some_and(|owner| owner.0 == faction)
        })
    })
}

fn unique_extreme(values: [f32; 3], choose_max: bool) -> Option<FactionId> {
    let mut indexed = values.into_iter().enumerate().collect::<Vec<_>>();
    indexed.sort_by(|left, right| left.1.partial_cmp(&right.1).expect("兵力统计不应出现 NaN"));
    if choose_max {
        indexed.reverse();
    }
    ((indexed[0].1 - indexed[1].1).abs() > 0.1).then_some(indexed[0].0 as FactionId + 1)
}

fn run_match(
    map_path: &Path,
    seed: u64,
    seat_factions: [FactionId; 3],
    controller_order: [FactionId; 3],
    max_seconds: f32,
    scan_permutation: Option<[FactionId; 3]>,
) -> Result<Outcome, String> {
    let mut app = App::new();
    app.add_plugins(GamePlugin);
    spawn_map_seeded(
        app.world_mut(),
        map_path,
        &assets_dir().join("subjects"),
        seed,
    )?;
    let seats = starting_seats(&app)?;
    assign_starting_factions(&mut app, &seats, seat_factions);
    if let Some(scan_permutation) = scan_permutation {
        permute_base_scan_order(&mut app, scan_permutation)?;
    }
    app.world_mut().insert_resource(AiControllers(
        controller_order
            .map(|faction| AiController::seeded(faction, AiParams::normal(), seed))
            .to_vec(),
    ));

    let lookup = app.world().resource::<GridLookup>().clone();
    let base_list = app.world().resource::<BaseList>().clone();
    let neutral_base_cells = base_list
        .0
        .iter()
        .filter(|entity| {
            app.world()
                .get::<Owner>(**entity)
                .is_some_and(|owner| owner.0 == NEUTRAL)
        })
        .map(|&entity| base_cell(&app, entity))
        .collect::<Result<Vec<_>, _>>()?;
    let linked_cells = lookup
        .cells
        .iter()
        .enumerate()
        .filter_map(|(cell, entity)| {
            (app.world().get::<CellKind>(*entity) == Some(&CellKind::LinkedTile)).then_some(cell)
        })
        .collect::<Vec<_>>();

    let steps = (max_seconds / SIM_DT) as usize;
    let sample_every = (1.0 / SIM_DT).round().max(1.0) as usize;
    let mut first_tile_time = None;
    let mut first_neutral_base_time = None;
    let mut first_elimination_time = None;
    let mut first_eliminated_seat = None;
    let mut previous_alive = [true; 3];
    let mut previous_leader = None;
    let mut leader_changes = 0;
    let mut max_pre_elimination_lead_ratio = 1.0f32;
    let mut was_trailing = [false; 3];

    for step in 0..steps {
        app.world_mut()
            .try_run_schedule(SimTick)
            .map_err(|error| format!("SimTick 运行失败: {error}"))?;
        if step % sample_every == 0 {
            let time = app.world().resource::<GameClock>().time;
            if first_tile_time.is_none()
                && linked_cells.iter().any(|&cell| {
                    app.world()
                        .get::<Owner>(lookup.entity(cell))
                        .is_some_and(|owner| owner.0 != NEUTRAL)
                })
            {
                first_tile_time = Some(time);
            }
            if first_neutral_base_time.is_none()
                && neutral_base_cells.iter().any(|&cell| {
                    app.world()
                        .get::<Owner>(lookup.entity(cell))
                        .is_some_and(|owner| owner.0 != NEUTRAL)
                })
            {
                first_neutral_base_time = Some(time);
            }

            let alive = alive_factions(&app);
            if first_elimination_time.is_none() {
                if let Some(faction_index) =
                    (0..3).find(|&index| previous_alive[index] && !alive[index])
                {
                    first_elimination_time = Some(time);
                    first_eliminated_seat = seat_factions
                        .iter()
                        .position(|&faction| faction as usize == faction_index + 1);
                }
            }
            previous_alive = alive;

            let totals =
                std::array::from_fn(|index| total_troops(app.world_mut(), index as FactionId + 1));
            if alive.into_iter().all(|value| value) {
                let mut descending = totals;
                descending
                    .sort_by(|left, right| right.partial_cmp(left).expect("兵力统计不应出现 NaN"));
                if descending[1] > 0.1 {
                    max_pre_elimination_lead_ratio =
                        max_pre_elimination_lead_ratio.max(descending[0] / descending[1]);
                }
                if let Some(leader) = unique_extreme(totals, true) {
                    if previous_leader.is_some_and(|previous| previous != leader) {
                        leader_changes += 1;
                    }
                    previous_leader = Some(leader);
                }
                if time >= 60.0 {
                    if let Some(trailer) = unique_extreme(totals, false) {
                        was_trailing[(trailer - 1) as usize] = true;
                    }
                }
            }
        }
        if app.world().resource::<Winner>().0.is_some() {
            break;
        }
    }

    let winner = app.world().resource::<Winner>().0;
    Ok(Outcome {
        winner,
        winner_seat: winner
            .and_then(|winner| seat_factions.iter().position(|&faction| faction == winner)),
        winner_submit_position: winner.and_then(|winner| {
            controller_order
                .iter()
                .position(|&faction| faction == winner)
        }),
        time: app.world().resource::<GameClock>().time,
        first_tile_time,
        first_neutral_base_time,
        first_elimination_time,
        first_eliminated_seat,
        leader_changes,
        max_pre_elimination_lead_ratio,
        winner_was_trailing: winner.is_some_and(|winner| was_trailing[(winner - 1) as usize]),
    })
}

fn evaluate_map(
    map_path: &Path,
    seeds: usize,
    max_seconds: f32,
    cross_scan_order: bool,
) -> Result<Summary, String> {
    let scan_permutations = if cross_scan_order {
        PERMUTATIONS.into_iter().map(Some).collect::<Vec<_>>()
    } else {
        vec![None]
    };
    let jobs = Arc::new(
        (1..=seeds)
            .flat_map(|seed| {
                let scan_permutations = scan_permutations.clone();
                PERMUTATIONS.into_iter().flat_map(move |seat_factions| {
                    let scan_permutations = scan_permutations.clone();
                    PERMUTATIONS.into_iter().flat_map(move |controller_order| {
                        scan_permutations
                            .clone()
                            .into_iter()
                            .map(move |scan_permutation| {
                                (
                                    seed as u64,
                                    seat_factions,
                                    controller_order,
                                    scan_permutation,
                                )
                            })
                    })
                })
            })
            .collect::<Vec<_>>(),
    );
    let worker_count = thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .min(jobs.len());
    let next = Arc::new(AtomicUsize::new(0));
    let (sender, receiver) = mpsc::channel();
    let map_path = map_path.to_path_buf();

    thread::scope(|scope| {
        for _ in 0..worker_count {
            let jobs = Arc::clone(&jobs);
            let next = Arc::clone(&next);
            let sender = sender.clone();
            let map_path = map_path.clone();
            scope.spawn(move || loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some(&(seed, seat_factions, controller_order, scan_permutation)) =
                    jobs.get(index)
                else {
                    break;
                };
                let outcome = run_match(
                    &map_path,
                    seed,
                    seat_factions,
                    controller_order,
                    max_seconds,
                    scan_permutation,
                );
                if sender.send(outcome).is_err() {
                    break;
                }
            });
        }
        drop(sender);
    });

    receiver
        .into_iter()
        .try_fold(Summary::default(), |mut summary, outcome| {
            summary.record(outcome?);
            Ok(summary)
        })
}

fn format_counts(counts: [usize; 3], prefix: &str) -> String {
    counts
        .into_iter()
        .enumerate()
        .map(|(index, count)| format!("{prefix}{} {count}", index + 1))
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let seeds = args
        .first()
        .and_then(|value| value.parse().ok())
        .unwrap_or(10);
    let max_seconds = args
        .get(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(600.0);
    let map_names = if args.len() > 2 {
        args[2..].iter().map(String::as_str).collect::<Vec<_>>()
    } else {
        DEFAULT_MAPS.to_vec()
    };
    let cross_scan_order = std::env::var("EASYWAR_SELFPLAY3_SCAN")
        .map(|value| value != "fixed")
        .unwrap_or(true);
    let scan_variants = if cross_scan_order {
        PERMUTATIONS.len()
    } else {
        1
    };

    println!(
        "[selfplay3] 每张地图 {seeds} 个种子 × 6 种出生轮换 × 6 种提交顺序 × {scan_variants} 种据点扫描，单局上限 {max_seconds:.0} 秒；据点扫描={} ",
        if cross_scan_order {
            "独立交叉"
        } else {
            "固定 TOML 顺序"
        }
    );
    for name in map_names {
        let path = resolve_map(name);
        let summary = evaluate_map(&path, seeds, max_seconds, cross_scan_order)?;
        let total = seeds * PERMUTATIONS.len() * PERMUTATIONS.len() * scan_variants;
        let map_name = path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or(name);
        println!(
            "[selfplay3] {map_name} | 完赛 {} / {total} 超时 {} | 完赛均时 {:.1}s",
            summary.completed,
            summary.timeouts,
            summary.average_completed_time(),
        );
        println!(
            "[selfplay3] {map_name} | {} | {} | {}",
            format_counts(summary.seat_wins, "席位"),
            format_counts(summary.faction_wins, "阵营"),
            format_counts(summary.submit_position_wins, "提交位"),
        );
        println!(
            "[selfplay3] {map_name} | {} | 首地块 {:.1}s 首中立据点 {:.1}s 首淘汰 {:.1}s",
            format_counts(summary.first_eliminated_seats, "首淘汰席位"),
            summary.first_tile.average(),
            summary.first_neutral_base.average(),
            summary.first_elimination.average(),
        );
        println!(
            "[selfplay3] {map_name} | 三方并存期最大领先比 {:.2} | 平均领先切换 {:.1} | 完赛翻盘 {} / {}",
            summary.max_pre_elimination_lead_ratio,
            summary.average_leader_changes(total),
            summary.comeback_wins,
            summary.completed,
        );
    }
    Ok(())
}
