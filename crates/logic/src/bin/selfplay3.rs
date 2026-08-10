//! 三人候选地图无头评估：独立交叉出生席位、提交顺序与据点扫描顺序。
//!
//! 用法：`cargo run -q -p easywar-logic --bin selfplay3 -- [种子数] [最大秒数] [地图文件...]`

use easywar_logic::evaluation3::{
    first_divergence, map_base_count, run_match, LinkedScanOrder, MatchConfig, MatchFactors,
    Outcome, PERMUTATIONS,
};
use easywar_logic::*;
use std::collections::BTreeMap;
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
        match self.count {
            0 => 0.0,
            count => self.total / count as f32,
        }
    }
}

#[derive(Default, Debug)]
struct LevelStats {
    wins: [usize; 3],
    completed: usize,
    timeouts: usize,
}

impl LevelStats {
    fn record(&mut self, winner_seat: Option<usize>) {
        if let Some(seat) = winner_seat {
            self.wins[seat] += 1;
            self.completed += 1;
        } else {
            self.timeouts += 1;
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
    by_submit_order: BTreeMap<[FactionId; 3], LevelStats>,
    by_scan_order: BTreeMap<Vec<usize>, LevelStats>,
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
        outcome
            .first_eliminated_seats
            .iter()
            .for_each(|&seat| self.first_eliminated_seats[seat] += 1);
        self.by_submit_order
            .entry(outcome.factors.submit_order)
            .or_default()
            .record(outcome.winner_seat);
        self.by_scan_order
            .entry(outcome.factors.base_scan_order.clone().unwrap_or_default())
            .or_default()
            .record(outcome.winner_seat);

        match outcome.winner_faction {
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
        match self.completed {
            0 => 0.0,
            completed => self.completed_time / completed as f32,
        }
    }

    fn average_leader_changes(&self, total: usize) -> f32 {
        match total {
            0 => 0.0,
            total => self.leader_changes as f32 / total as f32,
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

/// 三类三元角色地图的据点自同构排列：出生、共享边节点、本地节点分别同步置换。
fn role_aligned_base_order(permutation: [FactionId; 3]) -> Vec<usize> {
    let vertex_map = permutation.map(|vertex| (vertex - 1) as usize);
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
    vertex_map
        .into_iter()
        .chain(edge_map.into_iter().map(|index| index + 3))
        .chain(vertex_map.into_iter().map(|index| index + 6))
        .collect()
}

fn evaluate_map(
    map_path: &Path,
    seeds: usize,
    max_seconds: f32,
    cross_scan_order: bool,
) -> Result<Summary, String> {
    if cross_scan_order && map_base_count(map_path, &assets_dir().join("subjects"))? != 9 {
        return Err("据点扫描自同构交叉目前要求地图采用 3 出生 + 3 共享 + 3 本地的 9 据点角色结构；可设置 EASYWAR_SELFPLAY3_SCAN=fixed 仅测固定声明顺序".into());
    }
    let scan_orders = if cross_scan_order {
        PERMUTATIONS
            .into_iter()
            .map(|permutation| Some(role_aligned_base_order(permutation)))
            .collect::<Vec<_>>()
    } else {
        vec![None]
    };
    let jobs = Arc::new(
        (1..=seeds)
            .flat_map(|seed| {
                let scan_orders = scan_orders.clone();
                PERMUTATIONS.into_iter().flat_map(move |seat_factions| {
                    let scan_orders = scan_orders.clone();
                    PERMUTATIONS.into_iter().flat_map(move |submit_order| {
                        scan_orders
                            .clone()
                            .into_iter()
                            .map(move |base_scan_order| MatchFactors {
                                seed: seed as u64,
                                seat_factions,
                                submit_order,
                                base_scan_order,
                                linked_scan_order: LinkedScanOrder::Declared,
                                entity_declaration_noise: 0,
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
    let subjects_dir = assets_dir().join("subjects");

    thread::scope(|scope| {
        for _ in 0..worker_count {
            let jobs = Arc::clone(&jobs);
            let next = Arc::clone(&next);
            let sender = sender.clone();
            let map_path = map_path.clone();
            let subjects_dir = subjects_dir.clone();
            scope.spawn(move || loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some(factors) = jobs.get(index).cloned() else {
                    break;
                };
                let outcome = run_match(&MatchConfig {
                    map_path: map_path.clone(),
                    subjects_dir: subjects_dir.clone(),
                    max_seconds,
                    factors,
                    enable_ai: true,
                    capture_trace: false,
                    cell_permutation: None,
                });
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

fn format_order(order: &[FactionId; 3]) -> String {
    order
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("→")
}

fn print_stratified_report(map_name: &str, summary: &Summary) {
    println!("[selfplay3] {map_name} | 按提交顺序分层（胜数按出生席位归一）");
    summary.by_submit_order.iter().for_each(|(order, stats)| {
        println!(
            "[selfplay3] {map_name} | 提交 {} | {} | 完赛 {} 超时 {}",
            format_order(order),
            format_counts(stats.wins, "席位"),
            stats.completed,
            stats.timeouts,
        );
    });
    println!("[selfplay3] {map_name} | 按据点扫描自同构分层（胜数按出生席位归一）");
    summary
        .by_scan_order
        .values()
        .enumerate()
        .for_each(|(index, stats)| {
            println!(
                "[selfplay3] {map_name} | 扫描变体{} | {} | 完赛 {} 超时 {}",
                index + 1,
                format_counts(stats.wins, "席位"),
                stats.completed,
                stats.timeouts,
            );
        });
}

fn print_factor_probe(map_path: &Path, max_seconds: f32) -> Result<(), String> {
    let probe_seconds = max_seconds.min(30.0);
    let base_count = map_base_count(map_path, &assets_dir().join("subjects"))?;
    let base_config = MatchConfig {
        map_path: map_path.to_path_buf(),
        subjects_dir: assets_dir().join("subjects"),
        max_seconds: probe_seconds,
        factors: MatchFactors::default(),
        enable_ai: true,
        capture_trace: true,
        cell_permutation: None,
    };
    let reference = run_match(&base_config)?;
    let variants = [
        (
            "提交顺序",
            MatchFactors {
                submit_order: [3, 2, 1],
                ..MatchFactors::default()
            },
        ),
        (
            "BaseList 扫描",
            MatchFactors {
                base_scan_order: Some((0..base_count).rev().collect()),
                ..MatchFactors::default()
            },
        ),
        (
            "AI 关联地块候选",
            MatchFactors {
                linked_scan_order: LinkedScanOrder::Reversed,
                ..MatchFactors::default()
            },
        ),
    ];
    let map_name = map_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("未知地图");
    variants.into_iter().try_for_each(|(name, factors)| {
        let variant = run_match(&MatchConfig {
            factors,
            ..base_config.clone()
        })?;
        match first_divergence(&reference.trace, &variant.trace) {
            Some(divergence) => println!(
                "[selfplay3] {map_name} | 顺序探针 {name} | 首次分歧 tick={} ({:.3}s) 子系统={:?}",
                divergence.tick,
                divergence.tick as f32 * SIM_DT,
                divergence.parts,
            ),
            None => {
                println!("[selfplay3] {map_name} | 顺序探针 {name} | {probe_seconds:.0}s 内无分歧")
            }
        }
        Ok::<(), String>(())
    })
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
    let run_factor_probe = std::env::var("EASYWAR_SELFPLAY3_PROBE")
        .map(|value| value != "off")
        .unwrap_or(true);
    let scan_variants = if cross_scan_order {
        PERMUTATIONS.len()
    } else {
        1
    };

    println!(
        "[selfplay3] 每张地图 {seeds} 个种子 × 6 种出生轮换 × 6 种提交顺序 × {scan_variants} 种据点扫描，单局上限 {max_seconds:.0} 秒"
    );
    println!(
        "[selfplay3] 口径：席位、运行时阵营编号、提交位分别统计；同时输出提交顺序与扫描变体分层，禁止只用配对总胜率下结论"
    );
    for name in map_names {
        let path = resolve_map(name);
        if run_factor_probe {
            print_factor_probe(&path, max_seconds)?;
        }
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
        print_stratified_report(map_name, &summary);
    }
    Ok(())
}
