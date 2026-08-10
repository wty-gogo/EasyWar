//! 三人无头评估核心：实验因素、规范化轨迹、首次分歧与数值观测。
//!
//! 这里刻意不判断地图是否平衡。它只负责把席位、阵营、提交位和扫描顺序
//! 分开记录，并为评估器自身的不变量测试提供可复现证据。

use crate::*;
use bevy_app::App;
use bevy_ecs::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const PERMUTATIONS: [[FactionId; 3]; 6] = [
    [1, 2, 3],
    [1, 3, 2],
    [2, 1, 3],
    [2, 3, 1],
    [3, 1, 2],
    [3, 2, 1],
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LinkedScanOrder {
    Declared,
    Reversed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatchFactors {
    pub seed: u64,
    /// 数组下标是出生席位，值是该席位使用的运行时阵营编号。
    pub seat_factions: [FactionId; 3],
    /// 数组下标是提交位，值是该提交位对应的运行时阵营编号。
    pub submit_order: [FactionId; 3],
    /// `BaseList` 的下标排列；`None` 表示地图声明顺序。
    pub base_scan_order: Option<Vec<usize>>,
    pub linked_scan_order: LinkedScanOrder,
    /// 在加载地图前生成的无组件实体数，用来证明 ECS 实体编号不会泄漏进结果。
    pub entity_declaration_noise: usize,
}

impl Default for MatchFactors {
    fn default() -> Self {
        Self {
            seed: 1,
            seat_factions: [1, 2, 3],
            submit_order: [1, 2, 3],
            base_scan_order: None,
            linked_scan_order: LinkedScanOrder::Declared,
            entity_declaration_noise: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MatchConfig {
    pub map_path: PathBuf,
    pub subjects_dir: PathBuf,
    pub max_seconds: f32,
    pub factors: MatchFactors,
    pub enable_ai: bool,
    pub capture_trace: bool,
    /// 原格子下标到变形后格子下标的置换。用于镜像/旋转变形测试。
    pub cell_permutation: Option<Vec<CellIdx>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapshotPart {
    Cells,
    Streams,
    Squads,
    Intents,
    Winner,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateDigest {
    pub cells: u64,
    pub streams: u64,
    pub squads: u64,
    pub intents: u64,
    pub winner: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TracePoint {
    pub tick: usize,
    pub digest: StateDigest,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FirstDivergence {
    pub tick: usize,
    pub parts: Vec<SnapshotPart>,
}

#[derive(Clone, Debug)]
pub struct Outcome {
    pub factors: MatchFactors,
    pub winner_faction: Option<FactionId>,
    pub winner_seat: Option<usize>,
    pub winner_submit_position: Option<usize>,
    pub time: f32,
    pub first_tile_time: Option<f32>,
    pub first_neutral_base_time: Option<f32>,
    pub first_elimination_time: Option<f32>,
    pub first_eliminated_seats: Vec<usize>,
    pub leader_changes: usize,
    pub max_pre_elimination_lead_ratio: f32,
    pub winner_was_trailing: bool,
    pub trace: Vec<TracePoint>,
}

#[derive(Clone, Debug)]
struct Normalization {
    faction_to_seat: [FactionId; 4],
    cell_to_canonical: Vec<CellIdx>,
}

pub fn first_divergence(left: &[TracePoint], right: &[TracePoint]) -> Option<FirstDivergence> {
    left.iter()
        .zip(right)
        .find_map(|(left, right)| {
            (left.digest != right.digest).then(|| FirstDivergence {
                tick: left.tick.min(right.tick),
                parts: differing_parts(&left.digest, &right.digest),
            })
        })
        .or_else(|| {
            (left.len() != right.len()).then(|| FirstDivergence {
                tick: left.len().min(right.len()),
                parts: vec![SnapshotPart::Winner],
            })
        })
}

fn differing_parts(left: &StateDigest, right: &StateDigest) -> Vec<SnapshotPart> {
    [
        (SnapshotPart::Cells, left.cells != right.cells),
        (SnapshotPart::Streams, left.streams != right.streams),
        (SnapshotPart::Squads, left.squads != right.squads),
        (SnapshotPart::Intents, left.intents != right.intents),
        (SnapshotPart::Winner, left.winner != right.winner),
    ]
    .into_iter()
    .filter_map(|(part, differs)| differs.then_some(part))
    .collect()
}

pub fn horizontal_mirror(width: usize, height: usize) -> Vec<CellIdx> {
    (0..width * height)
        .map(|cell| {
            let (x, y) = (cell % width, cell / width);
            y * width + width - 1 - x
        })
        .collect()
}

pub fn run_match(config: &MatchConfig) -> Result<Outcome, String> {
    validate_factors(&config.factors)?;
    let mut app = App::new();
    app.add_plugins(GamePlugin);
    (0..config.factors.entity_declaration_noise).for_each(|_| {
        app.world_mut().spawn_empty();
    });
    spawn_map_seeded(
        app.world_mut(),
        &config.map_path,
        &config.subjects_dir,
        config.factors.seed,
    )?;

    let cell_to_canonical = if let Some(permutation) = &config.cell_permutation {
        apply_cell_permutation(&mut app, permutation)?;
        inverse_permutation(permutation)?
    } else {
        (0..app.world().resource::<GridLookup>().cells.len()).collect()
    };
    let seats = starting_seats(&app)?;
    assign_starting_factions(&mut app, &seats, config.factors.seat_factions);
    apply_base_scan_order(&mut app, config.factors.base_scan_order.as_deref())?;
    apply_linked_scan_order(&mut app, &config.factors.linked_scan_order);
    let faction_to_seat = faction_to_seat(config.factors.seat_factions);
    let normalization = Normalization {
        faction_to_seat,
        cell_to_canonical,
    };

    if config.enable_ai {
        app.world_mut().insert_resource(AiControllers(
            config
                .factors
                .submit_order
                .map(|faction| {
                    AiController::seeded(faction, AiParams::normal(), config.factors.seed)
                })
                .to_vec(),
        ));
    }

    let lookup = app.world().resource::<GridLookup>().clone();
    let bases = app.world().resource::<BaseList>().clone();
    let neutral_base_cells = bases
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

    let mut trace = Vec::new();
    if config.capture_trace {
        trace.push(TracePoint {
            tick: 0,
            digest: state_digest(app.world_mut(), &normalization),
        });
    }

    let steps = (config.max_seconds / SIM_DT) as usize;
    let sample_every = (1.0 / SIM_DT).round().max(1.0) as usize;
    let mut first_tile_time = None;
    let mut first_neutral_base_time = None;
    let mut first_elimination_time = None;
    let mut first_eliminated_seats = Vec::new();
    let mut previous_alive = [true; 3];
    let mut previous_leader = None;
    let mut leader_changes = 0;
    let mut max_pre_elimination_lead_ratio = 1.0f32;
    let mut was_trailing = [false; 3];

    for step in 0..steps {
        app.world_mut()
            .try_run_schedule(SimTick)
            .map_err(|error| format!("SimTick 运行失败: {error}"))?;
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

        let alive = alive_three_factions(&app);
        if first_elimination_time.is_none() {
            let eliminated = (0..3)
                .filter(|&index| previous_alive[index] && !alive[index])
                .filter_map(|index| {
                    config
                        .factors
                        .seat_factions
                        .iter()
                        .position(|&faction| faction as usize == index + 1)
                })
                .collect::<Vec<_>>();
            if !eliminated.is_empty() {
                first_elimination_time = Some(time);
                first_eliminated_seats = eliminated;
            }
        }
        previous_alive = alive;

        if step % sample_every == 0 && alive.into_iter().all(|value| value) {
            let totals =
                std::array::from_fn(|index| total_troops(app.world_mut(), index as FactionId + 1));
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
        if config.capture_trace {
            trace.push(TracePoint {
                tick: step + 1,
                digest: state_digest(app.world_mut(), &normalization),
            });
        }
        if app.world().resource::<Winner>().0.is_some() {
            break;
        }
    }

    let winner_faction = app.world().resource::<Winner>().0;
    Ok(Outcome {
        factors: config.factors.clone(),
        winner_faction,
        winner_seat: winner_faction.and_then(|winner| {
            config
                .factors
                .seat_factions
                .iter()
                .position(|&f| f == winner)
        }),
        winner_submit_position: winner_faction.and_then(|winner| {
            config
                .factors
                .submit_order
                .iter()
                .position(|&f| f == winner)
        }),
        time: app.world().resource::<GameClock>().time,
        first_tile_time,
        first_neutral_base_time,
        first_elimination_time,
        first_eliminated_seats,
        leader_changes,
        max_pre_elimination_lead_ratio,
        winner_was_trailing: winner_faction
            .is_some_and(|winner| was_trailing[(winner - 1) as usize]),
        trace,
    })
}

fn validate_factors(factors: &MatchFactors) -> Result<(), String> {
    let valid = |values: [FactionId; 3]| {
        let mut sorted = values;
        sorted.sort_unstable();
        sorted == [1, 2, 3]
    };
    if !valid(factors.seat_factions) {
        return Err("出生席位必须恰好分配阵营 1、2、3".into());
    }
    if !valid(factors.submit_order) {
        return Err("提交顺序必须恰好包含阵营 1、2、3".into());
    }
    Ok(())
}

fn base_cell(app: &App, entity: Entity) -> Result<CellIdx, String> {
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
    seats
        .iter()
        .zip(seat_factions)
        .for_each(|(&cell, faction)| {
            app.world_mut()
                .get_mut::<Owner>(lookup.entity(cell))
                .expect("出生据点缺少 Owner")
                .0 = faction;
        });
}

fn apply_base_scan_order(app: &mut App, order: Option<&[usize]>) -> Result<(), String> {
    let Some(order) = order else { return Ok(()) };
    let bases = app.world().resource::<BaseList>().clone();
    validate_permutation(order, bases.0.len(), "BaseList")?;
    app.world_mut().insert_resource(BaseList(
        order.iter().map(|&index| bases.0[index]).collect(),
    ));
    Ok(())
}

fn apply_linked_scan_order(app: &mut App, order: &LinkedScanOrder) {
    if *order == LinkedScanOrder::Declared {
        return;
    }
    let bases = app.world().resource::<BaseList>().clone();
    bases.0.into_iter().for_each(|entity| {
        app.world_mut()
            .get_mut::<Base>(entity)
            .expect("据点缺少 Base")
            .linked
            .reverse();
    });
}

fn apply_cell_permutation(app: &mut App, permutation: &[CellIdx]) -> Result<(), String> {
    let lookup = app.world().resource::<GridLookup>().clone();
    validate_permutation(permutation, lookup.cells.len(), "格子变形")?;
    let cells = lookup
        .cells
        .iter()
        .map(|&entity| {
            let world = app.world();
            (
                *world.get::<CellKind>(entity).expect("格子缺少 CellKind"),
                *world.get::<Owner>(entity).expect("格子缺少 Owner"),
                *world.get::<Garrison>(entity).expect("格子缺少 Garrison"),
                world.get::<Label>(entity).expect("格子缺少 Label").clone(),
                world.get::<Base>(entity).cloned(),
            )
        })
        .collect::<Vec<_>>();
    let old_base_list = app.world().resource::<BaseList>().clone();
    let old_base_cells = old_base_list
        .0
        .iter()
        .map(|&entity| base_cell(app, entity))
        .collect::<Result<Vec<_>, _>>()?;

    for (source, target) in permutation.iter().copied().enumerate() {
        let (kind, owner, garrison, label, base) = &cells[source];
        let entity = lookup.entity(target);
        let mut entity_mut = app.world_mut().entity_mut(entity);
        entity_mut.insert((*kind, *owner, *garrison, label.clone()));
        entity_mut.remove::<Base>();
        if let Some(base) = base {
            let mut transformed = base.clone();
            transformed.linked = transformed
                .linked
                .into_iter()
                .map(|cell| permutation[cell])
                .collect();
            entity_mut.insert(transformed);
        }
    }
    app.world_mut().insert_resource(BaseList(
        old_base_cells
            .into_iter()
            .map(|cell| lookup.entity(permutation[cell]))
            .collect(),
    ));
    Ok(())
}

fn validate_permutation(order: &[usize], len: usize, name: &str) -> Result<(), String> {
    let mut sorted = order.to_vec();
    sorted.sort_unstable();
    if sorted == (0..len).collect::<Vec<_>>() {
        Ok(())
    } else {
        Err(format!("{name} 必须是 0..{len} 的完整置换"))
    }
}

fn inverse_permutation(permutation: &[CellIdx]) -> Result<Vec<CellIdx>, String> {
    validate_permutation(permutation, permutation.len(), "格子变形")?;
    let mut inverse = vec![0; permutation.len()];
    permutation
        .iter()
        .copied()
        .enumerate()
        .for_each(|(source, target)| inverse[target] = source);
    Ok(inverse)
}

fn faction_to_seat(seat_factions: [FactionId; 3]) -> [FactionId; 4] {
    let mut result = [NEUTRAL; 4];
    seat_factions
        .iter()
        .copied()
        .enumerate()
        .for_each(|(seat, faction)| result[faction as usize] = seat as FactionId + 1);
    result
}

fn normalize_faction(faction: FactionId, normalization: &Normalization) -> FactionId {
    if faction == NEUTRAL {
        NEUTRAL
    } else {
        normalization.faction_to_seat[faction as usize]
    }
}

fn state_digest(world: &mut World, normalization: &Normalization) -> StateDigest {
    let lookup = world.resource::<GridLookup>().clone();
    let mut normalized_cells = vec![(0u8, 0u8, 0u32, 0u32); lookup.cells.len()];
    lookup.cells.iter().enumerate().for_each(|(cell, &entity)| {
        let kind = *world.get::<CellKind>(entity).expect("格子缺少 CellKind");
        let owner = world.get::<Owner>(entity).expect("格子缺少 Owner").0;
        let garrison = world.get::<Garrison>(entity).expect("格子缺少 Garrison");
        normalized_cells[normalization.cell_to_canonical[cell]] = (
            kind_code(kind),
            normalize_faction(owner, normalization),
            garrison.cur.to_bits(),
            garrison.max.to_bits(),
        );
    });
    let mut cells_hash = Hash64::new();
    cells_hash.add_usize(lookup.width);
    cells_hash.add_usize(lookup.height);
    normalized_cells.into_iter().for_each(|value| {
        cells_hash.add_u8(value.0);
        cells_hash.add_u8(value.1);
        cells_hash.add_u32(value.2);
        cells_hash.add_u32(value.3);
    });

    let mut stream_query = world.query::<(Entity, &Stream)>();
    let mut streams = stream_query
        .iter(world)
        .map(|(entity, stream)| (entity, stream.clone()))
        .collect::<Vec<_>>();
    streams.sort_by_key(|(_, stream)| stream.seq);
    let stream_seq = streams
        .iter()
        .map(|(entity, stream)| (*entity, stream.seq))
        .collect::<HashMap<_, _>>();
    let mut streams_hash = Hash64::new();
    streams.iter().for_each(|(_, stream)| {
        streams_hash.add_u64(stream.seq);
        streams_hash.add_u8(normalize_faction(stream.faction, normalization));
        streams_hash.add_usize(normalization.cell_to_canonical[stream.source]);
        streams_hash.add_usize(normalization.cell_to_canonical[stream.target]);
        add_path(&mut streams_hash, &stream.path, normalization);
        streams_hash.add_u32(stream.spawn_accum.to_bits());
        streams_hash.add_u32(stream.troop_carry.to_bits());
        streams_hash.add_u8(stream.active as u8);
    });

    let mut squad_query = world.query::<&Squad>();
    let mut squads = squad_query.iter(world).cloned().collect::<Vec<_>>();
    squads.sort_by_key(|squad| squad.seq);
    let mut squads_hash = Hash64::new();
    squads.iter().for_each(|squad| {
        squads_hash.add_u64(squad.seq);
        squads_hash.add_u64(stream_seq.get(&squad.stream).copied().unwrap_or(u64::MAX));
        squads_hash.add_u8(normalize_faction(squad.faction, normalization));
        squads_hash.add_u32(squad.troops.to_bits());
        add_path(&mut squads_hash, &squad.path, normalization);
        squads_hash.add_usize(squad.seg);
        squads_hash.add_u32(squad.t.to_bits());
        squads_hash.add_u8(match squad.mode {
            SquadMode::ToTarget => 0,
            SquadMode::Return => 1,
        });
        squads_hash.add_u8(squad.return_after_target as u8);
    });

    let intents = world.resource::<IntentQueue>().0.clone();
    let mut intents_hash = Hash64::new();
    intents.into_iter().for_each(|intent| match intent {
        Intent::SetStream {
            faction,
            source,
            target,
        } => {
            intents_hash.add_u8(0);
            intents_hash.add_u8(normalize_faction(faction, normalization));
            intents_hash.add_usize(normalization.cell_to_canonical[source]);
            intents_hash.add_usize(normalization.cell_to_canonical[target]);
        }
        Intent::StopStream { faction, source } => {
            intents_hash.add_u8(1);
            intents_hash.add_u8(normalize_faction(faction, normalization));
            intents_hash.add_usize(normalization.cell_to_canonical[source]);
        }
    });
    let mut winner_hash = Hash64::new();
    winner_hash.add_u32(world.resource::<GameClock>().time.to_bits());
    winner_hash.add_u8(
        world
            .resource::<Winner>()
            .0
            .map(|winner| normalize_faction(winner, normalization))
            .unwrap_or(NEUTRAL),
    );

    StateDigest {
        cells: cells_hash.finish(),
        streams: streams_hash.finish(),
        squads: squads_hash.finish(),
        intents: intents_hash.finish(),
        winner: winner_hash.finish(),
    }
}

fn add_path(hash: &mut Hash64, path: &[CellIdx], normalization: &Normalization) {
    hash.add_usize(path.len());
    path.iter()
        .for_each(|&cell| hash.add_usize(normalization.cell_to_canonical[cell]));
}

fn kind_code(kind: CellKind) -> u8 {
    match kind {
        CellKind::Void => 0,
        CellKind::Plain => 1,
        CellKind::LinkedTile => 2,
        CellKind::Base => 3,
    }
}

struct Hash64(u64);

impl Hash64 {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn add_u8(&mut self, value: u8) {
        self.0 ^= value as u64;
        self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
    }

    fn add_u32(&mut self, value: u32) {
        value
            .to_le_bytes()
            .into_iter()
            .for_each(|byte| self.add_u8(byte));
    }

    fn add_u64(&mut self, value: u64) {
        value
            .to_le_bytes()
            .into_iter()
            .for_each(|byte| self.add_u8(byte));
    }

    fn add_usize(&mut self, value: usize) {
        self.add_u64(value as u64);
    }

    fn finish(self) -> u64 {
        self.0
    }
}

fn alive_three_factions(app: &App) -> [bool; 3] {
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

pub fn map_dimensions(map_path: &Path, subjects_dir: &Path) -> Result<(usize, usize), String> {
    let mut app = App::new();
    app.add_plugins(GamePlugin);
    spawn_map_seeded(app.world_mut(), map_path, subjects_dir, 1)?;
    let lookup = app.world().resource::<GridLookup>();
    Ok((lookup.width, lookup.height))
}

pub fn map_base_count(map_path: &Path, subjects_dir: &Path) -> Result<usize, String> {
    let mut app = App::new();
    app.add_plugins(GamePlugin);
    spawn_map_seeded(app.world_mut(), map_path, subjects_dir, 1)?;
    Ok(app.world().resource::<BaseList>().0.len())
}
