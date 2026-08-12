//! 战术成本估算：把兵流速度、恢复/生产、路径防御与在途兵统一为纯计算结果。

use crate::board::Board;
use crate::components::*;
use bevy_ecs::prelude::*;

#[derive(Clone, Debug, PartialEq)]
pub struct AttackEstimate {
    pub path: Vec<CellIdx>,
    pub output_per_sec: f32,
    pub source_production_per_sec: f32,
    pub required_arrivals: f32,
    pub required_source_garrison: f32,
    pub friendly_committed: f32,
    pub hostile_committed: f32,
    pub intermediate_base: Option<CellIdx>,
    pub repeats_active_stream: bool,
}

impl AttackEstimate {
    pub fn can_launch(&self, source_garrison: f32, safety: f32) -> bool {
        self.intermediate_base.is_none()
            && self.required_source_garrison.is_finite()
            && source_garrison >= self.required_source_garrison * safety
    }
}

/// 当前驻军对应的理论单波兵力；整数派兵与小数累计仍由兵流系统负责。
pub(crate) fn theoretical_wave_troops(rules: Rules, garrison: f32) -> f32 {
    let step = rules.squad_growth_garrison_step;
    let linear_excess = (garrison.min(rules.squad_soft_cap_garrison) - step).max(0.0);
    let overflow = (garrison - rules.squad_soft_cap_garrison).max(0.0);
    rules.squad_max_size + linear_excess / step + (overflow / step).sqrt()
}

/// 连续兵流近似下，击穿会恢复的目标需要累计到达的兵力。
pub fn required_arrivals(strength: f32, recovery_per_sec: f32, output_per_sec: f32) -> f32 {
    if strength <= 0.0 {
        return 0.0;
    }
    if output_per_sec <= recovery_per_sec || output_per_sec <= 0.0 {
        return f32::INFINITY;
    }
    strength * output_per_sec / (output_per_sec - recovery_per_sec)
}

pub fn estimate_attack(
    world: &mut World,
    faction: FactionId,
    source: CellIdx,
    target: CellIdx,
) -> Option<AttackEstimate> {
    let board = Board::load(world);
    let squads = crate::world_ext::load_squads(world);
    let streams = crate::world_ext::load_streams(world);
    estimate_attack_board(&board, &squads, &streams, faction, source, target)
}

pub(crate) fn estimate_attack_board(
    board: &Board,
    squads: &[(Entity, Squad)],
    streams: &[(Entity, Stream)],
    faction: FactionId,
    source: CellIdx,
    target: CellIdx,
) -> Option<AttackEstimate> {
    if board.kind.get(source) != Some(&CellKind::Base)
        || board.owner.get(source) != Some(&faction)
        || !board.kind.get(target).is_some_and(CellKind::enterable)
        || source == target
    {
        return None;
    }
    let path = board.find_path(source, target, faction)?;
    let intermediate_base = path
        .iter()
        .copied()
        .skip(1)
        .take(path.len().saturating_sub(2))
        .find(|&cell| board.kind[cell] == CellKind::Base);
    let source_garrison = board.garrison[source];
    // 动态波次会随驻军被抽走而回落；用整段进攻都能维持的基础波次做保守估算，
    // 避免拿开局瞬时高速误判为全程吞吐。高驻军的实际损耗只会略低于该上界。
    let wave = board.rules.squad_max_size.min(source_garrison);
    let output_per_sec = wave / board.rules.squad_interval_sec;
    let source_production_per_sec = board
        .bases
        .iter()
        .find(|base| base.cell == source)
        .map(|base| board.base_production(base))
        .unwrap_or(0.0);
    let friendly_committed = squads
        .iter()
        .filter(|(_, squad)| {
            squad.faction == faction
                && squad.mode == SquadMode::ToTarget
                && squad.path.last() == Some(&target)
        })
        .map(|(_, squad)| squad.troops)
        .sum::<f32>();
    let hostile_committed = squads
        .iter()
        .filter(|(_, squad)| {
            squad.faction != faction
                && squad.faction != NEUTRAL
                && squad.mode == SquadMode::ToTarget
                && squad.path.last() == Some(&target)
        })
        .map(|(_, squad)| squad.troops)
        .sum::<f32>();
    let required_on_path = path
        .iter()
        .copied()
        .enumerate()
        .skip(1)
        .filter(|(_, cell)| board.owner[*cell] != faction)
        .map(|(step, cell)| {
            let recovery = board.cell_recovery_per_sec(cell);
            let contact_time = step as f32 * board.rules.squad_move_sec_per_cell;
            let projected = board.projected_garrison(cell, recovery, contact_time);
            required_arrivals(projected, recovery, output_per_sec)
        })
        .sum::<f32>();
    let required_arrivals = (required_on_path + hostile_committed - friendly_committed).max(0.0);
    let production_share = if output_per_sec > 0.0 {
        (source_production_per_sec / output_per_sec).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let required_source_garrison = required_arrivals * (1.0 - production_share);
    let repeats_active_stream = streams.iter().any(|(_, stream)| {
        stream.active
            && stream.faction == faction
            && stream.source == source
            && stream.target == target
    });
    Some(AttackEstimate {
        path,
        output_per_sec,
        source_production_per_sec,
        required_arrivals,
        required_source_garrison,
        friendly_committed,
        hostile_committed,
        intermediate_base,
        repeats_active_stream,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuous_cost_matches_attrition_formula() {
        assert!((required_arrivals(22.0, 1.0, 15.0) - 23.571_428).abs() < 0.000_1);
        assert!((required_arrivals(40.0, 2.5, 15.0) - 48.0).abs() < 0.000_1);
    }

    #[test]
    fn output_not_exceeding_recovery_can_never_capture() {
        assert!(required_arrivals(22.0, 1.0, 1.0).is_infinite());
        assert!(required_arrivals(40.0, 2.5, 2.5).is_infinite());
    }
}
