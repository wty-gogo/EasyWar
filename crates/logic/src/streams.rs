//! 阶段 2：兵流——终止判定 + 按节奏出兵。逐行移植自旧 sim.rs 的 update_streams。

use crate::board::Board;
use crate::components::*;
use crate::intents::recall_stream;
use crate::plugin::SIM_DT;
use crate::tactics::theoretical_wave_troops;
use crate::world_ext::{load_streams, write_stream};
use bevy_ecs::prelude::*;

/// 根据当前驻军计算下一波整数兵力，并返回留给下一波的小数额度。
///
/// `squad_max_size` 沿用地图字段名，动态规则里表示 40 兵以内的基础波次兵力。
fn next_wave_troops(rules: Rules, garrison: f32, carry: f32) -> (f32, f32) {
    let theoretical = theoretical_wave_troops(rules, garrison) + carry;
    let scheduled = theoretical.floor();
    let available = garrison.floor();
    (scheduled.min(available), theoretical - scheduled)
}

pub fn streams(world: &mut World) {
    if world.resource::<Winner>().0.is_some() {
        return;
    }
    let mut board = Board::load(world);
    let mut stream_list = load_streams(world);
    let mut squads = crate::world_ext::load_squads(world);
    // 本阶段新生成的小队：recall_stream 需要一并标记（旧实现里它们已在 self.squads 中）
    let mut new_squads: Vec<(Entity, Squad)> = Vec::new();

    for si in 0..stream_list.len() {
        if !stream_list[si].1.active {
            continue;
        }
        let (faction, source, target) = {
            let s = &stream_list[si].1;
            (s.faction, s.source, s.target)
        };

        // 源据点失守 → 兵流自然消亡（在途小队继续飞，生死自负）
        if board.owner[source] != faction {
            stream_list[si].1.active = false;
            continue;
        }

        // 目标是地块且已被本方占领 → 停止出兵，途中兵回家
        let tk = board.kind[target];
        if (tk == CellKind::Plain || tk == CellKind::LinkedTile) && board.owner[target] == faction {
            recall(&mut stream_list, &mut squads, &mut new_squads, si);
            continue;
        }
        // 目标是据点：占领后继续输送（增援流），不做任何处理

        // 出兵：固定间隔生成一队；每波兵力随源据点当前驻军增长，并累计取整余数。
        // 停止条件（两种，对所有目标类型一致）：
        //   - 目标是地块且已被本方占领 → 终止；
        //   - 驻军被抽到 0 → 立刻终止（在途小队到目标后再回家）。
        // 目标是据点则永不因占领而终止：打下来之后自动变成增援流，直到驻军归零。
        stream_list[si].1.spawn_accum += SIM_DT;
        let mut exhausted = false;
        while stream_list[si].1.spawn_accum >= board.rules.squad_interval_sec {
            stream_list[si].1.spawn_accum -= board.rules.squad_interval_sec;
            let (n, next_carry) = next_wave_troops(
                board.rules,
                board.garrison[source],
                stream_list[si].1.troop_carry,
            );
            stream_list[si].1.troop_carry = next_carry;
            if n >= 1.0 {
                board.garrison[source] -= n;
                board.touch(source);
                let seq = world.resource_mut::<SeqCounter>().next();
                let squad = Squad {
                    faction,
                    troops: n,
                    path: stream_list[si].1.path.clone(),
                    seg: 0,
                    t: 0.0,
                    mode: SquadMode::ToTarget,
                    stream: stream_list[si].0,
                    return_after_target: false,
                    seq,
                };
                let e = world.spawn(squad.clone()).id();
                new_squads.push((e, squad));
                if board.garrison[source].floor() < 1.0 {
                    exhausted = true;
                    break;
                }
            } else {
                exhausted = true;
                break;
            }
        }
        if exhausted {
            recall(&mut stream_list, &mut squads, &mut new_squads, si);
            continue;
        }
    }

    for (e, s) in &stream_list {
        write_stream(world, *e, s);
    }
    // 本阶段对 squads 的修改只有 recall 置的 return_after_target
    for (e, sq) in squads.iter().chain(new_squads.iter()) {
        crate::world_ext::write_squad(world, *e, sq);
    }
    board.flush(world);
}

/// recall_stream 的本阶段包装：同时覆盖刚生成、尚未进入 squads 列表的新小队
fn recall(
    stream_list: &mut [(Entity, Stream)],
    squads: &mut [(Entity, Squad)],
    new_squads: &mut [(Entity, Squad)],
    si: usize,
) {
    recall_stream(stream_list, squads, si);
    let entity = stream_list[si].0;
    for (_, sq) in new_squads.iter_mut() {
        if sq.stream == entity && sq.mode == SquadMode::ToTarget {
            sq.return_after_target = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rules() -> Rules {
        Rules {
            garrison_cap_base: 80.0,
            garrison_cap_per_tile: 10.0,
            regen_per_sec: 1.0,
            squad_interval_sec: 0.2,
            squad_max_size: 3.0,
            squad_growth_garrison_step: 40.0,
            squad_soft_cap_garrison: 120.0,
            squad_move_sec_per_cell: 0.4,
        }
    }

    #[test]
    fn low_garrison_keeps_base_wave_size() {
        assert_eq!(next_wave_troops(rules(), 40.0, 0.0), (3.0, 0.0));
        assert_eq!(next_wave_troops(rules(), 2.9, 0.0), (2.0, 0.0));
    }

    #[test]
    fn wave_size_grows_linearly_until_soft_cap() {
        assert_eq!(next_wave_troops(rules(), 80.0, 0.0), (4.0, 0.0));
        assert_eq!(next_wave_troops(rules(), 120.0, 0.0), (5.0, 0.0));
    }

    #[test]
    fn wave_size_uses_square_root_growth_above_soft_cap() {
        let at_four_hundred = theoretical_wave_troops(rules(), 400.0);
        let at_one_thousand = theoretical_wave_troops(rules(), 1000.0);
        assert!((at_four_hundred - 7.645_751).abs() < 0.000_1);
        assert!((at_one_thousand - 9.690_416).abs() < 0.000_1);
    }

    #[test]
    fn fractional_strength_is_carried_into_later_waves() {
        let (first, carry) = next_wave_troops(rules(), 60.0, 0.0);
        let (second, carry) = next_wave_troops(rules(), 60.0, carry);
        assert_eq!((first, second, carry), (3.0, 4.0, 0.0));
    }
}
