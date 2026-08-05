//! 阶段 4：同格敌对小队相遇互灭（双向抵扣），剩者前进；清理阵亡小队实体。
//! 移植自旧 sim.rs 的 resolve_squad_clashes + retain。
//! 注意：旧实现用 HashMap 聚合格子分组，迭代顺序随机；此处改用 BTreeMap，
//! 换取确定性（黄金快照容差足以吸收与原实现的细微差异）。

use crate::components::*;
use crate::world_ext::load_squads;
use bevy_ecs::prelude::*;
use std::collections::BTreeMap;

pub fn combat(world: &mut World) {
    if world.resource::<Winner>().0.is_some() {
        return;
    }
    let mut squads = load_squads(world);

    let mut by_cell: BTreeMap<CellIdx, Vec<usize>> = BTreeMap::new();
    for (i, (_, sq)) in squads.iter().enumerate() {
        if sq.troops > 0.0 {
            by_cell.entry(sq.current_cell()).or_default().push(i);
        }
    }
    for (_, group) in by_cell {
        // 聚合约每个阵营的兵力
        let mut sums: BTreeMap<FactionId, f32> = BTreeMap::new();
        for &i in &group {
            *sums.entry(squads[i].1.faction).or_default() += squads[i].1.troops;
        }
        loop {
            // 找兵力最强的两个不同阵营
            let mut top: Vec<(FactionId, f32)> = sums
                .iter()
                .filter(|(_, &v)| v > 0.0)
                .map(|(&k, &v)| (k, v))
                .collect();
            if top.len() < 2 {
                break;
            }
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let (fa, fb) = (top[0].0, top[1].0);
            let d = top[0].1.min(top[1].1);
            *sums.get_mut(&fa).unwrap() -= d;
            *sums.get_mut(&fb).unwrap() -= d;
            // 按序从各小队扣除
            drain(&mut squads, &group, fa, d);
            drain(&mut squads, &group, fb, d);
        }
    }

    // 写回 + 清理阵亡小队（旧实现的 retain(troops > 0)）
    for (e, sq) in &squads {
        if sq.troops > 0.0 {
            crate::world_ext::write_squad(world, *e, sq);
        } else {
            world.despawn(*e);
        }
    }
}

fn drain(squads: &mut [(Entity, Squad)], group: &[usize], faction: FactionId, mut amount: f32) {
    for &i in group {
        if amount <= 0.0 {
            break;
        }
        let sq = &mut squads[i].1;
        if sq.faction == faction && sq.troops > 0.0 {
            let d = sq.troops.min(amount);
            sq.troops -= d;
            amount -= d;
        }
    }
}
