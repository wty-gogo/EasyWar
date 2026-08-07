//! 内部棋盘快照：SimTick 系统从 World 读出稠密数组、跑旧算法、写回变更。
//! 这只是系统内部的移植策略——对外契约仍是纯 ECS 组件。

use crate::components::*;
use bevy_ecs::prelude::*;

/// 据点的可计算信息（从 Base 组件 + 格子状态派生）
#[derive(Clone, Debug)]
pub(crate) struct BaseInfo {
    pub cell: CellIdx,
    pub production_base: f32,
    pub production_bonus_per_tile: f32,
    pub linked: Vec<CellIdx>,
}

pub(crate) struct Board {
    pub width: usize,
    pub height: usize,
    pub kind: Vec<CellKind>,
    pub owner: Vec<FactionId>,
    pub garrison: Vec<f32>,
    pub garrison_max: Vec<f32>,
    pub bases: Vec<BaseInfo>,
    pub rules: Rules,
    /// 被本阶段修改过的格子（写回用）
    dirty: Vec<bool>,
}

impl Board {
    pub(crate) fn load(world: &mut World) -> Self {
        let lookup = world.resource::<GridLookup>().clone();
        let rules = *world.resource::<Rules>();
        let n = lookup.cells.len();
        let mut kind = vec![CellKind::Void; n];
        let mut owner = vec![NEUTRAL; n];
        let mut garrison = vec![0.0; n];
        let mut garrison_max = vec![0.0; n];
        // 按 lookup 逐格取组件（无组件的格子保留默认值）
        for (i, &e) in lookup.cells.iter().enumerate() {
            if let Some(k) = world.get::<CellKind>(e) {
                kind[i] = *k;
            }
            if let Some(o) = world.get::<Owner>(e) {
                owner[i] = o.0;
            }
            if let Some(g) = world.get::<Garrison>(e) {
                garrison[i] = g.cur;
                garrison_max[i] = g.max;
            }
        }
        let base_list = world.resource::<BaseList>().clone();
        let mut bases = Vec::new();
        for &e in &base_list.0 {
            let b = world.get::<Base>(e).expect("BaseList 中的实体缺 Base 组件");
            // 找该实体在 lookup 中的格子下标
            let cell = lookup
                .cells
                .iter()
                .position(|&c| c == e)
                .expect("Base 实体不在 GridLookup 中");
            bases.push(BaseInfo {
                cell,
                production_base: b.production_base,
                production_bonus_per_tile: b.production_bonus_per_tile,
                linked: b.linked.clone(),
            });
        }
        Board {
            width: lookup.width,
            height: lookup.height,
            kind,
            owner,
            garrison,
            garrison_max,
            bases,
            rules,
            dirty: vec![false; n],
        }
    }

    /// 把脏格子的 owner/garrison 写回组件（仅在值变化时写，保留变更检测语义）
    pub(crate) fn flush(self, world: &mut World) {
        let lookup = world.resource::<GridLookup>().clone();
        for (i, &e) in lookup.cells.iter().enumerate() {
            if !self.dirty[i] {
                continue;
            }
            if let Some(mut o) = world.get_mut::<Owner>(e) {
                if o.0 != self.owner[i] {
                    o.0 = self.owner[i];
                }
            }
            if let Some(mut g) = world.get_mut::<Garrison>(e) {
                if g.cur != self.garrison[i] || g.max != self.garrison_max[i] {
                    g.cur = self.garrison[i];
                    g.max = self.garrison_max[i];
                }
            }
        }
    }

    pub(crate) fn touch(&mut self, i: CellIdx) {
        self.dirty[i] = true;
    }

    pub(crate) fn xy(&self, i: CellIdx) -> (usize, usize) {
        (i % self.width, i / self.width)
    }

    fn in_bounds(&self, x: i64, y: i64) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height
    }

    /// 四向连通邻居（不含虚空）。顺序与旧实现一致：右、左、下、上。
    pub(crate) fn neighbors(&self, i: CellIdx) -> Vec<CellIdx> {
        let (x, y) = self.xy(i);
        let mut out = Vec::with_capacity(4);
        for (dx, dy) in [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)] {
            let (nx, ny) = (x as i64 + dx, y as i64 + dy);
            if self.in_bounds(nx, ny) {
                let j = ny as usize * self.width + nx as usize;
                if self.kind[j].enterable() {
                    out.push(j);
                }
            }
        }
        out
    }

    fn owned_linked(&self, base: &BaseInfo) -> usize {
        let owner = self.owner[base.cell];
        base.linked
            .iter()
            .filter(|&&c| self.owner[c] == owner)
            .count()
    }

    /// 据点实时产能 = 基础 + 加成 × 当前归属方占领的关联地块数
    pub(crate) fn base_production(&self, base: &BaseInfo) -> f32 {
        let owner = self.owner[base.cell];
        if owner == NEUTRAL {
            return 0.0;
        }
        base.production_base + base.production_bonus_per_tile * self.owned_linked(base) as f32
    }

    /// 据点驻军上限 = 基础上限 + 每块已占领关联地块 × 上限加成
    pub(crate) fn base_garrison_cap(&self, base: &BaseInfo) -> f32 {
        self.rules.garrison_cap_base
            + self.rules.garrison_cap_per_tile * self.owned_linked(base) as f32
    }

    /// 纯最短路径（按格数），等长时优先己方格子。
    /// Dijkstra：进入每格代价 = 1 + ε·(非己方)。逐行移植自旧 model.rs。
    pub(crate) fn find_path(
        &self,
        from: CellIdx,
        to: CellIdx,
        faction: FactionId,
    ) -> Option<Vec<CellIdx>> {
        if from == to || !self.kind[from].enterable() || !self.kind[to].enterable() {
            return None;
        }
        const EPS: f32 = 0.001;
        let n = self.kind.len();
        let mut dist = vec![f32::INFINITY; n];
        let mut prev: Vec<Option<CellIdx>> = vec![None; n];
        let mut done = vec![false; n];
        dist[from] = 0.0;
        for _ in 0..n {
            let mut u = None;
            let mut best = f32::INFINITY;
            for i in 0..n {
                if !done[i] && dist[i] < best {
                    best = dist[i];
                    u = Some(i);
                }
            }
            let u = u?;
            done[u] = true;
            if u == to {
                break;
            }
            for v in self.neighbors(u) {
                let step = 1.0 + if self.owner[v] == faction { 0.0 } else { EPS };
                let nd = dist[u] + step;
                if nd < dist[v] {
                    dist[v] = nd;
                    prev[v] = Some(u);
                }
            }
        }
        if !done[to] {
            return None;
        }
        let mut path = vec![to];
        let mut cur = to;
        while cur != from {
            cur = prev[cur]?;
            path.push(cur);
        }
        path.reverse();
        Some(path)
    }
}
