use std::collections::HashMap;

pub type FactionId = u8;
pub type CellIdx = usize;

/// 中立（无主）阵营 id
pub const NEUTRAL: FactionId = 0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CellKind {
    /// 虚空，不可进入，不构成地图
    Void,
    /// 普通中立地块（纯数值路障）
    Plain,
    /// 关联地块（知识点），绑定某个据点
    LinkedTile,
    /// 据点（产兵，兵力唯一来源）
    Base,
}

#[derive(Clone, Debug)]
pub struct Cell {
    pub kind: CellKind,
    pub owner: FactionId,
    /// 防御值（地块）或驻军（据点）
    pub garrison: f32,
    /// 回防上限（普通/关联地块/中立要塞）；己方据点不使用
    pub garrison_max: f32,
    /// 关联地块的知识点名称（如 "函数"）
    pub label: Option<String>,
}

impl Cell {
    pub fn enterable(&self) -> bool {
        self.kind != CellKind::Void
    }
}

#[derive(Clone, Debug)]
pub struct Base {
    pub cell: CellIdx,
    pub subject_id: String,
    pub subject_name: String,
    pub production_base: f32,
    pub production_bonus_per_tile: f32,
    /// 写死归属本据点的关联地块
    pub linked: Vec<CellIdx>,
}

#[derive(Clone, Debug)]
pub struct Faction {
    pub id: FactionId,
    pub name: String,
    pub color: [f32; 4],
    pub is_player: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SquadMode {
    /// 前往兵流目标
    ToTarget,
    /// 兵流终止后返回源据点
    Return,
}

#[derive(Clone, Debug)]
pub struct Squad {
    pub faction: FactionId,
    pub troops: f32,
    /// path[0] = 源，path[last] = 目标
    pub path: Vec<CellIdx>,
    /// 当前所在段：从 path[seg] 走向 path[seg+1]
    pub seg: usize,
    /// 段内进度 0..1
    pub t: f32,
    pub mode: SquadMode,
    pub stream: usize,
    /// 兵流已终止：先飞到目标完成战斗，幸存后再回家（不原地掉头）
    pub return_after_target: bool,
}

impl Squad {
    pub fn current_cell(&self) -> CellIdx {
        self.path[self.seg]
    }
}

#[derive(Clone, Debug)]
pub struct Stream {
    pub faction: FactionId,
    pub source: CellIdx,
    pub target: CellIdx,
    pub path: Vec<CellIdx>,
    pub spawn_accum: f32,
    pub active: bool,
}

/// 全局规则参数（与 assets/maps/*.toml 的 [rules] 对应）
#[derive(Clone, Copy, Debug)]
pub struct Rules {
    pub garrison_cap_base: f32,
    pub garrison_cap_per_tile: f32,
    pub regen_per_sec: f32,
    pub squad_interval_sec: f32,
    pub squad_max_size: f32,
    pub squad_move_sec_per_cell: f32,
}

#[derive(Clone, Debug)]
pub struct GameState {
    pub width: usize,
    pub height: usize,
    pub cells: Vec<Cell>,
    pub bases: Vec<Base>,
    /// cell -> bases 下标
    pub base_index: HashMap<CellIdx, usize>,
    pub factions: Vec<Faction>,
    pub squads: Vec<Squad>,
    pub streams: Vec<Stream>,
    pub rules: Rules,
    pub time: f32,
    pub winner: Option<FactionId>,
}

impl GameState {
    pub fn idx(&self, x: usize, y: usize) -> CellIdx {
        y * self.width + x
    }

    pub fn xy(&self, i: CellIdx) -> (usize, usize) {
        (i % self.width, i / self.width)
    }

    pub fn in_bounds(&self, x: i64, y: i64) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height
    }

    /// 四向连通邻居（不含虚空）
    pub fn neighbors(&self, i: CellIdx) -> impl Iterator<Item = CellIdx> + '_ {
        let (x, y) = self.xy(i);
        let mut out = Vec::with_capacity(4);
        for (dx, dy) in [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)] {
            let (nx, ny) = (x as i64 + dx, y as i64 + dy);
            if self.in_bounds(nx, ny) {
                let j = self.idx(nx as usize, ny as usize);
                if self.cells[j].enterable() {
                    out.push(j);
                }
            }
        }
        out.into_iter()
    }

    /// 据点实时产能 = 基础 + 加成 × 当前归属方占领的关联地块数
    pub fn base_production(&self, base: &Base) -> f32 {
        let owner = self.cells[base.cell].owner;
        if owner == NEUTRAL {
            return 0.0;
        }
        base.production_base + base.production_bonus_per_tile * self.owned_linked(base) as f32
    }

    /// 据点驻军上限 = 基础上限 + 每块已占领关联地块 × 上限加成
    pub fn base_garrison_cap(&self, base: &Base) -> f32 {
        self.rules.garrison_cap_base + self.rules.garrison_cap_per_tile * self.owned_linked(base) as f32
    }

    fn owned_linked(&self, base: &Base) -> usize {
        let owner = self.cells[base.cell].owner;
        base.linked
            .iter()
            .filter(|&&c| self.cells[c].owner == owner)
            .count()
    }

    /// 纯最短路径（按格数），等长时优先己方格子。
    /// 实现：Dijkstra，进入每格代价 = 1 + ε·(非己方)，ε 远小于 1 保证长度优先。
    pub fn find_path(&self, from: CellIdx, to: CellIdx, faction: FactionId) -> Option<Vec<CellIdx>> {
        if from == to || !self.cells[from].enterable() || !self.cells[to].enterable() {
            return None;
        }
        const EPS: f32 = 0.001;
        let n = self.cells.len();
        let mut dist = vec![f32::INFINITY; n];
        let mut prev: Vec<Option<CellIdx>> = vec![None; n];
        let mut done = vec![false; n];
        dist[from] = 0.0;
        for _ in 0..n {
            // 线性扫描取最小（网格仅数百格，足够快）
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
                let step = 1.0 + if self.cells[v].owner == faction { 0.0 } else { EPS };
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

    /// 建立或改道兵流（同一源据点只维持一条）
    pub fn set_stream(&mut self, faction: FactionId, source: CellIdx, target: CellIdx) -> bool {
        if self.cells[source].kind != CellKind::Base || self.cells[source].owner != faction {
            return false;
        }
        if !self.cells[target].enterable() || source == target {
            return false;
        }
        let Some(path) = self.find_path(source, target, faction) else {
            return false;
        };
        // 第一遍：找出要替换的旧兵流，并为其在途小队算好改道路径
        let mut replaced = Vec::new();
        for (si, s) in self.streams.iter().enumerate() {
            if s.active && s.faction == faction && s.source == source {
                replaced.push(si);
            }
        }
        for &si in &replaced {
            let mut new_paths = Vec::new();
            for (qi, sq) in self.squads.iter().enumerate() {
                if sq.stream == si && sq.mode == SquadMode::ToTarget {
                    if let Some(p) = self.find_path(sq.current_cell(), target, faction) {
                        new_paths.push((qi, p));
                    }
                }
            }
            for (qi, p) in new_paths {
                let sq = &mut self.squads[qi];
                sq.path = p;
                sq.seg = 0;
                sq.t = 0.0;
                sq.return_after_target = false; // 改道即新任务，取消"到点后回家"
            }
            self.streams[si].active = false;
        }
        self.streams.push(Stream {
            faction,
            source,
            target,
            path,
            spawn_accum: 0.0,
            active: true,
        });
        true
    }

    /// 停止兵流，途中兵回家
    pub fn stop_stream(&mut self, faction: FactionId, source: CellIdx) {
        let mut to_recall = Vec::new();
        for (si, s) in self.streams.iter().enumerate() {
            if s.active && s.faction == faction && s.source == source {
                to_recall.push(si);
            }
        }
        for si in to_recall {
            self.recall_stream(si);
        }
    }

    /// 停用兵流：在途小队**继续飞向目标**，到达后幸存的再返回源据点
    pub(crate) fn recall_stream(&mut self, si: usize) {
        self.streams[si].active = false;
        for sq in self.squads.iter_mut() {
            if sq.stream == si && sq.mode == SquadMode::ToTarget {
                sq.return_after_target = true;
            }
        }
    }

    pub fn stream_from(&self, faction: FactionId, source: CellIdx) -> Option<(usize, &Stream)> {
        self.streams
            .iter()
            .enumerate()
            .find(|(_, s)| s.active && s.faction == faction && s.source == source)
    }

    /// 仍拥有至少一个据点的非中立阵营
    pub fn alive_factions(&self) -> Vec<FactionId> {
        let mut alive: Vec<FactionId> = Vec::new();
        for b in &self.bases {
            let owner = self.cells[b.cell].owner;
            if owner != NEUTRAL && !alive.contains(&owner) {
                alive.push(owner);
            }
        }
        alive
    }

    /// 阵营总兵力 = 据点驻军 + 在途小队（AI 评估用）
    pub fn total_troops(&self, faction: FactionId) -> f32 {
        let garrison: f32 = self
            .bases
            .iter()
            .filter(|b| self.cells[b.cell].owner == faction)
            .map(|b| self.cells[b.cell].garrison)
            .sum();
        let transit: f32 = self
            .squads
            .iter()
            .filter(|s| s.faction == faction)
            .map(|s| s.troops)
            .sum();
        garrison + transit
    }
}
