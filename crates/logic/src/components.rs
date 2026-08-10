//! 纯数据组件与公共资源。无逻辑——逻辑在各领域系统里。

use bevy_ecs::prelude::*;

pub type FactionId = u8;
pub type CellIdx = usize;

/// 中立（无主）阵营 id
pub const NEUTRAL: FactionId = 0;

// ---------- 格子 ----------

#[derive(Component, Clone, Copy, PartialEq, Eq, Debug)]
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

impl CellKind {
    pub fn enterable(&self) -> bool {
        *self != CellKind::Void
    }
}

/// 格子归属
#[derive(Component, Clone, Copy, PartialEq, Eq, Debug)]
pub struct Owner(pub FactionId);

/// 防御值（地块）或驻军（据点）
#[derive(Component, Clone, Copy, Debug)]
pub struct Garrison {
    pub cur: f32,
    /// 回防上限（普通/关联地块/中立要塞）；己方据点为 f32::INFINITY
    pub max: f32,
}

/// 关联地块的知识点名称 / 据点的学科名
#[derive(Component, Clone, Debug)]
pub struct Label(pub Option<String>);

/// 据点附加数据（挂在据点格子实体上）
#[derive(Component, Clone, Debug)]
pub struct Base {
    pub subject_id: String,
    pub subject_name: String,
    pub production_base: f32,
    pub production_bonus_per_tile: f32,
    /// 写死归属本据点的关联地块
    pub linked: Vec<CellIdx>,
}

// ---------- 小队 ----------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SquadMode {
    /// 前往兵流目标
    ToTarget,
    /// 兵流终止后返回源据点
    Return,
}

#[derive(Component, Clone, Debug, PartialEq)]
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
    /// 所属兵流实体
    pub stream: Entity,
    /// 兵流已终止：先飞到目标完成战斗，幸存后再回家（不原地掉头）
    pub return_after_target: bool,
    /// 单调递增序号：保证迭代顺序 = 生成顺序（确定性）
    pub seq: u64,
}

impl Squad {
    pub fn current_cell(&self) -> CellIdx {
        self.path[self.seg]
    }
}

// ---------- 兵流 ----------

#[derive(Component, Clone, Debug, PartialEq)]
pub struct Stream {
    pub faction: FactionId,
    pub source: CellIdx,
    pub target: CellIdx,
    pub path: Vec<CellIdx>,
    pub spawn_accum: f32,
    /// 理论波次兵力取整后留下的小数额度；改道会创建新兵流并清零。
    pub troop_carry: f32,
    pub active: bool,
    /// 单调递增序号：保证迭代顺序 = 建立顺序（确定性）
    pub seq: u64,
}

// ---------- 资源 ----------

/// 格子索引 → 实体。地图加载时构建一次，运行期不变。
#[derive(Resource, Clone, Debug)]
pub struct GridLookup {
    pub width: usize,
    pub height: usize,
    /// 长度 = width * height，含虚空格
    pub cells: Vec<Entity>,
}

impl GridLookup {
    pub fn idx(&self, x: usize, y: usize) -> CellIdx {
        y * self.width + x
    }
    pub fn xy(&self, i: CellIdx) -> (usize, usize) {
        (i % self.width, i / self.width)
    }
    pub fn entity(&self, i: CellIdx) -> Entity {
        self.cells[i]
    }
}

/// 据点实体列表（按地图文件出现顺序，即旧的 bases 下标语义）
#[derive(Resource, Clone, Debug)]
pub struct BaseList(pub Vec<Entity>);

#[derive(Clone, Debug)]
pub struct Faction {
    pub id: FactionId,
    pub name: String,
    pub color: [f32; 4],
    pub is_player: bool,
}

#[derive(Resource, Clone, Debug, Default)]
pub struct Factions(pub Vec<Faction>);

/// 全局规则参数（与 assets/maps/*.toml 的 [rules] 对应）
#[derive(Resource, Clone, Copy, Debug)]
pub struct Rules {
    pub garrison_cap_base: f32,
    pub garrison_cap_per_tile: f32,
    pub regen_per_sec: f32,
    pub squad_interval_sec: f32,
    pub squad_max_size: f32,
    /// 驻军超过此值后，每再增加这么多驻军，理论每波兵力增加 1。
    pub squad_growth_garrison_step: f32,
    /// 驻军超过此值后改用平方根缓增长，抑制超上限驻军的瞬时吞吐。
    pub squad_soft_cap_garrison: f32,
    pub squad_move_sec_per_cell: f32,
}

/// 游戏内时间（秒），随 SimTick 推进；分出胜负后停走
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct GameClock {
    pub time: f32,
}

#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct Winner(pub Option<FactionId>);

/// 单调递增序号分配器（Squad/Stream 的 seq）
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct SeqCounter(pub u64);

impl SeqCounter {
    pub fn next(&mut self) -> u64 {
        let v = self.0;
        self.0 += 1;
        v
    }
}
