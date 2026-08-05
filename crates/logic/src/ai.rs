//! AI 决策系统：规则驱动、不作弊，与玩家共用意图管道。
//! 决策逻辑逐行移植自旧 ai.rs；差异仅在于读取 Board 快照、输出 Intent。

use crate::board::Board;
use crate::components::*;
use crate::intents::{Intent, IntentQueue};
use crate::plugin::SIM_DT;
use crate::world_ext::{load_squads, load_streams};
use bevy_ecs::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct AiParams {
    /// 决策间隔（秒）
    pub decision_interval: f32,
    /// 出手阈值：驻军 > 目标防御 × N
    pub attack_threshold: f32,
    /// 扩张野心：0=仅己方关联地块，1=会啃要塞，2=会抢玩家关联地块
    pub expansion: u8,
    /// 是否拦截对方兵流
    pub intercept: bool,
    /// 同时维持兵流数上限
    pub max_streams: usize,
    /// 总攻阈值（兵力优势倍数）
    pub total_attack_ratio: f32,
    /// 失误率：本次决策发呆/跳过的概率
    pub error_rate: f32,
}

impl AiParams {
    pub fn easy() -> Self {
        Self { decision_interval: 3.0, attack_threshold: 2.0, expansion: 0, intercept: false, max_streams: 1, total_attack_ratio: 1.8, error_rate: 0.20 }
    }
    pub fn normal() -> Self {
        Self { decision_interval: 2.0, attack_threshold: 1.5, expansion: 1, intercept: false, max_streams: 2, total_attack_ratio: 1.3, error_rate: 0.05 }
    }
    pub fn hard() -> Self {
        Self { decision_interval: 1.0, attack_threshold: 1.2, expansion: 2, intercept: true, max_streams: 3, total_attack_ratio: 1.15, error_rate: 0.0 }
    }
}

/// 单个 AI 玩家的状态（确定性 xorshift 随机源内嵌）
#[derive(Clone, Debug)]
pub struct AiController {
    pub faction: FactionId,
    pub params: AiParams,
    timer: f32,
    rng: u64,
}

impl AiController {
    pub fn new(faction: FactionId, params: AiParams) -> Self {
        Self {
            faction,
            // 首次决策在 decision_interval 之后（与旧实现一致；置 0 会让 RNG 序列错位一拍）
            timer: params.decision_interval,
            params,
            rng: 0x9E3779B97F4A7C15 ^ (faction as u64 + 1).wrapping_mul(0x2545F4914F6CDD1D),
        }
    }

    fn roll(&mut self) -> f32 {
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        (x.wrapping_mul(0x2545F4914F6CDD1D) % 10000) as f32 / 10000.0
    }
}

/// AI 玩家列表。宿主按需插入：无头对战塞满，真人局只塞 AI 阵营。
#[derive(Resource, Default, Debug)]
pub struct AiControllers(pub Vec<AiController>);

/// SimTick 链首（apply_intents 之前）：各 AI 到点决策，把意图推入队列。
pub fn ai_decide(world: &mut World) {
    if world.resource::<Winner>().0.is_some() {
        return;
    }
    // 先推进计时器，决定本 tick 哪些 AI 出手
    let mut due: Vec<usize> = Vec::new();
    {
        let mut ctrls = world.resource_mut::<AiControllers>();
        for (i, c) in ctrls.0.iter_mut().enumerate() {
            c.timer -= SIM_DT;
            if c.timer <= 0.0 {
                c.timer = c.params.decision_interval;
                if c.roll() >= c.params.error_rate {
                    due.push(i);
                }
            }
        }
    }
    if due.is_empty() {
        return;
    }

    let board = Board::load(world);
    let squads = load_squads(world);
    let streams = load_streams(world);
    let factions = world.resource::<Factions>().0.clone();

    let mut intents = Vec::new();
    {
        let ctrls = world.resource::<AiControllers>();
        for &i in &due {
            let c = &ctrls.0[i];
            if let Some(intent) = decide(c, &board, &squads, &streams, &factions) {
                intents.push(intent);
            }
        }
    }
    let mut q = world.resource_mut::<IntentQueue>();
    for intent in intents {
        q.push(intent);
    }
}

/// 单个 AI 的一次决策。移植自旧 AiController::decide。
fn decide(
    c: &AiController,
    board: &Board,
    squads: &[(Entity, Squad)],
    streams: &[(Entity, Stream)],
    factions: &[Faction],
) -> Option<Intent> {
    let me = c.faction;
    let my_bases: Vec<&crate::board::BaseInfo> =
        board.bases.iter().filter(|b| board.owner[b.cell] == me).collect();
    if my_bases.is_empty() {
        return None;
    }
    let garrison_of = |cell: CellIdx| board.garrison[cell];
    let stream_from = |source: CellIdx| {
        streams
            .iter()
            .find(|(_, s)| s.active && s.faction == me && s.source == source)
    };

    // ---- 1. 守：据点被敌小队瞄准且驻军不足 → 从最富的己方据点增援 ----
    for b in &my_bases {
        let threatened = squads.iter().any(|(_, sq)| {
            sq.faction != me && sq.mode == SquadMode::ToTarget && sq.path.last() == Some(&b.cell)
        });
        if threatened && garrison_of(b.cell) < 0.4 * board.base_garrison_cap(b) {
            if let Some(rich) = my_bases
                .iter()
                .filter(|r| r.cell != b.cell)
                .max_by(|a, c| garrison_of(a.cell).partial_cmp(&garrison_of(c.cell)).unwrap())
            {
                if garrison_of(rich.cell) > 10.0 {
                    return Some(Intent::SetStream { faction: me, source: rich.cell, target: b.cell });
                }
            }
        }
    }

    // 兵流数已满 → 不再开新战线
    let active = streams.iter().filter(|(_, s)| s.active && s.faction == me).count();
    if active >= c.params.max_streams {
        return None;
    }

    let strongest = |exclude: Option<CellIdx>| {
        my_bases
            .iter()
            .filter(|b| Some(b.cell) != exclude)
            // 已有兵流的据点不开第二条
            .filter(|b| stream_from(b.cell).is_none())
            .max_by(|a, cc| garrison_of(a.cell).partial_cmp(&garrison_of(cc.cell)).unwrap())
            .copied()
    };

    // ---- 1.5 拦截（困难）：朝对方兵流路径中点派兵对冲 ----
    if c.params.intercept {
        if let Some((_, ps)) = streams
            .iter()
            .find(|(_, s)| s.active && s.faction != me && s.faction != NEUTRAL)
        {
            let mid = ps.path[ps.path.len() / 2];
            if board.owner[mid] != me {
                if let Some(src) = strongest(None) {
                    if garrison_of(src.cell) > 30.0 {
                        return Some(Intent::SetStream { faction: me, source: src.cell, target: mid });
                    }
                }
            }
        }
    }

    // ---- 2. 吃产能：占领自己据点的未占领关联地块 ----
    let mut sorted = my_bases.clone();
    sorted.sort_by(|a, cc| garrison_of(cc.cell).partial_cmp(&garrison_of(a.cell)).unwrap());
    for b in sorted {
        if stream_from(b.cell).is_some() {
            continue; // 这个据点已经在干活
        }
        let mut targets: Vec<CellIdx> =
            b.linked.iter().copied().filter(|&t| board.owner[t] != me).collect();
        targets.sort_by(|&a, &cc| garrison_of(a).partial_cmp(&garrison_of(cc)).unwrap());
        if let Some(&t) = targets.first() {
            if garrison_of(b.cell) > garrison_of(t) * c.params.attack_threshold {
                return Some(Intent::SetStream { faction: me, source: b.cell, target: t });
            }
        }
    }

    // ---- 3. 扩张 ----
    if c.params.expansion >= 1 {
        // 啃中立要塞（驻军 > 要塞驻军 × 阈值）
        for b in &board.bases {
            if board.owner[b.cell] != NEUTRAL {
                continue;
            }
            if let Some(src) = strongest(None) {
                if garrison_of(src.cell) > garrison_of(b.cell) * c.params.attack_threshold {
                    return Some(Intent::SetStream { faction: me, source: src.cell, target: b.cell });
                }
            }
        }
    }
    if c.params.expansion >= 2 {
        // 抢对方手上的关联地块（断对方产能加成）
        for eb in &board.bases {
            let eo = board.owner[eb.cell];
            if eo == me || eo == NEUTRAL {
                continue;
            }
            for &t in &eb.linked {
                if board.owner[t] == eo {
                    if let Some(src) = strongest(None) {
                        if garrison_of(src.cell) > garrison_of(t) * c.params.attack_threshold {
                            return Some(Intent::SetStream { faction: me, source: src.cell, target: t });
                        }
                    }
                }
            }
        }
    }

    // ---- 4. 总攻：总兵力优势超过阈值 → 直捣对方最弱据点 ----
    // 饱和总攻：双方上限相同时优势阈值永远达不到 → 据点囤满且不明显劣势时也全军出击，
    // 防止对称均势下的无限囤兵死锁（"无平局"设计的 AI 侧保障）
    let my_total = total_troops_board(board, squads, me);
    let enemy = factions
        .iter()
        .filter(|f| f.id != me)
        .map(|f| f.id)
        .max_by(|&a, &cc| {
            total_troops_board(board, squads, a)
                .partial_cmp(&total_troops_board(board, squads, cc))
                .unwrap()
        });
    if let Some(e) = enemy {
        let enemy_total = total_troops_board(board, squads, e);
        let weakest_enemy_base = board
            .bases
            .iter()
            .filter(|b| board.owner[b.cell] == e)
            .min_by(|a, cc| garrison_of(a.cell).partial_cmp(&garrison_of(cc.cell)).unwrap());
        if let (Some(src), Some(tgt)) = (strongest(None), weakest_enemy_base) {
            let cap = board.base_garrison_cap(src);
            let saturated = garrison_of(src.cell) >= 0.95 * cap;
            let dominant = my_total > enemy_total * c.params.total_attack_ratio;
            let all_in = saturated && my_total >= enemy_total * 0.9;
            if (dominant || all_in) && garrison_of(src.cell) > 30.0 {
                return Some(Intent::SetStream { faction: me, source: src.cell, target: tgt.cell });
            }
        }
    }
    None
}

/// 阵营总兵力 = 据点驻军 + 在途小队（Board 版，AI 评估用）
pub(crate) fn total_troops_board(board: &Board, squads: &[(Entity, Squad)], faction: FactionId) -> f32 {
    let garrison: f32 = board
        .bases
        .iter()
        .filter(|b| board.owner[b.cell] == faction)
        .map(|b| board.garrison[b.cell])
        .sum();
    let transit: f32 = squads.iter().filter(|(_, s)| s.faction == faction).map(|(_, s)| s.troops).sum();
    garrison + transit
}
