//! 规则驱动 AI（GDD §6）：决策循环 + 优先级 + 三档难度行为参数。
//! 难度全部来自行为参数，不修改任何游戏数值（不作弊原则）。

use crate::model::*;

#[derive(Clone, Copy, Debug)]
pub struct AiParams {
    /// 决策间隔（秒）
    pub decision_interval: f32,
    /// 出手阈值：驻军 > 目标防御 × N 才行动
    pub attack_threshold: f32,
    /// 扩张野心：0 = 只吃关联地块；1 = +啃中立要塞；2 = +抢对方手上的关联地块
    pub expansion: u8,
    /// 是否拦截对方兵流
    pub intercept: bool,
    /// 同时维持的兵流数上限
    pub max_streams: usize,
    /// 总攻阈值：总兵力优势倍数
    pub total_attack_ratio: f32,
    /// 失误率：本次决策发呆/跳过的概率
    pub error_rate: f32,
}

impl AiParams {
    pub fn easy() -> Self {
        Self {
            decision_interval: 3.0,
            attack_threshold: 2.0,
            expansion: 0,
            intercept: false,
            max_streams: 1,
            total_attack_ratio: 1.8,
            error_rate: 0.2,
        }
    }
    pub fn normal() -> Self {
        Self {
            decision_interval: 2.0,
            attack_threshold: 1.5,
            expansion: 1,
            intercept: false,
            max_streams: 2,
            total_attack_ratio: 1.3,
            error_rate: 0.05,
        }
    }
    pub fn hard() -> Self {
        Self {
            decision_interval: 1.0,
            attack_threshold: 1.2,
            expansion: 2,
            intercept: true,
            max_streams: 3,
            total_attack_ratio: 1.15,
            error_rate: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum AiCommand {
    SetStream { source: CellIdx, target: CellIdx },
    #[allow(dead_code)]
    StopStream { source: CellIdx },
}

/// 一个 AI 控制器管一个阵营
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

    /// 每帧调用；到决策点时返回一条指令（也可能空手而归）
    pub fn update(&mut self, game: &GameState, dt: f32) -> Vec<AiCommand> {
        self.timer -= dt;
        if self.timer > 0.0 || game.winner.is_some() {
            return Vec::new();
        }
        self.timer = self.params.decision_interval;
        if self.roll() < self.params.error_rate {
            return Vec::new(); // 发呆
        }
        self.decide(game).into_iter().collect()
    }

    fn decide(&mut self, game: &GameState) -> Option<AiCommand> {
        let me = self.faction;
        let my_bases: Vec<&Base> = game
            .bases
            .iter()
            .filter(|b| game.cells[b.cell].owner == me)
            .collect();
        if my_bases.is_empty() {
            return None;
        }
        let garrison_of = |c: CellIdx| game.cells[c].garrison;

        // ---- 1. 守：据点被敌小队瞄准且驻军不足 → 从最富的己方据点增援 ----
        for b in &my_bases {
            let threatened = game.squads.iter().any(|sq| {
                sq.faction != me
                    && sq.mode == SquadMode::ToTarget
                    && sq.path.last() == Some(&b.cell)
            });
            if threatened && garrison_of(b.cell) < 0.4 * game.base_garrison_cap(b) {
                if let Some(rich) = my_bases
                    .iter()
                    .filter(|r| r.cell != b.cell)
                    .max_by(|a, c| garrison_of(a.cell).partial_cmp(&garrison_of(c.cell)).unwrap())
                {
                    if garrison_of(rich.cell) > 10.0 {
                        return Some(AiCommand::SetStream { source: rich.cell, target: b.cell });
                    }
                }
            }
        }

        // 兵流数已满 → 不再开新战线
        let active = game
            .streams
            .iter()
            .filter(|s| s.active && s.faction == me)
            .count();
        if active >= self.params.max_streams {
            return None;
        }

        let strongest = |exclude: Option<CellIdx>| {
            my_bases
                .iter()
                .filter(|b| Some(b.cell) != exclude)
                // 已有兵流的据点不开第二条
                .filter(|b| game.stream_from(me, b.cell).is_none())
                .max_by(|a, c| garrison_of(a.cell).partial_cmp(&garrison_of(c.cell)).unwrap())
                .copied()
        };

        // ---- 1.5 拦截（困难）：朝对方兵流路径中点派兵对冲 ----
        if self.params.intercept {
            if let Some(ps) = game
                .streams
                .iter()
                .find(|s| s.active && s.faction != me && s.faction != NEUTRAL)
            {
                let mid = ps.path[ps.path.len() / 2];
                if game.cells[mid].owner != me {
                    if let Some(src) = strongest(None) {
                        if garrison_of(src.cell) > 30.0 {
                            return Some(AiCommand::SetStream { source: src.cell, target: mid });
                        }
                    }
                }
            }
        }

        // ---- 2. 吃产能：占领自己据点的未占领关联地块 ----
        let mut sorted = my_bases.clone();
        sorted.sort_by(|a, c| garrison_of(c.cell).partial_cmp(&garrison_of(a.cell)).unwrap());
        for b in sorted {
            if game.stream_from(me, b.cell).is_some() {
                continue; // 这个据点已经在干活
            }
            let mut targets: Vec<CellIdx> = b
                .linked
                .iter()
                .copied()
                .filter(|&t| game.cells[t].owner != me)
                .collect();
            targets.sort_by(|&a, &c| garrison_of(a).partial_cmp(&garrison_of(c)).unwrap());
            if let Some(&t) = targets.first() {
                if garrison_of(b.cell) > garrison_of(t) * self.params.attack_threshold {
                    return Some(AiCommand::SetStream { source: b.cell, target: t });
                }
            }
        }

        // ---- 3. 扩张 ----
        if self.params.expansion >= 1 {
            // 啃中立要塞（驻军 > 要塞驻军 × 阈值）
            for b in &game.bases {
                if game.cells[b.cell].owner != NEUTRAL {
                    continue;
                }
                if let Some(src) = strongest(None) {
                    if garrison_of(src.cell) > garrison_of(b.cell) * self.params.attack_threshold {
                        return Some(AiCommand::SetStream { source: src.cell, target: b.cell });
                    }
                }
            }
        }
        if self.params.expansion >= 2 {
            // 抢对方手上的关联地块（断对方产能加成）
            for eb in &game.bases {
                let eo = game.cells[eb.cell].owner;
                if eo == me || eo == NEUTRAL {
                    continue;
                }
                for &t in &eb.linked {
                    if game.cells[t].owner == eo {
                        if let Some(src) = strongest(None) {
                            if garrison_of(src.cell) > garrison_of(t) * self.params.attack_threshold
                            {
                                return Some(AiCommand::SetStream { source: src.cell, target: t });
                            }
                        }
                    }
                }
            }
        }

        // ---- 4. 总攻：总兵力优势超过阈值 → 直捣对方最弱据点 ----
        // 饱和总攻：双方上限相同时优势阈值永远达不到 → 据点囤满且不明显劣势时也全军出击，
        // 防止对称均势下的无限囤兵死锁（"无平局"设计的 AI 侧保障）
        let my_total = game.total_troops(me);
        let enemy = game
            .factions
            .iter()
            .filter(|f| f.id != me)
            .map(|f| f.id)
            .max_by(|&a, &c| game.total_troops(a).partial_cmp(&game.total_troops(c)).unwrap());
        if let Some(e) = enemy {
            let enemy_total = game.total_troops(e);
            let weakest_enemy_base = game
                .bases
                .iter()
                .filter(|b| game.cells[b.cell].owner == e)
                .min_by(|a, c| garrison_of(a.cell).partial_cmp(&garrison_of(c.cell)).unwrap());
            if let (Some(src), Some(tgt)) = (strongest(None), weakest_enemy_base) {
                let cap = game.base_garrison_cap(src);
                let saturated = garrison_of(src.cell) >= 0.95 * cap;
                let dominant = my_total > enemy_total * self.params.total_attack_ratio;
                let all_in = saturated && my_total >= enemy_total * 0.9;
                if (dominant || all_in) && garrison_of(src.cell) > 30.0 {
                    return Some(AiCommand::SetStream { source: src.cell, target: tgt.cell });
                }
            }
        }
        None
    }
}
