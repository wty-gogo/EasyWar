use crate::model::*;
use std::collections::HashMap;

impl GameState {
    /// 推进 dt 秒游戏时间。所有战斗均为双向抵扣（见 GDD §3.3）。
    pub fn update(&mut self, dt: f32) {
        if self.winner.is_some() {
            return;
        }
        self.time += dt;

        self.produce_and_regen(dt);
        self.update_streams(dt);
        self.move_squads(dt);
        self.resolve_squad_clashes();
        self.squads.retain(|s| s.troops > 0.0);
        self.check_winner();
    }

    /// 1. 据点产兵 + 地块/中立据点回防
    fn produce_and_regen(&mut self, dt: f32) {
        // 据点产兵（封顶只阻止增长，不删除已有驻军：丢失地块导致上限下降时存量保留）
        for i in 0..self.bases.len() {
            let prod = self.base_production(&self.bases[i]);
            if prod > 0.0 {
                let cap = self.base_garrison_cap(&self.bases[i]);
                let cell = self.bases[i].cell;
                let g = self.cells[cell].garrison;
                if g < cap {
                    self.cells[cell].garrison = (g + prod * dt).min(cap);
                }
            }
        }
        // 回防：中立格子（含中立要塞）+ 任何阵营的地块（非据点）
        for c in self.cells.iter_mut() {
            let regens = c.owner == NEUTRAL || c.kind != CellKind::Base;
            if regens && c.enterable() && c.garrison < c.garrison_max {
                c.garrison = (c.garrison + self.rules.regen_per_sec * dt).min(c.garrison_max);
            }
        }
    }

    /// 2. 兵流：终止判定 + 按节奏出兵
    fn update_streams(&mut self, dt: f32) {
        for si in 0..self.streams.len() {
            if !self.streams[si].active {
                continue;
            }
            let (faction, source, target) = {
                let s = &self.streams[si];
                (s.faction, s.source, s.target)
            };

            // 源据点失守 → 兵流自然消亡（在途小队继续飞，生死自负）
            if self.cells[source].owner != faction {
                self.streams[si].active = false;
                continue;
            }

            // 目标是地块且已被本方占领 → 停止出兵，途中兵回家
            let tk = self.cells[target].kind;
            if (tk == CellKind::Plain || tk == CellKind::LinkedTile)
                && self.cells[target].owner == faction
            {
                self.recall_stream(si);
                continue;
            }
            // 目标是据点：占领后继续输送（增援流），不做任何处理

            // 出兵：每 squad_interval_sec 出一队，兵力 = min(上限, 当前驻军)，不足 3 也照出。
            // 停止条件（两种，对所有目标类型一致）：
            //   - 目标是地块且已被本方占领 → 终止；
            //   - 驻军被抽到 0 → 立刻终止（在途小队到目标后再回家）。
            // 目标是据点则永不因占领而终止：打下来之后自动变成增援流，直到驻军归零。
            self.streams[si].spawn_accum += dt;
            let mut exhausted = false;
            while self.streams[si].spawn_accum >= self.rules.squad_interval_sec {
                self.streams[si].spawn_accum -= self.rules.squad_interval_sec;
                let avail = self.cells[source].garrison.floor();
                if avail >= 1.0 {
                    let n = avail.min(self.rules.squad_max_size);
                    self.cells[source].garrison -= n;
                    let path = self.streams[si].path.clone();
                    self.squads.push(Squad {
                        faction,
                        troops: n,
                        path,
                        seg: 0,
                        t: 0.0,
                        mode: SquadMode::ToTarget,
                        stream: si,
                        return_after_target: false,
                    });
                    if self.cells[source].garrison.floor() < 1.0 {
                        exhausted = true;
                        break;
                    }
                }
            }
            if exhausted {
                self.recall_stream(si);
                continue;
            }
        }
    }

    /// 3. 小队移动与逐格结算（小队 vs 地块/据点：双向抵扣）
    fn move_squads(&mut self, dt: f32) {
        let speed = 1.0 / self.rules.squad_move_sec_per_cell; // 格/秒
        let mut needs_return: Vec<usize> = Vec::new();
        for (qi, sq) in self.squads.iter_mut().enumerate() {
            sq.t += speed * dt;
            while sq.t >= 1.0 && sq.troops > 0.0 {
                sq.t -= 1.0;
                let next = sq.path[sq.seg + 1];

                // 进入非己方格：双向抵扣
                if self.cells[next].owner != sq.faction {
                    let d = sq.troops.min(self.cells[next].garrison);
                    sq.troops -= d;
                    self.cells[next].garrison -= d;
                    if self.cells[next].garrison <= 0.0 {
                        // 占领：据点易主 / 地块易主
                        self.cells[next].garrison = 0.0;
                        self.cells[next].owner = sq.faction;
                    }
                } else if self.cells[next].kind == CellKind::Base {
                    // 与己方据点碰撞：直接进入据点并入驻军（无论是否兵流终点）。
                    // 并入无上限——上限只约束生产，不约束并入
                    self.cells[next].garrison += sq.troops;
                    sq.troops = 0.0;
                }

                if sq.troops <= 0.0 {
                    break; // 全灭
                }

                sq.seg += 1;
                if sq.seg + 1 >= sq.path.len() {
                    if sq.return_after_target {
                        // 兵流已终止：到达目标后不并入，掉头回家（循环外统一改道）
                        needs_return.push(qi);
                        break;
                    }
                    // 到达终点：并入驻军（增援/回家），并入无上限
                    self.cells[next].garrison += sq.troops;
                    sq.troops = 0.0;
                    break;
                }
            }
        }
        // 统一处理"到达目标后回家"的小队（避免循环内借用冲突）
        for qi in needs_return {
            let sq = &self.squads[qi];
            if sq.troops <= 0.0 {
                continue;
            }
            let (faction, from, stream_id) = (sq.faction, sq.current_cell(), sq.stream);
            let source = self.streams[stream_id].source;
            if let Some(p) = self.find_path(from, source, faction) {
                let sq = &mut self.squads[qi];
                sq.path = p;
                sq.seg = 0;
                sq.t = 0.0;
                sq.mode = SquadMode::Return;
                sq.return_after_target = false;
            } else {
                // 无路可回家：就地并入
                let troops = self.squads[qi].troops;
                self.cells[from].garrison += troops;
                self.squads[qi].troops = 0.0;
            }
        }
    }

    /// 4. 同格敌对小队相遇：互灭（双向抵扣），剩者前进
    fn resolve_squad_clashes(&mut self) {
        let mut by_cell: HashMap<CellIdx, Vec<usize>> = HashMap::new();
        for (i, sq) in self.squads.iter().enumerate() {
            if sq.troops > 0.0 {
                by_cell.entry(sq.current_cell()).or_default().push(i);
            }
        }
        for (_, group) in by_cell {
            // 聚合约每个阵营的兵力
            let mut sums: HashMap<FactionId, f32> = HashMap::new();
            for &i in &group {
                *sums.entry(self.squads[i].faction).or_default() += self.squads[i].troops;
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
                Self::drain(&mut self.squads, &group, fa, d);
                Self::drain(&mut self.squads, &group, fb, d);
            }
        }
    }

    fn drain(squads: &mut [Squad], group: &[usize], faction: FactionId, mut amount: f32) {
        for &i in group {
            if amount <= 0.0 {
                break;
            }
            let sq = &mut squads[i];
            if sq.faction == faction && sq.troops > 0.0 {
                let d = sq.troops.min(amount);
                sq.troops -= d;
                amount -= d;
            }
        }
    }

    /// 5. 胜负：只剩一个非中立阵营拥有据点 → 该阵营获胜
    fn check_winner(&mut self) {
        let alive = self.alive_factions();
        if alive.len() == 1 {
            self.winner = Some(alive[0]);
        }
    }
}
