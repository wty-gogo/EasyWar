//! 阶段 3：小队移动与逐格结算（小队 vs 地块/据点：双向抵扣）。
//! 逐行移植自旧 sim.rs 的 move_squads。

use crate::board::Board;
use crate::components::*;
use crate::plugin::SIM_DT;
use crate::world_ext::{load_squads, write_squad};
use bevy_ecs::prelude::*;

pub fn movement(world: &mut World) {
    if world.resource::<Winner>().0.is_some() {
        return;
    }
    let mut board = Board::load(world);
    let mut squads = load_squads(world);
    let streams = crate::world_ext::load_streams(world);

    let speed = 1.0 / board.rules.squad_move_sec_per_cell; // 格/秒
    let mut needs_return: Vec<usize> = Vec::new();
    for (qi, (_, sq)) in squads.iter_mut().enumerate() {
        sq.t += speed * SIM_DT;
        while sq.t >= 1.0 && sq.troops > 0.0 {
            sq.t -= 1.0;
            let next = sq.path[sq.seg + 1];

            // 进入非己方格：双向抵扣
            if board.owner[next] != sq.faction {
                let d = sq.troops.min(board.garrison[next]);
                sq.troops -= d;
                board.garrison[next] -= d;
                board.touch(next);
                if board.garrison[next] <= 0.0 {
                    // 占领：据点易主 / 地块易主
                    board.garrison[next] = 0.0;
                    board.owner[next] = sq.faction;
                }
            } else if board.kind[next] == CellKind::Base {
                // 与己方据点碰撞：直接进入据点并入驻军（无论是否兵流终点）。
                // 并入无上限——上限只约束生产，不约束并入
                board.garrison[next] += sq.troops;
                board.touch(next);
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
                board.garrison[next] += sq.troops;
                board.touch(next);
                sq.troops = 0.0;
                break;
            }
        }
    }

    // 统一处理"到达目标后回家"的小队（避免循环内借用冲突）
    for qi in needs_return {
        let sq = &squads[qi].1;
        if sq.troops <= 0.0 {
            continue;
        }
        let (faction, from, stream_entity) = (sq.faction, sq.current_cell(), sq.stream);
        let Some(source) = streams
            .iter()
            .find(|(e, _)| *e == stream_entity)
            .map(|(_, s)| s.source)
        else {
            continue; // 兵流实体已不存在（不应发生）
        };
        if let Some(p) = board.find_path(from, source, faction) {
            let sq = &mut squads[qi].1;
            sq.path = p;
            sq.seg = 0;
            sq.t = 0.0;
            sq.mode = SquadMode::Return;
            sq.return_after_target = false;
        } else {
            // 无路可回家：就地并入
            let troops = squads[qi].1.troops;
            board.garrison[from] += troops;
            board.touch(from);
            squads[qi].1.troops = 0.0;
        }
    }

    for (e, sq) in &squads {
        write_squad(world, *e, sq);
    }
    board.flush(world);
}
