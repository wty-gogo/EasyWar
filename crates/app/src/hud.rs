//! HUD 与调试信息。

use crate::common::*;
use crate::telemetry::TelemetryRecorder;
use bevy::prelude::*;
use easywar_logic::*;

#[derive(Debug, PartialEq)]
struct BaseEconomy {
    rate: f32,
    cap: f32,
    rate_label: &'static str,
}

fn base_economy(
    rules: &Rules,
    base: &Base,
    owner: FactionId,
    owned_linked: usize,
    neutral_cap: f32,
) -> BaseEconomy {
    if owner == NEUTRAL {
        return BaseEconomy {
            rate: base.production_base,
            cap: neutral_cap,
            rate_label: "恢复",
        };
    }
    BaseEconomy {
        rate: base.production_base + base.production_bonus_per_tile * owned_linked as f32,
        cap: rules.garrison_cap_base + rules.garrison_cap_per_tile * owned_linked as f32,
        rate_label: "产兵",
    }
}

fn owner_label(factions: &Factions, owner: FactionId) -> String {
    if owner == NEUTRAL {
        return "中立".into();
    }
    factions
        .0
        .iter()
        .find(|faction| faction.id == owner)
        .map(|faction| {
            if faction.is_player {
                "玩家".into()
            } else {
                format!("敌方·{}", faction.name)
            }
        })
        .unwrap_or_else(|| format!("阵营 {owner}"))
}

pub fn update_status_hud(
    drag: Res<DragState>,
    difficulty: Res<DifficultyName>,
    hud: Res<DebugHud>,
    telemetry: Res<TelemetryRecorder>,
    streams: Query<&Stream>,
    squads: Query<&Squad>,
    mut q: Query<&mut Text2d, With<HudText>>,
) {
    let Ok(mut text) = q.single_mut() else {
        return; // 棋盘渲染实体尚未生成
    };
    let active = streams.iter().filter(|s| s.active).count();
    let telemetry_status = if telemetry.is_active() {
        " · 埋点开启"
    } else {
        ""
    };
    text.0 = format!(
        "难度[{}](1～9/0切换) · 兵流 {} 条 · 小队 {} · 选中 {} 个据点{} · {}",
        difficulty.0,
        active,
        squads.iter().count(),
        drag.selected.len(),
        telemetry_status,
        hud.last_event
    );
}

pub fn update_base_info(
    drag: Res<DragState>,
    lookup: Res<GridLookup>,
    rules: Res<Rules>,
    factions: Res<Factions>,
    bases: Query<(&Owner, &Garrison, &Base)>,
    owners: Query<&Owner>,
    mut base_info: Query<&mut Text2d, With<BaseInfoText>>,
) {
    let Ok(mut base_info) = base_info.single_mut() else {
        return;
    };
    let Some(cell) = drag.inspected else {
        base_info.0 = "点击任意据点查看产兵速度与驻军上限".into();
        return;
    };
    let Ok((owner, garrison, base)) = bases.get(lookup.entity(cell)) else {
        base_info.0 = "点击任意据点查看产兵速度与驻军上限".into();
        return;
    };
    let owned_linked = base
        .linked
        .iter()
        .filter(|&&linked| {
            owners
                .get(lookup.entity(linked))
                .is_ok_and(|linked_owner| linked_owner.0 == owner.0)
        })
        .count();
    let economy = base_economy(&rules, base, owner.0, owned_linked, garrison.max);
    let linked =
        (owner.0 != NEUTRAL).then(|| format!("｜关联地 {owned_linked}/{}", base.linked.len()));
    base_info.0 = format!(
        "{}｜{}｜兵力 {}｜{} {:.1}/秒｜上限 {}{}",
        base.subject_name,
        owner_label(&factions, owner.0),
        fmt_num(garrison.cur),
        economy.rate_label,
        economy.rate,
        fmt_num(economy.cap),
        linked.unwrap_or_default(),
    );
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

    fn base() -> Base {
        Base {
            subject_id: "biology".into(),
            subject_name: "生物".into(),
            production_base: 2.5,
            production_bonus_per_tile: 0.2,
            linked: vec![1, 2, 3, 4],
        }
    }

    #[test]
    fn owned_base_details_use_live_production_and_capacity() {
        assert_eq!(
            base_economy(&rules(), &base(), PLAYER, 3, 40.0),
            BaseEconomy {
                rate: 3.1,
                cap: 110.0,
                rate_label: "产兵",
            }
        );
    }

    #[test]
    fn neutral_base_details_show_recovery_and_initial_cap() {
        assert_eq!(
            base_economy(&rules(), &base(), NEUTRAL, 4, 40.0),
            BaseEconomy {
                rate: 2.5,
                cap: 40.0,
                rate_label: "恢复",
            }
        );
    }
}
