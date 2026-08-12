//! 棋盘渲染：格子精灵/文字的生成与刷新、小队圆点。
//! 渲染系统只读逻辑组件，永不写（ARCHITECTURE.md §4 铁律 2）。

use crate::common::*;
use bevy::prelude::*;
use easywar_logic::Label as LogicLabel;
use easywar_logic::*;

type CellQuery<'w, 's> = Query<
    'w,
    's,
    (
        &'static CellKind,
        &'static Owner,
        &'static Garrison,
        &'static LogicLabel,
    ),
>;

fn format_cell_text(label: Option<&str>, garrison: f32) -> String {
    let num = fmt_num(garrison);
    label.map_or(num.clone(), |label| format!("{label}\n{num}"))
}

fn cell_text(cells: &CellQuery, e: Entity) -> String {
    let (_, _, garrison, label_text) = cells.get(e).unwrap();
    format_cell_text(label_text.0.as_deref(), garrison.cur)
}

fn border_color(factions: &Factions, tint: &RegionTint, owner: FactionId, idx: CellIdx) -> Color {
    if let Some(color) = tint.0.get(&idx) {
        return Color::srgba(color[0], color[1], color[2], 1.0);
    }
    if owner == NEUTRAL {
        return Color::srgb(0.55, 0.55, 0.58);
    }
    let c = faction_color(factions, owner);
    Color::srgba(c[0], c[1], c[2], 1.0)
}

fn fill_color(factions: &Factions, cells: &CellQuery, e: Entity) -> Color {
    let (_, owner, _, _) = cells.get(e).unwrap();
    fill_color_for_owner(factions, owner.0)
}

fn fill_color_for_owner(factions: &Factions, owner: FactionId) -> Color {
    if owner != NEUTRAL {
        let c = faction_color(factions, owner);
        return Color::srgba(c[0], c[1], c[2], 1.0);
    }
    Color::srgb(0.82, 0.83, 0.85)
}

fn text_color(owner: FactionId) -> Color {
    if owner != NEUTRAL {
        Color::WHITE
    } else {
        Color::srgb(0.25, 0.25, 0.28)
    }
}

/// 资源就绪后生成一次棋盘渲染实体
pub fn spawn_board_system(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    lookup: Option<Res<GridLookup>>,
    factions: Option<Res<Factions>>,
    tint: Option<Res<RegionTint>>,
    cells: CellQuery,
    spawned: Option<Res<BoardSpawned>>,
) {
    if spawned.is_some() {
        return;
    }
    let (Some(lookup), Some(factions), Some(tint)) = (lookup, factions, tint) else {
        return;
    };
    commands.insert_resource(BoardSpawned);
    let font: Handle<Font> = asset_server.load("fonts/NotoSansSC-Regular.ttf");
    let bold: Handle<Font> = asset_server.load("fonts/NotoSansSC-Bold.ttf");
    let origin = grid_origin(&lookup);

    for (i, &e) in lookup.cells.iter().enumerate() {
        let Ok((kind, owner, _, _)) = cells.get(e) else {
            continue;
        };
        if !kind.enterable() {
            continue;
        }
        let pos = cell_pos(&lookup, origin, i);
        commands.spawn((
            Sprite {
                color: border_color(&factions, &tint, owner.0, i),
                custom_size: Some(Vec2::splat(CELL)),
                ..default()
            },
            Transform::from_xyz(pos.x, pos.y, 0.0),
            BoardEntity,
            CellBorder(e, i),
        ));
        commands.spawn((
            Sprite {
                color: fill_color(&factions, &cells, e),
                custom_size: Some(Vec2::splat(CELL - BORDER * 2.0)),
                ..default()
            },
            Transform::from_xyz(pos.x, pos.y, 0.5),
            BoardEntity,
            CellFill(e),
        ));
        let is_base = *kind == CellKind::Base;
        commands.spawn((
            Text2d::new(cell_text(&cells, e)),
            TextFont {
                font: FontSource::Handle(if is_base { bold.clone() } else { font.clone() }),
                font_size: (if is_base { 14.0 } else { 11.0 }).into(),
                ..default()
            },
            TextColor(text_color(owner.0)),
            Transform::from_xyz(pos.x, pos.y, 1.0),
            BoardEntity,
            CellLabel(e, String::new()),
        ));
    }

    commands.spawn((
        Text2d::new("点击据点后点击目标派兵 · 拖框/Shift 多选 · 再点单选据点停止 · 右键/Esc 取消"),
        TextFont {
            font: FontSource::Handle(font.clone()),
            font_size: 15.0.into(),
            ..default()
        },
        TextColor(background_muted_text_color()),
        Transform::from_xyz(0.0, 360.0, 1.0),
        BoardEntity,
    ));
    commands.spawn((
        Text2d::new(""),
        TextFont {
            font: FontSource::Handle(font.clone()),
            font_size: 12.0.into(),
            ..default()
        },
        TextColor(background_muted_text_color()),
        Transform::from_xyz(0.0, 335.0, 1.0),
        BoardEntity,
        HudText,
    ));
    commands.spawn((
        Sprite {
            color: Color::srgba(1.0, 1.0, 1.0, 0.82),
            custom_size: Some(Vec2::new(650.0, 42.0)),
            ..default()
        },
        Transform::from_xyz(0.0, -350.0, 0.5),
        BoardEntity,
    ));
    commands.spawn((
        Text2d::new("点击任意据点查看产兵速度与驻军上限"),
        TextFont {
            font: FontSource::Handle(font),
            font_size: 15.0.into(),
            ..default()
        },
        TextColor(background_text_color()),
        Transform::from_xyz(0.0, -350.0, 1.0),
        BoardEntity,
        BaseInfoText,
    ));
}

pub fn sync_cells(
    factions: Res<Factions>,
    tint: Res<RegionTint>,
    cells: CellQuery,
    mut borders: Query<(&CellBorder, &mut Sprite), Without<CellFill>>,
    mut fills: Query<(&CellFill, &mut Sprite), Without<CellBorder>>,
    mut labels: Query<(&mut CellLabel, &mut Text2d, &mut TextColor)>,
) {
    for (cb, mut sprite) in borders.iter_mut() {
        if let Ok((_, owner, _, _)) = cells.get(cb.0) {
            sprite.color = border_color(&factions, &tint, owner.0, cb.1);
        }
    }
    for (cf, mut sprite) in fills.iter_mut() {
        if cells.get(cf.0).is_ok() {
            sprite.color = fill_color(&factions, &cells, cf.0);
        }
    }
    for (mut label, mut text, mut color) in labels.iter_mut() {
        let s = cell_text(&cells, label.0);
        if s != label.1 {
            text.0 = s.clone();
            label.1 = s;
        }
        if let Ok((_, owner, _, _)) = cells.get(label.0) {
            *color = TextColor(text_color(owner.0));
        }
    }
}

pub fn sync_squads(
    mut commands: Commands,
    lookup: Res<GridLookup>,
    factions: Res<Factions>,
    squads: Query<&Squad>,
    dots: Query<Entity, With<SquadDot>>,
) {
    for e in dots.iter() {
        commands.entity(e).despawn();
    }
    let origin = grid_origin(&lookup);
    for sq in squads.iter() {
        let a = cell_pos(&lookup, origin, sq.path[sq.seg]);
        let b = if sq.seg + 1 < sq.path.len() {
            cell_pos(&lookup, origin, sq.path[sq.seg + 1])
        } else {
            a
        };
        let pos = a.lerp(b, sq.t);
        let c = faction_color(&factions, sq.faction);
        let color = Color::srgba(c[0], c[1], c[2], 1.0);

        // 一兵一圆点，最多 3 个，垂直于行军方向排成一行
        let n = (sq.troops.round() as i32).clamp(1, 3) as usize;
        let dir = (b - a).try_normalize().unwrap_or(Vec2::X);
        let perp = Vec2::new(-dir.y, dir.x);
        const DOT: f32 = 6.0;
        const GAP: f32 = 8.0;
        for k in 0..n {
            let offset = (k as f32 - (n as f32 - 1.0) / 2.0) * GAP;
            let p = pos + perp * offset;
            commands.spawn((
                Sprite {
                    color,
                    custom_size: Some(Vec2::splat(DOT)),
                    ..default()
                },
                Transform::from_xyz(p.x, p.y, 2.0),
                BoardEntity,
                SquadDot,
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn neutral_region_uses_its_subject_border_color() {
        let subject_color = [0.18, 0.64, 0.42, 1.0];
        let tint = RegionTint(HashMap::from([(7, subject_color)]));

        assert_eq!(
            border_color(&Factions::default(), &tint, NEUTRAL, 7),
            Color::srgba(0.18, 0.64, 0.42, 1.0)
        );
    }

    #[test]
    fn neutral_cell_without_region_keeps_the_fallback_border() {
        assert_eq!(
            border_color(&Factions::default(), &RegionTint::default(), NEUTRAL, 7),
            Color::srgb(0.55, 0.55, 0.58)
        );
    }

    #[test]
    fn occupied_region_keeps_its_subject_border_color() {
        let subject_color = [0.72, 0.26, 0.58, 1.0];
        let tint = RegionTint(HashMap::from([(7, subject_color)]));
        let factions = Factions(vec![Faction {
            id: PLAYER,
            name: "玩家".into(),
            color: [0.23, 0.51, 0.96, 1.0],
            is_player: true,
        }]);

        assert_eq!(
            border_color(&factions, &tint, PLAYER, 7),
            Color::srgba(0.72, 0.26, 0.58, 1.0)
        );
    }

    #[test]
    fn occupied_cell_uses_the_full_faction_color() {
        let color = [0.23, 0.51, 0.96, 1.0];
        let factions = Factions(vec![Faction {
            id: PLAYER,
            name: "玩家".into(),
            color,
            is_player: true,
        }]);

        assert_eq!(
            fill_color_for_owner(&factions, PLAYER),
            Color::srgba(color[0], color[1], color[2], 1.0)
        );
    }

    #[test]
    fn base_cell_text_only_contains_current_garrison() {
        assert_eq!(format_cell_text(Some("生物"), 23.8), "生物\n23");
    }
}
