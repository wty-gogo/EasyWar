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
        Option<&'static Base>,
    ),
>;

/// 据点驻军上限（渲染侧重算：规则 + 已占领关联地块数）
fn base_cap(
    rules: &Rules,
    cells: &CellQuery,
    lookup: &GridLookup,
    base: &Base,
    owner: FactionId,
) -> f32 {
    let owned = base
        .linked
        .iter()
        .filter(|&&t| {
            cells
                .get(lookup.entity(t))
                .map(|(_, o, _, _, _)| o.0 == owner)
                .unwrap_or(false)
        })
        .count();
    rules.garrison_cap_base + rules.garrison_cap_per_tile * owned as f32
}

fn cell_text(rules: &Rules, cells: &CellQuery, lookup: &GridLookup, e: Entity) -> String {
    let (kind, owner, garrison, label_text, base) = cells.get(e).unwrap();
    let num = if *kind == CellKind::Base {
        let cap = match (base, owner.0) {
            (Some(b), o) if o != NEUTRAL => base_cap(rules, cells, lookup, b, o),
            _ => garrison.max,
        };
        format!("{}/{}", fmt_num(garrison.cur), fmt_num(cap))
    } else {
        fmt_num(garrison.cur)
    };
    match &label_text.0 {
        Some(l) => format!("{}\n{}", l, num),
        None => num,
    }
}

fn border_color(factions: &Factions, owner: FactionId) -> Color {
    if owner == NEUTRAL {
        return Color::srgb(0.55, 0.55, 0.58);
    }
    let c = faction_color(factions, owner);
    Color::srgba(c[0], c[1], c[2], 1.0)
}

fn fill_color(
    factions: &Factions,
    tint: &LinkedTint,
    cells: &CellQuery,
    e: Entity,
    idx: CellIdx,
) -> Color {
    let (kind, owner, _, _, _) = cells.get(e).unwrap();
    if owner.0 != NEUTRAL {
        let c = faction_color(factions, owner.0);
        if *kind == CellKind::Base {
            return Color::srgba(c[0], c[1], c[2], 1.0);
        }
        return Color::srgba(
            c[0] * 0.25 + 0.93 * 0.75,
            c[1] * 0.25 + 0.93 * 0.75,
            c[2] * 0.25 + 0.93 * 0.75,
            1.0,
        );
    }
    if let Some(t) = tint.0.get(&idx) {
        return Color::srgba(
            t[0] * 0.10 + 0.90 * 0.90,
            t[1] * 0.10 + 0.90 * 0.90,
            t[2] * 0.10 + 0.90 * 0.90,
            1.0,
        );
    }
    Color::srgb(0.90, 0.90, 0.90)
}

fn text_color(kind: &CellKind, owner: FactionId) -> Color {
    if *kind == CellKind::Base && owner != NEUTRAL {
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
    rules: Option<Res<Rules>>,
    factions: Option<Res<Factions>>,
    tint: Option<Res<LinkedTint>>,
    cells: CellQuery,
    spawned: Option<Res<BoardSpawned>>,
) {
    if spawned.is_some() {
        return;
    }
    let (Some(lookup), Some(rules), Some(factions), Some(tint)) = (lookup, rules, factions, tint)
    else {
        return;
    };
    commands.insert_resource(BoardSpawned);
    let font: Handle<Font> = asset_server.load("fonts/NotoSansSC-Regular.ttf");
    let bold: Handle<Font> = asset_server.load("fonts/NotoSansSC-Bold.ttf");
    let origin = grid_origin(&lookup);

    for (i, &e) in lookup.cells.iter().enumerate() {
        let Ok((kind, owner, _, _, _)) = cells.get(e) else {
            continue;
        };
        if !kind.enterable() {
            continue;
        }
        let pos = cell_pos(&lookup, origin, i);
        commands.spawn((
            Sprite {
                color: border_color(&factions, owner.0),
                custom_size: Some(Vec2::splat(CELL)),
                ..default()
            },
            Transform::from_xyz(pos.x, pos.y, 0.0),
            BoardEntity,
            CellBorder(e),
        ));
        commands.spawn((
            Sprite {
                color: fill_color(&factions, &tint, &cells, e, i),
                custom_size: Some(Vec2::splat(CELL - BORDER * 2.0)),
                ..default()
            },
            Transform::from_xyz(pos.x, pos.y, 0.5),
            BoardEntity,
            CellFill(e),
        ));
        let is_base = *kind == CellKind::Base;
        commands.spawn((
            Text2d::new(cell_text(&rules, &cells, &lookup, e)),
            TextFont {
                font: FontSource::Handle(if is_base { bold.clone() } else { font.clone() }),
                font_size: (if is_base { 14.0 } else { 11.0 }).into(),
                ..default()
            },
            TextColor(text_color(kind, owner.0)),
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
        TextColor(Color::srgb(0.5, 0.5, 0.5)),
        Transform::from_xyz(0.0, 360.0, 1.0),
        BoardEntity,
    ));
    commands.spawn((
        Text2d::new(""),
        TextFont {
            font: FontSource::Handle(font),
            font_size: 12.0.into(),
            ..default()
        },
        TextColor(Color::srgb(0.6, 0.6, 0.6)),
        Transform::from_xyz(0.0, 335.0, 1.0),
        BoardEntity,
        HudText,
    ));
}

pub fn sync_cells(
    factions: Res<Factions>,
    tint: Res<LinkedTint>,
    rules: Res<Rules>,
    lookup: Res<GridLookup>,
    cells: CellQuery,
    mut borders: Query<(&CellBorder, &mut Sprite), Without<CellFill>>,
    mut fills: Query<(&CellFill, &mut Sprite), Without<CellBorder>>,
    mut labels: Query<(&mut CellLabel, &mut Text2d, &mut TextColor)>,
) {
    for (cb, mut sprite) in borders.iter_mut() {
        if let Ok((_, owner, _, _, _)) = cells.get(cb.0) {
            sprite.color = border_color(&factions, owner.0);
        }
    }
    for (cf, mut sprite) in fills.iter_mut() {
        if let Ok((_, _, _, _, _)) = cells.get(cf.0) {
            let idx = lookup.cells.iter().position(|&e| e == cf.0).unwrap_or(0);
            sprite.color = fill_color(&factions, &tint, &cells, cf.0, idx);
        }
    }
    for (mut label, mut text, mut color) in labels.iter_mut() {
        let s = cell_text(&rules, &cells, &lookup, label.0);
        if s != label.1 {
            text.0 = s.clone();
            label.1 = s;
        }
        if let Ok((kind, owner, _, _, _)) = cells.get(label.0) {
            *color = TextColor(text_color(kind, owner.0));
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
