//! 兵流连线与据点选中框等 gizmos 覆盖层。

use crate::common::*;
use bevy::prelude::*;
use easywar_logic::*;

pub fn draw_overlays(
    lookup: Res<GridLookup>,
    factions: Res<Factions>,
    streams: Query<&Stream>,
    bases: Query<(&Owner, &Base)>,
    drag: Res<DragState>,
    mut gizmos: Gizmos,
) {
    let origin = grid_origin(&lookup);
    // 活跃兵流的路径连线
    for s in streams.iter() {
        if !s.active {
            continue;
        }
        let c = faction_color(&factions, s.faction);
        let color = Color::srgba(c[0], c[1], c[2], 0.55);
        for w in s.path.windows(2) {
            let a = cell_pos(&lookup, origin, w[0]);
            let b = cell_pos(&lookup, origin, w[1]);
            gizmos.line_2d(a, b, color);
        }
    }
    // 玩家据点金框
    for (i, &e) in lookup.cells.iter().enumerate() {
        if let Ok((owner, _)) = bases.get(e) {
            if owner.0 == PLAYER {
                let p = cell_pos(&lookup, origin, i);
                gizmos.rect_2d(
                    Isometry2d::from_translation(p),
                    Vec2::splat(CELL + 6.0),
                    Color::srgba(1.0, 0.9, 0.2, 0.8),
                );
            }
        }
    }
    // 选中据点橙框
    for &src in &drag.selected {
        let p = cell_pos(&lookup, origin, src);
        gizmos.rect_2d(
            Isometry2d::from_translation(p),
            Vec2::splat(CELL + 12.0),
            Color::srgb(1.0, 0.6, 0.1),
        );
    }
}
