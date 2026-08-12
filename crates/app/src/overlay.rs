//! 兵流连线与据点选中框等 gizmos 覆盖层。

use crate::common::*;
use bevy::prelude::*;
use easywar_logic::*;

fn selected_base_frames() -> [(f32, Color); 3] {
    [
        (CELL + 18.0, Color::srgba(0.03, 0.08, 0.10, 0.95)),
        (CELL + 14.0, Color::srgb(0.15, 0.95, 1.0)),
        (CELL + 10.0, Color::srgb(0.65, 1.0, 1.0)),
    ]
}

fn draw_selected_base(gizmos: &mut Gizmos, position: Vec2) {
    // 深色轮廓托住高亮，青色双框保证在任意阵营色和常驻金框上都清晰可见。
    for (size, color) in selected_base_frames() {
        gizmos.rect_2d(
            Isometry2d::from_translation(position),
            Vec2::splat(size),
            color,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_base_frames_stay_outside_the_player_frame() {
        assert!(selected_base_frames()
            .iter()
            .all(|(size, _)| *size > CELL + 6.0));
    }
}

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
    // 选中据点高对比双框
    for &src in &drag.selected {
        let p = cell_pos(&lookup, origin, src);
        draw_selected_base(&mut gizmos, p);
    }
}
