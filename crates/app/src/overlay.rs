//! 兵流连线与据点选中框等 gizmos 覆盖层。

use crate::common::*;
use bevy::prelude::*;
use easywar_logic::*;

const SELECTED_MARKER_SIZE: f32 = CELL + 16.0;
const SELECTED_MARKER_WIDTH: f32 = 8.0;

#[derive(Default, Reflect, GizmoConfigGroup)]
pub(crate) struct SelectedMarkerGizmos;

pub fn configure_gizmos(mut configs: ResMut<GizmoConfigStore>) {
    let (selected_marker, _) = configs.config_mut::<SelectedMarkerGizmos>();
    selected_marker.line.width = SELECTED_MARKER_WIDTH;
}

fn draw_selected_base(gizmos: &mut Gizmos<SelectedMarkerGizmos>, position: Vec2) {
    gizmos.rect_2d(
        Isometry2d::from_translation(position),
        Vec2::splat(SELECTED_MARKER_SIZE),
        Color::srgb(1.0, 0.48, 0.08),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_border_is_wider_than_the_cell_border() {
        assert!(SELECTED_MARKER_SIZE > CELL);
        assert!(SELECTED_MARKER_WIDTH > BORDER);
    }
}

pub fn draw_overlays(
    lookup: Res<GridLookup>,
    factions: Res<Factions>,
    streams: Query<&Stream>,
    drag: Res<DragState>,
    mut gizmos: Gizmos,
    mut selected_markers: Gizmos<SelectedMarkerGizmos>,
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
    // 选中据点只使用一圈高对比宽边框。
    for &src in &drag.selected {
        let p = cell_pos(&lookup, origin, src);
        draw_selected_base(&mut selected_markers, p);
    }
}
