//! 兵流连线与据点选中框等 gizmos 覆盖层。

use crate::common::*;
use bevy::prelude::*;
use easywar_logic::*;

const SELECTED_MARKER_HALF: f32 = CELL / 2.0 + 3.0;
const SELECTED_MARKER_LENGTH: f32 = 9.0;
const SELECTED_MARKER_WIDTH: f32 = 4.0;

#[derive(Default, Reflect, GizmoConfigGroup)]
pub(crate) struct SelectedMarkerGizmos;

pub fn configure_gizmos(mut configs: ResMut<GizmoConfigStore>) {
    let (selected_marker, _) = configs.config_mut::<SelectedMarkerGizmos>();
    selected_marker.line.width = SELECTED_MARKER_WIDTH;
}

fn draw_selected_base(gizmos: &mut Gizmos<SelectedMarkerGizmos>, position: Vec2) {
    let color = Color::srgb(1.0, 0.42, 0.05);
    for (x, y) in [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)] {
        let corner = position + Vec2::new(x, y) * SELECTED_MARKER_HALF;
        gizmos.line_2d(
            corner,
            corner - Vec2::new(x * SELECTED_MARKER_LENGTH, 0.0),
            color,
        );
        gizmos.line_2d(
            corner,
            corner - Vec2::new(0.0, y * SELECTED_MARKER_LENGTH),
            color,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_brackets_stay_close_to_the_cell() {
        assert!(SELECTED_MARKER_HALF < CELL / 2.0 + STEP - CELL);
        assert!(SELECTED_MARKER_WIDTH <= BORDER);
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
    // 出兵源和正在查看详情的据点使用紧贴格子的四角标记。
    for &src in &drag.selected {
        let p = cell_pos(&lookup, origin, src);
        draw_selected_base(&mut selected_markers, p);
    }
    if let Some(inspected) = drag
        .inspected
        .filter(|inspected| !drag.selected.contains(inspected))
    {
        let p = cell_pos(&lookup, origin, inspected);
        draw_selected_base(&mut selected_markers, p);
    }
}
