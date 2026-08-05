//! 玩家输入：拖拽/框选/点击 → 发 Intent（唯一的玩家写入口）、难度切换。

use crate::common::*;
use bevy::prelude::*;
use easywar_logic::*;

fn world_to_cell(world: Vec2, lookup: &GridLookup, cells: &Query<&CellKind>) -> Option<CellIdx> {
    let origin = grid_origin(lookup);
    let fx = (world.x - origin.x) / STEP;
    let fy = (origin.y - world.y) / STEP;
    let (x, y) = (fx.round() as i64, fy.round() as i64);
    if x < 0 || y < 0 || x as usize >= lookup.width || y as usize >= lookup.height {
        return None;
    }
    if (fx - x as f32).abs() > 0.5 || (fy - y as f32).abs() > 0.5 {
        return None;
    }
    let i = lookup.idx(x as usize, y as usize);
    let kind = cells.get(lookup.entity(i)).ok()?;
    kind.enterable().then_some(i)
}

#[allow(clippy::too_many_arguments)]
pub fn handle_input(
    mut intents: ResMut<IntentQueue>,
    mut drag: ResMut<DragState>,
    mut hud: ResMut<DebugHud>,
    lookup: Res<GridLookup>,
    kinds: Query<&CellKind>,
    owners: Query<&Owner>,
    streams: Query<&Stream>,
    buttons: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window>,
    camera: Query<(&Camera, &GlobalTransform)>,
    mut gizmos: Gizmos,
) {
    let window = windows.single();
    let (camera, cam_tf) = camera.single();
    let cursor_world = window
        .cursor_position()
        .and_then(|p| camera.viewport_to_world_2d(cam_tf, p).ok());
    let shift = keyboard.pressed(KeyCode::ShiftLeft) || keyboard.pressed(KeyCode::ShiftRight);

    if keyboard.just_pressed(KeyCode::Escape) {
        drag.selected.clear();
        hud.last_event = "取消选择".into();
    }

    if buttons.just_pressed(MouseButton::Left) {
        match cursor_world.and_then(|w| world_to_cell(w, &lookup, &kinds)) {
            Some(i) => {
                let kind = *kinds.get(lookup.entity(i)).unwrap();
                let owner = owners.get(lookup.entity(i)).unwrap().0;
                hud.last_event = format!("按下命中 {:?} kind={:?} owner={}", lookup.xy(i), kind, owner);
                if kind == CellKind::Base && owner == PLAYER {
                    if shift {
                        if !drag.selected.insert(i) {
                            drag.selected.remove(&i);
                        }
                        hud.last_event = format!("多选：{} 个据点", drag.selected.len());
                    } else {
                        drag.dragging = Some(i);
                        if !drag.selected.contains(&i) {
                            drag.selected.clear();
                            drag.selected.insert(i);
                        }
                    }
                } else {
                    drag.press_pos = cursor_world;
                }
            }
            None => {
                drag.press_pos = cursor_world;
            }
        }
    }

    if let (Some(src), Some(w)) = (drag.dragging, cursor_world) {
        let origin = grid_origin(&lookup);
        let a = cell_pos(&lookup, origin, src);
        gizmos.line_2d(a, w, Color::srgb(1.0, 0.85, 0.2));
    }
    if let (Some(p0), Some(w)) = (drag.press_pos, cursor_world) {
        if (w - p0).length() > 12.0 {
            gizmos.rect_2d(
                Isometry2d::from_translation((p0 + w) / 2.0),
                (w - p0).abs(),
                Color::srgba(1.0, 0.85, 0.2, 0.6),
            );
        }
    }

    if buttons.just_released(MouseButton::Left) {
        if let Some(src) = drag.dragging.take() {
            match cursor_world.and_then(|w| world_to_cell(w, &lookup, &kinds)) {
                Some(target) if target != src => {
                    let targets: Vec<CellIdx> = drag.selected.iter().copied().collect();
                    let ok_count = targets.len();
                    for b in targets {
                        intents.push(Intent::SetStream { faction: PLAYER, source: b, target });
                    }
                    hud.last_event = format!("{} 个据点出兵 → {:?}", ok_count, lookup.xy(target));
                }
                Some(_) => {
                    // 点击出兵中的己方据点（或拖到自己）= 停止出兵
                    let has_stream = streams
                        .iter()
                        .any(|s| s.active && s.faction == PLAYER && s.source == src);
                    if has_stream {
                        intents.push(Intent::StopStream { faction: PLAYER, source: src });
                        hud.last_event = format!("停止 {:?} 的兵流", lookup.xy(src));
                    } else {
                        hud.last_event = format!("已选中 {:?}，再点目标格派兵", lookup.xy(src));
                    }
                }
                None => {
                    hud.last_event = "释放在地图外，取消".into();
                }
            }
        } else if let Some(p0) = drag.press_pos.take() {
            if let Some(w) = cursor_world {
                if (w - p0).length() > 12.0 {
                    let min = p0.min(w);
                    let max = p0.max(w);
                    let origin = grid_origin(&lookup);
                    if !shift {
                        drag.selected.clear();
                    }
                    for (i, &e) in lookup.cells.iter().enumerate() {
                        let (Ok(kind), Ok(owner)) = (kinds.get(e), owners.get(e)) else { continue };
                        if *kind != CellKind::Base || owner.0 != PLAYER {
                            continue;
                        }
                        let p = cell_pos(&lookup, origin, i);
                        if p.x >= min.x && p.x <= max.x && p.y >= min.y && p.y <= max.y {
                            drag.selected.insert(i);
                        }
                    }
                    hud.last_event = format!("框选 {} 个据点", drag.selected.len());
                } else if let Some(target) = world_to_cell(w, &lookup, &kinds) {
                    let bases: Vec<CellIdx> = drag.selected.iter().copied().collect();
                    let ok_count = bases.len();
                    for b in bases {
                        intents.push(Intent::SetStream { faction: PLAYER, source: b, target });
                    }
                    hud.last_event = format!("{} 个据点出兵 → {:?}", ok_count, lookup.xy(target));
                } else {
                    drag.selected.clear();
                }
            }
        }
    }
}

/// 1/2/3 实时切换 AI 难度：重建全部 AI 控制器（行为参数全部来自行为，不作弊）
pub fn switch_difficulty(
    keyboard: Res<ButtonInput<KeyCode>>,
    factions: Res<Factions>,
    mut hud: ResMut<DebugHud>,
    mut commands: Commands,
) {
    let (idx, name) = if keyboard.just_pressed(KeyCode::Digit1) {
        (0usize, "简单")
    } else if keyboard.just_pressed(KeyCode::Digit2) {
        (1, "中等")
    } else if keyboard.just_pressed(KeyCode::Digit3) {
        (2, "困难")
    } else {
        return;
    };
    let params = DIFFICULTIES[idx].1();
    let controllers = factions
        .0
        .iter()
        .filter(|f| !f.is_player)
        .map(|f| AiController::new(f.id, params))
        .collect();
    commands.insert_resource(AiControllers(controllers));
    commands.insert_resource(DifficultyName(name));
    hud.last_event = format!("AI 难度切换为：{name}");
}
