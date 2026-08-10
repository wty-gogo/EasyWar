//! 玩家输入：桌面点击/框选与触屏拖拽分别转成 Intent；难度切换。

use crate::common::*;
use crate::neural_ai::{configured_controllers, NeuralModelResource};
use bevy::prelude::*;
use easywar_logic::*;
use std::collections::HashSet;

const BOX_THRESHOLD: f32 = 12.0;

#[derive(Debug)]
enum DesktopClickPlan {
    Select(CellIdx),
    Toggle(CellIdx),
    Send(Vec<Intent>),
    Stop(CellIdx),
    Clear,
    NeedSource,
}

pub fn desktop_input_mode(mode: Res<InputMode>) -> bool {
    *mode == InputMode::Desktop
}

pub fn touch_input_mode(mode: Res<InputMode>) -> bool {
    *mode == InputMode::Touch
}

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
    let cell = lookup.idx(x as usize, y as usize);
    cells
        .get(lookup.entity(cell))
        .ok()?
        .enterable()
        .then_some(cell)
}

fn set_stream_intents(selected: &HashSet<CellIdx>, target: CellIdx) -> Vec<Intent> {
    let mut sources: Vec<CellIdx> = selected
        .iter()
        .copied()
        .filter(|&source| source != target)
        .collect();
    sources.sort_unstable();
    sources
        .into_iter()
        .map(|source| Intent::SetStream {
            faction: PLAYER,
            source,
            target,
        })
        .collect()
}

fn desktop_click_plan(
    selected: &HashSet<CellIdx>,
    target: CellIdx,
    own_base: bool,
    shift: bool,
    active_stream: bool,
) -> DesktopClickPlan {
    if shift && own_base {
        return DesktopClickPlan::Toggle(target);
    }
    if selected.is_empty() {
        return if own_base {
            DesktopClickPlan::Select(target)
        } else {
            DesktopClickPlan::NeedSource
        };
    }
    if selected.len() == 1 && selected.contains(&target) {
        return if active_stream {
            DesktopClickPlan::Stop(target)
        } else {
            DesktopClickPlan::Clear
        };
    }
    DesktopClickPlan::Send(set_stream_intents(selected, target))
}

fn push_stream_intents(
    intents: &mut IntentQueue,
    selected: &HashSet<CellIdx>,
    target: CellIdx,
) -> usize {
    let commands = set_stream_intents(selected, target);
    let count = commands.len();
    for command in commands {
        intents.push(command);
    }
    count
}

fn toggle_selection(selected: &mut HashSet<CellIdx>, cell: CellIdx) {
    if !selected.insert(cell) {
        selected.remove(&cell);
    }
}

fn active_stream_from(streams: &Query<&Stream>, source: CellIdx) -> bool {
    streams
        .iter()
        .any(|stream| stream.active && stream.faction == PLAYER && stream.source == source)
}

fn select_bases_in_rect(
    selected: &mut HashSet<CellIdx>,
    from: Vec2,
    to: Vec2,
    append: bool,
    lookup: &GridLookup,
    kinds: &Query<&CellKind>,
    owners: &Query<&Owner>,
) {
    if !append {
        selected.clear();
    }
    let min = from.min(to);
    let max = from.max(to);
    let origin = grid_origin(lookup);
    for (cell, &entity) in lookup.cells.iter().enumerate() {
        let (Ok(kind), Ok(owner)) = (kinds.get(entity), owners.get(entity)) else {
            continue;
        };
        let position = cell_pos(lookup, origin, cell);
        if *kind == CellKind::Base
            && owner.0 == PLAYER
            && position.x >= min.x
            && position.x <= max.x
            && position.y >= min.y
            && position.y <= max.y
        {
            selected.insert(cell);
        }
    }
}

fn draw_selection_box(from: Option<Vec2>, cursor: Option<Vec2>, gizmos: &mut Gizmos) {
    let (Some(from), Some(cursor)) = (from, cursor) else {
        return;
    };
    if (cursor - from).length() > BOX_THRESHOLD {
        gizmos.rect_2d(
            Isometry2d::from_translation((from + cursor) / 2.0),
            (cursor - from).abs(),
            Color::srgba(1.0, 0.85, 0.2, 0.6),
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn handle_desktop_input(
    mut intents: ResMut<IntentQueue>,
    mut pointer: ResMut<DragState>,
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
    let (Ok(window), Ok((camera, camera_transform))) = (windows.single(), camera.single()) else {
        return;
    };
    let cursor = window
        .cursor_position()
        .and_then(|position| camera.viewport_to_world_2d(camera_transform, position).ok());
    let shift = keyboard.pressed(KeyCode::ShiftLeft) || keyboard.pressed(KeyCode::ShiftRight);

    if keyboard.just_pressed(KeyCode::Escape) || buttons.just_pressed(MouseButton::Right) {
        pointer.selected.clear();
        pointer.press_pos = None;
        hud.last_event = "已取消选择".into();
    }

    if buttons.just_pressed(MouseButton::Left) {
        pointer.press_pos = cursor;
    }
    draw_selection_box(pointer.press_pos, cursor, &mut gizmos);

    if !buttons.just_released(MouseButton::Left) {
        return;
    }
    let Some(press) = pointer.press_pos.take() else {
        return;
    };
    let Some(release) = cursor else {
        pointer.selected.clear();
        hud.last_event = "释放在窗口外，已取消选择".into();
        return;
    };

    if (release - press).length() > BOX_THRESHOLD {
        select_bases_in_rect(
            &mut pointer.selected,
            press,
            release,
            shift,
            &lookup,
            &kinds,
            &owners,
        );
        hud.last_event = format!("框选 {} 个据点", pointer.selected.len());
        return;
    }

    let Some(target) = world_to_cell(release, &lookup, &kinds) else {
        if !shift {
            pointer.selected.clear();
        }
        hud.last_event = "点击地图外，已取消选择".into();
        return;
    };
    let entity = lookup.entity(target);
    let kind = *kinds.get(entity).expect("可进入格缺少 CellKind");
    let owner = owners.get(entity).expect("可进入格缺少 Owner").0;
    let own_base = kind == CellKind::Base && owner == PLAYER;

    let plan = desktop_click_plan(
        &pointer.selected,
        target,
        own_base,
        shift,
        active_stream_from(&streams, target),
    );
    match plan {
        DesktopClickPlan::Select(source) => {
            pointer.selected.clear();
            pointer.selected.insert(source);
            hud.last_event = format!("已选中 {:?}，再点击目标地块派兵", lookup.xy(source));
        }
        DesktopClickPlan::Toggle(source) => {
            toggle_selection(&mut pointer.selected, source);
            hud.last_event = format!("已选择 {} 个据点", pointer.selected.len());
        }
        DesktopClickPlan::Send(commands) => {
            let count = commands.len();
            for command in commands {
                intents.push(command);
            }
            pointer.selected.clear();
            hud.last_event = format!("{count} 个据点出兵 → {:?}", lookup.xy(target));
        }
        DesktopClickPlan::Stop(source) => {
            intents.push(Intent::StopStream {
                faction: PLAYER,
                source,
            });
            pointer.selected.clear();
            hud.last_event = format!("停止 {:?} 的兵流", lookup.xy(source));
        }
        DesktopClickPlan::Clear => {
            pointer.selected.clear();
            hud.last_event = "已取消选择".into();
        }
        DesktopClickPlan::NeedSource => {
            hud.last_event = "请先选择己方据点".into();
        }
    }
}

/// 未来触屏端使用的原始拖拽交互。桌面端默认不运行；设置 `EASYWAR_INPUT=touch` 可测试。
#[allow(clippy::too_many_arguments)]
pub fn handle_touch_input(
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
    let (Ok(window), Ok((camera, camera_transform))) = (windows.single(), camera.single()) else {
        return;
    };
    let cursor = window
        .cursor_position()
        .and_then(|position| camera.viewport_to_world_2d(camera_transform, position).ok());
    let shift = keyboard.pressed(KeyCode::ShiftLeft) || keyboard.pressed(KeyCode::ShiftRight);

    if keyboard.just_pressed(KeyCode::Escape) {
        drag.selected.clear();
        hud.last_event = "取消选择".into();
    }

    if buttons.just_pressed(MouseButton::Left) {
        match cursor.and_then(|world| world_to_cell(world, &lookup, &kinds)) {
            Some(cell) => {
                let kind = *kinds.get(lookup.entity(cell)).expect("格子缺少 CellKind");
                let owner = owners.get(lookup.entity(cell)).expect("格子缺少 Owner").0;
                if kind == CellKind::Base && owner == PLAYER {
                    if shift {
                        toggle_selection(&mut drag.selected, cell);
                        hud.last_event = format!("多选：{} 个据点", drag.selected.len());
                    } else {
                        drag.dragging = Some(cell);
                        if !drag.selected.contains(&cell) {
                            drag.selected.clear();
                            drag.selected.insert(cell);
                        }
                    }
                } else {
                    drag.press_pos = cursor;
                }
            }
            None => drag.press_pos = cursor,
        }
    }

    if let (Some(source), Some(world)) = (drag.dragging, cursor) {
        let origin = grid_origin(&lookup);
        gizmos.line_2d(
            cell_pos(&lookup, origin, source),
            world,
            Color::srgb(1.0, 0.85, 0.2),
        );
    }
    draw_selection_box(drag.press_pos, cursor, &mut gizmos);

    if !buttons.just_released(MouseButton::Left) {
        return;
    }
    if let Some(source) = drag.dragging.take() {
        match cursor.and_then(|world| world_to_cell(world, &lookup, &kinds)) {
            Some(target) if target != source => {
                let count = push_stream_intents(&mut intents, &drag.selected, target);
                hud.last_event = format!("{count} 个据点出兵 → {:?}", lookup.xy(target));
            }
            Some(_) if active_stream_from(&streams, source) => {
                intents.push(Intent::StopStream {
                    faction: PLAYER,
                    source,
                });
                hud.last_event = format!("停止 {:?} 的兵流", lookup.xy(source));
            }
            Some(_) => {
                hud.last_event = format!("已选中 {:?}，拖到目标格派兵", lookup.xy(source));
            }
            None => hud.last_event = "释放在地图外，取消".into(),
        }
        return;
    }

    let Some(press) = drag.press_pos.take() else {
        return;
    };
    let Some(release) = cursor else {
        return;
    };
    if (release - press).length() > BOX_THRESHOLD {
        select_bases_in_rect(
            &mut drag.selected,
            press,
            release,
            shift,
            &lookup,
            &kinds,
            &owners,
        );
        hud.last_event = format!("框选 {} 个据点", drag.selected.len());
    } else if let Some(target) = world_to_cell(release, &lookup, &kinds) {
        let count = push_stream_intents(&mut intents, &drag.selected, target);
        hud.last_event = format!("{count} 个据点出兵 → {:?}", lookup.xy(target));
    } else {
        drag.selected.clear();
    }
}

/// 1/2/3/4 实时切换 AI：规则参数与神经模型都只提交玩家级合法意图。
pub fn switch_difficulty(
    keyboard: Res<ButtonInput<KeyCode>>,
    factions: Res<Factions>,
    current_map: Res<CurrentMapFile>,
    model: Res<NeuralModelResource>,
    mut selection: ResMut<MenuSelection>,
    mut hud: ResMut<DebugHud>,
    mut commands: Commands,
) {
    let idx = if keyboard.just_pressed(KeyCode::Digit1) {
        0usize
    } else if keyboard.just_pressed(KeyCode::Digit2) {
        1
    } else if keyboard.just_pressed(KeyCode::Digit3) {
        2
    } else if keyboard.just_pressed(KeyCode::Digit4) {
        3
    } else {
        return;
    };
    let (controllers, policy_controllers, name) =
        configured_controllers(idx, &current_map.0, &factions, &model);
    selection.difficulty = idx;
    commands.insert_resource(controllers);
    commands.insert_resource(policy_controllers);
    commands.insert_resource(DifficultyName(name));
    hud.last_event = format!("AI 难度切换为：{name}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiple_sources_are_sorted_and_target_source_is_skipped() {
        let selected = HashSet::from([9, 2, 5]);
        let commands = set_stream_intents(&selected, 5);
        let pairs: Vec<(CellIdx, CellIdx)> = commands
            .into_iter()
            .map(|intent| match intent {
                Intent::SetStream { source, target, .. } => (source, target),
                Intent::StopStream { .. } => panic!("不应生成停止命令"),
            })
            .collect();
        assert_eq!(pairs, vec![(2, 5), (9, 5)]);
    }

    #[test]
    fn toggling_selected_base_is_reversible() {
        let mut selected = HashSet::new();
        toggle_selection(&mut selected, 7);
        assert!(selected.contains(&7));
        toggle_selection(&mut selected, 7);
        assert!(selected.is_empty());
    }

    #[test]
    fn desktop_click_flow_selects_then_sends() {
        let empty = HashSet::new();
        assert!(matches!(
            desktop_click_plan(&empty, 3, true, false, false),
            DesktopClickPlan::Select(3)
        ));

        let selected = HashSet::from([3]);
        let DesktopClickPlan::Send(commands) =
            desktop_click_plan(&selected, 8, false, false, false)
        else {
            panic!("第二次点击目标应生成派兵命令");
        };
        assert!(matches!(
            commands.as_slice(),
            [Intent::SetStream {
                source: 3,
                target: 8,
                ..
            }]
        ));
    }

    #[test]
    fn clicking_only_selected_active_base_stops_stream() {
        let selected = HashSet::from([4]);
        assert!(matches!(
            desktop_click_plan(&selected, 4, true, false, true),
            DesktopClickPlan::Stop(4)
        ));
    }
}
