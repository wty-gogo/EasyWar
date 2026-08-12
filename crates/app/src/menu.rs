//! 开局界面：选学科、地图和难度。

use crate::common::*;
use bevy::prelude::*;
use easywar_logic::parse_hex_color;

const DIFFICULTY_CENTER: Vec2 = Vec2::new(0.0, -65.0);
const DIFFICULTY_HALF: Vec2 = Vec2::new(170.0, 22.0);
const DROPDOWN_OPTION_HALF: Vec2 = Vec2::new(170.0, 15.0);
const DROPDOWN_OPTION_STEP: f32 = 32.0;

#[derive(Component)]
struct DifficultyDropdownTrigger;

#[derive(Component)]
pub(crate) struct DifficultyDropdownLabel;

#[derive(Component)]
pub(crate) struct DifficultyDropdownEntity;

#[derive(Component)]
pub(crate) struct DifficultyDropdownOption {
    index: usize,
    center: Vec2,
    half: Vec2,
}

fn contains_point(center: Vec2, half: Vec2, point: Vec2) -> bool {
    let distance = point - center;
    distance.x.abs() <= half.x && distance.y.abs() <= half.y
}

fn dropdown_option_center(index: usize) -> Vec2 {
    let top =
        DIFFICULTY_CENTER.y + DIFFICULTY_HALF.y + DIFFICULTIES.len() as f32 * DROPDOWN_OPTION_STEP;
    Vec2::new(
        DIFFICULTY_CENTER.x,
        top - (index as f32 + 0.5) * DROPDOWN_OPTION_STEP,
    )
}

fn close_difficulty_dropdown(
    commands: &mut Commands,
    entities: &Query<Entity, With<DifficultyDropdownEntity>>,
) {
    entities.iter().for_each(|entity| {
        commands.entity(entity).despawn();
    });
}

fn open_difficulty_dropdown(commands: &mut Commands, font: &Handle<Font>, selected: usize) {
    let panel_height = DIFFICULTIES.len() as f32 * DROPDOWN_OPTION_STEP + 8.0;
    let panel_bottom = DIFFICULTY_CENTER.y + DIFFICULTY_HALF.y;
    commands.spawn((
        Sprite {
            color: Color::srgba(0.08, 0.10, 0.14, 0.98),
            custom_size: Some(Vec2::new(DIFFICULTY_HALF.x * 2.0 + 12.0, panel_height)),
            ..default()
        },
        Transform::from_xyz(DIFFICULTY_CENTER.x, panel_bottom + panel_height / 2.0, 20.0),
        MenuEntity,
        DifficultyDropdownEntity,
    ));
    DIFFICULTIES
        .iter()
        .enumerate()
        .for_each(|(index, difficulty)| {
            let center = dropdown_option_center(index);
            let color = if index == selected {
                Color::srgb(0.48, 0.42, 0.16)
            } else {
                Color::srgb(0.20, 0.23, 0.30)
            };
            commands.spawn((
                Sprite {
                    color,
                    custom_size: Some(DROPDOWN_OPTION_HALF * 2.0),
                    ..default()
                },
                Transform::from_xyz(center.x, center.y, 21.0),
                MenuEntity,
                DifficultyDropdownEntity,
                DifficultyDropdownOption {
                    index,
                    center,
                    half: DROPDOWN_OPTION_HALF,
                },
            ));
            commands.spawn((
                Text2d::new(difficulty.name),
                TextFont {
                    font: FontSource::Handle(font.clone()),
                    font_size: 17.0.into(),
                    ..default()
                },
                TextColor(Color::WHITE),
                Transform::from_xyz(center.x, center.y, 22.0),
                MenuEntity,
                DifficultyDropdownEntity,
            ));
        });
}

pub fn enter_menu(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    subjects: Res<SubjectList>,
) {
    let font: Handle<Font> = asset_server.load("fonts/NotoSansSC-Regular.ttf");
    let bold: Handle<Font> = asset_server.load("fonts/NotoSansSC-Bold.ttf");

    let spawn_text =
        |commands: &mut Commands, s: &str, font: &Handle<Font>, size: f32, y: f32, color: Color| {
            commands.spawn((
                Text2d::new(s.to_string()),
                TextFont {
                    font: FontSource::Handle(font.clone()),
                    font_size: size.into(),
                    ..default()
                },
                TextColor(color),
                Transform::from_xyz(0.0, y, 1.0),
                MenuEntity,
            ));
        };

    spawn_text(
        &mut commands,
        "EasyWar · 学科对抗",
        &bold,
        48.0,
        320.0,
        background_text_color(),
    );
    spawn_text(
        &mut commands,
        "选择你的学科",
        &font,
        20.0,
        258.0,
        background_muted_text_color(),
    );

    // 学科按钮（20 门：5 列 × 4 行网格，按行居中）
    const COLS: usize = 5;
    let total = subjects.0.len();
    for (i, s) in subjects.0.iter().enumerate() {
        let (row, col) = (i / COLS, i % COLS);
        let row_len = COLS.min(total - row * COLS);
        let x = (col as f32 - (row_len - 1) as f32 / 2.0) * 125.0;
        let center = Vec2::new(x, 195.0 - row as f32 * 60.0);
        let half = Vec2::new(55.0, 26.0);
        let c = parse_hex_color(&s.color);
        commands.spawn((
            Sprite {
                color: Color::srgba(c[0], c[1], c[2], 1.0),
                custom_size: Some(half * 2.0),
                ..default()
            },
            Transform::from_xyz(center.x, center.y, 0.0),
            MenuEntity,
            MenuButton {
                action: MenuAction::Subject(i),
                center,
                half,
            },
        ));
        commands.spawn((
            Text2d::new(s.name.clone()),
            TextFont {
                font: FontSource::Handle(bold.clone()),
                font_size: 20.0.into(),
                ..default()
            },
            TextColor(Color::WHITE),
            Transform::from_xyz(center.x, center.y, 1.0),
            MenuEntity,
        ));
    }

    spawn_text(
        &mut commands,
        "选择难度",
        &font,
        18.0,
        -25.0,
        background_muted_text_color(),
    );
    commands.spawn((
        Sprite {
            color: Color::srgb(0.30, 0.32, 0.38),
            custom_size: Some(DIFFICULTY_HALF * 2.0),
            ..default()
        },
        Transform::from_xyz(DIFFICULTY_CENTER.x, DIFFICULTY_CENTER.y, 2.0),
        MenuEntity,
        DifficultyDropdownTrigger,
    ));
    commands.spawn((
        Text2d::new(""),
        TextFont {
            font: FontSource::Handle(font.clone()),
            font_size: 18.0.into(),
            ..default()
        },
        TextColor(Color::WHITE),
        Transform::from_xyz(DIFFICULTY_CENTER.x, DIFFICULTY_CENTER.y, 3.0),
        MenuEntity,
        DifficultyDropdownLabel,
    ));

    spawn_text(
        &mut commands,
        "选择地图",
        &font,
        18.0,
        -108.0,
        background_muted_text_color(),
    );
    const MAP_COLUMNS: usize = 4;
    for (i, map) in MAPS.iter().enumerate() {
        let row = i / MAP_COLUMNS;
        let column = i % MAP_COLUMNS;
        let count_in_row = (MAPS.len() - row * MAP_COLUMNS).min(MAP_COLUMNS);
        let x = (column as f32 - (count_in_row - 1) as f32 / 2.0) * 170.0;
        let center = Vec2::new(x, -140.0 - row as f32 * 54.0);
        let half = Vec2::new(78.0, 22.0);
        commands.spawn((
            Sprite {
                color: Color::srgb(0.24, 0.29, 0.36),
                custom_size: Some(half * 2.0),
                ..default()
            },
            Transform::from_xyz(center.x, center.y, 0.0),
            MenuEntity,
            MenuButton {
                action: MenuAction::Map(i),
                center,
                half,
            },
        ));
        commands.spawn((
            Text2d::new(map.name),
            TextFont {
                font: FontSource::Handle(font.clone()),
                font_size: 15.0.into(),
                ..default()
            },
            TextColor(Color::WHITE),
            Transform::from_xyz(center.x, center.y, 1.0),
            MenuEntity,
        ));
    }

    // 开始按钮
    let center = Vec2::new(0.0, -258.0);
    let half = Vec2::new(110.0, 34.0);
    commands.spawn((
        Sprite {
            color: Color::srgb(0.85, 0.28, 0.30),
            custom_size: Some(half * 2.0),
            ..default()
        },
        Transform::from_xyz(center.x, center.y, 0.0),
        MenuEntity,
        MenuButton {
            action: MenuAction::Start,
            center,
            half,
        },
    ));
    commands.spawn((
        Text2d::new("开始对战"),
        TextFont {
            font: FontSource::Handle(bold.clone()),
            font_size: 26.0.into(),
            ..default()
        },
        TextColor(Color::WHITE),
        Transform::from_xyz(center.x, center.y, 1.0),
        MenuEntity,
    ));

    spawn_text(
        &mut commands,
        "点击己方据点，再点击目标地块派兵 · 拖框可多选据点 · 据点全占即胜",
        &font,
        14.0,
        -292.0,
        background_muted_text_color(),
    );
}

pub fn menu_input(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut selection: ResMut<MenuSelection>,
    mut next: ResMut<NextState<AppState>>,
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    camera: Query<(&Camera, &GlobalTransform)>,
    q: Query<&MenuButton>,
    dropdown_options: Query<&DifficultyDropdownOption>,
    dropdown_entities: Query<Entity, With<DifficultyDropdownEntity>>,
) {
    if !buttons.just_pressed(MouseButton::Left) {
        return;
    }
    let (Ok(window), Ok((camera, cam_tf))) = (windows.single(), camera.single()) else {
        return;
    };
    let Some(w) = window
        .cursor_position()
        .and_then(|p| camera.viewport_to_world_2d(cam_tf, p).ok())
    else {
        return;
    };
    let dropdown_open = !dropdown_entities.is_empty();
    if dropdown_open {
        if let Some(option) = dropdown_options
            .iter()
            .find(|option| contains_point(option.center, option.half, w))
        {
            selection.difficulty = option.index;
        }
        close_difficulty_dropdown(&mut commands, &dropdown_entities);
        return;
    }
    if contains_point(DIFFICULTY_CENTER, DIFFICULTY_HALF, w) {
        let font = asset_server.load("fonts/NotoSansSC-Regular.ttf");
        open_difficulty_dropdown(&mut commands, &font, selection.difficulty);
        return;
    }
    for btn in q.iter() {
        let d = w - btn.center;
        if d.x.abs() <= btn.half.x && d.y.abs() <= btn.half.y {
            match btn.action {
                MenuAction::Subject(i) => selection.subject = i,
                MenuAction::Map(i) => selection.map = i,
                MenuAction::Start => next.set(AppState::Playing),
            }
            return;
        }
    }
}

pub fn sync_difficulty_dropdown_label(
    selection: Res<MenuSelection>,
    mut labels: Query<&mut Text2d, With<DifficultyDropdownLabel>>,
) {
    let Some(choice) = DIFFICULTIES.get(selection.difficulty) else {
        return;
    };
    labels.iter_mut().for_each(|mut label| {
        label.0 = format!("{}  ▼", choice.name);
    });
}

pub fn menu_highlight(selection: Res<MenuSelection>, q: Query<&MenuButton>, mut gizmos: Gizmos) {
    for btn in q.iter() {
        let selected = match btn.action {
            MenuAction::Subject(i) => selection.subject == i,
            MenuAction::Map(i) => selection.map == i,
            MenuAction::Start => false,
        };
        if selected {
            gizmos.rect_2d(
                Isometry2d::from_translation(btn.center),
                btn.half * 2.0 + Vec2::splat(8.0),
                Color::srgb(1.0, 0.9, 0.2),
            );
        }
    }
    gizmos.rect_2d(
        Isometry2d::from_translation(DIFFICULTY_CENTER),
        DIFFICULTY_HALF * 2.0 + Vec2::splat(8.0),
        Color::srgb(1.0, 0.9, 0.2),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dropdown_options_fit_the_menu_and_keep_v11_near_the_trigger() {
        let first = dropdown_option_center(0);
        let last = dropdown_option_center(DIFFICULTIES.len() - 1);
        assert!(first.y < 390.0);
        assert!(last.y > DIFFICULTY_CENTER.y);
        assert!(last.y < first.y);
    }

    #[test]
    fn dropdown_hit_test_includes_edge_and_rejects_outside() {
        assert!(contains_point(
            DIFFICULTY_CENTER,
            DIFFICULTY_HALF,
            DIFFICULTY_CENTER + DIFFICULTY_HALF,
        ));
        assert!(!contains_point(
            DIFFICULTY_CENTER,
            DIFFICULTY_HALF,
            DIFFICULTY_CENTER + DIFFICULTY_HALF + Vec2::X,
        ));
    }
}
