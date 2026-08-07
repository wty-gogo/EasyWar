//! 开局界面：选学科、地图和难度。

use crate::common::*;
use bevy::prelude::*;
use easywar_logic::parse_hex_color;

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
        Color::WHITE,
    );
    spawn_text(
        &mut commands,
        "选择你的学科",
        &font,
        20.0,
        258.0,
        Color::srgb(0.7, 0.7, 0.7),
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
        Color::srgb(0.7, 0.7, 0.7),
    );
    for (i, (name, _)) in DIFFICULTIES.iter().enumerate() {
        let x = (i as f32 - 1.0) * 160.0;
        let center = Vec2::new(x, -65.0);
        let half = Vec2::new(65.0, 22.0);
        commands.spawn((
            Sprite {
                color: Color::srgb(0.30, 0.32, 0.38),
                custom_size: Some(half * 2.0),
                ..default()
            },
            Transform::from_xyz(center.x, center.y, 0.0),
            MenuEntity,
            MenuButton {
                action: MenuAction::Difficulty(i),
                center,
                half,
            },
        ));
        commands.spawn((
            Text2d::new(*name),
            TextFont {
                font: FontSource::Handle(font.clone()),
                font_size: 18.0.into(),
                ..default()
            },
            TextColor(Color::WHITE),
            Transform::from_xyz(center.x, center.y, 1.0),
            MenuEntity,
        ));
    }

    spawn_text(
        &mut commands,
        "选择地图",
        &font,
        18.0,
        -108.0,
        Color::srgb(0.7, 0.7, 0.7),
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
        Color::srgb(0.5, 0.5, 0.5),
    );
}

pub fn menu_input(
    mut selection: ResMut<MenuSelection>,
    mut next: ResMut<NextState<AppState>>,
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    camera: Query<(&Camera, &GlobalTransform)>,
    q: Query<&MenuButton>,
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
    for btn in q.iter() {
        let d = w - btn.center;
        if d.x.abs() <= btn.half.x && d.y.abs() <= btn.half.y {
            match btn.action {
                MenuAction::Subject(i) => selection.subject = i,
                MenuAction::Difficulty(i) => selection.difficulty = i,
                MenuAction::Map(i) => selection.map = i,
                MenuAction::Start => next.set(AppState::Playing),
            }
            return;
        }
    }
}

pub fn menu_highlight(selection: Res<MenuSelection>, q: Query<&MenuButton>, mut gizmos: Gizmos) {
    for btn in q.iter() {
        let selected = match btn.action {
            MenuAction::Subject(i) => selection.subject == i,
            MenuAction::Difficulty(i) => selection.difficulty == i,
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
}
