//! 结算画面（胜负 + 占领统计；按业主要求不含用时）。

use crate::common::*;
use bevy::prelude::*;

pub fn enter_ended(mut commands: Commands, asset_server: Res<AssetServer>, info: Res<EndInfo>) {
    let font: Handle<Font> = asset_server.load("fonts/NotoSansSC-Regular.ttf");
    let bold: Handle<Font> = asset_server.load("fonts/NotoSansSC-Bold.ttf");
    let won = info.winner == PLAYER;
    let (title, color) = if won {
        ("胜 利 !", Color::srgb(0.2, 0.85, 0.35))
    } else {
        ("战 败 …", Color::srgb(0.9, 0.3, 0.25))
    };

    // 半透明遮罩
    commands.spawn((
        Sprite {
            color: Color::srgba(0.05, 0.05, 0.08, 0.85),
            custom_size: Some(Vec2::new(2000.0, 2000.0)),
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 8.0),
        EndEntity,
    ));
    commands.spawn((
        Text2d::new(title),
        TextFont {
            font: FontSource::Handle(bold.clone()),
            font_size: 64.0.into(),
            ..default()
        },
        TextColor(color),
        Transform::from_xyz(0.0, 120.0, 9.0),
        EndEntity,
    ));
    commands.spawn((
        Text2d::new(format!(
            "你占领：据点 {} 座 · 地块 {} 块\n对方占领：据点 {} 座 · 地块 {} 块",
            info.player_bases, info.player_tiles, info.enemy_bases, info.enemy_tiles
        )),
        TextFont {
            font: FontSource::Handle(font.clone()),
            font_size: 20.0.into(),
            ..default()
        },
        TextColor(Color::srgb(0.85, 0.85, 0.85)),
        Transform::from_xyz(0.0, 30.0, 9.0),
        EndEntity,
    ));

    for (i, (label, restart)) in [("再来一局", true), ("回主菜单", false)].iter().enumerate()
    {
        let center = Vec2::new((i as f32 - 0.5) * 220.0, -80.0);
        let half = Vec2::new(90.0, 28.0);
        commands.spawn((
            Sprite {
                color: Color::srgb(0.30, 0.32, 0.38),
                custom_size: Some(half * 2.0),
                ..default()
            },
            Transform::from_xyz(center.x, center.y, 9.0),
            EndEntity,
            EndButton {
                restart: *restart,
                center,
                half,
            },
        ));
        commands.spawn((
            Text2d::new(*label),
            TextFont {
                font: FontSource::Handle(font.clone()),
                font_size: 20.0.into(),
                ..default()
            },
            TextColor(Color::WHITE),
            Transform::from_xyz(center.x, center.y, 10.0),
            EndEntity,
        ));
    }
}

pub fn end_input(
    mut next: ResMut<NextState<AppState>>,
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    camera: Query<(&Camera, &GlobalTransform)>,
    q: Query<&EndButton>,
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
            next.set(if btn.restart {
                AppState::Playing
            } else {
                AppState::Menu
            });
            return;
        }
    }
}
