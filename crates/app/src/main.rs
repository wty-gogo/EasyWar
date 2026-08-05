//! EasyWar 表现层组合根：插件装配 + 状态注册。
//! 对局生命周期见 driver.rs；模块划分见 docs/ARCHITECTURE.md §2。

mod common;
mod driver;
mod ending;
mod hud;
mod input;
mod menu;
mod overlay;
mod render;

use bevy::prelude::*;
use bevy::window::WindowResolution;
use common::*;
use easywar_logic::*;

fn main() {
    let root = workspace_assets();
    let subjects = load_subjects(&root.join("subjects")).expect("词库加载失败");
    let mut list: Vec<SubjectDef> = subjects.into_values().collect();
    list.sort_by(|a, b| a.id.cmp(&b.id)); // 稳定顺序

    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(AssetPlugin {
                file_path: "../../assets".into(),
                ..default()
            })
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "EasyWar · 学科对抗".into(),
                    resolution: WindowResolution::new(900.0, 780.0),
                    ..default()
                }),
                ..default()
            }),
    )
    .add_plugins(GamePlugin)
    .init_state::<AppState>()
    .insert_resource(SubjectList(list))
    .insert_resource(MenuSelection {
        subject: 1, // 默认语文
        difficulty: 1,
    })
    .insert_resource(DragState::default())
    .insert_resource(DebugHud::default())
    .insert_resource(SimAccum::default())
    .add_systems(Startup, |mut commands: Commands| {
        commands.spawn(Camera2d);
    })
    // 菜单
    .add_systems(OnEnter(AppState::Menu), menu::enter_menu)
    .add_systems(OnExit(AppState::Menu), cleanup::<MenuEntity>)
    .add_systems(Update, (menu::menu_input, menu::menu_highlight).run_if(in_state(AppState::Menu)))
    // 对局
    .add_systems(OnEnter(AppState::Playing), driver::enter_playing)
    .add_systems(OnExit(AppState::Playing), driver::exit_playing)
    .add_systems(
        Update,
        (
            driver::drive_sim,
            driver::check_end,
            input::handle_input,
            input::switch_difficulty,
            render::spawn_board_system,
            render::sync_cells,
            render::sync_squads,
            overlay::draw_overlays,
            hud::update_hud,
        )
            .run_if(in_state(AppState::Playing)),
    )
    // 结算
    .add_systems(OnEnter(AppState::Ended), ending::enter_ended)
    .add_systems(OnExit(AppState::Ended), cleanup::<EndEntity>)
    .add_systems(Update, ending::end_input.run_if(in_state(AppState::Ended)));

    // 冒烟模式：跳过菜单直接开局（AI 接管对手，玩家挂机）
    if std::env::args().any(|a| a == "--auto") {
        app.world_mut().resource_mut::<NextState<AppState>>().set(AppState::Playing);
    }
    app.run();
}
