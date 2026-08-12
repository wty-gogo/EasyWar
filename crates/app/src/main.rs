//! EasyWar 表现层组合根：插件装配 + 状态注册。
//! 对局生命周期见 driver.rs；模块划分见 docs/ARCHITECTURE.md §2。

mod common;
mod driver;
mod ending;
mod hud;
mod input;
mod menu;
mod neural_ai;
mod overlay;
mod render;
mod telemetry;

use bevy::prelude::*;
use bevy::window::WindowResolution;
use common::*;
use easywar_logic::*;

fn main() {
    let root = workspace_assets();
    let arguments = std::env::args().collect::<Vec<_>>();
    if let Some(index) = arguments
        .iter()
        .position(|argument| argument == "--verify-telemetry")
    {
        let path = arguments
            .get(index + 1)
            .map(std::path::Path::new)
            .expect("--verify-telemetry 后必须提供 JSONL 路径");
        println!(
            "{}",
            telemetry::verify_replay(path).expect("真人回放复验失败")
        );
        return;
    }
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
                    resolution: WindowResolution::new(900, 780),
                    ..default()
                }),
                ..default()
            })
            // icu_segmenter 缺中日韩断行字典（parley_data 未打包 cjdict），
            // 回退断行对中文足够用；告警走 log 后在此静音
            .set(bevy::log::LogPlugin {
                filter: format!("{},icu_provider=error", bevy::log::DEFAULT_FILTER),
                ..default()
            }),
    )
    .init_gizmo_group::<overlay::SelectedMarkerGizmos>()
    .add_plugins(GamePlugin)
    .init_state::<AppState>()
    .insert_resource(SubjectList(list))
    .insert_resource(ClearColor(app_background_color()))
    .insert_resource(MenuSelection {
        subject: 1, // 默认语文
        difficulty: configured_difficulty(),
        map: 0,
    })
    .insert_resource(InputMode::from_environment())
    .insert_resource(DragState::default())
    .insert_resource(DebugHud::default())
    .insert_resource(SimAccum::default())
    .insert_resource(neural_ai::NeuralModelResource::embedded())
    .insert_resource(telemetry::TelemetryRecorder::from_environment())
    .insert_resource(telemetry::PendingPlayerCommands::default())
    .add_systems(
        Startup,
        (
            |mut commands: Commands| {
                commands.spawn(Camera2d);
            },
            overlay::configure_gizmos,
        ),
    )
    // 菜单
    .add_systems(OnEnter(AppState::Menu), menu::enter_menu)
    .add_systems(OnExit(AppState::Menu), cleanup::<MenuEntity>)
    .add_systems(
        Update,
        (
            menu::menu_input,
            menu::sync_difficulty_dropdown_label,
            menu::menu_highlight,
        )
            .chain()
            .run_if(in_state(AppState::Menu)),
    )
    // 对局
    .add_systems(OnEnter(AppState::Playing), driver::enter_playing)
    .add_systems(OnExit(AppState::Playing), driver::exit_playing)
    .add_systems(
        SimTick,
        telemetry::capture_ai_commands.after(easywar_logic::rl::policy_decide),
    )
    .add_systems(
        Update,
        (
            input::handle_desktop_input.run_if(input::desktop_input_mode),
            input::handle_touch_input.run_if(input::touch_input_mode),
            input::switch_difficulty,
            telemetry::capture_player_commands,
            driver::drive_sim,
            telemetry::capture_periodic_and_terminal,
            driver::check_end,
        )
            .chain()
            .run_if(in_state(AppState::Playing)),
    )
    .add_systems(
        Update,
        (
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
        app.world_mut()
            .resource_mut::<NextState<AppState>>()
            .set(AppState::Playing);
    }
    app.run();
}
