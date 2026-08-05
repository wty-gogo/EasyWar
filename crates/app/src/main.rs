use bevy::prelude::*;
use bevy::window::WindowResolution;
use easywar_logic::*;
use std::collections::HashMap;
use std::path::PathBuf;

const CELL: f32 = 44.0;
const BORDER: f32 = 4.0;
const STEP: f32 = 48.0;
const FIXED_DT: f32 = 1.0 / 64.0;
const PLAYER: FactionId = 1;

const DIFFICULTIES: [(&str, fn() -> AiParams); 3] = [
    ("简单", AiParams::easy),
    ("中等", AiParams::normal),
    ("困难", AiParams::hard),
];

// ---------- 状态机 ----------

#[derive(States, Debug, Clone, PartialEq, Eq, Hash, Default)]
enum AppState {
    #[default]
    Menu,
    Playing,
    Ended,
}

// ---------- 资源与组件 ----------

#[derive(Resource)]
struct GameRes {
    game: GameState,
    linked_tint: HashMap<usize, [f32; 4]>,
}

#[derive(Resource)]
struct SubjectList(Vec<SubjectDef>);

#[derive(Resource)]
struct MenuSelection {
    subject: usize,
    difficulty: usize,
}

#[derive(Resource)]
struct AiRes {
    controllers: Vec<AiController>,
    difficulty: &'static str,
}

#[derive(Resource, Default)]
struct DragState {
    dragging: Option<usize>,
    selected: std::collections::HashSet<usize>,
    press_pos: Option<Vec2>,
}

#[derive(Resource, Default)]
struct DebugHud {
    last_event: String,
}

#[derive(Resource)]
struct EndInfo {
    winner: FactionId,
    player_bases: usize,
    player_tiles: usize,
    enemy_bases: usize,
    enemy_tiles: usize,
}

// 标记组件
#[derive(Component)]
struct MenuEntity;
#[derive(Component)]
struct BoardEntity;
#[derive(Component)]
struct EndEntity;

#[derive(Component)]
struct MenuButton {
    action: MenuAction,
    center: Vec2,
    half: Vec2,
}

#[derive(Clone, Copy)]
enum MenuAction {
    Subject(usize),
    Difficulty(usize),
    Start,
}

#[derive(Component)]
struct EndButton {
    restart: bool,
    center: Vec2,
    half: Vec2,
}

#[derive(Component)]
struct CellBorder(usize);
#[derive(Component)]
struct CellFill(usize);
#[derive(Component)]
struct CellLabel(usize, String);
#[derive(Component)]
struct SquadDot;
#[derive(Component)]
struct HudText;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--headless") {
        let secs: f32 = args.get(pos + 1).and_then(|s| s.parse().ok()).unwrap_or(60.0);
        headless_smoke(secs);
        return;
    }

    let root = workspace_assets();
    let subjects = load_subjects(&root.join("subjects")).expect("词库加载失败");
    let mut list: Vec<SubjectDef> = subjects.into_values().collect();
    list.sort_by(|a, b| a.id.cmp(&b.id)); // 稳定顺序

    App::new()
        .add_plugins(
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
        .init_state::<AppState>()
        .insert_resource(SubjectList(list))
        .insert_resource(MenuSelection {
            subject: 1, // 默认语文
            difficulty: 1,
        })
        .insert_resource(DragState::default())
        .insert_resource(DebugHud::default())
        .add_systems(Startup, |mut commands: Commands| {
            commands.spawn(Camera2d);
        })
        .add_systems(OnEnter(AppState::Menu), enter_menu)
        .add_systems(OnExit(AppState::Menu), cleanup::<MenuEntity>)
        .add_systems(Update, (menu_input, menu_highlight).run_if(in_state(AppState::Menu)))
        .add_systems(OnEnter(AppState::Playing), enter_playing)
        .add_systems(OnExit(AppState::Playing), exit_playing)
        .add_systems(
            FixedUpdate,
            (tick_sim, ai_tick).chain().run_if(in_state(AppState::Playing)),
        )
        .add_systems(
            Update,
            (
                spawn_board_system,
                handle_input,
                switch_difficulty,
                sync_cells,
                sync_squads,
                draw_overlays,
                update_hud,
                check_end,
            )
                .run_if(in_state(AppState::Playing)),
        )
        .add_systems(OnEnter(AppState::Ended), enter_ended)
        .add_systems(OnExit(AppState::Ended), cleanup::<EndEntity>)
        .add_systems(Update, end_input.run_if(in_state(AppState::Ended)))
        .run();
}

fn workspace_assets() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets")
}

fn cleanup<T: Component>(mut commands: Commands, q: Query<Entity, With<T>>) {
    for e in q.iter() {
        commands.entity(e).despawn();
    }
}

// ---------- 开局菜单 ----------

fn enter_menu(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    subjects: Res<SubjectList>,
) {
    let font: Handle<Font> = asset_server.load("fonts/NotoSansSC-Regular.ttf");
    let bold: Handle<Font> = asset_server.load("fonts/NotoSansSC-Bold.ttf");

    let spawn_text = |commands: &mut Commands, s: &str, font: &Handle<Font>, size: f32, y: f32, color: Color| {
        commands.spawn((
            Text2d::new(s.to_string()),
            TextFont { font: font.clone(), font_size: size, ..default() },
            TextColor(color),
            Transform::from_xyz(0.0, y, 1.0),
            MenuEntity,
        ));
    };

    spawn_text(&mut commands, "EasyWar · 学科对抗", &bold, 48.0, 300.0, Color::WHITE);
    spawn_text(&mut commands, "选择你的学科", &font, 20.0, 235.0, Color::srgb(0.7, 0.7, 0.7));

    // 学科按钮（6 个一排）
    let n = subjects.0.len() as f32;
    for (i, s) in subjects.0.iter().enumerate() {
        let x = (i as f32 - (n - 1.0) / 2.0) * 125.0;
        let center = Vec2::new(x, 160.0);
        let half = Vec2::new(55.0, 30.0);
        let c = parse_hex_color(&s.color);
        commands.spawn((
            Sprite { color: Color::srgba(c[0], c[1], c[2], 1.0), custom_size: Some(half * 2.0), ..default() },
            Transform::from_xyz(center.x, center.y, 0.0),
            MenuEntity,
            MenuButton { action: MenuAction::Subject(i), center, half },
        ));
        commands.spawn((
            Text2d::new(s.name.clone()),
            TextFont { font: bold.clone(), font_size: 20.0, ..default() },
            TextColor(Color::WHITE),
            Transform::from_xyz(center.x, center.y, 1.0),
            MenuEntity,
        ));
    }

    spawn_text(&mut commands, "选择难度", &font, 20.0, 80.0, Color::srgb(0.7, 0.7, 0.7));
    for (i, (name, _)) in DIFFICULTIES.iter().enumerate() {
        let x = (i as f32 - 1.0) * 160.0;
        let center = Vec2::new(x, 20.0);
        let half = Vec2::new(65.0, 26.0);
        commands.spawn((
            Sprite { color: Color::srgb(0.30, 0.32, 0.38), custom_size: Some(half * 2.0), ..default() },
            Transform::from_xyz(center.x, center.y, 0.0),
            MenuEntity,
            MenuButton { action: MenuAction::Difficulty(i), center, half },
        ));
        commands.spawn((
            Text2d::new(*name),
            TextFont { font: font.clone(), font_size: 20.0, ..default() },
            TextColor(Color::WHITE),
            Transform::from_xyz(center.x, center.y, 1.0),
            MenuEntity,
        ));
    }

    // 开始按钮
    let center = Vec2::new(0.0, -110.0);
    let half = Vec2::new(110.0, 34.0);
    commands.spawn((
        Sprite { color: Color::srgb(0.85, 0.28, 0.30), custom_size: Some(half * 2.0), ..default() },
        Transform::from_xyz(center.x, center.y, 0.0),
        MenuEntity,
        MenuButton { action: MenuAction::Start, center, half },
    ));
    commands.spawn((
        Text2d::new("开始对战"),
        TextFont { font: bold.clone(), font_size: 26.0, ..default() },
        TextColor(Color::WHITE),
        Transform::from_xyz(center.x, center.y, 1.0),
        MenuEntity,
    ));

    spawn_text(
        &mut commands,
        "拖动据点派兵 · 兵流撞到己方据点会并入 · 据点全占即胜",
        &font,
        14.0,
        -200.0,
        Color::srgb(0.5, 0.5, 0.5),
    );
}

fn menu_input(
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
    let window = windows.single();
    let (camera, cam_tf) = camera.single();
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
                MenuAction::Start => next.set(AppState::Playing),
            }
            return;
        }
    }
}

fn menu_highlight(selection: Res<MenuSelection>, q: Query<&MenuButton>, mut gizmos: Gizmos) {
    for btn in q.iter() {
        let selected = match btn.action {
            MenuAction::Subject(i) => selection.subject == i,
            MenuAction::Difficulty(i) => selection.difficulty == i,
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

// ---------- 对局 ----------

fn enter_playing(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    selection: Res<MenuSelection>,
    subjects: Res<SubjectList>,
) {
    let root = workspace_assets();
    let player_subject = subjects.0[selection.subject].id.clone();
    let ai_subject = subjects
        .0
        .iter()
        .find(|s| s.id != player_subject)
        .map(|s| s.id.clone())
        .unwrap();
    let game = build_game_custom(
        &root.join("maps/h_1v1.toml"),
        &root.join("subjects"),
        Some(&player_subject),
        Some(&ai_subject),
    )
    .expect("地图加载失败");

    let mut linked_tint = HashMap::new();
    for b in &game.bases {
        if let Some(s) = subjects.0.iter().find(|s| s.id == b.subject_id) {
            let c = parse_hex_color(&s.color);
            for &t in &b.linked {
                linked_tint.insert(t, c);
            }
        }
    }
    let (diff_name, diff_params) = DIFFICULTIES[selection.difficulty];
    let controllers = game
        .factions
        .iter()
        .filter(|f| !f.is_player)
        .map(|f| AiController::new(f.id, diff_params()))
        .collect();

    commands.insert_resource(GameRes { game, linked_tint });
    commands.insert_resource(AiRes { controllers, difficulty: diff_name });
    commands.insert_resource(DragState::default());
    commands.insert_resource(DebugHud::default());
    // 棋盘实体由 spawn_board_system 在下一帧生成（等资源就绪）
}

/// 资源就绪后生成一次棋盘实体
#[derive(Resource)]
struct BoardSpawned;

fn spawn_board_system(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    res: Res<GameRes>,
    spawned: Option<Res<BoardSpawned>>,
) {
    if spawned.is_some() {
        return;
    }
    commands.insert_resource(BoardSpawned);
    let font: Handle<Font> = asset_server.load("fonts/NotoSansSC-Regular.ttf");
    let bold: Handle<Font> = asset_server.load("fonts/NotoSansSC-Bold.ttf");
    let origin = grid_origin(&res.game);

    for (i, cell) in res.game.cells.iter().enumerate() {
        if !cell.enterable() {
            continue;
        }
        let pos = cell_pos(&res.game, origin, i);
        commands.spawn((
            Sprite {
                color: border_color(&res.game, i),
                custom_size: Some(Vec2::splat(CELL)),
                ..default()
            },
            Transform::from_xyz(pos.x, pos.y, 0.0),
            BoardEntity,
            CellBorder(i),
        ));
        commands.spawn((
            Sprite {
                color: fill_color(&res.game, &res.linked_tint, i),
                custom_size: Some(Vec2::splat(CELL - BORDER * 2.0)),
                ..default()
            },
            Transform::from_xyz(pos.x, pos.y, 0.5),
            BoardEntity,
            CellFill(i),
        ));
        let is_base = cell.kind == CellKind::Base;
        commands.spawn((
            Text2d::new(cell_text(&res.game, i)),
            TextFont {
                font: if is_base { bold.clone() } else { font.clone() },
                font_size: if is_base { 14.0 } else { 11.0 },
                ..default()
            },
            TextColor(text_color(cell)),
            Transform::from_xyz(pos.x, pos.y, 1.0),
            BoardEntity,
            CellLabel(i, String::new()),
        ));
    }

    commands.spawn((
        Text2d::new("拖动据点派兵 · Shift+点击多选 · 空白处拖框选 · 点出兵据点停止 · 1/2/3 换难度"),
        TextFont { font: font.clone(), font_size: 15.0, ..default() },
        TextColor(Color::srgb(0.5, 0.5, 0.5)),
        Transform::from_xyz(0.0, 360.0, 1.0),
        BoardEntity,
    ));
    commands.spawn((
        Text2d::new(""),
        TextFont { font, font_size: 12.0, ..default() },
        TextColor(Color::srgb(0.6, 0.6, 0.6)),
        Transform::from_xyz(0.0, 335.0, 1.0),
        BoardEntity,
        HudText,
    ));
}

fn exit_playing(mut commands: Commands, q: Query<Entity, With<BoardEntity>>) {
    for e in q.iter() {
        commands.entity(e).despawn();
    }
    commands.remove_resource::<GameRes>();
    commands.remove_resource::<AiRes>();
    commands.remove_resource::<BoardSpawned>();
}

// ---------- 对局：模拟与渲染 ----------

fn grid_origin(g: &GameState) -> Vec2 {
    Vec2::new(
        -(g.width as f32 - 1.0) * STEP / 2.0,
        (g.height as f32 - 1.0) * STEP / 2.0,
    )
}

fn cell_pos(g: &GameState, origin: Vec2, i: usize) -> Vec2 {
    let (x, y) = g.xy(i);
    Vec2::new(origin.x + x as f32 * STEP, origin.y - y as f32 * STEP)
}

fn fmt_num(v: f32) -> String {
    if v >= 1000.0 {
        format!("{:.1}k", v / 1000.0)
    } else {
        format!("{}", v.floor() as i32)
    }
}

fn cell_text(g: &GameState, i: usize) -> String {
    let cell = &g.cells[i];
    let num = if cell.kind == CellKind::Base {
        let cap = match g.base_index.get(&i) {
            Some(&bi) if cell.owner != NEUTRAL => g.base_garrison_cap(&g.bases[bi]),
            _ => cell.garrison_max,
        };
        format!("{}/{}", fmt_num(cell.garrison), fmt_num(cap))
    } else {
        fmt_num(cell.garrison)
    };
    match &cell.label {
        Some(l) => format!("{}\n{}", l, num),
        None => num,
    }
}

fn faction_color(g: &GameState, owner: FactionId) -> [f32; 4] {
    g.factions
        .iter()
        .find(|f| f.id == owner)
        .map(|f| f.color)
        .unwrap_or([0.5, 0.5, 0.5, 1.0])
}

fn border_color(g: &GameState, i: usize) -> Color {
    let cell = &g.cells[i];
    if cell.owner == NEUTRAL {
        return Color::srgb(0.55, 0.55, 0.58);
    }
    let c = faction_color(g, cell.owner);
    Color::srgba(c[0], c[1], c[2], 1.0)
}

fn fill_color(g: &GameState, tint: &HashMap<usize, [f32; 4]>, i: usize) -> Color {
    let cell = &g.cells[i];
    if cell.owner != NEUTRAL {
        let c = faction_color(g, cell.owner);
        if cell.kind == CellKind::Base {
            return Color::srgba(c[0], c[1], c[2], 1.0);
        }
        return Color::srgba(
            c[0] * 0.25 + 0.93 * 0.75,
            c[1] * 0.25 + 0.93 * 0.75,
            c[2] * 0.25 + 0.93 * 0.75,
            1.0,
        );
    }
    if let Some(t) = tint.get(&i) {
        return Color::srgba(
            t[0] * 0.10 + 0.90 * 0.90,
            t[1] * 0.10 + 0.90 * 0.90,
            t[2] * 0.10 + 0.90 * 0.90,
            1.0,
        );
    }
    Color::srgb(0.90, 0.90, 0.90)
}

fn text_color(cell: &Cell) -> Color {
    if cell.kind == CellKind::Base && cell.owner != NEUTRAL {
        Color::WHITE
    } else {
        Color::srgb(0.25, 0.25, 0.28)
    }
}

fn tick_sim(mut res: ResMut<GameRes>) {
    res.game.update(FIXED_DT);
}

fn ai_tick(mut res: ResMut<GameRes>, mut ai: ResMut<AiRes>) {
    let game = &mut res.game;
    let mut cmds = Vec::new();
    for c in ai.controllers.iter_mut() {
        cmds.extend(c.update(game, FIXED_DT).into_iter().map(|cmd| (c.faction, cmd)));
    }
    for (f, cmd) in cmds {
        match cmd {
            AiCommand::SetStream { source, target } => {
                game.set_stream(f, source, target);
            }
            AiCommand::StopStream { source } => game.stop_stream(f, source),
        }
    }
}

fn switch_difficulty(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut ai: ResMut<AiRes>,
    res: Res<GameRes>,
    mut hud: ResMut<DebugHud>,
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
    ai.controllers = res
        .game
        .factions
        .iter()
        .filter(|f| !f.is_player)
        .map(|f| AiController::new(f.id, params))
        .collect();
    ai.difficulty = name;
    hud.last_event = format!("AI 难度切换为：{name}");
}

fn sync_cells(
    res: Res<GameRes>,
    mut borders: Query<(&CellBorder, &mut Sprite), Without<CellFill>>,
    mut fills: Query<(&CellFill, &mut Sprite), Without<CellBorder>>,
    mut labels: Query<(&mut CellLabel, &mut Text2d, &mut TextColor)>,
) {
    for (cb, mut sprite) in borders.iter_mut() {
        sprite.color = border_color(&res.game, cb.0);
    }
    for (cf, mut sprite) in fills.iter_mut() {
        sprite.color = fill_color(&res.game, &res.linked_tint, cf.0);
    }
    for (mut label, mut text, mut color) in labels.iter_mut() {
        let s = cell_text(&res.game, label.0);
        if s != label.1 {
            text.0 = s.clone();
            label.1 = s;
        }
        *color = TextColor(text_color(&res.game.cells[label.0]));
    }
}

fn sync_squads(mut commands: Commands, res: Res<GameRes>, dots: Query<Entity, With<SquadDot>>) {
    for e in dots.iter() {
        commands.entity(e).despawn();
    }
    let origin = grid_origin(&res.game);
    for sq in &res.game.squads {
        let a = cell_pos(&res.game, origin, sq.path[sq.seg]);
        let b = if sq.seg + 1 < sq.path.len() {
            cell_pos(&res.game, origin, sq.path[sq.seg + 1])
        } else {
            a
        };
        let pos = a.lerp(b, sq.t);
        let c = faction_color(&res.game, sq.faction);
        let color = Color::srgba(c[0], c[1], c[2], 1.0);

        // 一兵一圆点，最多 3 个，垂直于行军方向排成一行
        let n = (sq.troops.round() as i32).clamp(1, 3) as usize;
        let dir = (b - a).try_normalize().unwrap_or(Vec2::X);
        let perp = Vec2::new(-dir.y, dir.x);
        const DOT: f32 = 6.0;
        const GAP: f32 = 8.0;
        for k in 0..n {
            let offset = (k as f32 - (n as f32 - 1.0) / 2.0) * GAP;
            let p = pos + perp * offset;
            commands.spawn((
                Sprite { color, custom_size: Some(Vec2::splat(DOT)), ..default() },
                Transform::from_xyz(p.x, p.y, 2.0),
                BoardEntity,
                SquadDot,
            ));
        }
    }
}

fn draw_overlays(res: Res<GameRes>, drag: Res<DragState>, mut gizmos: Gizmos) {
    let origin = grid_origin(&res.game);
    for s in &res.game.streams {
        if !s.active {
            continue;
        }
        let c = faction_color(&res.game, s.faction);
        let color = Color::srgba(c[0], c[1], c[2], 0.55);
        for w in s.path.windows(2) {
            let a = cell_pos(&res.game, origin, w[0]);
            let b = cell_pos(&res.game, origin, w[1]);
            gizmos.line_2d(a, b, color);
        }
    }
    for b in &res.game.bases {
        if res.game.cells[b.cell].owner == PLAYER {
            let p = cell_pos(&res.game, origin, b.cell);
            gizmos.rect_2d(
                Isometry2d::from_translation(p),
                Vec2::splat(CELL + 6.0),
                Color::srgba(1.0, 0.9, 0.2, 0.8),
            );
        }
    }
    for &src in &drag.selected {
        let p = cell_pos(&res.game, origin, src);
        gizmos.rect_2d(
            Isometry2d::from_translation(p),
            Vec2::splat(CELL + 12.0),
            Color::srgb(1.0, 0.6, 0.1),
        );
    }
}

// ---------- 对局：输入 ----------

fn world_to_cell(g: &GameState, world: Vec2) -> Option<usize> {
    let origin = grid_origin(g);
    let fx = (world.x - origin.x) / STEP;
    let fy = (origin.y - world.y) / STEP;
    let (x, y) = (fx.round() as i64, fy.round() as i64);
    if !g.in_bounds(x, y) {
        return None;
    }
    if (fx - x as f32).abs() > 0.5 || (fy - y as f32).abs() > 0.5 {
        return None;
    }
    let i = g.idx(x as usize, y as usize);
    g.cells[i].enterable().then_some(i)
}

fn handle_input(
    mut res: ResMut<GameRes>,
    mut drag: ResMut<DragState>,
    mut hud: ResMut<DebugHud>,
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
        match cursor_world.and_then(|w| world_to_cell(&res.game, w)) {
            Some(i) => {
                let c = &res.game.cells[i];
                hud.last_event = format!("按下命中 {:?} kind={:?} owner={}", res.game.xy(i), c.kind, c.owner);
                if c.kind == CellKind::Base && c.owner == PLAYER {
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
        let origin = grid_origin(&res.game);
        let a = cell_pos(&res.game, origin, src);
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
            match cursor_world.and_then(|w| world_to_cell(&res.game, w)) {
                Some(target) if target != src => {
                    let targets: Vec<usize> = drag.selected.iter().copied().collect();
                    let mut ok_count = 0;
                    for b in targets {
                        if res.game.set_stream(PLAYER, b, target) {
                            ok_count += 1;
                        }
                    }
                    hud.last_event = format!("{} 个据点出兵 → {:?}", ok_count, res.game.xy(target));
                }
                Some(_) => {
                    if res.game.stream_from(PLAYER, src).is_some() {
                        res.game.stop_stream(PLAYER, src);
                        hud.last_event = format!("停止 {:?} 的兵流", res.game.xy(src));
                    } else {
                        hud.last_event = format!("已选中 {:?}，再点目标格派兵", res.game.xy(src));
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
                    let origin = grid_origin(&res.game);
                    if !shift {
                        drag.selected.clear();
                    }
                    for b in &res.game.bases {
                        if res.game.cells[b.cell].owner != PLAYER {
                            continue;
                        }
                        let p = cell_pos(&res.game, origin, b.cell);
                        if p.x >= min.x && p.x <= max.x && p.y >= min.y && p.y <= max.y {
                            drag.selected.insert(b.cell);
                        }
                    }
                    hud.last_event = format!("框选 {} 个据点", drag.selected.len());
                } else if let Some(target) = world_to_cell(&res.game, w) {
                    let bases: Vec<usize> = drag.selected.iter().copied().collect();
                    let mut ok_count = 0;
                    for b in bases {
                        if res.game.set_stream(PLAYER, b, target) {
                            ok_count += 1;
                        }
                    }
                    hud.last_event = format!("{} 个据点出兵 → {:?}", ok_count, res.game.xy(target));
                } else {
                    drag.selected.clear();
                }
            }
        }
    }
}

fn update_hud(
    res: Res<GameRes>,
    drag: Res<DragState>,
    ai: Res<AiRes>,
    hud: Res<DebugHud>,
    mut q: Query<&mut Text2d, With<HudText>>,
) {
    let mut text = q.single_mut();
    let streams = res.game.streams.iter().filter(|s| s.active).count();
    text.0 = format!(
        "难度[{}](1/2/3切换) · 兵流 {} 条 · 小队 {} · 选中 {} 个据点 · {}",
        ai.difficulty,
        streams,
        res.game.squads.len(),
        drag.selected.len(),
        hud.last_event
    );
}

// ---------- 结算 ----------

fn check_end(
    mut commands: Commands,
    res: Res<GameRes>,
    mut next: ResMut<NextState<AppState>>,
) {
    if let Some(winner) = res.game.winner {
        let count = |f: FactionId, kind: CellKind| {
            res.game
                .cells
                .iter()
                .filter(|c| c.owner == f && c.kind == kind)
                .count()
        };
        commands.insert_resource(EndInfo {
            winner,
            player_bases: count(PLAYER, CellKind::Base),
            player_tiles: count(PLAYER, CellKind::LinkedTile),
            enemy_bases: count(2, CellKind::Base),
            enemy_tiles: count(2, CellKind::LinkedTile),
        });
        next.set(AppState::Ended);
    }
}

fn enter_ended(mut commands: Commands, asset_server: Res<AssetServer>, info: Res<EndInfo>) {
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
        TextFont { font: bold.clone(), font_size: 64.0, ..default() },
        TextColor(color),
        Transform::from_xyz(0.0, 120.0, 9.0),
        EndEntity,
    ));
    commands.spawn((
        Text2d::new(format!(
            "你占领：据点 {} 座 · 地块 {} 块\n对方占领：据点 {} 座 · 地块 {} 块",
            info.player_bases, info.player_tiles, info.enemy_bases, info.enemy_tiles
        )),
        TextFont { font: font.clone(), font_size: 20.0, ..default() },
        TextColor(Color::srgb(0.85, 0.85, 0.85)),
        Transform::from_xyz(0.0, 30.0, 9.0),
        EndEntity,
    ));

    for (i, (label, restart)) in [("再来一局", true), ("回主菜单", false)].iter().enumerate() {
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
            EndButton { restart: *restart, center, half },
        ));
        commands.spawn((
            Text2d::new(*label),
            TextFont { font: font.clone(), font_size: 20.0, ..default() },
            TextColor(Color::WHITE),
            Transform::from_xyz(center.x, center.y, 10.0),
            EndEntity,
        ));
    }
}

fn end_input(
    mut next: ResMut<NextState<AppState>>,
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    camera: Query<(&Camera, &GlobalTransform)>,
    q: Query<&EndButton>,
) {
    if !buttons.just_pressed(MouseButton::Left) {
        return;
    }
    let window = windows.single();
    let (camera, cam_tf) = camera.single();
    let Some(w) = window
        .cursor_position()
        .and_then(|p| camera.viewport_to_world_2d(cam_tf, p).ok())
    else {
        return;
    };
    for btn in q.iter() {
        let d = w - btn.center;
        if d.x.abs() <= btn.half.x && d.y.abs() <= btn.half.y {
            next.set(if btn.restart { AppState::Playing } else { AppState::Menu });
            return;
        }
    }
}

// ---------- 无头冒烟 ----------

fn headless_smoke(secs: f32) {
    let root = workspace_assets();
    let mut game = build_game(&root.join("maps/h_1v1.toml"), &root.join("subjects"))
        .expect("地图加载失败");
    println!(
        "[headless] 地图加载成功：{}x{}，据点 {} 个；AI vs AI 开打",
        game.width,
        game.height,
        game.bases.len()
    );
    let mut ais = vec![
        AiController::new(1, AiParams::normal()),
        AiController::new(2, AiParams::normal()),
    ];
    let steps = (secs / FIXED_DT) as usize;
    for i in 0..steps {
        game.update(FIXED_DT);
        let mut cmds = Vec::new();
        for ai in ais.iter_mut() {
            cmds.extend(ai.update(&game, FIXED_DT).into_iter().map(|c| (ai.faction, c)));
        }
        for (f, c) in cmds {
            match c {
                AiCommand::SetStream { source, target } => {
                    game.set_stream(f, source, target);
                }
                AiCommand::StopStream { source } => game.stop_stream(f, source),
            }
        }
        if i % ((60.0 / FIXED_DT) as usize) == 0 {
            println!(
                "[headless] t={:>5.1}s 语文总兵 {:>6.1} 数学总兵 {:>6.1} 小队 {} 胜者 {:?}",
                game.time,
                game.total_troops(1),
                game.total_troops(2),
                game.squads.len(),
                game.winner
            );
        }
        if game.winner.is_some() {
            break;
        }
    }
    println!("[headless] 结束：t={:.1}s winner={:?}", game.time, game.winner);
}
