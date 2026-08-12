//! 真人试玩埋点：保存可重算的玩家观察、合法动作、实际指令与终局。
//!
//! 默认写入 `training/telemetry/`；`EASYWAR_TELEMETRY=0` 可关闭，也可把变量值
//! 直接设为其他输出目录。日志采用逐行 JSON，单局一个文件，写入失败只停用本局埋点。

use crate::common::{workspace_assets, CurrentMapFile, DifficultyName, InputMode, PLAYER};
use bevy::prelude::*;
use easywar_logic::rl::{
    observe_world, observe_world_tactical, world_intent_to_action, RlObservation, RL_ACTION_COUNT,
    RL_OBSERVATION_CHANNELS,
};
use easywar_logic::{CellKind, FactionId, Factions, GameClock, GridLookup, Intent, Owner, Winner};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const SCHEMA_VERSION: u32 = 1;
const SNAPSHOT_INTERVAL_SECONDS: f32 = 1.0;

#[derive(Clone, Debug)]
struct TimedIntent {
    game_time: f32,
    intent: Intent,
}

#[derive(Deserialize)]
struct ReplayLine {
    event: String,
    map: Option<String>,
    game_time: Option<f32>,
    winner: Option<FactionId>,
    actor_faction: Option<FactionId>,
    actions: Option<Vec<ReplayAction>>,
}

#[derive(Deserialize)]
struct ReplayAction {
    command: ReplayCommand,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ReplayCommand {
    Wait,
    SetStream { source: usize, target: usize },
    StopStream { source: usize },
}

fn replay_tape(path: &Path) -> Result<(String, Vec<TimedIntent>, FactionId, f32), String> {
    let lines = std::fs::read_to_string(path)
        .map_err(|error| format!("读取真人回放失败：{error}"))?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            serde_json::from_str::<ReplayLine>(line)
                .map_err(|error| format!("解析真人回放失败：{error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let map = lines
        .iter()
        .find(|line| line.event == "session_started")
        .and_then(|line| line.map.clone())
        .ok_or_else(|| "真人回放缺少地图信息".to_string())?;
    let terminal = lines
        .iter()
        .rev()
        .find(|line| line.event == "session_ended")
        .ok_or_else(|| "真人回放没有正常终局".to_string())?;
    let winner = terminal
        .winner
        .ok_or_else(|| "真人回放终局缺少胜者".to_string())?;
    let terminal_time = terminal
        .game_time
        .ok_or_else(|| "真人回放终局缺少时间".to_string())?;
    let tape = lines
        .into_iter()
        .filter(|line| line.event == "decision")
        .flat_map(|line| {
            let game_time = line.game_time.unwrap_or_default();
            let faction = line.actor_faction.unwrap_or_default();
            line.actions
                .unwrap_or_default()
                .into_iter()
                .filter_map(move |action| {
                    let intent = match action.command {
                        ReplayCommand::Wait => return None,
                        ReplayCommand::SetStream { source, target } => Intent::SetStream {
                            faction,
                            source,
                            target,
                        },
                        ReplayCommand::StopStream { source } => {
                            Intent::StopStream { faction, source }
                        }
                    };
                    Some(TimedIntent { game_time, intent })
                })
        })
        .collect();
    Ok((map, tape, winner, terminal_time))
}

/// 同时播放已记录的玩家与 AI 指令，验证权威逻辑能复现原终局。
pub fn verify_replay(path: &Path) -> Result<String, String> {
    let (map, tape, expected_winner, expected_time) = replay_tape(path)?;
    let root = workspace_assets();
    let mut app = App::new();
    app.add_plugins(easywar_logic::GamePlugin);
    easywar_logic::spawn_map(
        app.world_mut(),
        &root.join("maps").join(&map),
        &root.join("subjects"),
    )?;
    let mut next_intent = 0usize;
    let maximum_ticks = (expected_time / easywar_logic::SIM_DT).ceil() as usize + 2;
    for _ in 0..maximum_ticks {
        let time = app.world().resource::<GameClock>().time;
        while tape
            .get(next_intent)
            .is_some_and(|entry| entry.game_time <= time + easywar_logic::SIM_DT / 2.0)
        {
            app.world_mut()
                .resource_mut::<easywar_logic::IntentQueue>()
                .push(tape[next_intent].intent);
            next_intent += 1;
        }
        app.world_mut().run_schedule(easywar_logic::SimTick);
        if app.world().resource::<Winner>().0.is_some() {
            break;
        }
    }
    let actual_winner = app
        .world()
        .resource::<Winner>()
        .0
        .ok_or_else(|| "按埋点时间线播放后没有得到终局".to_string())?;
    let actual_time = app.world().resource::<GameClock>().time;
    if actual_winner != expected_winner
        || (actual_time - expected_time).abs() > easywar_logic::SIM_DT * 2.0
    {
        return Err(format!(
            "回放不一致：期望胜者/时间 {expected_winner}/{expected_time:.3}，实际 {actual_winner}/{actual_time:.3}"
        ));
    }
    Ok(format!(
        "真人回放复验通过：{map}，胜者 {actual_winner}，{actual_time:.3} 秒，{} 条指令",
        tape.len()
    ))
}

#[derive(Resource, Default)]
pub struct PendingPlayerCommands(pub Vec<Intent>);

impl PendingPlayerCommands {
    pub fn push(&mut self, intent: Intent) {
        self.0.push(intent);
    }

    pub fn extend(&mut self, intents: impl IntoIterator<Item = Intent>) {
        self.0.extend(intents);
    }
}

enum TelemetryTarget {
    Disabled,
    Directory(PathBuf),
    #[cfg(test)]
    Memory,
}

fn telemetry_target(configured: Option<&str>) -> TelemetryTarget {
    match configured {
        Some("0" | "false" | "off") => TelemetryTarget::Disabled,
        None | Some("" | "1" | "true" | "on") => TelemetryTarget::Directory(
            workspace_assets()
                .parent()
                .expect("assets 必须位于工作区内")
                .join("training/telemetry"),
        ),
        Some(directory) => TelemetryTarget::Directory(PathBuf::from(directory)),
    }
}

enum TelemetrySink {
    File(BufWriter<File>),
    #[cfg(test)]
    Memory(Vec<u8>),
}

#[derive(Clone)]
struct ActiveSession {
    id: String,
    player: FactionId,
    opponent: FactionId,
    next_snapshot_time: f32,
    decision_index: u64,
    terminal_written: bool,
}

#[derive(Resource)]
pub struct TelemetryRecorder {
    target: TelemetryTarget,
    sink: Option<TelemetrySink>,
    session: Option<ActiveSession>,
    output_path: Option<PathBuf>,
    last_error: Option<String>,
}

impl TelemetryRecorder {
    pub fn from_environment() -> Self {
        let configured = std::env::var("EASYWAR_TELEMETRY").ok();
        let target = telemetry_target(configured.as_deref());
        Self {
            target,
            sink: None,
            session: None,
            output_path: None,
            last_error: None,
        }
    }

    #[cfg(test)]
    fn memory() -> Self {
        Self {
            target: TelemetryTarget::Memory,
            sink: None,
            session: None,
            output_path: None,
            last_error: None,
        }
    }

    pub fn start_session(&mut self, world: &mut World, input_mode: InputMode) -> Option<String> {
        self.sink = None;
        self.session = None;
        self.output_path = None;
        self.last_error = None;
        if matches!(self.target, TelemetryTarget::Disabled) {
            return None;
        }
        let factions = world.resource::<Factions>().0.clone();
        let Some(player) = factions.iter().find(|faction| faction.is_player) else {
            return self.fail("埋点无法找到玩家阵营");
        };
        let opponents = factions
            .iter()
            .filter(|faction| !faction.is_player)
            .collect::<Vec<_>>();
        if opponents.len() != 1 {
            return self.fail("当前真人训练埋点只支持双人地图");
        }
        let now = unix_millis();
        let map = world.resource::<CurrentMapFile>().0.clone();
        let difficulty = world.resource::<DifficultyName>().0.to_string();
        let id = format!("{now}-{}", std::process::id());
        match &self.target {
            TelemetryTarget::Disabled => return None,
            TelemetryTarget::Directory(directory) => {
                if let Err(error) = std::fs::create_dir_all(directory) {
                    return self.fail(format!("创建埋点目录失败：{error}"));
                }
                let path = directory.join(format!("{}-{}.jsonl", id, safe_name(&map)));
                let file = match OpenOptions::new().create_new(true).write(true).open(&path) {
                    Ok(file) => file,
                    Err(error) => return self.fail(format!("创建埋点文件失败：{error}")),
                };
                self.sink = Some(TelemetrySink::File(BufWriter::new(file)));
                self.output_path = Some(path);
            }
            #[cfg(test)]
            TelemetryTarget::Memory => self.sink = Some(TelemetrySink::Memory(Vec::new())),
        }
        self.session = Some(ActiveSession {
            id: id.clone(),
            player: player.id,
            opponent: opponents[0].id,
            next_snapshot_time: SNAPSHOT_INTERVAL_SECONDS,
            decision_index: 0,
            terminal_written: false,
        });
        let event = SessionStarted {
            schema_version: SCHEMA_VERSION,
            event: "session_started",
            session_id: &id,
            started_unix_ms: now,
            map: &map,
            difficulty: &difficulty,
            input_mode: input_mode_name(input_mode),
            player_faction: player.id,
            opponent_faction: opponents[0].id,
            faction_count: factions.len(),
            observation_channels: RL_OBSERVATION_CHANNELS,
            action_count: RL_ACTION_COUNT,
            snapshot_interval_seconds: SNAPSHOT_INTERVAL_SECONDS,
        };
        if let Err(error) = self.write_event(&event) {
            return self.fail(error);
        }
        if let Err(error) = self.record_decision(
            world,
            "initial_wait",
            "player",
            player.id,
            opponents[0].id,
            &[],
        ) {
            return self.fail(error);
        }
        self.output_path
            .as_ref()
            .map(|path| format!("埋点已开启：{}", path.display()))
            .or_else(|| Some("埋点已开启".into()))
    }

    fn record_decision(
        &mut self,
        world: &mut World,
        decision_kind: &'static str,
        actor_role: &'static str,
        actor: FactionId,
        opponent: FactionId,
        intents: &[Intent],
    ) -> Result<(), String> {
        let session = self
            .session
            .as_ref()
            .cloned()
            .ok_or_else(|| "埋点会话尚未开始".to_string())?;
        if session.terminal_written {
            return Ok(());
        }
        let observation = observation_record(world, actor, opponent)?;
        let actions = if intents.is_empty() {
            vec![ActionRecord {
                action_id: 0,
                command: TelemetryCommand::Wait,
                actor_legal: observation.actor_valid_actions.contains(&0),
                tactical_candidate: observation.tactical_candidate_actions.contains(&0),
            }]
        } else {
            intents
                .iter()
                .copied()
                .map(|intent| {
                    let action_id = world_intent_to_action(world, intent)?;
                    Ok(ActionRecord {
                        action_id,
                        command: TelemetryCommand::from(intent),
                        actor_legal: observation.actor_valid_actions.contains(&action_id),
                        tactical_candidate: observation
                            .tactical_candidate_actions
                            .contains(&action_id),
                    })
                })
                .collect::<Result<Vec<_>, String>>()?
        };
        let difficulty = world.resource::<DifficultyName>().0;
        let game_time = world.resource::<GameClock>().time;
        let event = DecisionEvent {
            schema_version: SCHEMA_VERSION,
            event: "decision",
            session_id: &session.id,
            decision_index: session.decision_index,
            decision_kind,
            game_time,
            difficulty,
            actor_role,
            actor_faction: actor,
            actions,
            observation,
        };
        self.write_event(&event)?;
        if let Some(active) = self.session.as_mut() {
            if decision_kind == "player_command" {
                active.next_snapshot_time = game_time + SNAPSHOT_INTERVAL_SECONDS;
            }
            active.decision_index += 1;
        }
        Ok(())
    }

    fn record_periodic_if_due(&mut self, world: &mut World) -> Result<(), String> {
        let Some(session) = self.session.as_ref() else {
            return Ok(());
        };
        let time = world.resource::<GameClock>().time;
        if session.terminal_written
            || world.resource::<Winner>().0.is_some()
            || time + f32::EPSILON < session.next_snapshot_time
        {
            return Ok(());
        }
        self.record_decision(
            world,
            "periodic_wait",
            "player",
            session.player,
            session.opponent,
            &[],
        )?;
        if let Some(active) = self.session.as_mut() {
            while active.next_snapshot_time <= time + f32::EPSILON {
                active.next_snapshot_time += SNAPSHOT_INTERVAL_SECONDS;
            }
        }
        Ok(())
    }

    fn record_terminal_if_needed(&mut self, world: &mut World) -> Result<(), String> {
        let Some(winner) = world.resource::<Winner>().0 else {
            return Ok(());
        };
        let session = self
            .session
            .as_ref()
            .cloned()
            .ok_or_else(|| "埋点会话尚未开始".to_string())?;
        if session.terminal_written {
            return Ok(());
        }
        let final_observation = observation_record(world, session.player, session.opponent)?;
        let event = SessionEnded {
            schema_version: SCHEMA_VERSION,
            event: "session_ended",
            session_id: &session.id,
            game_time: world.resource::<GameClock>().time,
            winner,
            player_won: winner == session.player,
            player_bases: owned_cell_count(world, session.player, CellKind::Base),
            player_tiles: owned_cell_count(world, session.player, CellKind::LinkedTile),
            opponent_bases: owned_cell_count(world, session.opponent, CellKind::Base),
            opponent_tiles: owned_cell_count(world, session.opponent, CellKind::LinkedTile),
            final_observation,
        };
        self.write_event(&event)?;
        if let Some(active) = self.session.as_mut() {
            active.terminal_written = true;
        }
        self.flush()
    }

    pub fn close_session(&mut self, world: &mut World) {
        if self.session.is_none() {
            return;
        }
        let terminal = self
            .session
            .as_ref()
            .is_some_and(|session| session.terminal_written);
        if !terminal {
            let session = self.session.as_ref().expect("会话存在").clone();
            let event = SessionAborted {
                schema_version: SCHEMA_VERSION,
                event: "session_aborted",
                session_id: &session.id,
                game_time: world.resource::<GameClock>().time,
            };
            let _ = self.write_event(&event);
        }
        let _ = self.flush();
        self.sink = None;
        self.session = None;
    }

    fn write_event(&mut self, event: &impl Serialize) -> Result<(), String> {
        let bytes =
            serde_json::to_vec(event).map_err(|error| format!("序列化埋点失败：{error}"))?;
        let sink = self
            .sink
            .as_mut()
            .ok_or_else(|| "埋点输出尚未打开".to_string())?;
        match sink {
            TelemetrySink::File(writer) => writer
                .write_all(&bytes)
                .and_then(|_| writer.write_all(b"\n"))
                .and_then(|_| writer.flush())
                .map_err(|error| format!("写入埋点失败：{error}")),
            #[cfg(test)]
            TelemetrySink::Memory(writer) => {
                writer.extend_from_slice(&bytes);
                writer.push(b'\n');
                Ok(())
            }
        }
    }

    fn flush(&mut self) -> Result<(), String> {
        match self.sink.as_mut() {
            Some(TelemetrySink::File(writer)) => writer
                .flush()
                .map_err(|error| format!("刷新埋点失败：{error}")),
            #[cfg(test)]
            Some(TelemetrySink::Memory(_)) | None => Ok(()),
            #[cfg(not(test))]
            None => Ok(()),
        }
    }

    fn fail(&mut self, message: impl Into<String>) -> Option<String> {
        let message = message.into();
        eprintln!("[telemetry] {message}");
        self.last_error = Some(message);
        self.sink = None;
        self.session = None;
        None
    }

    fn handle_result(&mut self, result: Result<(), String>) {
        if let Err(error) = result {
            self.fail(error);
        }
    }

    pub fn is_active(&self) -> bool {
        self.session.is_some() && self.sink.is_some()
    }

    fn factions(&self) -> Option<(FactionId, FactionId)> {
        self.session
            .as_ref()
            .map(|session| (session.player, session.opponent))
    }

    #[cfg(test)]
    fn memory_output(&self) -> String {
        match self.sink.as_ref() {
            Some(TelemetrySink::Memory(bytes)) => String::from_utf8(bytes.clone()).unwrap(),
            _ => String::new(),
        }
    }
}

pub fn capture_player_commands(world: &mut World) {
    let commands = std::mem::take(&mut world.resource_mut::<PendingPlayerCommands>().0);
    if commands.is_empty() || world.get_resource::<GridLookup>().is_none() {
        return;
    }
    let Some(mut recorder) = world.remove_resource::<TelemetryRecorder>() else {
        return;
    };
    if !commands.is_empty() && recorder.is_active() {
        let (player, opponent) = recorder.factions().expect("活动埋点会话必须记录双方阵营");
        let result = recorder.record_decision(
            world,
            "player_command",
            "player",
            player,
            opponent,
            &commands,
        );
        recorder.handle_result(result);
    }
    world.insert_resource(recorder);
}

/// 在 SimTick 链尾记录规则 AI 与神经 AI 真正提交的动作；此时动作尚未应用，
/// 因而观察与决策时看到的权威状态严格一致。
pub fn capture_ai_commands(world: &mut World) {
    let Some((player, opponent)) = world
        .get_resource::<TelemetryRecorder>()
        .and_then(TelemetryRecorder::factions)
    else {
        return;
    };
    let commands = world
        .resource::<easywar_logic::IntentQueue>()
        .0
        .iter()
        .copied()
        .filter(|intent| intent_faction(*intent) == opponent)
        .collect::<Vec<_>>();
    if commands.is_empty() {
        return;
    }
    let Some(mut recorder) = world.remove_resource::<TelemetryRecorder>() else {
        return;
    };
    let result =
        recorder.record_decision(world, "ai_command", "opponent", opponent, player, &commands);
    recorder.handle_result(result);
    world.insert_resource(recorder);
}

pub fn capture_periodic_and_terminal(world: &mut World) {
    if world.get_resource::<GridLookup>().is_none() {
        return;
    }
    let Some(mut recorder) = world.remove_resource::<TelemetryRecorder>() else {
        return;
    };
    let terminal = recorder.record_terminal_if_needed(world);
    recorder.handle_result(terminal);
    let periodic = recorder.record_periodic_if_due(world);
    recorder.handle_result(periodic);
    world.insert_resource(recorder);
}

#[derive(Serialize)]
struct SessionStarted<'a> {
    schema_version: u32,
    event: &'static str,
    session_id: &'a str,
    started_unix_ms: u128,
    map: &'a str,
    difficulty: &'a str,
    input_mode: &'static str,
    player_faction: FactionId,
    opponent_faction: FactionId,
    faction_count: usize,
    observation_channels: usize,
    action_count: usize,
    snapshot_interval_seconds: f32,
}

#[derive(Serialize)]
struct DecisionEvent<'a> {
    schema_version: u32,
    event: &'static str,
    session_id: &'a str,
    decision_index: u64,
    decision_kind: &'static str,
    game_time: f32,
    difficulty: &'a str,
    actor_role: &'static str,
    actor_faction: FactionId,
    actions: Vec<ActionRecord>,
    observation: ObservationRecord,
}

#[derive(Serialize)]
struct ActionRecord {
    action_id: usize,
    command: TelemetryCommand,
    actor_legal: bool,
    tactical_candidate: bool,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum TelemetryCommand {
    Wait,
    SetStream { source: usize, target: usize },
    StopStream { source: usize },
}

impl From<Intent> for TelemetryCommand {
    fn from(intent: Intent) -> Self {
        match intent {
            Intent::SetStream { source, target, .. } => Self::SetStream { source, target },
            Intent::StopStream { source, .. } => Self::StopStream { source },
        }
    }
}

#[derive(Serialize)]
struct ObservationRecord {
    width: usize,
    height: usize,
    base_cells: Vec<i32>,
    /// `[固定观察下标, 数值]`；未出现的项均为 0，可无损还原稠密观察。
    sparse_values: Vec<(usize, f32)>,
    /// 当前决策阵营按玩家规则允许的动作，不含模型的战术筛选。
    actor_valid_actions: Vec<usize>,
    /// 当前战术成本边界允许搜索或交给 V10 的候选动作。
    tactical_candidate_actions: Vec<usize>,
}

fn observation_record(
    world: &mut World,
    actor: FactionId,
    opponent: FactionId,
) -> Result<ObservationRecord, String> {
    let actor_observation = observe_world(world, actor, opponent)?;
    let tactical_observation = observe_world_tactical(world, actor, opponent)?;
    let sparse_values = sparse_values(&actor_observation);
    let actor_valid_actions = valid_action_indices(&actor_observation);
    let tactical_candidate_actions = valid_action_indices(&tactical_observation);
    Ok(ObservationRecord {
        width: actor_observation.width,
        height: actor_observation.height,
        base_cells: actor_observation.base_cells,
        sparse_values,
        actor_valid_actions,
        tactical_candidate_actions,
    })
}

fn sparse_values(observation: &RlObservation) -> Vec<(usize, f32)> {
    observation
        .values
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| *value != 0.0)
        .collect()
}

fn valid_action_indices(observation: &RlObservation) -> Vec<usize> {
    observation
        .action_mask
        .iter()
        .enumerate()
        .filter_map(|(action, &valid)| valid.then_some(action))
        .collect()
}

fn intent_faction(intent: Intent) -> FactionId {
    match intent {
        Intent::SetStream { faction, .. } | Intent::StopStream { faction, .. } => faction,
    }
}

#[derive(Serialize)]
struct SessionEnded<'a> {
    schema_version: u32,
    event: &'static str,
    session_id: &'a str,
    game_time: f32,
    winner: FactionId,
    player_won: bool,
    player_bases: usize,
    player_tiles: usize,
    opponent_bases: usize,
    opponent_tiles: usize,
    final_observation: ObservationRecord,
}

#[derive(Serialize)]
struct SessionAborted<'a> {
    schema_version: u32,
    event: &'static str,
    session_id: &'a str,
    game_time: f32,
}

fn owned_cell_count(world: &World, faction: FactionId, expected_kind: CellKind) -> usize {
    let lookup = world.resource::<GridLookup>();
    lookup
        .cells
        .iter()
        .filter(|&&entity| {
            world.get::<CellKind>(entity) == Some(&expected_kind)
                && world
                    .get::<Owner>(entity)
                    .is_some_and(|owner| owner.0 == faction)
        })
        .count()
}

fn input_mode_name(input_mode: InputMode) -> &'static str {
    match input_mode {
        InputMode::Desktop => "desktop",
        InputMode::Touch => "touch",
    }
}

fn unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn safe_name(name: &str) -> String {
    Path::new(name)
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("map")
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::DifficultyName;
    use bevy::prelude::App;
    use easywar_logic::{
        spawn_map_seeded, AiController, AiControllers, AiParams, GamePlugin, SimTick,
    };
    use serde_json::Value;

    fn telemetry_app() -> App {
        let root = workspace_assets();
        let mut app = App::new();
        app.add_plugins(GamePlugin);
        spawn_map_seeded(
            app.world_mut(),
            &root.join("maps/dual_ladder_1v1.toml"),
            &root.join("subjects"),
            42,
        )
        .expect("埋点测试地图应可加载");
        app.world_mut()
            .insert_resource(CurrentMapFile("dual_ladder_1v1.toml".into()));
        app.world_mut()
            .insert_resource(DifficultyName("神经模型 V10·长程实验"));
        app
    }

    #[test]
    fn telemetry_is_enabled_by_default_and_can_be_disabled() {
        assert!(matches!(
            telemetry_target(None),
            TelemetryTarget::Directory(_)
        ));
        ["0", "false", "off"].into_iter().for_each(|value| {
            assert!(matches!(
                telemetry_target(Some(value)),
                TelemetryTarget::Disabled
            ));
        });
        let TelemetryTarget::Directory(path) = telemetry_target(Some("tmp/my-telemetry")) else {
            panic!("自定义埋点路径应保持启用");
        };
        assert_eq!(path, PathBuf::from("tmp/my-telemetry"));
    }

    #[test]
    fn records_initial_wait_multi_command_periodic_wait_and_terminal() {
        let mut app = telemetry_app();
        let mut recorder = TelemetryRecorder::memory();
        recorder.start_session(app.world_mut(), InputMode::Desktop);
        let lookup = app.world().resource::<GridLookup>().clone();
        let source = lookup.idx(1, 6);
        let commands = [
            Intent::SetStream {
                faction: PLAYER,
                source,
                target: lookup.idx(2, 6),
            },
            Intent::SetStream {
                faction: PLAYER,
                source,
                target: lookup.idx(2, 5),
            },
        ];
        recorder
            .record_decision(app.world_mut(), "player_command", "player", 1, 2, &commands)
            .expect("应记录多选命令");
        let ai_command = [Intent::SetStream {
            faction: 2,
            source: lookup.idx(15, 6),
            target: source,
        }];
        recorder
            .record_decision(app.world_mut(), "ai_command", "opponent", 2, 1, &ai_command)
            .expect("应记录 AI 命令");
        app.world_mut().resource_mut::<GameClock>().time = 1.0;
        recorder
            .record_periodic_if_due(app.world_mut())
            .expect("应记录周期等待");
        app.world_mut().resource_mut::<Winner>().0 = Some(PLAYER);
        recorder
            .record_terminal_if_needed(app.world_mut())
            .expect("应记录终局");

        let events = recorder
            .memory_output()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(events[0]["event"], "session_started");
        assert_eq!(events[1]["decision_kind"], "initial_wait");
        assert_eq!(events[1]["actions"][0]["action_id"], 0);
        assert_eq!(events[2]["decision_kind"], "player_command");
        assert_eq!(events[2]["actions"].as_array().unwrap().len(), 2);
        assert!(events[2]["observation"]["actor_valid_actions"]
            .as_array()
            .is_some_and(|actions| !actions.is_empty()));
        assert_eq!(events[3]["decision_kind"], "ai_command");
        assert_eq!(events[3]["actor_role"], "opponent");
        assert_eq!(events[3]["actor_faction"], 2);
        assert_eq!(events[4]["decision_kind"], "periodic_wait");
        assert_eq!(events[5]["event"], "session_ended");
        assert_eq!(events[5]["player_won"], true);
    }

    #[test]
    fn sparse_observation_can_restore_every_nonzero_value() {
        let mut app = telemetry_app();
        let observation = observe_world_tactical(app.world_mut(), 1, 2).unwrap();
        let mut restored = vec![0.0; observation.values.len()];
        sparse_values(&observation)
            .into_iter()
            .for_each(|(index, value)| restored[index] = value);
        assert_eq!(restored, observation.values);
    }

    #[test]
    fn periodic_wait_requires_one_second_without_player_command() {
        let mut app = telemetry_app();
        let mut recorder = TelemetryRecorder::memory();
        recorder.start_session(app.world_mut(), InputMode::Desktop);
        let lookup = app.world().resource::<GridLookup>().clone();
        app.world_mut().resource_mut::<GameClock>().time = 0.5;
        recorder
            .record_decision(
                app.world_mut(),
                "player_command",
                "player",
                1,
                2,
                &[Intent::SetStream {
                    faction: 1,
                    source: lookup.idx(1, 6),
                    target: lookup.idx(2, 6),
                }],
            )
            .unwrap();
        app.world_mut().resource_mut::<GameClock>().time = 1.0;
        recorder.record_periodic_if_due(app.world_mut()).unwrap();
        assert_eq!(recorder.memory_output().lines().count(), 3);

        app.world_mut().resource_mut::<GameClock>().time = 1.5;
        recorder.record_periodic_if_due(app.world_mut()).unwrap();
        assert_eq!(recorder.memory_output().lines().count(), 4);
    }

    #[test]
    fn sim_tick_records_real_rule_ai_command_before_application() {
        let mut app = telemetry_app();
        let mut recorder = TelemetryRecorder::memory();
        recorder.start_session(app.world_mut(), InputMode::Desktop);
        app.world_mut().insert_resource(recorder);
        app.world_mut()
            .insert_resource(AiControllers(vec![AiController::seeded(
                2,
                AiParams::hard(),
                7,
            )]));
        app.add_systems(
            SimTick,
            capture_ai_commands.after(easywar_logic::rl::policy_decide),
        );

        for _ in 0..(64 * 12) {
            app.world_mut().run_schedule(SimTick);
            if app
                .world()
                .resource::<TelemetryRecorder>()
                .memory_output()
                .contains("\"decision_kind\":\"ai_command\"")
            {
                break;
            }
        }

        let output = app.world().resource::<TelemetryRecorder>().memory_output();
        assert!(output.contains("\"decision_kind\":\"ai_command\""));
        assert!(output.contains("\"actor_role\":\"opponent\""));
    }
}
