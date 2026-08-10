//! 游戏内神经模型 V5：读取训练导出的轻量权重，以 Rust 原生前向计算选择合法动作。

use crate::common::{DifficultyKind, DIFFICULTIES};
use bevy::prelude::Resource;
use easywar_logic::rl::{
    RlObservation, RL_ACTION_COUNT, RL_MAX_BASES, RL_MAX_CELLS, RL_MAX_HEIGHT, RL_MAX_WIDTH,
    RL_OBSERVATION_CHANNELS,
};
use easywar_logic::{
    AiController, AiControllers, AiParams, Factions, Policy, PolicyController, PolicyControllers,
};
use std::sync::Arc;

const MAGIC: &[u8; 8] = b"EWNNv1\0\0";
const EMBEDDED_WEIGHTS: &[u8] = include_bytes!("../../../assets/models/neural_v5.ewnn");
const EXPERT_MAPS: [&str; 2] = ["dual_ladder_1v1.toml", "braided_rings_1v1.toml"];

pub fn configured_controllers(
    difficulty: usize,
    map_file: &str,
    factions: &Factions,
    model: &NeuralModelResource,
) -> (AiControllers, PolicyControllers, &'static str) {
    let choice = DIFFICULTIES.get(difficulty).unwrap_or(&DIFFICULTIES[1]);
    match choice.kind {
        DifficultyKind::Rule(params) => (
            rule_controllers(factions, params()),
            PolicyControllers::default(),
            choice.name,
        ),
        DifficultyKind::NeuralV5 if EXPERT_MAPS.contains(&map_file) && factions.0.len() == 2 => {
            let player = factions
                .0
                .iter()
                .find(|faction| faction.is_player)
                .expect("1v1 地图必须存在玩家阵营")
                .id;
            let policies = factions
                .0
                .iter()
                .filter(|faction| !faction.is_player)
                .map(|faction| {
                    PolicyController::new(
                        faction.id,
                        player,
                        1.0,
                        Box::new(NeuralPolicy::new(model.0.clone())),
                    )
                })
                .collect();
            (
                AiControllers::default(),
                PolicyControllers(policies),
                choice.name,
            )
        }
        DifficultyKind::NeuralV5 => (
            rule_controllers(factions, AiParams::hard()),
            PolicyControllers::default(),
            "神经模型 V5（此图回退困难）",
        ),
    }
}

fn rule_controllers(factions: &Factions, params: AiParams) -> AiControllers {
    AiControllers(
        factions
            .0
            .iter()
            .filter(|faction| !faction.is_player)
            .map(|faction| AiController::new(faction.id, params))
            .collect(),
    )
}

#[derive(Resource, Clone)]
pub struct NeuralModelResource(pub Arc<NeuralModel>);

impl NeuralModelResource {
    pub fn embedded() -> Self {
        Self(Arc::new(
            NeuralModel::from_bytes(EMBEDDED_WEIGHTS).expect("内嵌神经模型 V5 权重损坏"),
        ))
    }
}

pub struct NeuralPolicy {
    model: Arc<NeuralModel>,
}

impl NeuralPolicy {
    pub fn new(model: Arc<NeuralModel>) -> Self {
        Self { model }
    }
}

impl Policy for NeuralPolicy {
    fn select_action(&mut self, observation: &RlObservation) -> usize {
        self.model.select_action(observation).unwrap_or(0)
    }
}

pub struct NeuralModel {
    hidden: usize,
    conv1_weight: Vec<f32>,
    conv1_bias: Vec<f32>,
    conv2_weight: Vec<f32>,
    conv2_bias: Vec<f32>,
    source_weight: Vec<f32>,
    source_bias: Vec<f32>,
    target_weight: Vec<f32>,
    target_bias: Vec<f32>,
    stop_weight: Vec<f32>,
    stop_bias: f32,
    no_op_weight: Vec<f32>,
    no_op_bias: f32,
    strategy: Vec<f32>,
}

impl NeuralModel {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut reader = WeightReader::new(bytes);
        if reader.take(MAGIC.len())? != MAGIC {
            return Err("神经模型 V5 权重文件标识不匹配".into());
        }
        let shape = [
            reader.u32()? as usize,
            reader.u32()? as usize,
            reader.u32()? as usize,
            reader.u32()? as usize,
            reader.u32()? as usize,
        ];
        let expected = [
            RL_OBSERVATION_CHANNELS,
            RL_MAX_HEIGHT,
            RL_MAX_WIDTH,
            RL_MAX_BASES,
            64,
        ];
        if shape != expected {
            return Err(format!(
                "神经模型 V5 形状不兼容：{shape:?}，期望 {expected:?}"
            ));
        }
        let hidden = shape[4];
        let model = Self {
            hidden,
            conv1_weight: reader.tensor(hidden * shape[0] * 9)?,
            conv1_bias: reader.tensor(hidden)?,
            conv2_weight: reader.tensor(hidden * hidden * 9)?,
            conv2_bias: reader.tensor(hidden)?,
            source_weight: reader.tensor(hidden * hidden)?,
            source_bias: reader.tensor(hidden)?,
            target_weight: reader.tensor(hidden * hidden)?,
            target_bias: reader.tensor(hidden)?,
            stop_weight: reader.tensor(hidden)?,
            stop_bias: reader.scalar()?,
            no_op_weight: reader.tensor(hidden)?,
            no_op_bias: reader.scalar()?,
            strategy: reader.tensor(hidden)?,
        };
        if !reader.is_finished() {
            return Err("神经模型 V5 权重文件包含未识别的尾部数据".into());
        }
        Ok(model)
    }

    pub fn select_action(&self, observation: &RlObservation) -> Result<usize, String> {
        if observation.values.len() != RL_OBSERVATION_CHANNELS * RL_MAX_CELLS
            || observation.action_mask.len() != RL_ACTION_COUNT
            || observation.base_cells.len() != RL_MAX_BASES
        {
            return Err("神经模型 V5 观察形状与训练契约不一致".into());
        }
        let first = conv3x3_relu(
            &observation.values,
            RL_OBSERVATION_CHANNELS,
            self.hidden,
            &self.conv1_weight,
            &self.conv1_bias,
        );
        let mut features = conv3x3_relu(
            &first,
            self.hidden,
            self.hidden,
            &self.conv2_weight,
            &self.conv2_bias,
        );
        features
            .chunks_mut(RL_MAX_CELLS)
            .zip(&self.strategy)
            .for_each(|(channel, &style)| channel.iter_mut().for_each(|value| *value += style));

        let pooled = (0..self.hidden)
            .map(|channel| {
                features[channel * RL_MAX_CELLS..(channel + 1) * RL_MAX_CELLS]
                    .iter()
                    .sum::<f32>()
                    / RL_MAX_CELLS as f32
            })
            .collect::<Vec<_>>();
        let sources = observation
            .base_cells
            .iter()
            .map(|&cell| cell.max(0) as usize)
            .map(|cell| {
                let base = (0..self.hidden)
                    .map(|channel| features[channel * RL_MAX_CELLS + cell])
                    .collect::<Vec<_>>();
                linear(&base, &self.source_weight, &self.source_bias, self.hidden)
            })
            .collect::<Vec<_>>();
        let targets = pointwise_linear(
            &features,
            &self.target_weight,
            &self.target_bias,
            self.hidden,
        );
        let stop_logits = sources
            .iter()
            .map(|source| dot(source, &self.stop_weight) + self.stop_bias)
            .collect::<Vec<_>>();
        let no_op = dot(&pooled, &self.no_op_weight) + self.no_op_bias;
        let stop_start = RL_ACTION_COUNT - RL_MAX_BASES;
        let mut best = (0usize, f32::NEG_INFINITY);
        if observation.action_mask[0] {
            best = (0, no_op);
        }
        let scale = (self.hidden as f32).sqrt();
        for (source_slot, source) in sources.iter().enumerate() {
            for target in 0..RL_MAX_CELLS {
                let action = 1 + source_slot * RL_MAX_CELLS + target;
                if !observation.action_mask[action] {
                    continue;
                }
                let score = (0..self.hidden)
                    .map(|channel| source[channel] * targets[channel * RL_MAX_CELLS + target])
                    .sum::<f32>()
                    / scale;
                if score > best.1 {
                    best = (action, score);
                }
            }
            let action = stop_start + source_slot;
            if observation.action_mask[action] && stop_logits[source_slot] > best.1 {
                best = (action, stop_logits[source_slot]);
            }
        }
        Ok(best.0)
    }
}

fn conv3x3_relu(
    input: &[f32],
    input_channels: usize,
    output_channels: usize,
    weights: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    let mut output = vec![0.0; output_channels * RL_MAX_CELLS];
    for output_channel in 0..output_channels {
        for y in 0..RL_MAX_HEIGHT {
            for x in 0..RL_MAX_WIDTH {
                let mut value = bias[output_channel];
                for input_channel in 0..input_channels {
                    for kernel_y in 0..3 {
                        let input_y = y as isize + kernel_y as isize - 1;
                        if !(0..RL_MAX_HEIGHT as isize).contains(&input_y) {
                            continue;
                        }
                        for kernel_x in 0..3 {
                            let input_x = x as isize + kernel_x as isize - 1;
                            if !(0..RL_MAX_WIDTH as isize).contains(&input_x) {
                                continue;
                            }
                            let weight = (((output_channel * input_channels + input_channel) * 3
                                + kernel_y)
                                * 3)
                                + kernel_x;
                            let cell = input_y as usize * RL_MAX_WIDTH + input_x as usize;
                            value += weights[weight] * input[input_channel * RL_MAX_CELLS + cell];
                        }
                    }
                }
                output[output_channel * RL_MAX_CELLS + y * RL_MAX_WIDTH + x] = value.max(0.0);
            }
        }
    }
    output
}

fn linear(input: &[f32], weights: &[f32], bias: &[f32], output: usize) -> Vec<f32> {
    (0..output)
        .map(|row| dot(input, &weights[row * input.len()..(row + 1) * input.len()]) + bias[row])
        .collect()
}

fn pointwise_linear(input: &[f32], weights: &[f32], bias: &[f32], channels: usize) -> Vec<f32> {
    let mut output = vec![0.0; channels * RL_MAX_CELLS];
    for output_channel in 0..channels {
        for cell in 0..RL_MAX_CELLS {
            output[output_channel * RL_MAX_CELLS + cell] = bias[output_channel]
                + (0..channels)
                    .map(|input_channel| {
                        weights[output_channel * channels + input_channel]
                            * input[input_channel * RL_MAX_CELLS + cell]
                    })
                    .sum::<f32>();
        }
    }
    output
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(&a, &b)| a * b).sum()
}

struct WeightReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> WeightReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], String> {
        let end = self.offset.saturating_add(count);
        if end > self.bytes.len() {
            return Err("神经模型 V5 权重文件意外结束".into());
        }
        let result = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(result)
    }

    fn u32(&mut self) -> Result<u32, String> {
        let bytes: [u8; 4] = self.take(4)?.try_into().expect("长度已经验证");
        Ok(u32::from_le_bytes(bytes))
    }

    fn tensor(&mut self, expected: usize) -> Result<Vec<f32>, String> {
        let count = self.u32()? as usize;
        if count != expected {
            return Err(format!("神经模型 V5 张量长度为 {count}，期望 {expected}"));
        }
        self.take(count * 4)?
            .chunks_exact(4)
            .map(|bytes| {
                let value: [u8; 4] = bytes.try_into().expect("长度已经验证");
                Ok(f32::from_le_bytes(value))
            })
            .collect()
    }

    fn scalar(&mut self) -> Result<f32, String> {
        self.tensor(1).map(|values| values[0])
    }

    fn is_finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::prelude::App;
    use easywar_logic::rl::{RlConfig, RlEnv, SeatTransform, SubmitOrder};
    use easywar_logic::{spawn_map_seeded, AiParams, Factions, GamePlugin, SimTick, Stream};
    use std::path::PathBuf;

    #[test]
    fn embedded_neural_model_loads_and_obeys_action_mask() {
        let model = NeuralModel::from_bytes(EMBEDDED_WEIGHTS).expect("权重应可读取");
        let observation = RlObservation {
            values: vec![0.0; RL_OBSERVATION_CHANNELS * RL_MAX_CELLS],
            action_mask: std::iter::once(true)
                .chain(std::iter::repeat(false))
                .take(RL_ACTION_COUNT)
                .collect(),
            base_cells: vec![-1; RL_MAX_BASES],
            width: RL_MAX_WIDTH,
            height: RL_MAX_HEIGHT,
            time: 0.0,
        };
        assert_eq!(model.select_action(&observation).unwrap(), 0);
    }

    #[test]
    fn rust_inference_matches_python_on_real_opponent_observation() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets");
        let mut environment = RlEnv::new(RlConfig {
            map_path: root.join("maps/dual_ladder_1v1.toml"),
            subjects_dir: root.join("subjects"),
            seed: 42,
            learner_faction: 1,
            opponent_faction: 2,
            opponent_params: AiParams::normal(),
            external_opponent: true,
            submit_order: SubmitOrder::LearnerFirst,
            seat_transform: SeatTransform::Identity,
            decision_interval_seconds: 1.0,
            stagnation_seconds: 300.0,
            max_decisions: 1200,
        })
        .expect("测试环境应能创建");
        let model = NeuralModel::from_bytes(EMBEDDED_WEIGHTS).expect("权重应可读取");
        let expected = [988, 0, 988, 988, 988, 988, 988, 984];
        for expected_action in expected {
            let observation = environment.observe_opponent().expect("应能读取对手视角");
            let action = model.select_action(&observation).unwrap();
            assert_eq!(action, expected_action);
            environment
                .step_external(0, action)
                .expect("双方动作应能推进环境");
        }
    }

    #[test]
    fn neural_model_v5_submits_intent_through_live_game_schedule() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets");
        let mut app = App::new();
        app.add_plugins(GamePlugin);
        spawn_map_seeded(
            app.world_mut(),
            &root.join("maps/dual_ladder_1v1.toml"),
            &root.join("subjects"),
            42,
        )
        .expect("地图应能加载");
        let model = NeuralModelResource::embedded();
        let (rules, policies, name) = configured_controllers(
            3,
            "dual_ladder_1v1.toml",
            app.world().resource::<Factions>(),
            &model,
        );
        assert_eq!(name, "神经模型 V5");
        assert!(rules.0.is_empty());
        assert_eq!(policies.0.len(), 1);
        let (fallback_rules, fallback_policies, fallback_name) =
            configured_controllers(3, "h_1v1.toml", app.world().resource::<Factions>(), &model);
        assert_eq!(fallback_name, "神经模型 V5（此图回退困难）");
        assert_eq!(fallback_rules.0.len(), 1);
        assert!(fallback_policies.0.is_empty());
        app.world_mut().insert_resource(rules);
        app.world_mut().insert_resource(policies);
        for _ in 0..65 {
            app.world_mut()
                .try_run_schedule(SimTick)
                .expect("实时调度应可运行");
        }
        assert!(
            app.world_mut()
                .query::<&Stream>()
                .iter(app.world())
                .any(|stream| stream.faction == 2 && stream.active),
            "神经模型动作应经意图队列建立 AI 兵流"
        );
    }
}
