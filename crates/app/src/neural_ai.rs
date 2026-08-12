//! 游戏内神经模型 V5～V11：读取训练导出的轻量权重，以 Rust 原生前向计算选择合法动作。

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

const MAGIC_V1: &[u8; 8] = b"EWNNv1\0\0";
const MAGIC_V2: &[u8; 8] = b"EWNNv2\0\0";
const EMBEDDED_V5_WEIGHTS: &[u8] = include_bytes!("../../../assets/models/neural_v5.ewnn");
const EMBEDDED_V6_WEIGHTS: &[u8] = include_bytes!("../../../assets/models/neural_v6.ewnn");
const EMBEDDED_V7_WEIGHTS: &[u8] = include_bytes!("../../../assets/models/neural_v7.ewnn");
const EMBEDDED_V8_WEIGHTS: &[u8] = include_bytes!("../../../assets/models/neural_v8.ewnn");
const EMBEDDED_V9_WEIGHTS: &[u8] = include_bytes!("../../../assets/models/neural_v9.ewnn");
const EMBEDDED_V10_WEIGHTS: &[u8] = include_bytes!("../../../assets/models/neural_v10.ewnn");
const EMBEDDED_V11_WEIGHTS: &[u8] = include_bytes!("../../../assets/models/neural_v11.ewnn");
const EXPERT_MAPS: [&str; 2] = ["dual_ladder_1v1.toml", "braided_rings_1v1.toml"];
const SELFPLAY_MAPS: [&str; 3] = [
    "dual_ladder_1v1.toml",
    "braided_rings_1v1.toml",
    "ring_chord_1v1.toml",
];
const SELFPLAY_TEMPERATURE: f32 = 0.5;
const SELFPLAY_SEED: u64 = 0x45_57_56_31_31;

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
        DifficultyKind::NeuralV5
        | DifficultyKind::NeuralV6
        | DifficultyKind::NeuralV7
        | DifficultyKind::NeuralV8
        | DifficultyKind::NeuralV9
        | DifficultyKind::NeuralV10
        | DifficultyKind::NeuralV11
            if (EXPERT_MAPS.contains(&map_file)
                || matches!(choice.kind, DifficultyKind::NeuralV11)
                    && SELFPLAY_MAPS.contains(&map_file))
                && factions.0.len() == 2 =>
        {
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
                    let selected = model.selected(choice.kind);
                    let policy: Box<dyn Policy> =
                        if matches!(choice.kind, DifficultyKind::NeuralV11) {
                            Box::new(NeuralPolicy::sampled(
                                selected,
                                SELFPLAY_TEMPERATURE,
                                SELFPLAY_SEED ^ u64::from(faction.id),
                            ))
                        } else {
                            Box::new(NeuralPolicy::new(selected))
                        };
                    if matches!(
                        choice.kind,
                        DifficultyKind::NeuralV7
                            | DifficultyKind::NeuralV8
                            | DifficultyKind::NeuralV9
                            | DifficultyKind::NeuralV10
                            | DifficultyKind::NeuralV11
                    ) {
                        PolicyController::new_tactical(faction.id, player, 1.0, policy)
                    } else {
                        PolicyController::new(faction.id, player, 1.0, policy)
                    }
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
        DifficultyKind::NeuralV6 => (
            rule_controllers(factions, AiParams::hard()),
            PolicyControllers::default(),
            "神经模型 V6（此图回退困难）",
        ),
        DifficultyKind::NeuralV7 => (
            rule_controllers(factions, AiParams::hard()),
            PolicyControllers::default(),
            "神经模型 V7·战术实验（此图回退困难）",
        ),
        DifficultyKind::NeuralV8 => (
            rule_controllers(factions, AiParams::hard()),
            PolicyControllers::default(),
            "神经模型 V8·强化实验（此图回退困难）",
        ),
        DifficultyKind::NeuralV9 => (
            rule_controllers(factions, AiParams::hard()),
            PolicyControllers::default(),
            "神经模型 V9·蓄兵实验（此图回退困难）",
        ),
        DifficultyKind::NeuralV10 => (
            rule_controllers(factions, AiParams::hard()),
            PolicyControllers::default(),
            "神经模型 V10·长程实验（此图回退困难）",
        ),
        DifficultyKind::NeuralV11 => (
            rule_controllers(factions, AiParams::hard()),
            PolicyControllers::default(),
            "神经模型 V11·自博弈（此图回退困难）",
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
pub struct NeuralModelResource {
    v5: Arc<NeuralModel>,
    v6: Arc<NeuralModel>,
    v7: Arc<NeuralModel>,
    v8: Arc<NeuralModel>,
    v9: Arc<NeuralModel>,
    v10: Arc<NeuralModel>,
    v11: Arc<NeuralModel>,
}

impl NeuralModelResource {
    pub fn embedded() -> Self {
        Self {
            v5: Arc::new(
                NeuralModel::from_bytes(EMBEDDED_V5_WEIGHTS).expect("内嵌神经模型 V5 权重损坏"),
            ),
            v6: Arc::new(
                NeuralModel::from_bytes(EMBEDDED_V6_WEIGHTS).expect("内嵌神经模型 V6 权重损坏"),
            ),
            v7: Arc::new(
                NeuralModel::from_bytes(EMBEDDED_V7_WEIGHTS).expect("内嵌神经模型 V7 权重损坏"),
            ),
            v8: Arc::new(
                NeuralModel::from_bytes(EMBEDDED_V8_WEIGHTS).expect("内嵌神经模型 V8 权重损坏"),
            ),
            v9: Arc::new(
                NeuralModel::from_bytes(EMBEDDED_V9_WEIGHTS).expect("内嵌神经模型 V9 权重损坏"),
            ),
            v10: Arc::new(
                NeuralModel::from_bytes(EMBEDDED_V10_WEIGHTS).expect("内嵌神经模型 V10 权重损坏"),
            ),
            v11: Arc::new(
                NeuralModel::from_bytes(EMBEDDED_V11_WEIGHTS).expect("内嵌神经模型 V11 权重损坏"),
            ),
        }
    }

    fn selected(&self, kind: DifficultyKind) -> Arc<NeuralModel> {
        match kind {
            DifficultyKind::NeuralV5 => self.v5.clone(),
            DifficultyKind::NeuralV6 => self.v6.clone(),
            DifficultyKind::NeuralV7 => self.v7.clone(),
            DifficultyKind::NeuralV8 => self.v8.clone(),
            DifficultyKind::NeuralV9 => self.v9.clone(),
            DifficultyKind::NeuralV10 => self.v10.clone(),
            DifficultyKind::NeuralV11 => self.v11.clone(),
            DifficultyKind::Rule(_) => unreachable!("规则难度不读取神经模型"),
        }
    }
}

pub struct NeuralPolicy {
    model: Arc<NeuralModel>,
    sampler: Option<PolicySampler>,
}

impl NeuralPolicy {
    pub fn new(model: Arc<NeuralModel>) -> Self {
        Self {
            model,
            sampler: None,
        }
    }

    pub fn sampled(model: Arc<NeuralModel>, temperature: f32, seed: u64) -> Self {
        Self {
            model,
            sampler: Some(PolicySampler::new(temperature, seed)),
        }
    }
}

impl Policy for NeuralPolicy {
    fn select_action(&mut self, observation: &RlObservation) -> usize {
        self.sampler
            .as_mut()
            .map(|sampler| {
                let random = sampler.next_unit();
                self.model
                    .sample_action(observation, sampler.temperature, random)
                    .unwrap_or(0)
            })
            .unwrap_or_else(|| self.model.select_action(observation).unwrap_or(0))
    }
}

struct PolicySampler {
    temperature: f32,
    state: u64,
}

impl PolicySampler {
    fn new(temperature: f32, seed: u64) -> Self {
        assert!(temperature > 0.0, "神经策略采样温度必须大于 0");
        Self {
            temperature,
            state: seed,
        }
    }

    fn next_unit(&mut self) -> f32 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        ((value >> 40) as f32) / ((1u32 << 24) as f32)
    }
}

pub struct NeuralModel {
    input_channels: usize,
    hidden: usize,
    conv1_weight: Vec<f32>,
    conv1_bias: Vec<f32>,
    conv2_weight: Vec<f32>,
    conv2_bias: Vec<f32>,
    source_weight: Vec<f32>,
    source_bias: Vec<f32>,
    target_weight: Vec<f32>,
    target_bias: Vec<f32>,
    source_context_weight: Vec<f32>,
    source_context_bias: Vec<f32>,
    target_context_weight: Vec<f32>,
    target_context_bias: Vec<f32>,
    stop_weight: Vec<f32>,
    stop_bias: f32,
    no_op_weight: Vec<f32>,
    no_op_bias: f32,
    strategy: Vec<f32>,
}

impl NeuralModel {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut reader = WeightReader::new(bytes);
        let magic = reader.take(MAGIC_V1.len())?;
        let has_global_context = match magic {
            value if value == MAGIC_V1 => false,
            value if value == MAGIC_V2 => true,
            _ => return Err("神经模型权重文件标识不匹配".into()),
        };
        let shape = [
            reader.u32()? as usize,
            reader.u32()? as usize,
            reader.u32()? as usize,
            reader.u32()? as usize,
            reader.u32()? as usize,
        ];
        let compatible = (1..=RL_OBSERVATION_CHANNELS).contains(&shape[0])
            && shape[1..] == [RL_MAX_HEIGHT, RL_MAX_WIDTH, RL_MAX_BASES, 64];
        if !compatible {
            return Err(format!(
                "神经模型形状不兼容：{shape:?}，观察通道应不超过 {RL_OBSERVATION_CHANNELS}"
            ));
        }
        let hidden = shape[4];
        let conv1_weight = reader.tensor(hidden * shape[0] * 9)?;
        let conv1_bias = reader.tensor(hidden)?;
        let conv2_weight = reader.tensor(hidden * hidden * 9)?;
        let conv2_bias = reader.tensor(hidden)?;
        let source_weight = reader.tensor(hidden * hidden)?;
        let source_bias = reader.tensor(hidden)?;
        let target_weight = reader.tensor(hidden * hidden)?;
        let target_bias = reader.tensor(hidden)?;
        let (
            source_context_weight,
            source_context_bias,
            target_context_weight,
            target_context_bias,
        ) = if has_global_context {
            (
                reader.tensor(hidden * hidden)?,
                reader.tensor(hidden)?,
                reader.tensor(hidden * hidden)?,
                reader.tensor(hidden)?,
            )
        } else {
            (
                vec![0.0; hidden * hidden],
                vec![0.0; hidden],
                vec![0.0; hidden * hidden],
                vec![0.0; hidden],
            )
        };
        let model = Self {
            input_channels: shape[0],
            hidden,
            conv1_weight,
            conv1_bias,
            conv2_weight,
            conv2_bias,
            source_weight,
            source_bias,
            target_weight,
            target_bias,
            source_context_weight,
            source_context_bias,
            target_context_weight,
            target_context_bias,
            stop_weight: reader.tensor(hidden)?,
            stop_bias: reader.scalar()?,
            no_op_weight: reader.tensor(hidden)?,
            no_op_bias: reader.scalar()?,
            strategy: reader.tensor(hidden)?,
        };
        if !reader.is_finished() {
            return Err("神经模型权重文件包含未识别的尾部数据".into());
        }
        Ok(model)
    }

    pub fn select_action(&self, observation: &RlObservation) -> Result<usize, String> {
        let logits = self.action_logits(observation)?;
        Ok(logits
            .iter()
            .copied()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |best, current| {
                if current.1 > best.1 {
                    current
                } else {
                    best
                }
            })
            .0)
    }

    pub fn sample_action(
        &self,
        observation: &RlObservation,
        temperature: f32,
        random: f32,
    ) -> Result<usize, String> {
        if temperature <= 0.0 {
            return Err("神经策略采样温度必须大于 0".into());
        }
        let logits = self.action_logits(observation)?;
        let maximum = logits
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(f32::NEG_INFINITY, f32::max);
        if !maximum.is_finite() {
            return Ok(0);
        }
        let weights = logits
            .iter()
            .map(|&logit| {
                if logit.is_finite() {
                    ((logit - maximum) / temperature).exp()
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        let total = weights.iter().sum::<f32>();
        let mut threshold = random.clamp(0.0, 1.0 - f32::EPSILON) * total;
        let mut fallback = 0usize;
        for (action, weight) in weights.into_iter().enumerate() {
            if weight == 0.0 {
                continue;
            }
            fallback = action;
            if threshold < weight {
                return Ok(action);
            }
            threshold -= weight;
        }
        Ok(fallback)
    }

    fn action_logits(&self, observation: &RlObservation) -> Result<Vec<f32>, String> {
        if observation.values.len() != RL_OBSERVATION_CHANNELS * RL_MAX_CELLS
            || observation.action_mask.len() != RL_ACTION_COUNT
            || observation.base_cells.len() != RL_MAX_BASES
        {
            return Err("神经模型观察形状与训练契约不一致".into());
        }
        let first = conv3x3_relu(
            &observation.values,
            self.input_channels,
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
        let source_context = linear(
            &pooled,
            &self.source_context_weight,
            &self.source_context_bias,
            self.hidden,
        );
        let target_context = linear(
            &pooled,
            &self.target_context_weight,
            &self.target_context_bias,
            self.hidden,
        );
        let sources = observation
            .base_cells
            .iter()
            .map(|&cell| cell.max(0) as usize)
            .map(|cell| {
                let base = (0..self.hidden)
                    .map(|channel| features[channel * RL_MAX_CELLS + cell])
                    .collect::<Vec<_>>();
                let mut projected =
                    linear(&base, &self.source_weight, &self.source_bias, self.hidden);
                projected
                    .iter_mut()
                    .zip(&source_context)
                    .for_each(|(value, context)| *value += context);
                projected
            })
            .collect::<Vec<_>>();
        let mut targets = pointwise_linear(
            &features,
            &self.target_weight,
            &self.target_bias,
            self.hidden,
        );
        targets
            .chunks_mut(RL_MAX_CELLS)
            .zip(&target_context)
            .for_each(|(channel, context)| channel.iter_mut().for_each(|value| *value += context));
        let stop_logits = sources
            .iter()
            .map(|source| dot(source, &self.stop_weight) + self.stop_bias)
            .collect::<Vec<_>>();
        let no_op = dot(&pooled, &self.no_op_weight) + self.no_op_bias;
        let stop_start = RL_ACTION_COUNT - RL_MAX_BASES;
        let mut logits = vec![f32::NEG_INFINITY; RL_ACTION_COUNT];
        if observation.action_mask[0] {
            logits[0] = no_op;
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
                logits[action] = score;
            }
            let action = stop_start + source_slot;
            if observation.action_mask[action] {
                logits[action] = stop_logits[source_slot];
            }
        }
        Ok(logits)
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
            return Err("神经模型权重文件意外结束".into());
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
            return Err(format!("神经模型张量长度为 {count}，期望 {expected}"));
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
    fn embedded_neural_models_load_and_obey_action_mask() {
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
        for weights in [
            EMBEDDED_V5_WEIGHTS,
            EMBEDDED_V6_WEIGHTS,
            EMBEDDED_V7_WEIGHTS,
            EMBEDDED_V8_WEIGHTS,
            EMBEDDED_V9_WEIGHTS,
            EMBEDDED_V10_WEIGHTS,
            EMBEDDED_V11_WEIGHTS,
        ] {
            let model = NeuralModel::from_bytes(weights).expect("权重应可读取");
            assert_eq!(model.select_action(&observation).unwrap(), 0);
        }
    }

    fn action_sequence(weights: &[u8], tactical_actions: bool) -> Vec<usize> {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets");
        let mut environment = RlEnv::new(RlConfig {
            map_path: root.join("maps/dual_ladder_1v1.toml"),
            subjects_dir: root.join("subjects"),
            seed: 42,
            learner_faction: 1,
            opponent_faction: 2,
            opponent_params: AiParams::normal(),
            external_opponent: true,
            tactical_actions,
            submit_order: SubmitOrder::LearnerFirst,
            seat_transform: SeatTransform::Identity,
            decision_interval_seconds: 1.0,
            stagnation_seconds: 300.0,
            max_decisions: 1200,
        })
        .expect("测试环境应能创建");
        let model = NeuralModel::from_bytes(weights).expect("权重应可读取");
        (0..8)
            .map(|_| {
                let observation = environment.observe_opponent().expect("应能读取对手视角");
                let action = model.select_action(&observation).unwrap();
                environment
                    .step_external(0, action)
                    .expect("双方动作应能推进环境");
                action
            })
            .collect()
    }

    fn sampled_action_sequence(weights: &[u8], temperature: f32, seed: u64) -> Vec<usize> {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets");
        let mut environment = RlEnv::new(RlConfig {
            map_path: root.join("maps/dual_ladder_1v1.toml"),
            subjects_dir: root.join("subjects"),
            seed: 42,
            learner_faction: 1,
            opponent_faction: 2,
            opponent_params: AiParams::normal(),
            external_opponent: true,
            tactical_actions: true,
            submit_order: SubmitOrder::LearnerFirst,
            seat_transform: SeatTransform::Identity,
            decision_interval_seconds: 1.0,
            stagnation_seconds: 300.0,
            max_decisions: 1200,
        })
        .expect("测试环境应能创建");
        let model = NeuralModel::from_bytes(weights).expect("权重应可读取");
        let mut sampler = PolicySampler::new(temperature, seed);
        (0..8)
            .map(|_| {
                let observation = environment.observe_opponent().expect("应能读取对手视角");
                let action = model
                    .sample_action(&observation, temperature, sampler.next_unit())
                    .unwrap();
                environment
                    .step_external(0, action)
                    .expect("双方动作应能推进环境");
                action
            })
            .collect()
    }

    #[test]
    fn rust_inference_matches_python_for_v5_through_v11() {
        assert_eq!(
            action_sequence(EMBEDDED_V5_WEIGHTS, false),
            [988, 0, 988, 988, 988, 988, 988, 984]
        );
        assert_eq!(
            action_sequence(EMBEDDED_V6_WEIGHTS, false),
            [988, 0, 988, 988, 988, 988, 988, 988]
        );
        assert_eq!(
            action_sequence(EMBEDDED_V7_WEIGHTS, true),
            [985, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            action_sequence(EMBEDDED_V8_WEIGHTS, true),
            [985, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            action_sequence(EMBEDDED_V9_WEIGHTS, true),
            [985, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            action_sequence(EMBEDDED_V10_WEIGHTS, true),
            [985, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            action_sequence(EMBEDDED_V11_WEIGHTS, true),
            [985, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            sampled_action_sequence(
                EMBEDDED_V11_WEIGHTS,
                SELFPLAY_TEMPERATURE,
                SELFPLAY_SEED ^ 2,
            ),
            [1019, 0, 0, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn sampled_policy_is_seeded_and_obeys_action_mask() {
        let mut observation = RlObservation {
            values: vec![0.0; RL_OBSERVATION_CHANNELS * RL_MAX_CELLS],
            action_mask: vec![false; RL_ACTION_COUNT],
            base_cells: vec![-1; RL_MAX_BASES],
            width: RL_MAX_WIDTH,
            height: RL_MAX_HEIGHT,
            time: 0.0,
        };
        observation.action_mask[0] = true;
        observation.action_mask[1] = true;
        let model =
            Arc::new(NeuralModel::from_bytes(EMBEDDED_V11_WEIGHTS).expect("V11 权重应可读取"));
        let mut left = NeuralPolicy::sampled(model.clone(), 0.5, 42);
        let mut right = NeuralPolicy::sampled(model, 0.5, 42);
        let left_actions = (0..32)
            .map(|_| left.select_action(&observation))
            .collect::<Vec<_>>();
        let right_actions = (0..32)
            .map(|_| right.select_action(&observation))
            .collect::<Vec<_>>();
        assert_eq!(left_actions, right_actions);
        assert!(left_actions
            .iter()
            .all(|&action| observation.action_mask[action]));
    }

    #[test]
    fn neural_model_v6_submits_intent_through_live_game_schedule() {
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
            4,
            "dual_ladder_1v1.toml",
            app.world().resource::<Factions>(),
            &model,
        );
        assert_eq!(name, "神经模型 V6");
        assert!(rules.0.is_empty());
        assert_eq!(policies.0.len(), 1);
        let (v5_rules, v5_policies, v5_name) = configured_controllers(
            3,
            "dual_ladder_1v1.toml",
            app.world().resource::<Factions>(),
            &model,
        );
        assert_eq!(v5_name, "神经模型 V5");
        assert!(v5_rules.0.is_empty());
        assert_eq!(v5_policies.0.len(), 1);
        let (v7_rules, v7_policies, v7_name) = configured_controllers(
            5,
            "dual_ladder_1v1.toml",
            app.world().resource::<Factions>(),
            &model,
        );
        assert_eq!(v7_name, "神经模型 V7·战术实验");
        assert!(v7_rules.0.is_empty());
        assert_eq!(v7_policies.0.len(), 1);
        let (v8_rules, v8_policies, v8_name) = configured_controllers(
            6,
            "dual_ladder_1v1.toml",
            app.world().resource::<Factions>(),
            &model,
        );
        assert_eq!(v8_name, "神经模型 V8·强化实验");
        assert!(v8_rules.0.is_empty());
        assert_eq!(v8_policies.0.len(), 1);
        let (v9_rules, v9_policies, v9_name) = configured_controllers(
            7,
            "dual_ladder_1v1.toml",
            app.world().resource::<Factions>(),
            &model,
        );
        assert_eq!(v9_name, "神经模型 V9·蓄兵实验");
        assert!(v9_rules.0.is_empty());
        assert_eq!(v9_policies.0.len(), 1);
        let (v10_rules, v10_policies, v10_name) = configured_controllers(
            8,
            "dual_ladder_1v1.toml",
            app.world().resource::<Factions>(),
            &model,
        );
        assert_eq!(v10_name, "神经模型 V10·长程实验");
        assert!(v10_rules.0.is_empty());
        assert_eq!(v10_policies.0.len(), 1);
        let (v11_rules, v11_policies, v11_name) = configured_controllers(
            9,
            "dual_ladder_1v1.toml",
            app.world().resource::<Factions>(),
            &model,
        );
        assert_eq!(v11_name, "神经模型 V11·自博弈");
        assert!(v11_rules.0.is_empty());
        assert_eq!(v11_policies.0.len(), 1);
        let (ring_rules, ring_policies, ring_name) = configured_controllers(
            9,
            "ring_chord_1v1.toml",
            app.world().resource::<Factions>(),
            &model,
        );
        assert_eq!(ring_name, "神经模型 V11·自博弈");
        assert!(ring_rules.0.is_empty());
        assert_eq!(ring_policies.0.len(), 1);
        let (fallback_rules, fallback_policies, fallback_name) =
            configured_controllers(4, "h_1v1.toml", app.world().resource::<Factions>(), &model);
        assert_eq!(fallback_name, "神经模型 V6（此图回退困难）");
        assert_eq!(fallback_rules.0.len(), 1);
        assert!(fallback_policies.0.is_empty());
        let (_, _, v5_fallback_name) =
            configured_controllers(3, "h_1v1.toml", app.world().resource::<Factions>(), &model);
        assert_eq!(v5_fallback_name, "神经模型 V5（此图回退困难）");
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
