use easywar_logic::rl::{
    step_batch, step_batch_external, EpisodeEnd, RlConfig, RlEnv, RlObservation, RlStep,
    SeatTransform, SubmitOrder, RL_ACTION_COUNT, RL_MAX_BASES, RL_MAX_HEIGHT, RL_MAX_WIDTH,
    RL_OBSERVATION_CHANNELS,
};
use easywar_logic::AiParams;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::path::PathBuf;

#[pyclass]
struct BatchObservation {
    #[pyo3(get)]
    observations: Vec<Vec<f32>>,
    #[pyo3(get)]
    action_masks: Vec<Vec<bool>>,
    #[pyo3(get)]
    base_cells: Vec<Vec<i32>>,
    #[pyo3(get)]
    widths: Vec<usize>,
    #[pyo3(get)]
    heights: Vec<usize>,
    #[pyo3(get)]
    times: Vec<f32>,
}

#[pyclass]
struct BatchStep {
    #[pyo3(get)]
    observations: Vec<Vec<f32>>,
    #[pyo3(get)]
    action_masks: Vec<Vec<bool>>,
    #[pyo3(get)]
    base_cells: Vec<Vec<i32>>,
    #[pyo3(get)]
    rewards: Vec<f32>,
    #[pyo3(get)]
    end_codes: Vec<u8>,
    #[pyo3(get)]
    end_names: Vec<String>,
    #[pyo3(get)]
    action_applied: Vec<bool>,
    #[pyo3(get)]
    opponent_action_applied: Vec<bool>,
    #[pyo3(get)]
    decisions: Vec<usize>,
}

#[pyclass(name = "BatchEnv")]
struct PyBatchEnv {
    environments: Vec<RlEnv>,
}

#[pymethods]
impl PyBatchEnv {
    #[new]
    #[pyo3(signature = (
        map_paths,
        subjects_dir,
        num_envs,
        seed=1,
        opponent="normal",
        decision_interval_seconds=1.0,
        stagnation_seconds=300.0,
        max_decisions=1200,
        map_transforms=None,
        alternate_seats=true,
        external_opponent=false,
        alternate_submit_order=true,
        variant_offset=0,
        rule_opponents=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        map_paths: Vec<String>,
        subjects_dir: String,
        num_envs: usize,
        seed: u64,
        opponent: &str,
        decision_interval_seconds: f32,
        stagnation_seconds: f32,
        max_decisions: usize,
        map_transforms: Option<Vec<String>>,
        alternate_seats: bool,
        external_opponent: bool,
        alternate_submit_order: bool,
        variant_offset: usize,
        rule_opponents: Option<Vec<String>>,
    ) -> PyResult<Self> {
        if map_paths.is_empty() {
            return Err(PyValueError::new_err("训练地图列表不能为空"));
        }
        if num_envs == 0 {
            return Err(PyValueError::new_err("批量环境数量必须大于 0"));
        }
        let opponent_names = rule_opponents.unwrap_or_else(|| vec![opponent.to_string()]);
        if opponent_names.is_empty() {
            return Err(PyValueError::new_err("规则对手池不能为空"));
        }
        let opponent_pool = opponent_names
            .iter()
            .map(|name| opponent_params(name))
            .collect::<PyResult<Vec<_>>>()?;
        let transforms = parse_transforms(map_transforms, map_paths.len())?;
        let subjects_dir = PathBuf::from(subjects_dir);
        let map_count = map_paths.len();
        let opponent_count = if external_opponent {
            1
        } else {
            opponent_pool.len()
        };
        let environments = (0..num_envs)
            .map(|index| {
                let (map_index, opponent_index, variant) =
                    map_opponent_variant(index, map_count, opponent_count, variant_offset);
                RlEnv::new(RlConfig {
                    map_path: PathBuf::from(&map_paths[map_index]),
                    subjects_dir: subjects_dir.clone(),
                    seed: seed.wrapping_add(index as u64),
                    learner_faction: 1,
                    opponent_faction: 2,
                    opponent_params: opponent_pool[opponent_index],
                    external_opponent,
                    submit_order: if alternate_submit_order && variant / 2 % 2 == 1 {
                        SubmitOrder::OpponentFirst
                    } else {
                        SubmitOrder::LearnerFirst
                    },
                    seat_transform: if alternate_seats && variant % 2 == 1 {
                        transforms[map_index]
                    } else {
                        SeatTransform::Identity
                    },
                    decision_interval_seconds,
                    stagnation_seconds,
                    max_decisions,
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(PyValueError::new_err)?;
        Ok(Self { environments })
    }

    fn observe(&mut self) -> PyResult<BatchObservation> {
        let observations = self
            .environments
            .iter_mut()
            .map(RlEnv::observe)
            .collect::<Result<Vec<_>, _>>()
            .map_err(PyValueError::new_err)?;
        Ok(pack_observations(observations))
    }

    fn observe_opponents(&mut self) -> PyResult<BatchObservation> {
        let observations = self
            .environments
            .iter_mut()
            .map(RlEnv::observe_opponent)
            .collect::<Result<Vec<_>, _>>()
            .map_err(PyValueError::new_err)?;
        Ok(pack_observations(observations))
    }

    #[pyo3(signature = (difficulty="normal"))]
    fn expert_actions(&mut self, difficulty: &str) -> PyResult<Vec<usize>> {
        let params = opponent_params(difficulty)?;
        self.environments
            .iter_mut()
            .map(|environment| environment.expert_action(params))
            .collect::<Result<Vec<_>, _>>()
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (actions, threads=0))]
    fn step(&mut self, actions: Vec<usize>, threads: usize) -> PyResult<BatchStep> {
        let workers = worker_count(threads);
        let steps = step_batch(&mut self.environments, &actions, workers)
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .map_err(PyValueError::new_err)?;
        Ok(pack_steps(steps))
    }

    #[pyo3(signature = (learner_actions, opponent_actions, threads=0))]
    fn step_external(
        &mut self,
        learner_actions: Vec<usize>,
        opponent_actions: Vec<usize>,
        threads: usize,
    ) -> PyResult<BatchStep> {
        let workers = worker_count(threads);
        let steps = step_batch_external(
            &mut self.environments,
            &learner_actions,
            &opponent_actions,
            workers,
        )
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map_err(PyValueError::new_err)?;
        Ok(pack_steps(steps))
    }

    fn reset_indices(
        &mut self,
        indices: Vec<usize>,
        seeds: Vec<u64>,
    ) -> PyResult<BatchObservation> {
        if indices.len() != seeds.len() {
            return Err(PyValueError::new_err("重置下标与种子数量不一致"));
        }
        indices
            .into_iter()
            .zip(seeds)
            .try_for_each(|(index, seed)| {
                self.environments
                    .get_mut(index)
                    .ok_or_else(|| format!("环境下标 {index} 越界"))?
                    .reset(seed)
                    .map(|_| ())
            })
            .map_err(PyValueError::new_err)?;
        self.observe()
    }

    fn __len__(&self) -> usize {
        self.environments.len()
    }
}

fn map_opponent_variant(
    index: usize,
    map_count: usize,
    opponent_count: usize,
    variant_offset: usize,
) -> (usize, usize, usize) {
    let map_index = index % map_count;
    let opponent_index = index / map_count % opponent_count;
    let variant = variant_offset + index / (map_count * opponent_count);
    (map_index, opponent_index, variant)
}

fn parse_transforms(
    configured: Option<Vec<String>>,
    map_count: usize,
) -> PyResult<Vec<SeatTransform>> {
    let names = configured.unwrap_or_else(|| vec!["identity".into(); map_count]);
    if names.len() != map_count {
        return Err(PyValueError::new_err(
            "地图自同构数量必须与训练地图数量一致",
        ));
    }
    names
        .into_iter()
        .map(|name| match name.as_str() {
            "identity" => Ok(SeatTransform::Identity),
            "vertical" => Ok(SeatTransform::Vertical),
            "rotational" => Ok(SeatTransform::Rotational),
            _ => Err(PyValueError::new_err(format!("未知地图自同构: {name}"))),
        })
        .collect()
}

fn worker_count(threads: usize) -> usize {
    if threads == 0 {
        std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1)
    } else {
        threads
    }
}

fn opponent_params(name: &str) -> PyResult<AiParams> {
    match name {
        "easy" => Ok(AiParams::easy()),
        "normal" => Ok(AiParams::normal()),
        "hard" => Ok(AiParams::hard()),
        _ => Err(PyValueError::new_err(format!("未知规则 AI 难度: {name}"))),
    }
}

fn pack_observations(observations: Vec<RlObservation>) -> BatchObservation {
    BatchObservation {
        observations: observations
            .iter()
            .map(|observation| observation.values.clone())
            .collect(),
        action_masks: observations
            .iter()
            .map(|observation| observation.action_mask.clone())
            .collect(),
        base_cells: observations
            .iter()
            .map(|observation| observation.base_cells.clone())
            .collect(),
        widths: observations
            .iter()
            .map(|observation| observation.width)
            .collect(),
        heights: observations
            .iter()
            .map(|observation| observation.height)
            .collect(),
        times: observations
            .iter()
            .map(|observation| observation.time)
            .collect(),
    }
}

fn pack_steps(steps: Vec<RlStep>) -> BatchStep {
    BatchStep {
        observations: steps
            .iter()
            .map(|step| step.observation.values.clone())
            .collect(),
        action_masks: steps
            .iter()
            .map(|step| step.observation.action_mask.clone())
            .collect(),
        base_cells: steps
            .iter()
            .map(|step| step.observation.base_cells.clone())
            .collect(),
        rewards: steps.iter().map(|step| step.reward).collect(),
        end_codes: steps.iter().map(|step| end_code(step.end)).collect(),
        end_names: steps.iter().map(|step| format!("{:?}", step.end)).collect(),
        action_applied: steps.iter().map(|step| step.action_applied).collect(),
        opponent_action_applied: steps
            .iter()
            .map(|step| step.opponent_action_applied)
            .collect(),
        decisions: steps.iter().map(|step| step.decision).collect(),
    }
}

fn end_code(end: EpisodeEnd) -> u8 {
    match end {
        EpisodeEnd::Ongoing => 0,
        EpisodeEnd::Won => 1,
        EpisodeEnd::Lost => 2,
        EpisodeEnd::Stalemate => 3,
        EpisodeEnd::CycleDetected => 4,
        EpisodeEnd::BudgetExceeded => 5,
    }
}

#[pymodule]
fn easywar_rl(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyBatchEnv>()?;
    module.add_class::<BatchObservation>()?;
    module.add_class::<BatchStep>()?;
    module.add("OBSERVATION_CHANNELS", RL_OBSERVATION_CHANNELS)?;
    module.add("MAX_WIDTH", RL_MAX_WIDTH)?;
    module.add("MAX_HEIGHT", RL_MAX_HEIGHT)?;
    module.add("MAX_BASES", RL_MAX_BASES)?;
    module.add("ACTION_COUNT", RL_ACTION_COUNT)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::map_opponent_variant;

    #[test]
    fn two_maps_cover_four_variants_orthogonally() {
        let assignments = (0..8)
            .map(|index| map_opponent_variant(index, 2, 1, 0))
            .collect::<Vec<_>>();
        assert_eq!(
            assignments,
            vec![
                (0, 0, 0),
                (1, 0, 0),
                (0, 0, 1),
                (1, 0, 1),
                (0, 0, 2),
                (1, 0, 2),
                (0, 0, 3),
                (1, 0, 3),
            ]
        );
    }

    #[test]
    fn maps_difficulties_and_seats_are_independent_factors() {
        let assignments = (0..12)
            .map(|index| map_opponent_variant(index, 2, 3, 0))
            .collect::<Vec<_>>();
        for map in 0..2 {
            for opponent in 0..3 {
                assert!(assignments.contains(&(map, opponent, 0)));
                assert!(assignments.contains(&(map, opponent, 1)));
            }
        }
    }
}
