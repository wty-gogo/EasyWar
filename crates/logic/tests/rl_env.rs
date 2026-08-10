use easywar_logic::rl::{
    decode_action, step_batch, step_batch_external, EpisodeEnd, RlAction, RlConfig, RlEnv,
    SeatTransform, SubmitOrder, RL_ACTION_COUNT, RL_MAX_WIDTH, RL_OBSERVATION_LEN,
};
use easywar_logic::AiParams;
use std::path::PathBuf;

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets")
}

fn config(seed: u64) -> RlConfig {
    RlConfig {
        map_path: assets_dir().join("maps/h_1v1.toml"),
        subjects_dir: assets_dir().join("subjects"),
        seed,
        learner_faction: 1,
        opponent_faction: 2,
        opponent_params: AiParams::normal(),
        external_opponent: false,
        submit_order: SubmitOrder::LearnerFirst,
        seat_transform: SeatTransform::Identity,
        decision_interval_seconds: 1.0,
        stagnation_seconds: 300.0,
        max_decisions: 1_200,
    }
}

#[test]
fn observation_and_action_space_are_fixed_and_masked() {
    let mut environment = RlEnv::new(config(7)).expect("创建强化学习环境失败");
    let observation = environment.observe().expect("读取观察失败");
    assert_eq!(observation.values.len(), RL_OBSERVATION_LEN);
    assert_eq!(observation.action_mask.len(), RL_ACTION_COUNT);
    assert_eq!((observation.width, observation.height), (14, 13));
    assert!(observation.action_mask[0], "不操作必须始终合法");
    assert!(
        observation
            .action_mask
            .iter()
            .filter(|&&valid| valid)
            .count()
            > 1,
        "开局必须存在至少一个合法派兵动作"
    );
    observation
        .action_mask
        .iter()
        .enumerate()
        .filter(|(_, valid)| **valid)
        .for_each(|(action, _)| {
            if let RlAction::SetStream { target_grid, .. } =
                decode_action(action).expect("合法掩码中的动作必须可解码")
            {
                assert!(target_grid % RL_MAX_WIDTH < observation.width);
                assert!(target_grid / RL_MAX_WIDTH < observation.height);
            }
        });
}

#[test]
fn seat_transform_swaps_factions_and_aligns_base_scan_order() {
    let mut warmup = config(9);
    warmup.seat_transform = SeatTransform::Rotational;
    RlEnv::new(warmup).expect("H 图应支持 180 度席位自同构");

    let mut main = config(9);
    main.map_path = assets_dir().join("maps/dual_ladder_1v1.toml");
    main.seat_transform = SeatTransform::Vertical;
    RlEnv::new(main).expect("双线梯形应支持左右席位自同构");
}

#[test]
fn legal_player_level_action_is_applied_by_authoritative_logic() {
    let mut environment = RlEnv::new(config(11)).expect("创建强化学习环境失败");
    let observation = environment.observe().expect("读取观察失败");
    let action = observation
        .action_mask
        .iter()
        .enumerate()
        .find_map(|(action, &valid)| (action > 0 && valid).then_some(action))
        .expect("应存在合法派兵动作");
    let step = environment.step(action).expect("执行动作失败");
    assert!(step.action_applied);
    assert_eq!(step.decision, 1);
    assert_eq!(step.end, EpisodeEnd::Ongoing);
    assert!((step.observation.time - 1.0).abs() < 0.001);
}

#[test]
fn rule_expert_only_returns_legal_player_actions() {
    let mut environment = RlEnv::new(config(12)).expect("创建强化学习环境失败");
    for _ in 0..20 {
        let observation = environment.observe().expect("读取观察失败");
        let action = environment
            .expert_action(AiParams::normal())
            .expect("读取规则老师动作失败");
        assert!(
            observation.action_mask[action],
            "规则老师不得绕过玩家合法动作掩码"
        );
        let step = environment.step(action).expect("执行规则老师动作失败");
        if step.end.is_terminal() {
            break;
        }
    }
}

#[test]
fn stagnation_is_distinct_from_engineering_budget() {
    let mut idle = config(14);
    idle.stagnation_seconds = 1.0;
    let mut environment = RlEnv::new(idle).expect("创建强化学习环境失败");
    assert_eq!(environment.step(0).unwrap().end, EpisodeEnd::Stalemate);
}

#[test]
fn engineering_budget_is_not_reported_as_a_normal_timeout() {
    let mut short = config(13);
    short.max_decisions = 2;
    let mut environment = RlEnv::new(short).expect("创建强化学习环境失败");
    assert_eq!(environment.step(0).unwrap().end, EpisodeEnd::Ongoing);
    assert_eq!(environment.step(0).unwrap().end, EpisodeEnd::BudgetExceeded);
}

#[test]
fn fixed_seed_and_actions_reproduce_observation_and_reward() {
    let mut left = RlEnv::new(config(17)).expect("创建左侧环境失败");
    let mut right = RlEnv::new(config(17)).expect("创建右侧环境失败");
    for _ in 0..20 {
        let left_step = left.step(0).expect("推进左侧环境失败");
        let right_step = right.step(0).expect("推进右侧环境失败");
        assert_eq!(left_step, right_step);
        if left_step.end.is_terminal() {
            break;
        }
    }
}

#[test]
fn thread_count_does_not_change_batched_results() {
    let configs = (21..29).map(config).collect::<Vec<_>>();
    let mut sequential = configs
        .iter()
        .cloned()
        .map(RlEnv::new)
        .collect::<Result<Vec<_>, _>>()
        .expect("创建串行环境失败");
    let mut parallel = configs
        .into_iter()
        .map(RlEnv::new)
        .collect::<Result<Vec<_>, _>>()
        .expect("创建并行环境失败");
    let actions = vec![0; sequential.len()];

    for _ in 0..10 {
        let sequential_steps = step_batch(&mut sequential, &actions, 1)
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("串行批次推进失败");
        let parallel_steps = step_batch(&mut parallel, &actions, 4)
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("并行批次推进失败");
        assert_eq!(sequential_steps, parallel_steps);
    }
}

#[test]
fn external_opponent_receives_legal_symmetric_observation() {
    let mut external = config(31);
    external.external_opponent = true;
    let mut environment = RlEnv::new(external).expect("创建外部对手环境失败");
    let learner = environment.observe().expect("读取学习者观察失败");
    let opponent = environment.observe_opponent().expect("读取对手观察失败");
    assert!(learner.action_mask.iter().any(|&valid| valid));
    assert!(opponent.action_mask.iter().filter(|&&valid| valid).count() > 1);
    assert_eq!(learner.time, opponent.time);
}

#[test]
fn fixed_external_actions_reproduce_for_both_submit_orders() {
    for order in [SubmitOrder::LearnerFirst, SubmitOrder::OpponentFirst] {
        let mut external = config(37);
        external.external_opponent = true;
        external.submit_order = order;
        let mut left = RlEnv::new(external.clone()).expect("创建左侧外部环境失败");
        let mut right = RlEnv::new(external).expect("创建右侧外部环境失败");
        for _ in 0..20 {
            let left_step = left.step_external(0, 0).expect("推进左侧外部环境失败");
            let right_step = right.step_external(0, 0).expect("推进右侧外部环境失败");
            assert_eq!(left_step, right_step);
            if left_step.end.is_terminal() {
                break;
            }
        }
    }
}

#[test]
fn thread_count_does_not_change_external_batched_results() {
    let configs = (41..49)
        .map(|seed| {
            let mut external = config(seed);
            external.external_opponent = true;
            external.submit_order = if seed % 2 == 0 {
                SubmitOrder::LearnerFirst
            } else {
                SubmitOrder::OpponentFirst
            };
            external
        })
        .collect::<Vec<_>>();
    let mut sequential = configs
        .iter()
        .cloned()
        .map(RlEnv::new)
        .collect::<Result<Vec<_>, _>>()
        .expect("创建串行外部环境失败");
    let mut parallel = configs
        .into_iter()
        .map(RlEnv::new)
        .collect::<Result<Vec<_>, _>>()
        .expect("创建并行外部环境失败");
    let actions = vec![0; sequential.len()];

    for _ in 0..10 {
        let sequential_steps = step_batch_external(&mut sequential, &actions, &actions, 1)
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("串行外部批次推进失败");
        let parallel_steps = step_batch_external(&mut parallel, &actions, &actions, 4)
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("并行外部批次推进失败");
        assert_eq!(sequential_steps, parallel_steps);
    }
}
