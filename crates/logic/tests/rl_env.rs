use easywar_logic::rl::{
    decode_action, encode_set_stream_action, step_batch, step_batch_external, EpisodeEnd, RlAction,
    RlConfig, RlEnv, SeatTransform, SubmitOrder, RL_ACTION_COUNT, RL_MAX_CELLS, RL_MAX_WIDTH,
    RL_OBSERVATION_LEN,
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
        tactical_actions: false,
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
fn observation_exposes_target_recovery_rate() {
    let mut environment = RlEnv::new(config(8)).expect("创建强化学习环境失败");
    let observation = environment.observe().expect("读取观察失败");
    let physics_fixed_grid = 6 * RL_MAX_WIDTH + 2;
    assert!(
        (observation.values[17 * RL_MAX_CELLS + physics_fixed_grid] - 0.5).abs() < 0.001,
        "中立物理据点应暴露 2.5/秒、按5归一的恢复特征"
    );
}

#[test]
fn tactically_impossible_long_attack_receives_penalty() {
    let mut external = config(10);
    external.external_opponent = true;
    let mut idle = RlEnv::new(external.clone()).expect("创建等待环境失败");
    let mut attack = RlEnv::new(external).expect("创建进攻环境失败");
    let observation = attack.observe().expect("读取观察失败");
    let player_slot = observation
        .base_cells
        .iter()
        .position(|&cell| cell >= 0 && observation.values[3 * RL_MAX_CELLS + cell as usize] > 0.5)
        .expect("应找到玩家据点槽位");
    let math_fixed_grid = RL_MAX_WIDTH + 11;
    let action = encode_set_stream_action(player_slot, math_fixed_grid).expect("动作应可编码");
    assert!(observation.action_mask[action]);
    let idle_step = idle.step_external(0, 0).expect("推进等待环境失败");
    let attack_step = attack.step_external(action, 0).expect("推进进攻环境失败");
    assert!(
        attack_step.reward < idle_step.reward - 0.005,
        "跨越中间据点的无效远征应受到战术惩罚"
    );
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
    for tactical_actions in [false, true] {
        let mut configured = config(12);
        configured.tactical_actions = tactical_actions;
        let mut environment = RlEnv::new(configured).expect("创建强化学习环境失败");
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
}

#[test]
fn tactical_hard_expert_keeps_a_nonzero_hard_ceiling() {
    let maps = ["dual_ladder_1v1.toml", "braided_rings_1v1.toml"];
    let variants = [
        (SeatTransform::Identity, SubmitOrder::LearnerFirst),
        (SeatTransform::Identity, SubmitOrder::OpponentFirst),
        (SeatTransform::Vertical, SubmitOrder::LearnerFirst),
        (SeatTransform::Vertical, SubmitOrder::OpponentFirst),
    ];
    let wins = maps
        .into_iter()
        .flat_map(|map| variants.into_iter().map(move |variant| (map, variant)))
        .filter(|(map, (seat, order))| {
            let mut configured = config(940_000);
            configured.map_path = assets_dir().join("maps").join(map);
            configured.opponent_params = AiParams::hard();
            configured.tactical_actions = true;
            configured.seat_transform = *seat;
            configured.submit_order = *order;
            configured.max_decisions = 600;
            let mut environment = RlEnv::new(configured).expect("创建困难专家上限环境失败");
            (0..600)
                .find_map(|_| {
                    let action = environment
                        .expert_action(AiParams::hard())
                        .expect("读取困难规则专家动作失败");
                    let step = environment.step(action).expect("推进困难专家对局失败");
                    step.end
                        .is_terminal()
                        .then_some(step.end == EpisodeEnd::Won)
                })
                .unwrap_or(false)
        })
        .count();

    assert!(
        wins > 0,
        "困难规则专家经过战术动作边界后不应在两张训练地图的四向控制中全败"
    );
}

#[test]
fn tactical_mask_keeps_rule_expert_frontline_staging_actions() {
    let mut configured = config(19);
    configured.map_path = assets_dir().join("maps/dual_ladder_1v1.toml");
    configured.opponent_params = AiParams::hard();
    configured.tactical_actions = true;
    let mut environment = RlEnv::new(configured).expect("创建战术训练环境失败");
    let mut found_staging = false;

    for _ in 0..600 {
        let observation = environment.observe().expect("读取战术观察失败");
        let action = environment
            .expert_action(AiParams::hard())
            .expect("读取困难规则老师动作失败");
        if let RlAction::SetStream { target_grid, .. } =
            decode_action(action).expect("规则老师动作必须可解码")
        {
            let targets_owned_base = observation.values[2 * RL_MAX_CELLS + target_grid] > 0.5
                && observation.values[3 * RL_MAX_CELLS + target_grid] > 0.5;
            if targets_owned_base {
                assert!(
                    observation.action_mask[action],
                    "规则老师的前线蓄兵动作必须被战术掩码保留"
                );
                found_staging = true;
                break;
            }
        }
        let step = environment.step(action).expect("推进规则老师轨迹失败");
        if step.end.is_terminal() {
            break;
        }
    }

    assert!(found_staging, "困难规则老师应在双线地图上执行前线蓄兵");
}

#[test]
fn stagnation_is_distinct_from_engineering_budget() {
    let mut idle = config(14);
    idle.stagnation_seconds = 1.0;
    let mut environment = RlEnv::new(idle).expect("创建强化学习环境失败");
    let step = environment.step(0).unwrap();
    assert_eq!(step.end, EpisodeEnd::Stalemate);
    assert_eq!(step.reward, -1.1);
}

#[test]
fn engineering_budget_is_not_reported_as_a_normal_timeout() {
    let mut short = config(13);
    short.max_decisions = 2;
    let mut environment = RlEnv::new(short).expect("创建强化学习环境失败");
    assert_eq!(environment.step(0).unwrap().end, EpisodeEnd::Ongoing);
    let step = environment.step(0).unwrap();
    assert_eq!(step.end, EpisodeEnd::BudgetExceeded);
    assert_eq!(step.reward, -1.1);
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
