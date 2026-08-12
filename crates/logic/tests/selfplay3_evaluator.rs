use easywar_logic::evaluation3::{
    first_divergence, horizontal_mirror, run_match, LinkedScanOrder, MatchConfig, MatchFactors,
    SnapshotPart,
};
use std::path::PathBuf;

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets")
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/selfplay3_symmetric.toml")
}

fn config(factors: MatchFactors, seconds: f32, enable_ai: bool) -> MatchConfig {
    MatchConfig {
        map_path: fixture_path(),
        subjects_dir: assets_dir().join("subjects"),
        max_seconds: seconds,
        factors,
        enable_ai,
        capture_trace: true,
        cell_permutation: None,
    }
}

#[test]
fn fixed_seed_reproduces_every_tick() {
    let config = config(MatchFactors::default(), 8.0, true);
    let first = run_match(&config).expect("第一次运行夹具失败");
    let second = run_match(&config).expect("第二次运行夹具失败");
    assert_eq!(first.trace, second.trace);
    assert_eq!(first.winner_faction, second.winner_faction);
}

#[test]
fn faction_renaming_keeps_seat_normalized_trace() {
    let reference =
        run_match(&config(MatchFactors::default(), 8.0, true)).expect("运行基准夹具失败");
    let renamed = run_match(&config(
        MatchFactors {
            seat_factions: [2, 3, 1],
            submit_order: [2, 3, 1],
            ..MatchFactors::default()
        },
        8.0,
        true,
    ))
    .expect("运行阵营重命名夹具失败");
    assert_eq!(first_divergence(&reference.trace, &renamed.trace), None);
}

#[test]
fn horizontal_mirror_permutates_trace_back_to_same_seats() {
    let reference =
        run_match(&config(MatchFactors::default(), 8.0, true)).expect("运行基准夹具失败");
    let mut mirrored_config = config(MatchFactors::default(), 8.0, true);
    mirrored_config.cell_permutation = Some(horizontal_mirror(11, 3));
    let mirrored = run_match(&mirrored_config).expect("运行镜像夹具失败");
    assert_eq!(first_divergence(&reference.trace, &mirrored.trace), None);
}

#[test]
fn entity_declaration_order_does_not_leak_into_grid_lookup() {
    let reference =
        run_match(&config(MatchFactors::default(), 8.0, true)).expect("运行基准夹具失败");
    let noisy = run_match(&config(
        MatchFactors {
            entity_declaration_noise: 37,
            ..MatchFactors::default()
        },
        8.0,
        true,
    ))
    .expect("运行实体声明噪声夹具失败");
    assert_eq!(first_divergence(&reference.trace, &noisy.trace), None);
}

#[test]
fn submit_and_ai_candidate_order_are_measured_as_independent_factors() {
    let reference =
        run_match(&config(MatchFactors::default(), 8.0, true)).expect("运行基准夹具失败");
    let reversed_submit = run_match(&config(
        MatchFactors {
            submit_order: [3, 2, 1],
            ..MatchFactors::default()
        },
        8.0,
        true,
    ))
    .expect("运行反向提交夹具失败");
    let reversed_linked = run_match(&config(
        MatchFactors {
            linked_scan_order: LinkedScanOrder::Reversed,
            ..MatchFactors::default()
        },
        8.0,
        true,
    ))
    .expect("运行反向候选扫描夹具失败");

    let submit_divergence = first_divergence(&reference.trace, &reversed_submit.trace)
        .expect("提交顺序变化应被轨迹记录");
    let linked_divergence = first_divergence(&reference.trace, &reversed_linked.trace)
        .expect("AI 候选顺序变化应被轨迹记录");
    assert!(submit_divergence.parts.contains(&SnapshotPart::Intents));
    assert!(linked_divergence.parts.contains(&SnapshotPart::Intents));
    assert!(submit_divergence.tick > 0);
    assert!(linked_divergence.tick > 0);
}

#[test]
fn base_list_order_does_not_change_cost_ranked_ai_trace() {
    let reference =
        run_match(&config(MatchFactors::default(), 20.0, true)).expect("运行基准夹具失败");
    let reversed = run_match(&config(
        MatchFactors {
            base_scan_order: Some(vec![4, 3, 2, 1, 0]),
            ..MatchFactors::default()
        },
        20.0,
        true,
    ))
    .expect("运行反向 BaseList 夹具失败");
    assert_eq!(
        first_divergence(&reference.trace, &reversed.trace),
        None,
        "按成本和坐标稳定排序后，BaseList 声明顺序不应再改变规则 AI 轨迹"
    );
}
