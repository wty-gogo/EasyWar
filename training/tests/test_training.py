from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch
import easywar_rl

from behavior import (
    episode_behavior_from_dict,
    finish_behavior,
    initial_behavior_state,
    record_behavior,
    summarize_behaviors,
)
from behavior_report import render_behavior_report
from evaluation import (
    EvaluationResult,
    ReplayTrace,
    aggregate_results,
    model_selection_key,
    passes_completion_gate,
    passes_validation_gate,
)
from runtime import (
    build_model,
    checkpoint_strategy_count,
    load_model_weights,
    repository_root,
    training_map_names,
)
from train_ppo import (
    action_target_distribution,
    categorical_kl,
    configure_strategy_adapter_training,
    environment_strategy_ids,
    load_checkpoint,
    run_validation,
    save_checkpoint,
    strategy_js_divergence,
    strategy_specialization_score,
    training_factor_label,
    validate_args,
    validate_resume_checkpoint,
)


class EvaluationResultTests(unittest.TestCase):
    def test_reports_completion_and_total_sample_rates_separately(self) -> None:
        result = EvaluationResult(
            map_name="fixture.toml",
            opponent="easy",
            episodes=10,
            seed=100,
            outcomes={"Won": 3, "Lost": 2, "Stalemate": 5},
        )
        self.assertEqual(result.completion_rate, 0.5)
        self.assertEqual(result.completed_win_rate, 0.6)
        self.assertEqual(result.overall_win_rate, 0.3)

    def test_aggregate_keeps_all_terminal_categories(self) -> None:
        results = [
            EvaluationResult("a", "normal", 4, 10, {"Won": 1, "Lost": 3}),
            EvaluationResult(
                "b", "normal", 4, 20, {"Won": 2, "CycleDetected": 2}
            ),
        ]
        aggregate = aggregate_results(results)
        self.assertEqual(aggregate.episodes, 8)
        self.assertEqual(
            aggregate.outcomes, {"Won": 3, "Lost": 3, "CycleDetected": 2}
        )

    def test_model_selection_prioritizes_total_sample_wins_after_gate(self) -> None:
        lower_wins = EvaluationResult("a", "normal", 10, 1, {"Won": 4, "Lost": 6})
        higher_wins = EvaluationResult(
            "a", "normal", 10, 1, {"Won": 5, "CycleDetected": 5}
        )
        self.assertGreater(
            model_selection_key(higher_wins), model_selection_key(lower_wins)
        )

    def test_completion_gate_applies_to_every_training_map(self) -> None:
        stable = EvaluationResult("a", "normal", 10, 1, {"Won": 5, "Lost": 5})
        unstable = EvaluationResult(
            "b", "normal", 10, 1, {"Won": 2, "CycleDetected": 8}
        )
        self.assertFalse(passes_completion_gate([stable, unstable], 0.8))
        self.assertTrue(passes_completion_gate([stable], 0.8))

    def test_validation_gate_applies_both_rates_to_every_factor(self) -> None:
        qualified = EvaluationResult("a", "easy", 10, 1, {"Won": 5, "Lost": 5})
        low_completion = EvaluationResult(
            "a", "hard", 10, 2, {"Won": 5, "CycleDetected": 5}
        )
        low_win_rate = EvaluationResult(
            "b", "normal", 10, 3, {"Won": 4, "Lost": 6}
        )
        self.assertTrue(passes_validation_gate([qualified], 0.8, 0.5))
        self.assertFalse(
            passes_validation_gate([qualified, low_completion], 0.8, 0.5)
        )
        self.assertFalse(
            passes_validation_gate([qualified, low_win_rate], 0.8, 0.5)
        )

    def test_replay_trace_is_kept_in_structured_result(self) -> None:
        replay = ReplayTrace(
            map_name="fixture.toml",
            opponent="easy",
            seed=3,
            outcome="Won",
            variant=1,
            seat_transform="vertical",
            actions=(0, 12),
            decisions=2,
        )
        result = EvaluationResult(
            "fixture.toml", "easy", 1, 3, {"Won": 1}, (replay,)
        )
        self.assertEqual(
            result.as_dict()["representative_replays"][0]["actions"], [0, 12]
        )


class BehaviorMetricsTests(unittest.TestCase):
    @staticmethod
    def observation() -> torch.Tensor:
        observation = torch.zeros((17, 13, 17))
        observation[0, 0, :4] = 1.0
        observation[2, 0, 0] = 1.0
        observation[3, 0, 0] = 1.0
        return observation

    @staticmethod
    def base_cells() -> torch.Tensor:
        cells = torch.full((16,), -1, dtype=torch.long)
        cells[0] = 0
        return cells

    @staticmethod
    def set_stream(target: int) -> int:
        return 1 + target

    def test_tracks_opening_retarget_and_counterattack_from_visible_state(self) -> None:
        state = initial_behavior_state("fixture.toml", "hard", 0, "identity")
        opening = self.observation()
        opening[3, 0, 2] = 1.0
        opening[1, 0, 3] = 1.0
        opening[5, 0, 3] = 1.0
        state = record_behavior(
            state, opening, self.base_cells(), self.set_stream(3)
        )

        counterattack = self.observation()
        counterattack[13, 0, 0] = 1.0
        counterattack[1, 0, 2] = 1.0
        counterattack[4, 0, 2] = 1.0
        state = record_behavior(
            state, counterattack, self.base_cells(), self.set_stream(2)
        )
        state = record_behavior(
            state, counterattack, self.base_cells(), self.set_stream(2)
        )
        episode = finish_behavior(state, "Won")

        self.assertEqual(episode.first_command_decision, 1)
        self.assertEqual(episode.first_offensive_decision, 1)
        self.assertEqual(episode.first_offensive_target.grid, 3)
        self.assertEqual(episode.retarget_count, 1)
        self.assertEqual(episode.counterattack_count, 1)
        self.assertEqual(episode.command_count, 3)
        self.assertEqual(episode.effective_command_count, 2)
        self.assertEqual(episode.redundant_command_count, 1)
        self.assertEqual(episode.distinct_offensive_targets, 2)
        self.assertEqual(episode.distinct_counterattack_targets, 1)

    def test_seat_transform_normalizes_mirrored_opening_target(self) -> None:
        observation = self.observation()
        observation[1, 0, 3] = 1.0
        observation[5, 0, 3] = 1.0
        state = initial_behavior_state("fixture.toml", "easy", 1, "vertical")
        episode = finish_behavior(
            record_behavior(
                state, observation, self.base_cells(), self.set_stream(3)
            ),
            "Won",
        )
        self.assertEqual(episode.first_offensive_target.grid, 0)
        self.assertEqual(episode.seat_transform, "vertical")

    def test_summary_preserves_seat_submit_factors_and_opening_entropy(self) -> None:
        episodes = []
        for variant, target in [(0, 2), (3, 3)]:
            observation = self.observation()
            observation[1, 0, target] = 1.0
            observation[5, 0, target] = 1.0
            state = initial_behavior_state(
                "fixture.toml",
                "normal",
                variant,
                "vertical" if variant % 2 else "identity",
            )
            episodes.append(
                finish_behavior(
                    record_behavior(
                        state,
                        observation,
                        self.base_cells(),
                        self.set_stream(target),
                    ),
                    "Won",
                )
            )
        summary = summarize_behaviors(tuple(episodes))
        self.assertEqual(summary["unique_opening_targets"], 2)
        self.assertAlmostEqual(summary["opening_target_entropy"], 0.693147, places=5)
        self.assertEqual(
            set(summary["by_map_opponent_strategy_seat_submit"]),
            {
                "fixture.toml|normal|策略0|identity|learner_first",
                "fixture.toml|normal|策略0|vertical|opponent_first",
            },
        )
        self.assertEqual(summary["factors_with_multiple_openings"], 0)
        self.assertEqual(
            episode_behavior_from_dict(episodes[0].as_dict()), episodes[0]
        )

    def test_human_report_explains_results_without_raw_field_knowledge(self) -> None:
        payload = {
            "results": [
                {
                    "map": "fixture.toml",
                    "opponent": "hard",
                    "strategy_id": 1,
                    "episodes": 2,
                    "outcomes": {"Won": 1, "Lost": 1},
                    "completion_rate": 1.0,
                    "overall_win_rate": 0.5,
                    "behavior_summary": {
                        "retarget_rate": 0.1,
                        "counterattack_episode_rate": 0.5,
                        "mean_distinct_offensive_targets": 3.0,
                    },
                    "behavior_episodes": [
                        {
                            "first_offensive_target": {
                                "x": 2,
                                "y": 3,
                                "owner": "neutral",
                                "cell_kind": "base",
                            }
                        }
                    ],
                }
            ]
        }
        report = render_behavior_report([payload], "测试报告", 0.8, 0.5)
        self.assertIn("一句话结论", report)
        self.assertIn("中立据点(2,3)", report)
        self.assertIn("不评价地图是否平衡或有趣", report)

    def test_human_report_does_not_treat_tied_openings_as_differentiation(self) -> None:
        def result(strategy_id: int, targets: list[tuple[int, int]]) -> dict[str, object]:
            return {
                "map": "fixture.toml",
                "opponent": "normal",
                "strategy_id": strategy_id,
                "episodes": len(targets),
                "outcomes": {"Won": len(targets)},
                "completion_rate": 1.0,
                "overall_win_rate": 1.0,
                "behavior_summary": {},
                "behavior_episodes": [
                    {
                        "first_offensive_target": {
                            "x": x,
                            "y": y,
                            "owner": "neutral",
                            "cell_kind": "linked",
                        }
                    }
                    for x, y in targets
                ],
            }

        report = render_behavior_report(
            [{"results": [result(0, [(1, 1), (2, 2)]), result(1, [(2, 2), (1, 1)])]}],
            "平票测试",
            0.8,
            0.5,
        )
        self.assertIn("0/1 个地图与难度组合", report)
        self.assertIn("不能宣称多策略训练成功", report)


class TrainingBoundaryTests(unittest.TestCase):
    def test_h_and_held_out_maps_are_never_part_of_training_phase(self) -> None:
        self.assertEqual(
            training_map_names("main"),
            [
                "dual_ladder_1v1.toml",
                "braided_rings_1v1.toml",
            ],
        )
        self.assertNotIn(
            "layered_triangle_duel_1v1.toml", training_map_names("main")
        )
        self.assertNotIn("h_1v1.toml", training_map_names("main"))
        self.assertNotIn("ring_chord_1v1.toml", training_map_names("main"))
        with self.assertRaisesRegex(ValueError, "H 图已退出训练课程"):
            training_map_names("warmup")

    def test_historical_pool_requires_orthogonal_batch_size(self) -> None:
        args = Namespace(
            phase="main",
            historical_opponent=[Path("opponent.pt")],
            rule_opponents=["normal"],
            num_envs=6,
            rollout_steps=8,
            epochs=1,
            minibatch_size=8,
            validation_episodes=8,
            validation_num_envs=8,
            dagger_model_prob=0.5,
            updates=1,
            imitation_updates=0,
            checkpoint_every=1,
            validation_every=1,
            early_stop_patience=0,
            anchor_checkpoint=None,
            anchor_kl_coef=0.0,
            strategy_count=1,
            strategy_diversity_coef=0.0,
            strategy_specialization_coef=0.0,
            strategy_adapter_only=False,
            strategy_diversity_samples=8,
            minimum_validation_completion=0.8,
            minimum_validation_win_rate=0.5,
        )
        with self.assertRaisesRegex(ValueError, "正交覆盖"):
            validate_args(args)

    def test_checkpoint_round_trip_keeps_optimizer_and_counters(self) -> None:
        device = torch.device("cpu")
        model = build_model(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        args = Namespace(
            phase="main",
            rule_opponents=["normal"],
            teacher="normal",
            seed=7,
            resume=None,
            initialize_from=None,
            historical_opponent=[],
            anchor_checkpoint=None,
            anchor_kl_coef=0.0,
            strategy_count=1,
            strategy_diversity_coef=0.0,
            strategy_specialization_coef=0.0,
            strategy_adapter_only=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, args, model, optimizer, 99, 12, 34, (0.5, 0.8))
            checkpoint = load_checkpoint(path, device)

        self.assertIn("optimizer", checkpoint)
        self.assertEqual(checkpoint["training_state"]["next_seed"], 99)
        self.assertEqual(
            checkpoint["training_state"]["imitation_updates_completed"], 12
        )
        self.assertEqual(checkpoint["training_state"]["ppo_updates_completed"], 34)

    def test_resume_rejects_silent_opponent_change(self) -> None:
        device = torch.device("cpu")
        model = build_model(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        saved_args = Namespace(
            phase="main",
            rule_opponents=["easy"],
            teacher="normal",
            seed=7,
            resume=None,
            initialize_from=None,
            historical_opponent=[],
            anchor_checkpoint=None,
            anchor_kl_coef=0.0,
            strategy_count=1,
            strategy_diversity_coef=0.0,
            strategy_specialization_coef=0.0,
            strategy_adapter_only=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, saved_args, model, optimizer, 99, 12, 34, (0.5, 0.8))
            checkpoint = load_checkpoint(path, device)
        resumed_args = Namespace(
            phase="main",
            rule_opponents=["normal"],
            teacher="normal",
            imitation_updates=12,
            updates=34,
            historical_opponent=[],
            anchor_checkpoint=None,
            anchor_kl_coef=0.0,
            strategy_count=1,
            strategy_diversity_coef=0.0,
            strategy_specialization_coef=0.0,
            strategy_adapter_only=False,
        )
        with self.assertRaisesRegex(ValueError, "相同规则对手池"):
            validate_resume_checkpoint(checkpoint, resumed_args)

    def test_resume_rejects_historical_opponent_pool_change(self) -> None:
        device = torch.device("cpu")
        model = build_model(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        saved_args = Namespace(
            phase="main",
            rule_opponents=["normal"],
            teacher="normal",
            seed=7,
            resume=None,
            initialize_from=None,
            historical_opponent=[],
            anchor_checkpoint=None,
            anchor_kl_coef=0.0,
            strategy_count=1,
            strategy_diversity_coef=0.0,
            strategy_specialization_coef=0.0,
            strategy_adapter_only=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, saved_args, model, optimizer, 99, 12, 34, (0.5, 0.8))
            checkpoint = load_checkpoint(path, device)
            resumed_args = Namespace(
                phase="main",
                rule_opponents=["normal"],
                teacher="normal",
                imitation_updates=12,
                updates=34,
                historical_opponent=[Path(directory) / "other.pt"],
                anchor_checkpoint=None,
                anchor_kl_coef=0.0,
                strategy_count=1,
                strategy_diversity_coef=0.0,
                strategy_specialization_coef=0.0,
                strategy_adapter_only=False,
            )
            with self.assertRaisesRegex(ValueError, "历史模型对手池"):
                validate_resume_checkpoint(checkpoint, resumed_args)

    def test_rule_difficulties_are_visible_in_training_factor_labels(self) -> None:
        args = Namespace(
            phase="main",
            rule_opponents=["easy", "normal", "hard"],
            historical_opponent=[],
            strategy_count=1,
        )
        labels = [training_factor_label(args, index) for index in range(24)]
        for map_name in training_map_names("main"):
            for opponent in args.rule_opponents:
                self.assertTrue(
                    any(map_name in label and opponent in label for label in labels)
                )
        self.assertTrue(any("学习者先提交" in label for label in labels))
        self.assertTrue(any("对手先提交" in label for label in labels))

    def test_action_target_distribution_uses_visible_target_ownership(self) -> None:
        observations = torch.zeros((3, 17, 13, 17))
        cells = 13 * 17
        targets = [5, 6, 7]
        observations[0, 4].flatten()[targets[0]] = 1.0
        observations[1, 5].flatten()[targets[1]] = 1.0
        observations[2, 3].flatten()[targets[2]] = 1.0
        actions = torch.tensor([1 + target for target in targets])
        result = action_target_distribution(actions, observations)
        self.assertAlmostEqual(result["enemy_target_rate"], 1 / 3)
        self.assertAlmostEqual(result["neutral_target_rate"], 1 / 3)
        self.assertAlmostEqual(result["friendly_target_rate"], 1 / 3)

    def test_anchor_kl_is_zero_for_same_policy_and_positive_after_change(self) -> None:
        reference = torch.tensor([[1.0, 0.0, float("-inf")]])
        same = categorical_kl(reference, reference)
        changed = categorical_kl(reference, torch.tensor([[0.0, 1.0, float("-inf")]]))
        self.assertAlmostEqual(same.item(), 0.0)
        self.assertGreater(changed.item(), 0.0)

    def test_legacy_checkpoint_preserves_strategy_zero_and_initializes_new_strategies(self) -> None:
        device = torch.device("cpu")
        legacy = build_model(device)
        legacy_state = {
            key: value
            for key, value in legacy.state_dict().items()
            if key != "strategy_embedding.weight"
        }
        checkpoint = {"model": legacy_state}
        expanded = build_model(device, strategy_count=4)
        load_model_weights(expanded, checkpoint)
        self.assertEqual(checkpoint_strategy_count(checkpoint), 1)
        self.assertEqual(tuple(expanded.strategy_embedding.weight.shape), (4, 64))
        self.assertTrue(torch.equal(
            expanded.strategy_embedding.weight[0],
            torch.zeros_like(expanded.strategy_embedding.weight[0]),
        ))
        self.assertTrue(
            all(
                expanded.strategy_embedding.weight[strategy].norm() > 0.0
                for strategy in range(1, 4)
            )
        )
        self.assertEqual(
            len({
                tuple(expanded.strategy_embedding.weight[strategy].tolist())
                for strategy in range(4)
            }),
            4,
        )
        observations = torch.zeros((4, 17, 13, 17))
        bases = torch.zeros((4, 16), dtype=torch.long)
        masks = torch.ones((4, 3553), dtype=torch.bool)
        actions, _, _ = expanded.act(
            observations,
            bases,
            masks,
            torch.arange(4),
            deterministic=True,
        )
        self.assertEqual(tuple(actions.shape), (4,))

    def test_strategy_js_detects_controlled_policy_difference(self) -> None:
        model = build_model(torch.device("cpu"), strategy_count=2)
        observations = torch.randn((2, 17, 13, 17))
        bases = torch.zeros((2, 16), dtype=torch.long)
        masks = torch.zeros((2, 3553), dtype=torch.bool)
        masks[:, :100] = True
        identical = strategy_js_divergence(
            model, observations, bases, masks, 2, 2
        )
        with torch.no_grad():
            model.strategy_embedding.weight[1].fill_(0.5)
        different = strategy_js_divergence(
            model, observations, bases, masks, 2, 2
        )
        different.backward()
        self.assertAlmostEqual(identical.item(), 0.0, places=6)
        self.assertGreater(different.item(), 0.0)
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_strategy_specialization_rewards_interpretable_target_preferences(self) -> None:
        observations = torch.zeros((3, 17, 13, 17))
        preferred_target = 10
        other_target = 11
        observations[0, 4].flatten()[preferred_target] = 1.0
        observations[1, 5].flatten()[preferred_target] = 1.0
        observations[2, 1].flatten()[preferred_target] = 1.0
        masks = torch.zeros((3, easywar_rl.ACTION_COUNT), dtype=torch.bool)
        masks[:, 0] = True
        masks[:, 1 + preferred_target] = True
        masks[:, 1 + other_target] = True
        aligned_logits = torch.full((3, easywar_rl.ACTION_COUNT), -100.0)
        aligned_logits[:, 0] = 0.0
        aligned_logits[:, 1 + preferred_target] = 3.0
        aligned_logits[:, 1 + other_target] = -3.0
        aligned_logits.requires_grad_()
        opposite_logits = aligned_logits.detach().clone()
        opposite_logits[:, 1 + preferred_target] = -3.0
        opposite_logits[:, 1 + other_target] = 3.0
        strategy_ids = torch.tensor([1, 2, 3])
        aligned = strategy_specialization_score(
            aligned_logits, observations, masks, strategy_ids
        )
        opposite = strategy_specialization_score(
            opposite_logits, observations, masks, strategy_ids
        )
        aligned.backward()
        self.assertGreater(aligned.item(), opposite.item())
        self.assertTrue(torch.isfinite(aligned_logits.grad).all())

    def test_strategy_adapter_only_updates_nonbaseline_strategy_rows(self) -> None:
        model = build_model(torch.device("cpu"), strategy_count=4)
        parameters = configure_strategy_adapter_training(model)
        self.assertEqual(len(parameters), 1)
        self.assertIs(parameters[0], model.strategy_embedding.weight)
        loss = model.strategy_embedding.weight.sum()
        loss.backward()
        gradient = model.strategy_embedding.weight.grad
        self.assertTrue(torch.equal(gradient[0], torch.zeros_like(gradient[0])))
        self.assertTrue(torch.equal(gradient[1:], torch.ones_like(gradient[1:])))
        self.assertTrue(
            all(
                not parameter.requires_grad
                for name, parameter in model.named_parameters()
                if name != "strategy_embedding.weight"
            )
        )

    def test_strategy_ids_are_orthogonal_blocks_after_all_engine_variants(self) -> None:
        args = Namespace(
            phase="main",
            historical_opponent=[],
            rule_opponents=["easy", "normal", "hard"],
            num_envs=96,
            strategy_count=4,
        )
        strategies = environment_strategy_ids(args, torch.device("cpu"))
        self.assertEqual(strategies.tolist(), [style for style in range(4) for _ in range(24)])

    def test_anchor_coefficient_requires_checkpoint(self) -> None:
        args = Namespace(
            phase="main",
            historical_opponent=[],
            rule_opponents=["easy", "normal", "hard"],
            num_envs=12,
            rollout_steps=8,
            epochs=1,
            minibatch_size=8,
            validation_episodes=8,
            validation_num_envs=8,
            dagger_model_prob=0.5,
            updates=1,
            imitation_updates=0,
            checkpoint_every=1,
            validation_every=1,
            early_stop_patience=0,
            anchor_checkpoint=None,
            anchor_kl_coef=0.05,
            strategy_count=1,
            strategy_diversity_coef=0.0,
            strategy_specialization_coef=0.0,
            strategy_adapter_only=False,
            strategy_diversity_samples=8,
            minimum_validation_completion=0.8,
            minimum_validation_win_rate=0.5,
        )
        with self.assertRaisesRegex(ValueError, "anchor-checkpoint"):
            validate_args(args)

    def test_validation_covers_every_map_and_rule_difficulty(self) -> None:
        args = Namespace(
            phase="main",
            validation_opponents=["easy", "normal", "hard"],
            validation_episodes=10,
            validation_num_envs=2,
            validation_seed=100,
            threads=1,
            minimum_validation_completion=0.8,
            minimum_validation_win_rate=0.5,
            strategy_count=2,
        )
        calls: list[tuple[str, str, int, int]] = []

        def fake_evaluate_model(**kwargs: object) -> EvaluationResult:
            map_name = str(kwargs["map_name"])
            opponent = str(kwargs["opponent"])
            seed = int(kwargs["seed"])
            strategy_id = int(kwargs["strategy_id"])
            calls.append((map_name, opponent, seed, strategy_id))
            return EvaluationResult(
                map_name,
                opponent,
                10,
                seed,
                {"Won": 5, "Lost": 5},
                strategy_id=strategy_id,
            )

        with patch("train_ppo.evaluate_model", side_effect=fake_evaluate_model):
            results, aggregate, _, eligible = run_validation(
                args, build_model(torch.device("cpu")), torch.device("cpu")
            )

        self.assertEqual(len(results), 12)
        self.assertEqual(aggregate["opponent"], "mixed")
        self.assertTrue(eligible)
        self.assertEqual(
            {(map_name, opponent, strategy) for map_name, opponent, _, strategy in calls},
            {
                (map_name, opponent, strategy)
                for map_name in training_map_names("main")
                for opponent in args.validation_opponents
                for strategy in range(args.strategy_count)
            },
        )
        self.assertEqual(len({seed for _, _, seed, _ in calls}), 12)


class ExternalBatchIntegrationTests(unittest.TestCase):
    def test_external_models_can_submit_both_actions_in_parallel(self) -> None:
        import easywar_rl

        root = repository_root()
        environment = easywar_rl.BatchEnv(
            [str(root / "assets" / "maps" / "dual_ladder_1v1.toml")],
            str(root / "assets" / "subjects"),
            4,
            seed=77,
            opponent="normal",
            map_transforms=["vertical"],
            external_opponent=True,
            alternate_submit_order=True,
        )
        opponent = environment.observe_opponents()
        self.assertTrue(all(mask[0] for mask in opponent.action_masks))
        transition = environment.step_external([0] * 4, [0] * 4, 4)
        self.assertEqual(transition.decisions, [1] * 4)
        self.assertTrue(all(transition.opponent_action_applied))


if __name__ == "__main__":
    unittest.main()
