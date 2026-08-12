from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import easywar_rl
import torch

from human_replay import (
    action_mask,
    dense_observation,
    load_human_samples,
    load_replay,
    replay_samples,
    split_sessions,
)
from runtime import build_model
from train_human_shadow import train


def observation() -> dict[str, object]:
    return {
        "width": 17,
        "height": 13,
        "base_cells": [0] + [-1] * (easywar_rl.MAX_BASES - 1),
        "sparse_values": [[0, 1.0], [221, 0.5]],
        "actor_valid_actions": [0, 1],
        "tactical_candidate_actions": [0],
    }


def events(session: str, command: bool = True) -> list[dict[str, object]]:
    result = [
        {
            "schema_version": 1,
            "event": "session_started",
            "session_id": session,
            "map": "dual_ladder_1v1.toml",
            "difficulty": "神经模型 V11·自博弈",
        }
    ]
    if command:
        result.append(
            {
                "schema_version": 1,
                "event": "decision",
                "session_id": session,
                "actor_role": "player",
                "decision_kind": "player_command",
                "actions": [{"action_id": 1, "actor_legal": True}],
                "observation": observation(),
            }
        )
    result.extend(
        [
            {
                "schema_version": 1,
                "event": "decision",
                "session_id": session,
                "actor_role": "player",
                "decision_kind": "periodic_wait",
                "actions": [{"action_id": 0, "actor_legal": True}],
                "observation": observation(),
            },
            {
                "schema_version": 1,
                "event": "session_ended",
                "session_id": session,
                "player_won": True,
            },
        ]
    )
    return result


def write_replay(path: Path, session: str) -> None:
    path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events(session))
        + "\n",
        encoding="utf-8",
    )


class HumanReplayTests(unittest.TestCase):
    def test_restores_sparse_observation_and_both_masks(self) -> None:
        restored = dense_observation(observation())
        self.assertEqual(restored.shape, (easywar_rl.OBSERVATION_CHANNELS, 13, 17))
        self.assertEqual(restored.flat[0], 1.0)
        self.assertEqual(restored.flat[221], 0.5)
        self.assertEqual(action_mask(observation(), "player").nonzero()[0].tolist(), [0, 1])
        self.assertEqual(action_mask(observation(), "tactical").nonzero()[0].tolist(), [0])

    def test_only_player_legal_samples_enter_shadow_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "match.jsonl"
            write_replay(path, "one")
            replay = load_replay(path)
            player_samples = replay_samples(replay, "player")
            tactical_samples = replay_samples(replay, "tactical")
        self.assertEqual([sample.action for sample in player_samples], [1, 0])
        self.assertEqual([sample.action for sample in tactical_samples], [0])

    def test_directory_loading_and_split_keep_whole_sessions_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_replay(root / "one.jsonl", "one")
            write_replay(root / "two.jsonl", "two")
            replays, samples = load_human_samples([root])
            training, validation = split_sessions(replays, 0.5, 7)
        self.assertEqual(len(replays), 2)
        self.assertEqual(len(samples), 4)
        self.assertFalse(training & validation)
        self.assertEqual(training | validation, {"one", "two"})

    def test_human_shadow_checkpoint_can_train_from_whole_session_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_replay(root / "one.jsonl", "one")
            write_replay(root / "two.jsonl", "two")
            initial = root / "initial.pt"
            output = root / "shadow.pt"
            torch.save({"model": build_model(torch.device("cpu")).state_dict()}, initial)
            result, training, validation = train(
                Namespace(
                    inputs=[root],
                    initialize_from=initial,
                    checkpoint=output,
                    mask_mode="player",
                    wait_to_command_ratio=1.0,
                    validation_fraction=0.5,
                    epochs=1,
                    batch_size=4,
                    learning_rate=1e-4,
                    command_weight=2.0,
                    seed=7,
                    device="cpu",
                )
            )
            checkpoint = torch.load(output, map_location="cpu", weights_only=True)
        self.assertEqual(result, output)
        self.assertEqual(training.samples, 2)
        self.assertEqual(validation.samples, 2)
        self.assertEqual(
            checkpoint["training_state"]["source"], "human_telemetry_shadow"
        )


if __name__ == "__main__":
    unittest.main()
