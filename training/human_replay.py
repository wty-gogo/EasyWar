"""读取真人试玩 JSONL，并按整局生成可训练的人类策略样本。"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import easywar_rl
import numpy as np


@dataclass(frozen=True)
class HumanSample:
    session_id: str
    map_name: str
    difficulty: str
    player_won: bool
    values: np.ndarray
    mask: np.ndarray
    bases: np.ndarray
    action: int
    is_wait: bool


@dataclass(frozen=True)
class HumanReplay:
    path: Path
    session_id: str
    map_name: str
    difficulty: str
    player_won: bool
    completed: bool
    decisions: tuple[dict[str, object], ...]


def telemetry_paths(inputs: list[Path]) -> list[Path]:
    paths = [
        candidate
        for path in inputs
        for candidate in (
            sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
        )
    ]
    return list(dict.fromkeys(path.resolve() for path in paths))


def load_replay(path: Path) -> HumanReplay:
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    starts = [event for event in events if event.get("event") == "session_started"]
    terminals = [event for event in events if event.get("event") == "session_ended"]
    if len(starts) != 1:
        raise ValueError(f"{path} 必须且只能包含一个 session_started")
    start = starts[0]
    if int(start.get("schema_version", -1)) != 1:
        raise ValueError(f"{path} 的埋点 schema 不受支持")
    terminal = terminals[-1] if terminals else None
    return HumanReplay(
        path=path,
        session_id=str(start["session_id"]),
        map_name=str(start["map"]),
        difficulty=str(start["difficulty"]),
        player_won=bool(terminal and terminal.get("player_won", False)),
        completed=terminal is not None,
        decisions=tuple(
            event
            for event in events
            if event.get("event") == "decision"
            and event.get("actor_role") == "player"
        ),
    )


def dense_observation(record: dict[str, object]) -> np.ndarray:
    cells = easywar_rl.MAX_HEIGHT * easywar_rl.MAX_WIDTH
    size = easywar_rl.OBSERVATION_CHANNELS * cells
    values = np.zeros(size, dtype=np.float32)
    for raw_index, raw_value in record["sparse_values"]:
        index = int(raw_index)
        value = float(raw_value)
        if not 0 <= index < size or not math.isfinite(value):
            raise ValueError(f"观察稀疏项非法：{raw_index}/{raw_value}")
        values[index] = value
    return values.reshape(
        easywar_rl.OBSERVATION_CHANNELS,
        easywar_rl.MAX_HEIGHT,
        easywar_rl.MAX_WIDTH,
    )


def action_mask(record: dict[str, object], mode: str) -> np.ndarray:
    field = {
        "player": "actor_valid_actions",
        "tactical": "tactical_candidate_actions",
    }.get(mode)
    if field is None:
        raise ValueError(f"未知动作掩码模式：{mode}")
    mask = np.zeros(easywar_rl.ACTION_COUNT, dtype=np.bool_)
    indices = np.asarray(record[field], dtype=np.int64)
    if indices.size and (indices.min() < 0 or indices.max() >= easywar_rl.ACTION_COUNT):
        raise ValueError("埋点包含越界动作")
    mask[indices] = True
    return mask


def replay_samples(
    replay: HumanReplay,
    mask_mode: str = "player",
) -> tuple[HumanSample, ...]:
    samples: list[HumanSample] = []
    for decision in replay.decisions:
        observation = decision["observation"]
        values = dense_observation(observation)
        mask = action_mask(observation, mask_mode)
        bases = np.asarray(observation["base_cells"], dtype=np.int64)
        if bases.shape != (easywar_rl.MAX_BASES,):
            raise ValueError(f"{replay.path} 的据点槽位形状错误：{bases.shape}")
        for action_record in decision["actions"]:
            action = int(action_record["action_id"])
            if not 0 <= action < easywar_rl.ACTION_COUNT:
                raise ValueError(f"{replay.path} 包含越界动作 {action}")
            if not bool(action_record.get("actor_legal", False)):
                raise ValueError(f"{replay.path} 包含玩家规则下非法的已执行动作 {action}")
            if not mask[action]:
                continue
            samples.append(
                HumanSample(
                    session_id=replay.session_id,
                    map_name=replay.map_name,
                    difficulty=replay.difficulty,
                    player_won=replay.player_won,
                    values=values,
                    mask=mask,
                    bases=bases,
                    action=action,
                    is_wait=action == 0,
                )
            )
    return tuple(samples)


def downsample_waits(
    samples: tuple[HumanSample, ...], wait_to_command_ratio: float
) -> tuple[HumanSample, ...]:
    if wait_to_command_ratio < 0.0:
        raise ValueError("等待样本比例不能小于 0")
    commands = [sample for sample in samples if not sample.is_wait]
    waits = [sample for sample in samples if sample.is_wait]
    maximum_waits = max(1, math.ceil(len(commands) * wait_to_command_ratio))
    if len(waits) <= maximum_waits:
        return samples
    positions = np.linspace(0, len(waits) - 1, maximum_waits, dtype=np.int64)
    selected_waits = {id(waits[index]) for index in positions.tolist()}
    return tuple(
        sample
        for sample in samples
        if not sample.is_wait or id(sample) in selected_waits
    )


def load_human_samples(
    inputs: list[Path],
    mask_mode: str = "player",
    wait_to_command_ratio: float = 2.0,
    completed_only: bool = True,
) -> tuple[tuple[HumanReplay, ...], tuple[HumanSample, ...]]:
    replays = tuple(load_replay(path) for path in telemetry_paths(inputs))
    selected = tuple(
        replay for replay in replays if replay.completed or not completed_only
    )
    samples = tuple(
        sample
        for replay in selected
        for sample in downsample_waits(
            replay_samples(replay, mask_mode), wait_to_command_ratio
        )
    )
    return selected, samples


def split_sessions(
    replays: tuple[HumanReplay, ...], validation_fraction: float, seed: int
) -> tuple[set[str], set[str]]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("验证比例必须位于 0 到 1 之间")
    ids = np.asarray([replay.session_id for replay in replays], dtype=object)
    if len(ids) < 2 or validation_fraction == 0.0:
        return set(ids.tolist()), set()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    validation_count = min(len(ids) - 1, max(1, round(len(ids) * validation_fraction)))
    return set(ids[validation_count:].tolist()), set(ids[:validation_count].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 EasyWar 真人试玩埋点")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--mask-mode", choices=["player", "tactical"], default="player")
    parser.add_argument("--wait-to-command-ratio", type=float, default=2.0)
    args = parser.parse_args()
    replays, samples = load_human_samples(
        args.inputs, args.mask_mode, args.wait_to_command_ratio
    )
    commands = sum(not sample.is_wait for sample in samples)
    print(
        f"完整对局 {len(replays)} | 胜局 {sum(replay.player_won for replay in replays)} | "
        f"训练样本 {len(samples)} | 指令 {commands} | 等待 {len(samples) - commands}"
    )
    for replay in replays:
        print(
            f"{replay.path.name} | {replay.map_name} | {replay.difficulty} | "
            f"{'玩家胜' if replay.player_won else '玩家负'}"
        )


if __name__ == "__main__":
    main()
