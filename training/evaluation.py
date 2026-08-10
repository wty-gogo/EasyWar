from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import easywar_rl
import torch

from behavior import (
    BehaviorState,
    EpisodeBehavior,
    finish_behavior,
    initial_behavior_state,
    record_behavior,
    summarize_behaviors,
)
from model import EasyWarActorCritic
from runtime import map_transform, repository_root, to_tensors


@dataclass(frozen=True)
class ReplayTrace:
    map_name: str
    opponent: str
    seed: int
    outcome: str
    variant: int
    seat_transform: str
    actions: tuple[int, ...]
    decisions: int
    strategy_id: int = 0
    behavior: EpisodeBehavior | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "map": self.map_name,
            "opponent": self.opponent,
            "seed": self.seed,
            "outcome": self.outcome,
            "variant": self.variant,
            "seat_transform": self.seat_transform,
            "actions": list(self.actions),
            "decisions": self.decisions,
            "strategy_id": self.strategy_id,
            "behavior": self.behavior.as_dict() if self.behavior is not None else None,
        }


@dataclass(frozen=True)
class EvaluationResult:
    map_name: str
    opponent: str
    episodes: int
    seed: int
    outcomes: dict[str, int]
    representative_replays: tuple[ReplayTrace, ...] = field(default_factory=tuple)
    behavior_episodes: tuple[EpisodeBehavior, ...] = field(default_factory=tuple)
    strategy_id: int = 0

    @property
    def completed(self) -> int:
        return self.outcomes.get("Won", 0) + self.outcomes.get("Lost", 0)

    @property
    def completion_rate(self) -> float:
        return self.completed / self.episodes if self.episodes else 0.0

    @property
    def completed_win_rate(self) -> float:
        return self.outcomes.get("Won", 0) / self.completed if self.completed else 0.0

    @property
    def overall_win_rate(self) -> float:
        return self.outcomes.get("Won", 0) / self.episodes if self.episodes else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "map": self.map_name,
            "opponent": self.opponent,
            "strategy_id": self.strategy_id,
            "episodes": self.episodes,
            "seed": self.seed,
            "outcomes": self.outcomes,
            "completion_rate": self.completion_rate,
            "completed_win_rate": self.completed_win_rate,
            "overall_win_rate": self.overall_win_rate,
            "behavior_summary": summarize_behaviors(self.behavior_episodes),
            "behavior_episodes": [
                episode.as_dict() for episode in self.behavior_episodes
            ],
            "representative_replays": [
                replay.as_dict() for replay in self.representative_replays
            ],
        }


def aggregate_results(results: list[EvaluationResult]) -> EvaluationResult:
    outcomes = Counter[str]()
    for result in results:
        outcomes.update(result.outcomes)
    opponents = {result.opponent for result in results}
    strategies = {result.strategy_id for result in results}
    return EvaluationResult(
        map_name="aggregate",
        opponent=next(iter(opponents)) if len(opponents) == 1 else "mixed",
        episodes=sum(result.episodes for result in results),
        seed=min((result.seed for result in results), default=0),
        outcomes=dict(outcomes),
        representative_replays=tuple(
            replay
            for result in results
            for replay in result.representative_replays
        ),
        behavior_episodes=tuple(
            episode for result in results for episode in result.behavior_episodes
        ),
        strategy_id=next(iter(strategies)) if len(strategies) == 1 else -1,
    )


def model_selection_key(result: EvaluationResult) -> tuple[float, float]:
    """先最大化所有样本取胜率，再用正常完赛率打破平局。"""

    return result.overall_win_rate, result.completion_rate


def passes_completion_gate(
    results: list[EvaluationResult], minimum_completion_rate: float
) -> bool:
    return bool(results) and all(
        result.completion_rate >= minimum_completion_rate for result in results
    )


def passes_validation_gate(
    results: list[EvaluationResult],
    minimum_completion_rate: float,
    minimum_overall_win_rate: float,
) -> bool:
    """每个地图与难度因子都必须达到完赛率和全样本胜率门槛。"""

    return bool(results) and all(
        result.completion_rate >= minimum_completion_rate
        and result.overall_win_rate >= minimum_overall_win_rate
        for result in results
    )


def evaluate_model(
    model: EasyWarActorCritic,
    device: torch.device,
    map_name: str,
    opponent: str,
    episodes: int,
    num_envs: int,
    threads: int,
    seed: int,
    seat_transform: str | None = None,
    capture_replays: bool = False,
    strategy_id: int = 0,
) -> EvaluationResult:
    if episodes <= 0 or num_envs <= 0:
        raise ValueError("评测局数和并行环境数必须大于 0")
    root = repository_root()
    environment_count = min(num_envs, episodes)
    transform_name = seat_transform or map_transform(map_name)
    environment = easywar_rl.BatchEnv(
        [str(root / "assets" / "maps" / map_name)],
        str(root / "assets" / "subjects"),
        environment_count,
        seed=seed,
        opponent=opponent,
        map_transforms=[transform_name],
        alternate_seats=True,
    )
    current = to_tensors(environment.observe(), device)
    strategy_ids = torch.full(
        (environment_count,), strategy_id, dtype=torch.long, device=device
    )
    outcomes = Counter[str]()
    active_seeds = [seed + index for index in range(environment_count)]
    active_actions: list[list[int]] = [[] for _ in range(environment_count)]
    active_behaviors: list[BehaviorState] = [
        initial_behavior_state(
            map_name,
            opponent,
            index,
            transform_name if index % 2 else "identity",
            strategy_id,
        )
        for index in range(environment_count)
    ]
    behavior_episodes: list[EpisodeBehavior] = []
    representative: dict[str, ReplayTrace] = {}
    next_seed = seed + environment_count
    was_training = model.training
    model.eval()

    while sum(outcomes.values()) < episodes:
        with torch.no_grad():
            actions, _, _ = model.act(
                current.values,
                current.bases,
                current.masks,
                strategy_ids,
                deterministic=True,
            )
        action_ids = actions.cpu().tolist()
        current_values = current.values.detach().cpu()
        current_bases = current.bases.detach().cpu()
        active_behaviors = [
            record_behavior(
                behavior,
                current_values[index],
                current_bases[index],
                action_ids[index],
            )
            for index, behavior in enumerate(active_behaviors)
        ]
        transition = environment.step(action_ids, threads)
        if capture_replays:
            for index, action in enumerate(action_ids):
                active_actions[index].append(action)
        terminal_indices: list[int] = []
        for index, name in enumerate(transition.end_names):
            if name != "Ongoing":
                terminal_indices.append(index)
                if sum(outcomes.values()) < episodes:
                    episode_behavior = finish_behavior(active_behaviors[index], name)
                    behavior_episodes.append(episode_behavior)
                    outcomes[name] += 1
                    if capture_replays:
                        representative.setdefault(
                            name,
                            ReplayTrace(
                                map_name=map_name,
                                opponent=opponent,
                                seed=active_seeds[index],
                                outcome=name,
                                variant=index,
                                seat_transform=(
                                    transform_name if index % 2 == 1 else "identity"
                                ),
                                actions=tuple(active_actions[index]),
                                decisions=transition.decisions[index],
                                strategy_id=strategy_id,
                                behavior=episode_behavior,
                            ),
                        )
        if terminal_indices and sum(outcomes.values()) < episodes:
            seeds = list(range(next_seed, next_seed + len(terminal_indices)))
            next_seed += len(terminal_indices)
            if capture_replays:
                for index, reset_seed in zip(terminal_indices, seeds, strict=True):
                    active_seeds[index] = reset_seed
                    active_actions[index] = []
            for index in terminal_indices:
                active_behaviors[index] = initial_behavior_state(
                    map_name,
                    opponent,
                    index,
                    transform_name if index % 2 else "identity",
                    strategy_id,
                )
            batch = environment.reset_indices(terminal_indices, seeds)
        else:
            batch = transition
        current = to_tensors(batch, device)

    model.train(was_training)
    return EvaluationResult(
        map_name=map_name,
        opponent=opponent,
        episodes=episodes,
        seed=seed,
        outcomes=dict(outcomes),
        representative_replays=tuple(representative.values()),
        behavior_episodes=tuple(behavior_episodes),
        strategy_id=strategy_id,
    )


def verify_replay(replay: ReplayTrace, threads: int = 1) -> str:
    root = repository_root()
    environment = easywar_rl.BatchEnv(
        [str(root / "assets" / "maps" / replay.map_name)],
        str(root / "assets" / "subjects"),
        1,
        seed=replay.seed,
        opponent=replay.opponent,
        map_transforms=[replay.seat_transform],
        alternate_seats=True,
        variant_offset=replay.variant,
    )
    current = to_tensors(environment.observe(), torch.device("cpu"))
    behavior = initial_behavior_state(
        replay.map_name,
        replay.opponent,
        replay.variant,
        replay.seat_transform,
        replay.strategy_id,
    )
    actual = "Ongoing"
    decisions = 0
    for action in replay.actions:
        behavior = record_behavior(
            behavior,
            current.values[0],
            current.bases[0],
            action,
        )
        transition = environment.step([action], threads)
        actual = transition.end_names[0]
        decisions = transition.decisions[0]
        if actual != "Ongoing":
            break
        current = to_tensors(transition, torch.device("cpu"))
    if actual != replay.outcome or decisions != replay.decisions:
        raise ValueError(
            f"回放不一致：期望 {replay.outcome}/{replay.decisions}，"
            f"实际 {actual}/{decisions}"
        )
    if replay.behavior is not None:
        actual_behavior = finish_behavior(behavior, actual)
        if actual_behavior != replay.behavior:
            raise ValueError("回放行为指标不一致")
    return actual


def format_result(result: EvaluationResult) -> str:
    behavior = summarize_behaviors(result.behavior_episodes)
    factor_count = len(behavior["by_map_opponent_strategy_seat_submit"])
    strategy = "全部策略" if result.strategy_id < 0 else f"策略 {result.strategy_id}"
    return (
        f"地图 {result.map_name} | 对手 {result.opponent} | "
        f"{strategy} | 结果 {result.outcomes}\n"
        f"正常完赛率：{result.completion_rate:.1%}\n"
        f"正常完赛内部胜率：{result.completed_win_rate:.1%}\n"
        f"总样本取胜率：{result.overall_win_rate:.1%}\n"
        f"首攻覆盖：{behavior['unique_opening_targets']} 个目标 | "
        f"同因子多开局："
        f"{behavior['factors_with_multiple_openings']}/{factor_count}\n"
        f"换线代理率：{behavior['retarget_rate']:.1%} | "
        f"出现反攻的对局：{behavior['counterattack_episode_rate']:.1%} | "
        f"重复指令率：{behavior['redundant_command_rate']:.1%}"
    )
