from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, replace

import easywar_rl
import torch
from torch import Tensor


@dataclass(frozen=True)
class OpeningTarget:
    grid: int
    x: int
    y: int
    owner: str
    cell_kind: str

    def as_dict(self) -> dict[str, object]:
        return {
            "grid": self.grid,
            "x": self.x,
            "y": self.y,
            "owner": self.owner,
            "cell_kind": self.cell_kind,
        }


@dataclass(frozen=True)
class BehaviorState:
    map_name: str
    opponent: str
    strategy_id: int
    variant: int
    seat_transform: str
    submit_order: str
    decisions: int
    command_count: int
    effective_command_count: int
    effective_offensive_command_count: int
    redundant_command_count: int
    offensive_command_count: int
    enemy_target_count: int
    neutral_target_count: int
    linked_target_count: int
    retarget_count: int
    counterattack_count: int
    first_command_decision: int | None
    first_offensive_decision: int | None
    first_offensive_target: OpeningTarget | None
    offensive_targets: frozenset[int]
    counterattack_targets: frozenset[int]
    source_targets: tuple[int | None, ...]
    previous_friendly_cells: frozenset[int]
    lost_cells: frozenset[int]


@dataclass(frozen=True)
class EpisodeBehavior:
    map_name: str
    opponent: str
    strategy_id: int
    outcome: str
    variant: int
    seat_transform: str
    submit_order: str
    decisions: int
    command_count: int
    effective_command_count: int
    effective_offensive_command_count: int
    redundant_command_count: int
    offensive_command_count: int
    enemy_target_count: int
    neutral_target_count: int
    linked_target_count: int
    retarget_count: int
    counterattack_count: int
    first_command_decision: int | None
    first_offensive_decision: int | None
    first_offensive_target: OpeningTarget | None
    distinct_offensive_targets: int
    distinct_counterattack_targets: int

    def as_dict(self) -> dict[str, object]:
        return {
            "map": self.map_name,
            "opponent": self.opponent,
            "strategy_id": self.strategy_id,
            "outcome": self.outcome,
            "variant": self.variant,
            "seat_transform": self.seat_transform,
            "submit_order": self.submit_order,
            "decisions": self.decisions,
            "command_count": self.command_count,
            "effective_command_count": self.effective_command_count,
            "effective_offensive_command_count": self.effective_offensive_command_count,
            "redundant_command_count": self.redundant_command_count,
            "offensive_command_count": self.offensive_command_count,
            "enemy_target_count": self.enemy_target_count,
            "neutral_target_count": self.neutral_target_count,
            "linked_target_count": self.linked_target_count,
            "retarget_count": self.retarget_count,
            "counterattack_count": self.counterattack_count,
            "first_command_decision": self.first_command_decision,
            "first_offensive_decision": self.first_offensive_decision,
            "first_offensive_target": (
                self.first_offensive_target.as_dict()
                if self.first_offensive_target is not None
                else None
            ),
            "distinct_offensive_targets": self.distinct_offensive_targets,
            "distinct_counterattack_targets": self.distinct_counterattack_targets,
        }


def initial_behavior_state(
    map_name: str,
    opponent: str,
    variant: int,
    seat_transform: str,
    strategy_id: int = 0,
) -> BehaviorState:
    return BehaviorState(
        map_name=map_name,
        opponent=opponent,
        strategy_id=strategy_id,
        variant=variant,
        seat_transform=seat_transform,
        submit_order=("opponent_first" if variant // 2 % 2 else "learner_first"),
        decisions=0,
        command_count=0,
        effective_command_count=0,
        effective_offensive_command_count=0,
        redundant_command_count=0,
        offensive_command_count=0,
        enemy_target_count=0,
        neutral_target_count=0,
        linked_target_count=0,
        retarget_count=0,
        counterattack_count=0,
        first_command_decision=None,
        first_offensive_decision=None,
        first_offensive_target=None,
        offensive_targets=frozenset(),
        counterattack_targets=frozenset(),
        source_targets=(None,) * easywar_rl.MAX_BASES,
        previous_friendly_cells=frozenset(),
        lost_cells=frozenset(),
    )


def _active_cells(channel: Tensor) -> frozenset[int]:
    return frozenset(
        torch.nonzero(channel.flatten() > 0.5, as_tuple=False).flatten().tolist()
    )


def _map_dimensions(observation: Tensor) -> tuple[int, int]:
    enterable = torch.nonzero(observation[0] > 0.5, as_tuple=False)
    if enterable.numel() == 0:
        return easywar_rl.MAX_WIDTH, easywar_rl.MAX_HEIGHT
    return int(enterable[:, 1].max().item()) + 1, int(enterable[:, 0].max().item()) + 1


def _canonical_grid(
    grid: int,
    observation: Tensor,
    seat_transform: str,
) -> int:
    x = grid % easywar_rl.MAX_WIDTH
    y = grid // easywar_rl.MAX_WIDTH
    width, height = _map_dimensions(observation)
    if seat_transform == "vertical":
        x = width - 1 - x
    elif seat_transform == "rotational":
        x = width - 1 - x
        y = height - 1 - y
    return y * easywar_rl.MAX_WIDTH + x


def _target_description(
    target: int,
    observation: Tensor,
    seat_transform: str,
) -> OpeningTarget:
    canonical = _canonical_grid(target, observation, seat_transform)
    y, x = divmod(target, easywar_rl.MAX_WIDTH)
    owner = (
        "enemy"
        if observation[4, y, x] > 0.5
        else "neutral"
        if observation[5, y, x] > 0.5
        else "friendly"
        if observation[3, y, x] > 0.5
        else "unknown"
    )
    cell_kind = (
        "base"
        if observation[2, y, x] > 0.5
        else "linked"
        if observation[1, y, x] > 0.5
        else "other"
    )
    canonical_y, canonical_x = divmod(canonical, easywar_rl.MAX_WIDTH)
    return OpeningTarget(
        grid=canonical,
        x=canonical_x,
        y=canonical_y,
        owner=owner,
        cell_kind=cell_kind,
    )


def record_behavior(
    state: BehaviorState,
    observation: Tensor,
    base_cells: Tensor,
    action: int,
) -> BehaviorState:
    """从动作前的玩家可见观察推进行为状态，不读取隐藏数据。"""

    decision = state.decisions + 1
    friendly_cells = _active_cells(observation[3])
    newly_lost = state.previous_friendly_cells - friendly_cells
    lost_cells = (state.lost_cells | newly_lost) - friendly_cells
    common = {
        "decisions": decision,
        "previous_friendly_cells": friendly_cells,
        "lost_cells": lost_cells,
    }
    source_targets = tuple(
        target
        if int(base_cells[slot].item()) >= 0
        and observation[
            13,
            int(base_cells[slot].item()) // easywar_rl.MAX_WIDTH,
            int(base_cells[slot].item()) % easywar_rl.MAX_WIDTH,
        ]
        > 0.5
        else None
        for slot, target in enumerate(state.source_targets)
    )
    stop_start = easywar_rl.ACTION_COUNT - easywar_rl.MAX_BASES
    if action == 0:
        return replace(state, source_targets=source_targets, **common)
    if action >= stop_start:
        slot = action - stop_start
        if slot >= easywar_rl.MAX_BASES:
            return replace(state, **common)
        updated_targets = list(source_targets)
        updated_targets[slot] = None
        return replace(state, source_targets=tuple(updated_targets), **common)

    encoded = action - 1
    cells = easywar_rl.MAX_WIDTH * easywar_rl.MAX_HEIGHT
    slot, target = divmod(encoded, cells)
    if slot >= easywar_rl.MAX_BASES or int(base_cells[slot].item()) < 0:
        return replace(state, **common)
    description = _target_description(target, observation, state.seat_transform)
    offensive = description.owner in {"enemy", "neutral"}
    previous_target = source_targets[slot]
    effective = previous_target != target
    updated_targets = list(source_targets)
    updated_targets[slot] = target
    first_command = state.first_command_decision or decision
    first_offensive_decision = state.first_offensive_decision
    first_offensive_target = state.first_offensive_target
    if offensive and first_offensive_decision is None:
        first_offensive_decision = decision
        first_offensive_target = description
    counterattack = effective and offensive and target in lost_cells
    return replace(
        state,
        command_count=state.command_count + 1,
        effective_command_count=state.effective_command_count + int(effective),
        effective_offensive_command_count=(
            state.effective_offensive_command_count + int(effective and offensive)
        ),
        redundant_command_count=state.redundant_command_count + int(not effective),
        offensive_command_count=state.offensive_command_count + int(offensive),
        enemy_target_count=(
            state.enemy_target_count + int(effective and description.owner == "enemy")
        ),
        neutral_target_count=(
            state.neutral_target_count
            + int(effective and description.owner == "neutral")
        ),
        linked_target_count=(
            state.linked_target_count
            + int(effective and description.cell_kind == "linked")
        ),
        retarget_count=(
            state.retarget_count
            + int(previous_target is not None and previous_target != target)
        ),
        counterattack_count=(
            state.counterattack_count + int(counterattack)
        ),
        first_command_decision=first_command,
        first_offensive_decision=first_offensive_decision,
        first_offensive_target=first_offensive_target,
        offensive_targets=(
            state.offensive_targets | {description.grid}
            if offensive
            else state.offensive_targets
        ),
        counterattack_targets=(
            state.counterattack_targets | {description.grid}
            if counterattack
            else state.counterattack_targets
        ),
        source_targets=tuple(updated_targets),
        **common,
    )


def finish_behavior(state: BehaviorState, outcome: str) -> EpisodeBehavior:
    return EpisodeBehavior(
        map_name=state.map_name,
        opponent=state.opponent,
        strategy_id=state.strategy_id,
        outcome=outcome,
        variant=state.variant,
        seat_transform=state.seat_transform,
        submit_order=state.submit_order,
        decisions=state.decisions,
        command_count=state.command_count,
        effective_command_count=state.effective_command_count,
        effective_offensive_command_count=state.effective_offensive_command_count,
        redundant_command_count=state.redundant_command_count,
        offensive_command_count=state.offensive_command_count,
        enemy_target_count=state.enemy_target_count,
        neutral_target_count=state.neutral_target_count,
        linked_target_count=state.linked_target_count,
        retarget_count=state.retarget_count,
        counterattack_count=state.counterattack_count,
        first_command_decision=state.first_command_decision,
        first_offensive_decision=state.first_offensive_decision,
        first_offensive_target=state.first_offensive_target,
        distinct_offensive_targets=len(state.offensive_targets),
        distinct_counterattack_targets=len(state.counterattack_targets),
    )


def episode_behavior_from_dict(payload: dict[str, object]) -> EpisodeBehavior:
    target_payload = payload.get("first_offensive_target")
    target = (
        OpeningTarget(
            grid=int(target_payload["grid"]),
            x=int(target_payload["x"]),
            y=int(target_payload["y"]),
            owner=str(target_payload["owner"]),
            cell_kind=str(target_payload["cell_kind"]),
        )
        if isinstance(target_payload, dict)
        else None
    )
    return EpisodeBehavior(
        map_name=str(payload["map"]),
        opponent=str(payload["opponent"]),
        strategy_id=int(payload.get("strategy_id", 0)),
        outcome=str(payload["outcome"]),
        variant=int(payload["variant"]),
        seat_transform=str(payload["seat_transform"]),
        submit_order=str(payload["submit_order"]),
        decisions=int(payload["decisions"]),
        command_count=int(payload["command_count"]),
        effective_command_count=int(
            payload.get("effective_command_count", payload["command_count"])
        ),
        effective_offensive_command_count=int(
            payload.get(
                "effective_offensive_command_count",
                payload["offensive_command_count"],
            )
        ),
        redundant_command_count=int(payload.get("redundant_command_count", 0)),
        offensive_command_count=int(payload["offensive_command_count"]),
        enemy_target_count=int(payload.get("enemy_target_count", 0)),
        neutral_target_count=int(payload.get("neutral_target_count", 0)),
        linked_target_count=int(payload.get("linked_target_count", 0)),
        retarget_count=int(payload["retarget_count"]),
        counterattack_count=int(payload["counterattack_count"]),
        first_command_decision=(
            int(payload["first_command_decision"])
            if payload.get("first_command_decision") is not None
            else None
        ),
        first_offensive_decision=(
            int(payload["first_offensive_decision"])
            if payload.get("first_offensive_decision") is not None
            else None
        ),
        first_offensive_target=target,
        distinct_offensive_targets=int(payload["distinct_offensive_targets"]),
        distinct_counterattack_targets=int(
            payload.get("distinct_counterattack_targets", 0)
        ),
    )


def _mean(values: list[int]) -> float | None:
    return sum(values) / len(values) if values else None


def _opening_key(episode: EpisodeBehavior) -> str | None:
    target = episode.first_offensive_target
    return (
        f"{episode.map_name}|{episode.opponent}|策略{episode.strategy_id}|{target.grid}"
        if target is not None
        else None
    )


def _summary_core(episodes: tuple[EpisodeBehavior, ...]) -> dict[str, object]:
    episode_count = len(episodes)
    commands = sum(episode.command_count for episode in episodes)
    effective_commands = sum(episode.effective_command_count for episode in episodes)
    effective_offensive_commands = sum(
        episode.effective_offensive_command_count for episode in episodes
    )
    redundant_commands = sum(episode.redundant_command_count for episode in episodes)
    offensive_commands = sum(episode.offensive_command_count for episode in episodes)
    enemy_targets = sum(episode.enemy_target_count for episode in episodes)
    neutral_targets = sum(episode.neutral_target_count for episode in episodes)
    linked_targets = sum(episode.linked_target_count for episode in episodes)
    retargets = sum(episode.retarget_count for episode in episodes)
    counterattacks = sum(episode.counterattack_count for episode in episodes)
    opening_counts = Counter(
        key for episode in episodes if (key := _opening_key(episode)) is not None
    )
    opening_total = sum(opening_counts.values())
    opening_entropy = (
        -sum(
            (count / opening_total) * math.log(count / opening_total)
            for count in opening_counts.values()
        )
        if opening_total
        else 0.0
    )
    return {
        "episode_count": episode_count,
        "episodes_with_offense": opening_total,
        "offensive_episode_rate": (
            opening_total / episode_count if episode_count else 0.0
        ),
        "mean_first_command_decision": _mean(
            [
                episode.first_command_decision
                for episode in episodes
                if episode.first_command_decision is not None
            ]
        ),
        "mean_first_offensive_decision": _mean(
            [
                episode.first_offensive_decision
                for episode in episodes
                if episode.first_offensive_decision is not None
            ]
        ),
        "commands_per_episode": commands / episode_count if episode_count else 0.0,
        "effective_commands_per_episode": (
            effective_commands / episode_count if episode_count else 0.0
        ),
        "redundant_command_rate": (
            redundant_commands / commands if commands else 0.0
        ),
        "offensive_commands_per_episode": (
            offensive_commands / episode_count if episode_count else 0.0
        ),
        "effective_offensive_commands_per_episode": (
            effective_offensive_commands / episode_count if episode_count else 0.0
        ),
        "enemy_target_rate": (
            enemy_targets / effective_commands if effective_commands else 0.0
        ),
        "neutral_target_rate": (
            neutral_targets / effective_commands if effective_commands else 0.0
        ),
        "linked_target_rate": (
            linked_targets / effective_commands if effective_commands else 0.0
        ),
        "retarget_count": retargets,
        "retarget_rate": retargets / effective_commands if effective_commands else 0.0,
        "counterattack_count": counterattacks,
        "counterattack_rate": (
            counterattacks / effective_offensive_commands
            if effective_offensive_commands
            else 0.0
        ),
        "counterattack_episode_rate": (
            sum(episode.counterattack_count > 0 for episode in episodes) / episode_count
            if episode_count
            else 0.0
        ),
        "mean_distinct_counterattack_targets": (
            sum(episode.distinct_counterattack_targets for episode in episodes)
            / episode_count
            if episode_count
            else 0.0
        ),
        "mean_distinct_offensive_targets": (
            sum(episode.distinct_offensive_targets for episode in episodes)
            / episode_count
            if episode_count
            else 0.0
        ),
        "unique_opening_targets": len(opening_counts),
        "opening_target_entropy": opening_entropy,
        "opening_target_counts": dict(sorted(opening_counts.items())),
    }


def summarize_behaviors(episodes: tuple[EpisodeBehavior, ...]) -> dict[str, object]:
    summary = _summary_core(episodes)
    factors = sorted(
        {
            (
                episode.map_name,
                episode.opponent,
                episode.strategy_id,
                episode.seat_transform,
                episode.submit_order,
            )
            for episode in episodes
        }
    )
    outcomes = sorted({episode.outcome for episode in episodes})
    factor_summaries = {
        f"{map_name}|{opponent}|策略{strategy_id}|{seat}|{submit}": _summary_core(
            tuple(
                episode
                for episode in episodes
                if episode.map_name == map_name
                and episode.opponent == opponent
                and episode.strategy_id == strategy_id
                and episode.seat_transform == seat
                and episode.submit_order == submit
            )
        )
        for map_name, opponent, strategy_id, seat, submit in factors
    }
    return {
        **summary,
        "factors_with_multiple_openings": sum(
            factor["unique_opening_targets"] > 1 for factor in factor_summaries.values()
        ),
        "mean_within_factor_opening_entropy": (
            sum(
                float(factor["opening_target_entropy"])
                for factor in factor_summaries.values()
            )
            / len(factor_summaries)
            if factor_summaries
            else 0.0
        ),
        "by_map_opponent_strategy_seat_submit": factor_summaries,
        "by_outcome": {
            outcome: _summary_core(
                tuple(episode for episode in episodes if episode.outcome == outcome)
            )
            for outcome in outcomes
        },
    }
