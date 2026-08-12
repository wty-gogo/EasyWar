"""通过冻结历史联盟和冠军门槛进行无老师自博弈训练。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import easywar_rl
import torch

from evaluation import EvaluationResult, evaluate_model
from runtime import (
    build_model,
    checkpoint_observation_channels,
    checkpoint_strategy_count,
    load_model_weights,
    repository_root,
    seed_everything,
    to_model_tensors,
)


TRAINING_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = TRAINING_DIR / "checkpoints"
RUN_DIR = TRAINING_DIR / "runs"
TRAINING_MAPS = ("dual_ladder_1v1.toml", "braided_rings_1v1.toml")
RULE_OPPONENTS = ("easy", "normal", "hard")


@dataclass(frozen=True)
class LeagueState:
    generation: int
    champion: str
    champion_temperature: float
    archive: tuple[str, ...]
    decisions: int
    games: int


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    reasons: tuple[str, ...]
    direct_outcomes: dict[str, int]
    candidate_rules: tuple[dict[str, object], ...]
    champion_rules: tuple[dict[str, object], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EasyWar 无老师自博弈冠军联盟")
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--updates", type=int, default=16, help="每个候选的 PPO 更新数")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--recent-window", type=int, default=6)
    parser.add_argument("--direct-episodes", type=int, default=64)
    parser.add_argument("--rule-episodes", type=int, default=8)
    parser.add_argument("--candidate-temperature", type=float, default=0.5)
    parser.add_argument("--opponent-temperature", type=float, default=0.8)
    parser.add_argument("--minimum-direct-win-rate", type=float, default=0.52)
    parser.add_argument("--minimum-direct-completion", type=float, default=0.75)
    parser.add_argument("--maximum-rule-drop", type=float, default=0.125)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20_260_812)
    parser.add_argument(
        "--initial-champion",
        type=Path,
        default=Path("checkpoints/tactical-v10-blend-50.pt"),
    )
    parser.add_argument("--extra-opponent", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--state", type=Path, default=Path("runs/selfplay-league-state.json")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("runs/selfplay-league.jsonl")
    )
    return parser.parse_args()


def resolve_training_path(path: Path) -> Path:
    return (path if path.is_absolute() else TRAINING_DIR / path).resolve()


def default_opponents() -> tuple[Path, ...]:
    return tuple(
        (CHECKPOINT_DIR / name).resolve()
        for name in (
            "unified-v6-focused2-best.pt",
            "tactical-v8-ppo-best.pt",
            "tactical-v9-logistics-bc.pt",
            "tactical-v10-blend-50.pt",
        )
    )


def unique_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    return tuple(dict.fromkeys(path.resolve() for path in paths))


def initial_state(args: argparse.Namespace) -> LeagueState:
    champion = resolve_training_path(args.initial_champion)
    extras = tuple(resolve_training_path(path) for path in args.extra_opponent)
    archive = unique_paths((*default_opponents(), *extras, champion))
    missing = [str(path) for path in archive if not path.exists()]
    if missing:
        raise FileNotFoundError(f"自博弈对手不存在：{missing}")
    return LeagueState(0, str(champion), 0.0, tuple(map(str, archive)), 0, 0)


def load_state(path: Path, args: argparse.Namespace) -> LeagueState:
    if not path.exists():
        return initial_state(args)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return LeagueState(
        generation=int(payload["generation"]),
        champion=str(payload["champion"]),
        champion_temperature=float(payload["champion_temperature"]),
        archive=tuple(payload["archive"]),
        decisions=int(payload["decisions"]),
        games=int(payload["games"]),
    )


def save_state(path: Path, state: LeagueState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(asdict(state), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_report(path: Path, event: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as report:
        report.write(json.dumps(event, ensure_ascii=False) + "\n")


def selected_opponents(state: LeagueState, recent_window: int) -> tuple[Path, ...]:
    archive = tuple(Path(path) for path in state.archive)
    stable = default_opponents()
    recent = archive[-recent_window:] if recent_window else ()
    return unique_paths((*stable, *recent, Path(state.champion)))


def training_command(
    args: argparse.Namespace,
    generation: int,
    state: LeagueState,
    opponents: tuple[Path, ...],
    candidate: Path,
    training_report: Path,
) -> list[str]:
    return [
        sys.executable,
        "train_ppo.py",
        "--historical-opponent",
        *(str(path) for path in opponents),
        "--historical-opponent-temperature",
        str(args.opponent_temperature),
        "--num-envs",
        str(args.num_envs),
        "--tactical-actions",
        "--initialize-from",
        state.champion,
        "--anchor-checkpoint",
        state.champion,
        "--anchor-kl-coef",
        "0.005",
        "--terminal-only-reward",
        "--updates",
        str(args.updates),
        "--rollout-steps",
        str(args.rollout_steps),
        "--epochs",
        "2",
        "--minibatch-size",
        "512",
        "--learning-rate",
        "0.000003",
        "--gamma",
        "0.9995",
        "--gae-lambda",
        "0.99",
        "--entropy",
        "0.005",
        "--rollout-temperature",
        "0.8",
        "--threads",
        str(args.threads),
        "--device",
        args.device,
        "--checkpoint",
        str(candidate),
        "--checkpoint-every",
        str(args.updates),
        "--report",
        str(training_report),
        "--seed",
        str(args.seed + generation * 1_000_003),
    ]


def load_policy(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = build_model(
        device,
        checkpoint_strategy_count(checkpoint),
        checkpoint_observation_channels(checkpoint),
    )
    load_model_weights(model, checkpoint)
    model.eval()
    return model, checkpoint


def paired_match(
    candidate_path: Path,
    champion_path: Path,
    device: torch.device,
    seed: int,
    episodes: int,
    candidate_temperature: float,
    champion_temperature: float,
    threads: int,
) -> Counter[str]:
    candidate, _ = load_policy(candidate_path, device)
    champion, _ = load_policy(champion_path, device)
    root = repository_root()
    environment_count = min(episodes, 16)
    environment = easywar_rl.BatchEnv(
        [str(root / "assets/maps" / name) for name in TRAINING_MAPS],
        str(root / "assets/subjects"),
        environment_count,
        seed=seed,
        opponent="hard",
        map_transforms=["vertical", "vertical"],
        alternate_seats=True,
        external_opponent=True,
        tactical_actions=True,
        alternate_submit_order=True,
        max_decisions=1_200,
    )
    learner, learner_previous = to_model_tensors(
        environment.observe(), device, candidate.observation_channels
    )
    opponent, opponent_previous = to_model_tensors(
        environment.observe_opponents(), device, champion.observation_channels
    )
    outcomes: Counter[str] = Counter()
    next_seed = seed + environment_count
    while sum(outcomes.values()) < episodes:
        with torch.no_grad():
            learner_actions = candidate.act(
                learner.values,
                learner.bases,
                learner.masks,
                deterministic=candidate_temperature == 0.0,
                temperature=candidate_temperature or 1.0,
            )[0]
            opponent_actions = champion.act(
                opponent.values,
                opponent.bases,
                opponent.masks,
                deterministic=champion_temperature == 0.0,
                temperature=champion_temperature or 1.0,
            )[0]
        transition = environment.step_external(
            learner_actions.cpu().tolist(),
            opponent_actions.cpu().tolist(),
            threads,
        )
        terminal = [
            index
            for index, name in enumerate(transition.end_names)
            if name != "Ongoing"
        ]
        for index in terminal:
            if sum(outcomes.values()) < episodes:
                outcomes[transition.end_names[index]] += 1
        if terminal and sum(outcomes.values()) < episodes:
            seeds = list(range(next_seed, next_seed + len(terminal)))
            next_seed += len(terminal)
            batch = environment.reset_indices(terminal, seeds)
        else:
            batch = transition
        learner, learner_previous = to_model_tensors(
            batch,
            device,
            candidate.observation_channels,
            learner_previous,
            terminal,
        )
        opponent, opponent_previous = to_model_tensors(
            environment.observe_opponents(),
            device,
            champion.observation_channels,
            opponent_previous,
            terminal,
        )
    return outcomes


def rule_results(
    path: Path,
    temperature: float,
    args: argparse.Namespace,
    seed: int,
) -> tuple[EvaluationResult, ...]:
    device = torch.device(args.device)
    model, checkpoint = load_policy(path, device)
    tactical_actions = bool(
        checkpoint.get("training_state", {}).get("tactical_actions", False)
    )
    results: list[EvaluationResult] = []
    for opponent_index, opponent in enumerate(RULE_OPPONENTS):
        for map_index, map_name in enumerate(TRAINING_MAPS):
            factor_seed = seed + opponent_index * 100_000 + map_index * 10_000
            seed_everything(factor_seed)
            results.append(
                evaluate_model(
                    model,
                    device,
                    map_name,
                    opponent,
                    args.rule_episodes,
                    min(args.rule_episodes, 8),
                    args.threads,
                    factor_seed,
                    tactical_actions=tactical_actions,
                    policy_temperature=temperature,
                )
            )
    return tuple(results)


def promotion_decision(
    direct: Counter[str],
    candidate_rules: tuple[EvaluationResult, ...],
    champion_rules: tuple[EvaluationResult, ...],
    minimum_direct_win_rate: float,
    minimum_direct_completion: float,
    maximum_rule_drop: float,
) -> PromotionDecision:
    episodes = sum(direct.values())
    direct_win_rate = direct["Won"] / episodes if episodes else 0.0
    direct_completion = (direct["Won"] + direct["Lost"]) / episodes if episodes else 0.0
    reasons: list[str] = []
    if direct_win_rate < minimum_direct_win_rate:
        reasons.append(
            f"对冠军总样本胜率 {direct_win_rate:.1%} < {minimum_direct_win_rate:.1%}"
        )
    if direct_completion < minimum_direct_completion:
        reasons.append(
            f"对冠军完赛率 {direct_completion:.1%} < {minimum_direct_completion:.1%}"
        )
    for candidate, champion in zip(candidate_rules, champion_rules, strict=True):
        if candidate.overall_win_rate + maximum_rule_drop < champion.overall_win_rate:
            reasons.append(
                f"{candidate.map_name}/{candidate.opponent} 胜率 "
                f"{candidate.overall_win_rate:.1%}，冠军为 {champion.overall_win_rate:.1%}"
            )
        if candidate.completion_rate + maximum_rule_drop < champion.completion_rate:
            reasons.append(
                f"{candidate.map_name}/{candidate.opponent} 完赛率 "
                f"{candidate.completion_rate:.1%}，冠军为 {champion.completion_rate:.1%}"
            )
    return PromotionDecision(
        promoted=not reasons,
        reasons=tuple(reasons),
        direct_outcomes=dict(direct),
        candidate_rules=tuple(result.as_dict() for result in candidate_rules),
        champion_rules=tuple(result.as_dict() for result in champion_rules),
    )


def training_summary(report: Path, args: argparse.Namespace) -> tuple[int, Counter[str]]:
    events = [
        json.loads(line)
        for line in report.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    updates = [event for event in events if event.get("event") == "ppo_update"]
    outcomes = Counter(updates[-1].get("outcomes", {})) if updates else Counter()
    decisions = len(updates) * args.rollout_steps * args.num_envs
    return decisions, outcomes


def run(args: argparse.Namespace) -> None:
    state_path = resolve_training_path(args.state)
    report_path = resolve_training_path(args.report)
    state = load_state(state_path, args)
    device = torch.device(args.device)
    print(
        f"自博弈联盟从第 {state.generation} 代继续，当前冠军："
        f"{Path(state.champion).name}"
    )
    for _ in range(args.generations):
        generation = state.generation + 1
        opponents = selected_opponents(state, args.recent_window)
        candidate = CHECKPOINT_DIR / f"selfplay-league-g{generation:03d}.pt"
        training_report = RUN_DIR / f"selfplay-league-g{generation:03d}-training.jsonl"
        training_report.unlink(missing_ok=True)
        print(
            f"第 {generation} 代开始：{len(opponents)} 个冻结对手，"
            f"{args.num_envs * args.rollout_steps * args.updates:,} 次训练决策"
        )
        started = time.monotonic()
        subprocess.run(
            training_command(
                args, generation, state, opponents, candidate, training_report
            ),
            cwd=TRAINING_DIR,
            check=True,
        )
        decisions, outcomes = training_summary(training_report, args)
        evaluation_seed = args.seed + generation * 10_000_019
        seed_everything(evaluation_seed)
        direct = paired_match(
            candidate,
            Path(state.champion),
            device,
            evaluation_seed,
            args.direct_episodes,
            args.candidate_temperature,
            state.champion_temperature,
            args.threads,
        )
        candidate_rules = rule_results(
            candidate, args.candidate_temperature, args, evaluation_seed + 1_000_000
        )
        champion_rules = rule_results(
            Path(state.champion),
            state.champion_temperature,
            args,
            evaluation_seed + 1_000_000,
        )
        decision = promotion_decision(
            direct,
            candidate_rules,
            champion_rules,
            args.minimum_direct_win_rate,
            args.minimum_direct_completion,
            args.maximum_rule_drop,
        )
        archive = unique_paths(
            (*tuple(Path(path) for path in state.archive), candidate.resolve())
        )
        state = LeagueState(
            generation=generation,
            champion=str(candidate.resolve()) if decision.promoted else state.champion,
            champion_temperature=(
                args.candidate_temperature
                if decision.promoted
                else state.champion_temperature
            ),
            archive=tuple(map(str, archive)),
            decisions=state.decisions + decisions,
            games=state.games + sum(outcomes.values()),
        )
        save_state(state_path, state)
        append_report(
            report_path,
            {
                "generation": generation,
                "candidate": str(candidate.resolve()),
                "training_decisions": decisions,
                "training_outcomes": dict(outcomes),
                "elapsed_seconds": time.monotonic() - started,
                **asdict(decision),
            },
        )
        verdict = "晋级冠军" if decision.promoted else "加入对手库但不晋级"
        print(f"第 {generation} 代：{verdict}；直面对局 {dict(direct)}")
        for reason in decision.reasons:
            print(f"  - {reason}")


if __name__ == "__main__":
    run(parse_args())
