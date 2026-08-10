from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import easywar_rl
import torch
from torch import Tensor
from torch.nn import functional as F

from evaluation import (
    aggregate_results,
    evaluate_model,
    model_selection_key,
    passes_validation_gate,
)
from runtime import (
    HistoricalOpponentPool,
    TensorObservation,
    build_model,
    checkpoint_strategy_count,
    choose_device,
    environment_signature,
    load_model_weights,
    repository_root,
    resolve_artifact_path,
    seed_everything,
    to_tensors,
    training_map_names,
    training_maps,
    training_transforms,
)


CHECKPOINT_SCHEMA_VERSION = 7
TERMINAL_NAMES = ("Won", "Lost", "Stalemate", "CycleDetected", "BudgetExceeded")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 EasyWar 1v1 PPO AI")
    parser.add_argument("--phase", choices=["main"], default="main")
    parser.add_argument(
        "--rule-opponents",
        choices=["easy", "normal", "hard"],
        nargs="+",
        default=["easy", "normal", "hard"],
        help="内部规则对手池；按地图、难度和席位正交分配",
    )
    parser.add_argument(
        "--historical-opponent",
        type=Path,
        nargs="+",
        default=[],
        help="使用一个或多个冻结检查点替代规则对手，并按环境稳定轮换",
    )
    parser.add_argument("--num-envs", type=int, default=24)
    parser.add_argument(
        "--strategy-count",
        type=int,
        default=1,
        help="可控策略编号数量；大于 1 时按完整实验因素正交采样",
    )
    parser.add_argument(
        "--strategy-diversity-coef",
        type=float,
        default=0.0,
        help="最大化同一观察下不同策略分布的 JS 散度系数",
    )
    parser.add_argument(
        "--strategy-diversity-samples",
        type=int,
        default=32,
        help="每个小批量用于计算策略多样性目标的观察数",
    )
    parser.add_argument(
        "--strategy-specialization-coef",
        type=float,
        default=0.0,
        help="可解释策略意图的条件目标偏好系数",
    )
    parser.add_argument(
        "--strategy-adapter-only",
        action="store_true",
        help="冻结主体网络，只训练新增策略向量并保持基准策略不变",
    )
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--updates", type=int, default=10, help="PPO 目标累计更新数")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--imitation-updates", type=int, default=0, help="模仿目标累计更新数"
    )
    parser.add_argument(
        "--command-weight",
        type=float,
        default=6.0,
        help="模仿阶段非等待动作的样本权重",
    )
    parser.add_argument(
        "--dagger-model-prob",
        type=float,
        default=0.5,
        help="模仿采样时执行模型动作的概率，老师仍为到达状态提供标签",
    )
    parser.add_argument(
        "--teacher", choices=["easy", "normal", "hard"], default="normal"
    )
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ppo.pt"))
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--report", type=Path, default=Path("runs/training.jsonl"))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--resume", type=Path, help="恢复模型、优化器和累计更新数")
    source.add_argument("--initialize-from", type=Path, help="只加载模型权重，开始新课程")
    parser.add_argument(
        "--anchor-checkpoint",
        type=Path,
        help="冻结的旧模型锚点，用 KL 约束降低新课程中的策略遗忘",
    )
    parser.add_argument(
        "--anchor-kl-coef",
        type=float,
        default=0.0,
        help="旧模型策略 KL 约束系数；大于 0 时必须提供锚点检查点",
    )
    parser.add_argument("--validation-every", type=int, default=0)
    parser.add_argument("--validation-episodes", type=int, default=16)
    parser.add_argument("--validation-num-envs", type=int, default=8)
    parser.add_argument("--validation-seed", type=int, default=1_000_000)
    parser.add_argument(
        "--validation-opponents",
        choices=["easy", "normal", "hard"],
        nargs="+",
        default=["easy", "normal", "hard"],
        help="逐地图验证的规则对手难度矩阵",
    )
    parser.add_argument(
        "--best-checkpoint", type=Path, default=Path("checkpoints/best.pt")
    )
    parser.add_argument("--minimum-validation-completion", type=float, default=0.8)
    parser.add_argument("--minimum-validation-win-rate", type=float, default=0.5)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="连续多少次验证无改进后早停，0 表示禁用",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.dagger_model_prob <= 1.0:
        raise ValueError("--dagger-model-prob 必须位于 0 到 1 之间")
    positive = {
        "--num-envs": args.num_envs,
        "--rollout-steps": args.rollout_steps,
        "--epochs": args.epochs,
        "--minibatch-size": args.minibatch_size,
        "--validation-episodes": args.validation_episodes,
        "--validation-num-envs": args.validation_num_envs,
        "--strategy-count": args.strategy_count,
        "--strategy-diversity-samples": args.strategy_diversity_samples,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"以下参数必须大于 0：{', '.join(invalid)}")
    if args.updates < 0 or args.imitation_updates < 0:
        raise ValueError("训练更新数不能为负数")
    if args.checkpoint_every < 0 or args.validation_every < 0:
        raise ValueError("保存与验证间隔不能为负数")
    if args.early_stop_patience < 0:
        raise ValueError("早停耐心值不能为负数")
    if args.anchor_kl_coef < 0.0:
        raise ValueError("锚点 KL 系数不能为负数")
    if args.strategy_diversity_coef < 0.0:
        raise ValueError("策略多样性系数不能为负数")
    if args.strategy_specialization_coef < 0.0:
        raise ValueError("策略专门化系数不能为负数")
    if args.anchor_kl_coef > 0.0 and args.anchor_checkpoint is None:
        raise ValueError("启用锚点 KL 约束时必须提供 --anchor-checkpoint")
    if not 0.0 <= args.minimum_validation_completion <= 1.0:
        raise ValueError("最低验证完赛率必须位于 0 到 1 之间")
    if not 0.0 <= args.minimum_validation_win_rate <= 1.0:
        raise ValueError("最低验证胜率必须位于 0 到 1 之间")
    map_count = len(training_map_names(args.phase))
    opponent_count = 1 if args.historical_opponent else len(args.rule_opponents)
    orthogonal_batch = map_count * opponent_count * 4 * args.strategy_count
    if args.num_envs % orthogonal_batch != 0:
        raise ValueError(
            f"当前课程的环境数必须是 {orthogonal_batch} 的倍数，"
            "才能正交覆盖地图、席位和提交顺序"
        )


def json_safe_args(args: argparse.Namespace) -> dict[str, object]:
    def json_safe(value: object) -> object:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        return value

    return {
        key: json_safe(value)
        for key, value in vars(args).items()
    }


def historical_opponent_paths(args: argparse.Namespace) -> list[Path]:
    return [resolve_artifact_path(path).resolve() for path in args.historical_opponent]


def anchor_checkpoint_path(args: argparse.Namespace) -> str | None:
    return (
        str(resolve_artifact_path(args.anchor_checkpoint).resolve())
        if args.anchor_checkpoint is not None
        else None
    )


def categorical_kl(reference_logits: Tensor, current_logits: Tensor) -> Tensor:
    """计算 KL(冻结旧策略 || 当前策略)，并安全忽略双方都屏蔽的动作。"""

    reference_log_probs = F.log_softmax(reference_logits, dim=-1)
    current_log_probs = F.log_softmax(current_logits, dim=-1)
    reference_probs = reference_log_probs.exp()
    valid = reference_probs > 0.0
    safe_reference_log_probs = torch.where(
        valid, reference_log_probs, torch.zeros_like(reference_log_probs)
    )
    safe_current_log_probs = torch.where(
        valid, current_log_probs, torch.zeros_like(current_log_probs)
    )
    terms = reference_probs * (safe_reference_log_probs - safe_current_log_probs)
    return terms.sum(dim=-1).mean()


def environment_strategy_ids(
    args: argparse.Namespace, device: torch.device
) -> Tensor:
    map_count = len(training_map_names(args.phase))
    opponent_count = 1 if args.historical_opponent else len(args.rule_opponents)
    factor_width = map_count * opponent_count * 4
    return (
        torch.arange(args.num_envs, device=device) // factor_width
    ) % args.strategy_count


def strategy_js_divergence(
    model: torch.nn.Module,
    observations: Tensor,
    bases: Tensor,
    masks: Tensor,
    strategy_count: int,
    sample_count: int,
) -> Tensor:
    """同一批观察上各可控策略相对混合策略的 Jensen-Shannon 散度。"""

    if strategy_count <= 1 or observations.shape[0] == 0:
        return torch.zeros((), device=observations.device)
    count = min(sample_count, observations.shape[0])
    selected_observations = observations[:count]
    selected_bases = bases[:count]
    selected_masks = masks[:count]
    logits = torch.stack(
        [
            model.logits_and_value(
                selected_observations,
                selected_bases,
                selected_masks,
                torch.full(
                    (count,), style, dtype=torch.long, device=observations.device
                ),
            )[0]
            for style in range(strategy_count)
        ]
    )
    log_probs = F.log_softmax(logits, dim=-1)
    probabilities = log_probs.exp()
    mixture = probabilities.mean(dim=0)
    mixture_log = mixture.clamp_min(torch.finfo(mixture.dtype).tiny).log()
    safe_log_probs = torch.where(
        probabilities > 0.0,
        log_probs,
        torch.zeros_like(log_probs),
    )
    terms = probabilities * (safe_log_probs - mixture_log.unsqueeze(0))
    return terms.sum(dim=-1).mean().clamp_min(0.0)


def strategy_specialization_score(
    logits: Tensor,
    observations: Tensor,
    action_masks: Tensor,
    strategy_ids: Tensor,
) -> Tensor:
    """衡量压制、扩张与铺路策略是否把指令投向对应类别。"""

    cells = easywar_rl.MAX_WIDTH * easywar_rl.MAX_HEIGHT
    stop_start = easywar_rl.ACTION_COUNT - easywar_rl.MAX_BASES
    command_probabilities = F.softmax(logits, dim=-1)[:, 1:stop_start].reshape(
        -1, easywar_rl.MAX_BASES, cells
    )
    target_probabilities = command_probabilities.sum(dim=1)
    valid_targets = action_masks[:, 1:stop_start].reshape(
        -1, easywar_rl.MAX_BASES, cells
    ).any(dim=1)
    flat_observations = observations.flatten(2)
    preferred_channels = torch.tensor(
        [0, 4, 5, 1], dtype=torch.long, device=observations.device
    )
    supported = (strategy_ids > 0) & (strategy_ids < preferred_channels.shape[0])
    resolved_channels = preferred_channels[strategy_ids.clamp_max(3)]
    preferred_targets = flat_observations.gather(
        1,
        resolved_channels.view(-1, 1, 1).expand(-1, 1, cells),
    ).squeeze(1) * valid_targets
    available = preferred_targets.sum(dim=1) > 0.0
    command_mass = target_probabilities.sum(dim=1)
    scores = (target_probabilities * preferred_targets).sum(dim=1) / command_mass.clamp_min(
        torch.finfo(target_probabilities.dtype).eps
    )
    selected = supported & available & (command_mass > 0.0)
    return (
        scores[selected].mean()
        if selected.any()
        else logits.sum() * 0.0
    )


def configure_strategy_adapter_training(
    model: torch.nn.Module,
) -> list[torch.nn.Parameter]:
    """冻结主体网络，并让基准策略行在反向传播中保持不变。"""

    model.requires_grad_(False)
    embedding = model.strategy_embedding.weight
    embedding.requires_grad_(True)

    def preserve_baseline(gradient: Tensor) -> Tensor:
        return torch.cat([torch.zeros_like(gradient[:1]), gradient[1:]], dim=0)

    embedding.register_hook(preserve_baseline)
    return [embedding]


def append_report(path: Path, event: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as report:
        report.write(json.dumps(event, ensure_ascii=False) + "\n")


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=True)


def validate_resume_checkpoint(
    checkpoint: dict[str, Any], args: argparse.Namespace
) -> None:
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("恢复训练只支持当前检查点格式；旧模型可改用 --initialize-from")
    state = checkpoint["training_state"]
    if state["phase"] != args.phase:
        raise ValueError("恢复训练必须保持同一课程阶段；跨阶段请使用 --initialize-from")
    if state["rule_opponents"] != args.rule_opponents or state["teacher"] != args.teacher:
        raise ValueError(
            "恢复训练必须保持相同规则对手池和规则老师；"
            "切换课程请使用 --initialize-from"
        )
    configured_pool = [str(path) for path in historical_opponent_paths(args)]
    if state.get("historical_opponents", []) != configured_pool:
        raise ValueError("恢复训练必须保持相同历史模型对手池")
    if state.get("anchor_checkpoint") != anchor_checkpoint_path(args):
        raise ValueError("恢复训练必须保持相同抗遗忘锚点")
    if state.get("anchor_kl_coef") != args.anchor_kl_coef:
        raise ValueError("恢复训练必须保持相同锚点 KL 系数")
    if state.get("strategy_count", 1) != args.strategy_count:
        raise ValueError("恢复训练必须保持相同可控策略数量")
    if state.get("strategy_diversity_coef", 0.0) != args.strategy_diversity_coef:
        raise ValueError("恢复训练必须保持相同策略多样性系数")
    if state.get("strategy_specialization_coef", 0.0) != args.strategy_specialization_coef:
        raise ValueError("恢复训练必须保持相同策略专门化系数")
    if state.get("strategy_adapter_only", False) != args.strategy_adapter_only:
        raise ValueError("恢复训练必须保持相同策略适配器模式")
    if checkpoint.get("environment_signature") != environment_signature():
        raise ValueError("检查点的观察或动作空间与当前环境不兼容")
    if args.imitation_updates < state["imitation_updates_completed"]:
        raise ValueError("模仿目标更新数不能小于检查点已完成数量")
    if args.updates < state["ppo_updates_completed"]:
        raise ValueError("PPO 目标更新数不能小于检查点已完成数量")


def checkpoint_payload(
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    next_seed: int,
    imitation_updates_completed: int,
    ppo_updates_completed: int,
    best_selection_key: tuple[float, float],
) -> dict[str, object]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "environment_signature": environment_signature(),
        "training_state": {
            "phase": args.phase,
            "rule_opponents": args.rule_opponents,
            "teacher": args.teacher,
            "historical_opponents": [
                str(path) for path in historical_opponent_paths(args)
            ],
            "anchor_checkpoint": anchor_checkpoint_path(args),
            "anchor_kl_coef": args.anchor_kl_coef,
            "strategy_count": args.strategy_count,
            "strategy_diversity_coef": args.strategy_diversity_coef,
            "strategy_specialization_coef": args.strategy_specialization_coef,
            "strategy_adapter_only": args.strategy_adapter_only,
            "seed": args.seed,
            "next_seed": next_seed,
            "imitation_updates_completed": imitation_updates_completed,
            "ppo_updates_completed": ppo_updates_completed,
            "best_selection_key": list(best_selection_key),
        },
        "training_args": json_safe_args(args),
    }


def save_checkpoint(
    path: Path,
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    next_seed: int,
    imitation_updates_completed: int,
    ppo_updates_completed: int,
    best_selection_key: tuple[float, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        checkpoint_payload(
            args,
            model,
            optimizer,
            next_seed,
            imitation_updates_completed,
            ppo_updates_completed,
            best_selection_key,
        ),
        temporary,
    )
    temporary.replace(path)


def build_environment(args: argparse.Namespace, seed: int) -> easywar_rl.BatchEnv:
    root = repository_root()
    return easywar_rl.BatchEnv(
        training_maps(args.phase),
        str(root / "assets" / "subjects"),
        args.num_envs,
        seed=seed,
        opponent=args.rule_opponents[0],
        rule_opponents=args.rule_opponents,
        map_transforms=training_transforms(args.phase),
        alternate_seats=True,
        external_opponent=bool(args.historical_opponent),
        alternate_submit_order=True,
    )


def reset_finished(
    environment: easywar_rl.BatchEnv,
    transition: object,
    next_seed: int,
) -> tuple[object, int]:
    terminal_indices = [
        index for index, code in enumerate(transition.end_codes) if code != 0
    ]
    if not terminal_indices:
        return transition, next_seed
    reset_seeds = list(range(next_seed, next_seed + len(terminal_indices)))
    return (
        environment.reset_indices(terminal_indices, reset_seeds),
        next_seed + len(terminal_indices),
    )


def advance_environment(
    environment: easywar_rl.BatchEnv,
    learner_actions: Tensor,
    opponent_observation: TensorObservation | None,
    opponent_pool: HistoricalOpponentPool | None,
    threads: int,
    next_seed: int,
) -> tuple[object, object, TensorObservation | None, int]:
    if opponent_pool is None:
        transition = environment.step(
            learner_actions.detach().cpu().tolist(),
            threads,
        )
    else:
        if opponent_observation is None:
            raise ValueError("历史模型对手缺少当前观察")
        transition = environment.step_external(
            learner_actions.detach().cpu().tolist(),
            opponent_pool.actions(opponent_observation).cpu().tolist(),
            threads,
        )
    batch, next_seed = reset_finished(environment, transition, next_seed)
    next_opponent = (
        None
        if opponent_pool is None
        else to_tensors(environment.observe_opponents(), learner_actions.device)
    )
    return transition, batch, next_opponent, next_seed


def discrete_entropy(values: list[int]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    return (
        -sum((count / total) * math.log(count / total) for count in counts.values())
        if total
        else 0.0
    )


def action_distribution(actions: Tensor) -> dict[str, float | int]:
    stop_start = easywar_rl.ACTION_COUNT - easywar_rl.MAX_BASES
    total = max(actions.numel(), 1)
    action_ids = actions.detach().cpu().tolist()
    commands = [action - 1 for action in action_ids if 0 < action < stop_start]
    cells = easywar_rl.MAX_WIDTH * easywar_rl.MAX_HEIGHT
    sources = [command // cells for command in commands]
    targets = [command % cells for command in commands]
    return {
        "no_op_rate": (actions == 0).sum().item() / total,
        "set_stream_rate": ((actions > 0) & (actions < stop_start)).sum().item()
        / total,
        "stop_stream_rate": (actions >= stop_start).sum().item() / total,
        "unique_command_sources": len(set(sources)),
        "unique_command_targets": len(set(targets)),
        "command_source_entropy": discrete_entropy(sources),
        "command_target_entropy": discrete_entropy(targets),
    }


def action_target_distribution(
    actions: Tensor, observations: Tensor
) -> dict[str, float]:
    stop_start = easywar_rl.ACTION_COUNT - easywar_rl.MAX_BASES
    command_indices = torch.nonzero(
        (actions > 0) & (actions < stop_start), as_tuple=False
    ).flatten()
    if command_indices.numel() == 0:
        return {
            "enemy_target_rate": 0.0,
            "neutral_target_rate": 0.0,
            "friendly_target_rate": 0.0,
            "base_target_rate": 0.0,
            "linked_target_rate": 0.0,
        }
    cells = easywar_rl.MAX_WIDTH * easywar_rl.MAX_HEIGHT
    targets = (actions.index_select(0, command_indices) - 1) % cells
    selected = observations.index_select(0, command_indices).flatten(2)
    target_values = selected.gather(
        2,
        targets.view(-1, 1, 1).expand(-1, selected.shape[1], 1),
    ).squeeze(-1)
    return {
        "enemy_target_rate": target_values[:, 4].mean().item(),
        "neutral_target_rate": target_values[:, 5].mean().item(),
        "friendly_target_rate": target_values[:, 3].mean().item(),
        "base_target_rate": target_values[:, 2].mean().item(),
        "linked_target_rate": target_values[:, 1].mean().item(),
    }


def training_factor_label(args: argparse.Namespace, index: int) -> str:
    map_names = training_map_names(args.phase)
    map_count = len(map_names)
    map_name = map_names[index % map_count]
    if args.historical_opponent:
        opponent = historical_opponent_paths(args)[
            index % len(args.historical_opponent)
        ].name
        variant = index // map_count
        seat = "变换席位" if variant % 2 else "原始席位"
        submit = "对手先提交" if variant // 2 % 2 else "学习者先提交"
        strategy = variant // 4 % args.strategy_count
        return f"{map_name}|历史:{opponent}|策略{strategy}|{seat}|{submit}"
    opponent_count = len(args.rule_opponents)
    opponent = args.rule_opponents[index // map_count % opponent_count]
    variant = index // (map_count * opponent_count)
    seat = "变换席位" if variant % 2 else "原始席位"
    submit = "对手先提交" if variant // 2 % 2 else "学习者先提交"
    strategy = variant // 4 % args.strategy_count
    return f"{map_name}|规则:{opponent}|策略{strategy}|{seat}|{submit}"


def run_validation(
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[
    list[dict[str, object]], dict[str, object], tuple[float, float], bool
]:
    results = [
        evaluate_model(
            model=model,
            device=device,
            map_name=map_name,
            opponent=opponent,
            episodes=args.validation_episodes,
            num_envs=args.validation_num_envs,
            threads=args.threads,
            seed=(
                args.validation_seed
                + strategy_id * 1_000_000
                + map_index * 100_000
                + opponent_index * 10_000
            ),
            strategy_id=strategy_id,
        )
        for map_index, map_name in enumerate(training_map_names(args.phase))
        for opponent_index, opponent in enumerate(args.validation_opponents)
        for strategy_id in range(args.strategy_count)
    ]
    aggregate = aggregate_results(results)
    return (
        [result.as_dict() for result in results],
        aggregate.as_dict(),
        model_selection_key(aggregate),
        passes_validation_gate(
            results,
            args.minimum_validation_completion,
            args.minimum_validation_win_rate,
        ),
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = choose_device(args.device)
    checkpoint_path = resolve_artifact_path(args.checkpoint)
    best_checkpoint_path = resolve_artifact_path(args.best_checkpoint)
    report_path = resolve_artifact_path(args.report)
    model = build_model(device, args.strategy_count)
    optimized_parameters = (
        configure_strategy_adapter_training(model)
        if args.strategy_adapter_only
        else list(model.parameters())
    )
    optimizer = torch.optim.Adam(optimized_parameters, lr=args.learning_rate)
    anchor_model = None
    imitation_completed = 0
    ppo_completed = 0
    best_key = (-1.0, -1.0)
    environment_seed = args.seed

    if args.resume:
        resume_path = resolve_artifact_path(args.resume)
        checkpoint = load_checkpoint(resume_path, device)
        validate_resume_checkpoint(checkpoint, args)
        load_model_weights(model, checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer"])
        state = checkpoint["training_state"]
        args.seed = state["seed"]
        imitation_completed = state["imitation_updates_completed"]
        ppo_completed = state["ppo_updates_completed"]
        best_key = tuple(state["best_selection_key"])
        environment_seed = state["next_seed"]
        print(
            f"恢复训练：{resume_path} | 模仿 {imitation_completed} | "
            f"PPO {ppo_completed} | 新环境种子 {environment_seed}"
        )
    elif args.initialize_from:
        initialization_path = resolve_artifact_path(args.initialize_from)
        load_model_weights(model, load_checkpoint(initialization_path, device))
        print(f"加载初始权重：{initialization_path}")

    if args.anchor_checkpoint is not None:
        configured_anchor_path = resolve_artifact_path(args.anchor_checkpoint)
        anchor_checkpoint = load_checkpoint(configured_anchor_path, device)
        anchor_model = build_model(
            device, checkpoint_strategy_count(anchor_checkpoint)
        )
        load_model_weights(anchor_model, anchor_checkpoint)
        anchor_model.eval()
        anchor_model.requires_grad_(False)
        print(
            f"加载冻结锚点：{configured_anchor_path} | "
            f"KL 系数 {args.anchor_kl_coef:g}"
        )

    continuation_seed = (
        args.seed
        + environment_seed
        + imitation_completed * 1_000_003
        + ppo_completed * 10_000_019
    )
    seed_everything(continuation_seed)
    environment = build_environment(args, environment_seed)
    opponent_paths = historical_opponent_paths(args)
    opponent_pool = (
        HistoricalOpponentPool(opponent_paths, device, args.num_envs)
        if opponent_paths
        else None
    )
    next_seed = environment_seed + args.num_envs
    current = to_tensors(environment.observe(), device)
    strategy_ids = environment_strategy_ids(args, device)
    opponent_current = (
        to_tensors(environment.observe_opponents(), device)
        if opponent_pool is not None
        else None
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"-{args.seed}"
    append_report(
        report_path,
        {
            "event": "run_started",
            "run_id": run_id,
            "device": str(device),
            "args": json_safe_args(args),
            "imitation_updates_completed": imitation_completed,
            "ppo_updates_completed": ppo_completed,
        },
    )

    if args.validation_every and imitation_completed == 0 and ppo_completed == 0:
        results, aggregate, candidate_key, eligible = run_validation(
            args, model, device
        )
        append_report(
            report_path,
            {
                "event": "validation",
                "run_id": run_id,
                "update": 0,
                "stage": "initial",
                "results": results,
                "aggregate": aggregate,
                "selection_key": list(candidate_key),
                "validation_gate_passed": eligible,
                "minimum_completion_rate": args.minimum_validation_completion,
                "minimum_overall_win_rate": args.minimum_validation_win_rate,
            },
        )
        print(
            f"初始验证 | 总样本取胜率 {candidate_key[0]:.1%} | "
            f"正常完赛率 {candidate_key[1]:.1%}"
        )
        if eligible:
            best_key = candidate_key
            save_checkpoint(
                best_checkpoint_path,
                args,
                model,
                optimizer,
                next_seed,
                imitation_completed,
                ppo_completed,
                best_key,
            )
            print(f"初始模型已通过门禁并保存：{best_checkpoint_path}")

    for update in range(imitation_completed + 1, args.imitation_updates + 1):
        acted_on_observation = current.values
        expert = torch.as_tensor(
            environment.expert_actions(args.teacher), dtype=torch.long, device=device
        )
        logits, _ = model.logits_and_value(
            current.values, current.bases, current.masks, strategy_ids
        )
        sample_weights = torch.where(expert == 0, 1.0, args.command_weight)
        imitation_loss = (
            F.cross_entropy(logits, expert, reduction="none") * sample_weights
        ).mean()
        if anchor_model is None:
            anchor_kl = torch.zeros((), device=device)
        else:
            with torch.no_grad():
                anchor_logits, _ = anchor_model.logits_and_value(
                    current.values,
                    current.bases,
                    current.masks,
                    torch.zeros_like(strategy_ids),
                )
            anchor_kl = categorical_kl(anchor_logits, logits)
        loss = imitation_loss + args.anchor_kl_coef * anchor_kl
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        prediction = logits.argmax(dim=-1)
        use_model = torch.rand(args.num_envs, device=device) < args.dagger_model_prob
        rollout_action = torch.where(use_model, prediction, expert)
        transition, batch, opponent_current, next_seed = advance_environment(
            environment,
            rollout_action,
            opponent_current,
            opponent_pool,
            args.threads,
            next_seed,
        )
        current = to_tensors(batch, device)
        imitation_completed = update

        if update == 1 or update % 50 == 0 or update == args.imitation_updates:
            accuracy = (prediction == expert).float().mean().item()
            command_examples = expert != 0
            command_accuracy = (
                (prediction[command_examples] == expert[command_examples])
                .float()
                .mean()
                .item()
                if command_examples.any()
                else None
            )
            command_summary = (
                f"{command_accuracy:.1%}" if command_accuracy is not None else "无样本"
            )
            print(
                f"模仿 {update}/{args.imitation_updates} | 老师 {args.teacher} | "
                f"损失 {loss.item():.4f} | 动作准确率 {accuracy:.1%} | "
                f"指令准确率 {command_summary}"
            )
            append_report(
                report_path,
                {
                    "event": "imitation_update",
                    "run_id": run_id,
                    "update": update,
                    "loss": loss.item(),
                    "imitation_loss": imitation_loss.item(),
                    "anchor_kl": anchor_kl.item(),
                    "accuracy": accuracy,
                    "command_accuracy": command_accuracy,
                    **action_distribution(rollout_action),
                    **action_target_distribution(rollout_action, acted_on_observation),
                },
            )
        if args.checkpoint_every and update % args.checkpoint_every == 0:
            save_checkpoint(
                checkpoint_path,
                args,
                model,
                optimizer,
                next_seed,
                imitation_completed,
                ppo_completed,
                best_key,
            )

    completed = Counter({name: 0 for name in TERMINAL_NAMES})
    completed_by_factor: defaultdict[str, Counter[str]] = defaultdict(Counter)
    validations_without_improvement = 0
    for update in range(ppo_completed + 1, args.updates + 1):
        observations: list[Tensor] = []
        masks: list[Tensor] = []
        bases: list[Tensor] = []
        actions: list[Tensor] = []
        log_probs: list[Tensor] = []
        values: list[Tensor] = []
        rewards: list[Tensor] = []
        dones: list[Tensor] = []

        for _ in range(args.rollout_steps):
            with torch.no_grad():
                action, log_prob, value = model.act(
                    current.values, current.bases, current.masks, strategy_ids
                )
            transition, batch, opponent_current, next_seed = advance_environment(
                environment,
                action,
                opponent_current,
                opponent_pool,
                args.threads,
                next_seed,
            )
            reward = torch.as_tensor(transition.rewards, device=device)
            done = torch.as_tensor(
                [code != 0 for code in transition.end_codes], device=device
            ).float()
            observations.append(current.values)
            masks.append(current.masks)
            bases.append(current.bases)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            dones.append(done)
            completed.update(name for name in transition.end_names if name != "Ongoing")
            for index, name in enumerate(transition.end_names):
                if name != "Ongoing":
                    completed_by_factor[training_factor_label(args, index)][name] += 1
            current = to_tensors(batch, device)

        with torch.no_grad():
            _, next_value = model.logits_and_value(
                current.values, current.bases, current.masks, strategy_ids
            )
        reward_tensor = torch.stack(rewards)
        done_tensor = torch.stack(dones)
        value_tensor = torch.stack(values)
        advantages = torch.zeros_like(reward_tensor)
        gae = torch.zeros(args.num_envs, device=device)
        for step in reversed(range(args.rollout_steps)):
            following_value = (
                next_value
                if step == args.rollout_steps - 1
                else value_tensor[step + 1]
            )
            alive = 1.0 - done_tensor[step]
            delta = (
                reward_tensor[step]
                + args.gamma * following_value * alive
                - value_tensor[step]
            )
            gae = delta + args.gamma * args.gae_lambda * alive * gae
            advantages[step] = gae
        returns = advantages + value_tensor

        flat_observations = torch.cat(observations)
        flat_masks = torch.cat(masks)
        flat_bases = torch.cat(bases)
        flat_actions = torch.cat(actions)
        flat_log_probs = torch.cat(log_probs)
        flat_strategies = strategy_ids.repeat(args.rollout_steps)
        flat_advantages = advantages.flatten()
        flat_returns = returns.flatten()
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (
            flat_advantages.std() + 1e-8
        )
        batch_size = flat_actions.shape[0]
        optimization_metrics: list[
            tuple[float, float, float, float, float, float, float, float]
        ] = []

        for _ in range(args.epochs):
            batches = torch.randperm(batch_size, device=device).split(
                args.minibatch_size
            )
            for indices in batches:
                new_log_prob, entropy, new_value, current_logits = (
                    model.evaluate_actions(
                        flat_observations[indices],
                        flat_bases[indices],
                        flat_masks[indices],
                        flat_actions[indices],
                        flat_strategies[indices],
                    )
                )
                if anchor_model is None:
                    anchor_kl = torch.zeros((), device=device)
                else:
                    with torch.no_grad():
                        anchor_logits, _ = anchor_model.logits_and_value(
                            flat_observations[indices],
                            flat_bases[indices],
                            flat_masks[indices],
                            torch.zeros_like(flat_strategies[indices]),
                        )
                    anchor_kl = categorical_kl(anchor_logits, current_logits)
                strategy_js = strategy_js_divergence(
                    model,
                    flat_observations[indices],
                    flat_bases[indices],
                    flat_masks[indices],
                    args.strategy_count,
                    args.strategy_diversity_samples,
                )
                specialization = strategy_specialization_score(
                    current_logits,
                    flat_observations[indices],
                    flat_masks[indices],
                    flat_strategies[indices],
                )
                log_ratio = new_log_prob - flat_log_probs[indices]
                ratio = log_ratio.exp()
                unclipped = ratio * flat_advantages[indices]
                clipped = ratio.clamp(1.0 - args.clip, 1.0 + args.clip) * flat_advantages[
                    indices
                ]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = 0.5 * (new_value - flat_returns[indices]).square().mean()
                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + args.value_coef * value_loss
                    - args.entropy * entropy_mean
                    + args.anchor_kl_coef * anchor_kl
                    - args.strategy_diversity_coef * strategy_js
                    - args.strategy_specialization_coef * specialization
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > args.clip).float().mean()
                optimization_metrics.append(
                    (
                        policy_loss.item(),
                        value_loss.item(),
                        entropy_mean.item(),
                        approximate_kl.item(),
                        clip_fraction.item(),
                        anchor_kl.item(),
                        strategy_js.item(),
                        specialization.item(),
                    )
                )

        ppo_completed = update
        mean_metric = lambda index: sum(row[index] for row in optimization_metrics) / len(
            optimization_metrics
        )
        update_metrics = {
            "reward_mean": reward_tensor.mean().item(),
            "policy_loss": mean_metric(0),
            "value_loss": mean_metric(1),
            "entropy": mean_metric(2),
            "approximate_kl": mean_metric(3),
            "clip_fraction": mean_metric(4),
            "anchor_kl": mean_metric(5),
            "strategy_js_divergence": mean_metric(6),
            "strategy_specialization_score": mean_metric(7),
            **action_distribution(flat_actions),
            **action_target_distribution(flat_actions, flat_observations),
        }
        print(
            f"更新 {update}/{args.updates} | 设备 {device} | "
            f"平均奖励 {update_metrics['reward_mean']:.4f} | 终局 {dict(completed)}"
        )
        append_report(
            report_path,
            {
                "event": "ppo_update",
                "run_id": run_id,
                "update": update,
                "outcomes": dict(completed),
                "outcomes_by_training_factor": {
                    factor: dict(outcomes)
                    for factor, outcomes in sorted(completed_by_factor.items())
                },
                **update_metrics,
            },
        )

        should_validate = args.validation_every and (
            update % args.validation_every == 0 or update == args.updates
        )
        stop_after_update = False
        if should_validate:
            results, aggregate, candidate_key, eligible = run_validation(
                args, model, device
            )
            append_report(
                report_path,
                {
                    "event": "validation",
                    "run_id": run_id,
                    "update": update,
                    "results": results,
                    "aggregate": aggregate,
                    "selection_key": list(candidate_key),
                    "validation_gate_passed": eligible,
                    "minimum_completion_rate": args.minimum_validation_completion,
                    "minimum_overall_win_rate": args.minimum_validation_win_rate,
                },
            )
            print(
                f"验证 {update} | 总样本取胜率 {candidate_key[0]:.1%} | "
                f"正常完赛率 {candidate_key[1]:.1%}"
            )
            if eligible and candidate_key > best_key:
                best_key = candidate_key
                validations_without_improvement = 0
                save_checkpoint(
                    best_checkpoint_path,
                    args,
                    model,
                    optimizer,
                    next_seed,
                    imitation_completed,
                    ppo_completed,
                    best_key,
                )
                print(f"训练内最佳模型已保存：{best_checkpoint_path}")
            else:
                validations_without_improvement += 1
                if not eligible:
                    print(
                        "验证未通过逐地图×难度的完赛率/全样本胜率门禁，"
                        "当前模型不参与最佳模型选择"
                    )
                else:
                    print("验证通过硬门禁，但没有优于当前最佳模型")
                stop_after_update = bool(
                    args.early_stop_patience
                    and validations_without_improvement >= args.early_stop_patience
                )

        if args.checkpoint_every and update % args.checkpoint_every == 0:
            save_checkpoint(
                checkpoint_path,
                args,
                model,
                optimizer,
                next_seed,
                imitation_completed,
                ppo_completed,
                best_key,
            )
        if stop_after_update:
            append_report(
                report_path,
                {
                    "event": "early_stopped",
                    "run_id": run_id,
                    "update": update,
                    "validations_without_improvement": validations_without_improvement,
                },
            )
            print(
                f"连续 {validations_without_improvement} 次验证无改进，训练提前停止"
            )
            break

    save_checkpoint(
        checkpoint_path,
        args,
        model,
        optimizer,
        next_seed,
        imitation_completed,
        ppo_completed,
        best_key,
    )
    append_report(
        report_path,
        {
            "event": "run_completed",
            "run_id": run_id,
            "checkpoint": str(checkpoint_path),
            "imitation_updates_completed": imitation_completed,
            "ppo_updates_completed": ppo_completed,
            "best_selection_key": list(best_key),
        },
    )
    print(f"模型已保存：{checkpoint_path}")
    print(f"训练报告：{report_path}")


if __name__ == "__main__":
    main()
