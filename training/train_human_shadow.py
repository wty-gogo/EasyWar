"""把多局真人状态动作埋点训练为可加入自博弈联盟的玩家影子。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from human_replay import HumanSample, load_human_samples, split_sessions
from runtime import (
    build_model,
    checkpoint_observation_channels,
    checkpoint_strategy_count,
    choose_device,
    load_model_weights,
    resolve_artifact_path,
    seed_everything,
)


@dataclass(frozen=True)
class ShadowMetrics:
    loss: float
    accuracy: float
    command_accuracy: float | None
    samples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 EasyWar 真人玩家影子")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument(
        "--initialize-from",
        type=Path,
        default=Path("checkpoints/selfplay-league-g001.pt"),
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/human-shadow.pt")
    )
    parser.add_argument("--mask-mode", choices=["player", "tactical"], default="player")
    parser.add_argument("--wait-to-command-ratio", type=float, default=2.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--command-weight", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20_260_812)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def sample_tensors(
    samples: tuple[HumanSample, ...], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(np.stack([sample.values for sample in samples]), device=device),
        torch.as_tensor(np.stack([sample.bases for sample in samples]), device=device),
        torch.as_tensor(np.stack([sample.mask for sample in samples]), device=device),
        torch.as_tensor(
            [sample.action for sample in samples], dtype=torch.long, device=device
        ),
    )


def policy_metrics(
    model: torch.nn.Module,
    samples: tuple[HumanSample, ...],
    device: torch.device,
    batch_size: int,
) -> ShadowMetrics:
    if not samples:
        return ShadowMetrics(0.0, 0.0, None, 0)
    losses: list[torch.Tensor] = []
    correct = 0
    command_correct = 0
    commands = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            observations, bases, masks, actions = sample_tensors(batch, device)
            logits, _ = model.logits_and_value(observations, bases, masks)
            losses.append(F.cross_entropy(logits, actions, reduction="sum"))
            predictions = logits.argmax(dim=-1)
            correct += int((predictions == actions).sum().item())
            command_mask = actions != 0
            commands += int(command_mask.sum().item())
            command_correct += int(
                (predictions[command_mask] == actions[command_mask]).sum().item()
            )
    return ShadowMetrics(
        loss=sum(loss.item() for loss in losses) / len(samples),
        accuracy=correct / len(samples),
        command_accuracy=command_correct / commands if commands else None,
        samples=len(samples),
    )


def train(args: argparse.Namespace) -> tuple[Path, ShadowMetrics, ShadowMetrics]:
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0.0:
        raise ValueError("训练轮数、批量大小和学习率必须大于 0")
    if args.command_weight <= 0.0:
        raise ValueError("指令权重必须大于 0")
    seed_everything(args.seed)
    replays, samples = load_human_samples(
        args.inputs,
        args.mask_mode,
        args.wait_to_command_ratio,
        completed_only=True,
    )
    if not replays:
        raise ValueError("没有包含正常终局的真人埋点")
    if not any(not sample.is_wait for sample in samples):
        raise ValueError("真人埋点中没有可训练的指令动作")
    training_ids, validation_ids = split_sessions(
        replays, args.validation_fraction, args.seed
    )
    training = tuple(sample for sample in samples if sample.session_id in training_ids)
    validation = tuple(sample for sample in samples if sample.session_id in validation_ids)
    device = choose_device(args.device)
    initial_path = resolve_artifact_path(args.initialize_from)
    checkpoint = torch.load(initial_path, map_location=device, weights_only=True)
    model = build_model(
        device,
        checkpoint_strategy_count(checkpoint),
        checkpoint_observation_channels(checkpoint),
    )
    load_model_weights(model, checkpoint)
    actor_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("value_head.")
    ]
    optimizer = torch.optim.Adam(actor_parameters, lr=args.learning_rate)
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(len(training)).tolist()
        epoch_loss = 0.0
        for start in range(0, len(training), args.batch_size):
            batch = tuple(
                training[index]
                for index in permutation[start : start + args.batch_size]
            )
            observations, bases, masks, actions = sample_tensors(batch, device)
            logits, _ = model.logits_and_value(observations, bases, masks)
            weights = torch.where(
                actions == 0,
                torch.ones_like(actions, dtype=torch.float32),
                torch.full_like(actions, args.command_weight, dtype=torch.float32),
            )
            losses = F.cross_entropy(logits, actions, reduction="none")
            loss = (losses * weights).sum() / weights.sum()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_parameters, 0.5)
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"玩家影子 {epoch}/{args.epochs} | 加权损失 {epoch_loss / len(training):.4f}")
    training_metrics = policy_metrics(model, training, device, args.batch_size)
    validation_metrics = policy_metrics(model, validation, device, args.batch_size)
    output = resolve_artifact_path(args.checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "model": model.state_dict(),
        "training_state": {
            "observation_channels": model.observation_channels,
            "strategy_count": model.strategy_count,
            "tactical_actions": args.mask_mode == "tactical",
            "source": "human_telemetry_shadow",
            "sessions": [replay.session_id for replay in replays],
            "mask_mode": args.mask_mode,
            "training_metrics": training_metrics.__dict__,
            "validation_metrics": validation_metrics.__dict__,
        },
        "training_args": {
            key: (
                [str(item) if isinstance(item, Path) else item for item in value]
                if isinstance(value, list)
                else str(value) if isinstance(value, Path) else value
            )
            for key, value in vars(args).items()
        },
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    print(
        json.dumps(
            {
                "checkpoint": str(output),
                "training": training_metrics.__dict__,
                "validation": validation_metrics.__dict__,
            },
            ensure_ascii=False,
        )
    )
    return output, training_metrics, validation_metrics


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
