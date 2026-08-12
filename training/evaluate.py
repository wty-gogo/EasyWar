from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluation import aggregate_results, evaluate_model, format_result, verify_replay
from runtime import (
    build_model,
    checkpoint_observation_channels,
    checkpoint_strategy_count,
    choose_device,
    load_model_weights,
    resolve_artifact_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 EasyWar 1v1 神经网络 AI")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--map", dest="maps", nargs="+", default=["ring_chord_1v1.toml"])
    parser.add_argument("--opponent", choices=["easy", "normal", "hard"], default="normal")
    parser.add_argument(
        "--seat-transform",
        choices=["auto", "identity", "vertical", "rotational"],
        default="auto",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument(
        "--policy-temperature",
        type=float,
        default=0.0,
        help="评测动作采样温度；0 使用确定性最高分动作",
    )
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--strategy-ids",
        type=int,
        nargs="+",
        help="指定要评测的可控策略编号；默认评测检查点中的全部策略",
    )
    parser.add_argument("--json", type=Path, help="写入结构化评测报告")
    parser.add_argument("--replays", type=Path, help="写入每种终局的一局确定性代表回放")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint_path = resolve_artifact_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    tactical_actions = bool(
        checkpoint.get("training_state", {}).get("tactical_actions", False)
    )
    strategy_count = checkpoint_strategy_count(checkpoint)
    strategy_ids = args.strategy_ids or list(range(strategy_count))
    invalid_strategies = [
        strategy for strategy in strategy_ids if not 0 <= strategy < strategy_count
    ]
    if invalid_strategies:
        raise ValueError(f"策略编号超出 0..{strategy_count}：{invalid_strategies}")
    model = build_model(
        device,
        strategy_count,
        checkpoint_observation_channels(checkpoint),
    )
    load_model_weights(model, checkpoint)
    transform = None if args.seat_transform == "auto" else args.seat_transform
    results = [
        evaluate_model(
            model=model,
            device=device,
            map_name=map_name,
            opponent=args.opponent,
            episodes=args.episodes,
            num_envs=args.num_envs,
            threads=args.threads,
            seed=args.seed + index * 100_000,
            seat_transform=transform,
            capture_replays=bool(args.replays),
            strategy_id=strategy_id,
            tactical_actions=tactical_actions,
            policy_temperature=args.policy_temperature,
        )
        for index, map_name in enumerate(args.maps)
        for strategy_id in strategy_ids
    ]
    for result in results:
        print(format_result(result))
    aggregate = aggregate_results(results)
    if len(results) > 1:
        print(format_result(aggregate))
    print("注意：神经网络胜率只能验证策略泛化，不能单独证明地图平衡或有趣。")

    replays = [
        replay
        for result in results
        for replay in result.representative_replays
    ]
    if args.replays:
        for replay in replays:
            verify_replay(replay, args.threads or 1)
        replay_path = resolve_artifact_path(args.replays)
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint_path),
                    "traces": [replay.as_dict() for replay in replays],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"代表回放已复验并保存：{replay_path}")

    if args.json:
        report_path = resolve_artifact_path(args.json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint_path),
                    "results": [result.as_dict() for result in results],
                    "aggregate": aggregate.as_dict(),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"结构化报告已保存：{report_path}")


if __name__ == "__main__":
    main()
