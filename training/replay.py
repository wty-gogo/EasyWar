from __future__ import annotations

import argparse
import json
from pathlib import Path

from behavior import episode_behavior_from_dict
from evaluation import ReplayTrace, verify_replay
from runtime import resolve_artifact_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="复验 EasyWar 神经网络评测代表回放"
    )
    parser.add_argument("replays", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(
        resolve_artifact_path(args.replays).read_text(encoding="utf-8")
    )
    traces = [
        ReplayTrace(
            map_name=trace["map"],
            opponent=trace["opponent"],
            seed=trace["seed"],
            outcome=trace["outcome"],
            variant=trace["variant"],
            seat_transform=trace["seat_transform"],
            actions=tuple(trace["actions"]),
            decisions=trace["decisions"],
            strategy_id=trace.get("strategy_id", 0),
            behavior=(
                episode_behavior_from_dict(trace["behavior"])
                if trace.get("behavior") is not None
                else None
            ),
        )
        for trace in payload["traces"]
    ]
    for trace in traces:
        verify_replay(trace, args.threads)
        print(
            f"回放通过：{trace.map_name} | {trace.outcome} | "
            f"种子 {trace.seed} | 决策 {trace.decisions}"
        )


if __name__ == "__main__":
    main()
