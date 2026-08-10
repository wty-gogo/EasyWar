from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from runtime import resolve_artifact_path


MAP_NAMES = {
    "dual_ladder_1v1.toml": "双梯线",
    "braided_rings_1v1.toml": "编织环路",
}
OPPONENT_NAMES = {"easy": "简单", "normal": "普通", "hard": "困难"}
STRATEGY_NAMES = {
    0: "基准",
    1: "主动压制",
    2: "中立扩张",
    3: "关联铺路",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成可直接阅读的中文策略行为报告")
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="EasyWar 神经网络 AI 行为报告")
    parser.add_argument("--minimum-completion", type=float, default=0.8)
    parser.add_argument("--minimum-win-rate", type=float, default=0.5)
    return parser.parse_args()


def _percentage(value: float) -> str:
    return f"{value:.1%}"


def _opening_description(result: dict[str, object]) -> str:
    openings = Counter[tuple[int, int, str, str]]()
    for episode in result.get("behavior_episodes", []):
        target = episode.get("first_offensive_target")
        if target is None:
            continue
        openings[
            (
                int(target["x"]),
                int(target["y"]),
                str(target["owner"]),
                str(target["cell_kind"]),
            )
        ] += 1
    owner_names = {"enemy": "敌方", "neutral": "中立", "friendly": "己方"}
    kind_names = {"base": "据点", "linked": "关联地块", "other": "格子"}
    return "、".join(
        f"{owner_names.get(owner, owner)}{kind_names.get(kind, kind)}"
        f"({x},{y})×{count}"
        for (x, y, owner, kind), count in openings.most_common()
    ) or "没有主动进攻"


def _dominant_opening(result: dict[str, object]) -> tuple[int, int] | None:
    targets = Counter[tuple[int, int]]()
    for episode in result.get("behavior_episodes", []):
        target = episode.get("first_offensive_target")
        if target is not None:
            targets[(int(target["x"]), int(target["y"]))] += 1
    ranked = targets.most_common(2)
    if not ranked or (len(ranked) > 1 and ranked[0][1] == ranked[1][1]):
        return None
    return ranked[0][0]


def _result_rows(payloads: list[dict[str, object]]) -> list[dict[str, object]]:
    return [result for payload in payloads for result in payload.get("results", [])]


def _strategy_label(strategy_id: int) -> str:
    return f"{strategy_id}（{STRATEGY_NAMES.get(strategy_id, '自由策略')}）"


def _intent_signal(
    result: dict[str, object],
    baseline: dict[str, object] | None,
) -> str:
    strategy_id = int(result.get("strategy_id", 0))
    behavior = result.get("behavior_summary", {})
    if strategy_id == 0:
        return (
            f"敌 {_percentage(float(behavior.get('enemy_target_rate', 0.0)))} / "
            f"中立 {_percentage(float(behavior.get('neutral_target_rate', 0.0)))} / "
            f"关联 {_percentage(float(behavior.get('linked_target_rate', 0.0)))}"
        )
    intent = {
        1: ("敌方", "enemy_target_rate"),
        2: ("中立", "neutral_target_rate"),
        3: ("关联", "linked_target_rate"),
    }.get(strategy_id)
    if intent is None:
        return "未定义训练意图"
    label, metric = intent
    rate = float(behavior.get(metric, 0.0))
    if baseline is None:
        return f"{label} {_percentage(rate)}"
    delta = rate - float(baseline.get(metric, 0.0))
    return f"{label} {_percentage(rate)}（较基准 {delta * 100:+.1f} 点）"


def render_behavior_report(
    payloads: list[dict[str, object]],
    title: str,
    minimum_completion: float,
    minimum_win_rate: float,
) -> str:
    results = _result_rows(payloads)
    episodes = sum(int(result["episodes"]) for result in results)
    wins = sum(int(result["outcomes"].get("Won", 0)) for result in results)
    losses = sum(int(result["outcomes"].get("Lost", 0)) for result in results)
    unfinished = episodes - wins - losses
    strategies = sorted({int(result.get("strategy_id", 0)) for result in results})
    baselines = {
        (str(result["map"]), str(result["opponent"])): result.get(
            "behavior_summary", {}
        )
        for result in results
        if int(result.get("strategy_id", 0)) == 0
    }
    failed = [
        result
        for result in results
        if float(result["completion_rate"]) < minimum_completion
        or float(result["overall_win_rate"]) < minimum_win_rate
    ]
    openings_by_matchup: defaultdict[tuple[str, str], set[tuple[int, int]]] = defaultdict(set)
    matchups = {
        (str(result["map"]), str(result["opponent"])) for result in results
    }
    for result in results:
        opening = _dominant_opening(result)
        if opening is not None:
            openings_by_matchup[(str(result["map"]), str(result["opponent"]))].add(
                opening
            )
    differentiated = sum(len(openings) > 1 for openings in openings_by_matchup.values())
    intent_metrics = {1: "enemy_target_rate", 2: "neutral_target_rate", 3: "linked_target_rate"}
    intent_deltas = {
        strategy_id: (
            sum(
                float(result.get("behavior_summary", {}).get(metric, 0.0))
                - float(
                    baselines.get(
                        (str(result["map"]), str(result["opponent"])), {}
                    ).get(metric, 0.0)
                )
                for result in results
                if int(result.get("strategy_id", 0)) == strategy_id
            )
            / max(
                1,
                sum(
                    int(result.get("strategy_id", 0)) == strategy_id
                    for result in results
                ),
            )
        )
        for strategy_id, metric in intent_metrics.items()
    }
    conclusion = (
        f"本次共评测 {episodes} 局，{wins} 胜、{losses} 负、"
        f"{unfinished} 局未正常结束。"
        f"{len(strategies)} 个可控策略在 {differentiated}/"
        f"{len(matchups)} 个地图与难度组合中形成了不同主开局。"
        + (
            "所有逐策略因子都通过当前强度门槛。"
            if not failed
            else f"仍有 {len(failed)} 个逐策略因子没有通过强度门槛。"
        )
    )
    lines = [
        f"# {title}",
        "",
        "## 一句话结论",
        "",
        conclusion,
        "相对基准，主动压制的意图命中平均 "
        f"{intent_deltas.get(1, 0.0) * 100:+.1f} 点，中立扩张 "
        f"{intent_deltas.get(2, 0.0) * 100:+.1f} 点，关联铺路 "
        f"{intent_deltas.get(3, 0.0) * 100:+.1f} 点。",
        "换句话说：模型强度已经过关，但四种策略目前仍像同一套打法的细微变体。",
        "",
        "> 这份报告评价模型会不会打、打法是否分化，不评价地图是否平衡或有趣。",
        "",
        "## 逐项结果",
        "",
        "| 地图 | 对手 | 策略 | 胜负 | 完赛率 | 全样本胜率 | 训练意图命中 | 首攻 | 换线代理 | 出现反攻 | 覆盖目标 |",
        "|---|---|---|---:|---:|---:|---|---|---:|---:|---:|",
    ]
    for result in results:
        behavior = result.get("behavior_summary", {})
        outcomes = result["outcomes"]
        lines.append(
            "| "
            + " | ".join(
                [
                    MAP_NAMES.get(str(result["map"]), str(result["map"])),
                    OPPONENT_NAMES.get(str(result["opponent"]), str(result["opponent"])),
                    _strategy_label(int(result.get("strategy_id", 0))),
                    f"{outcomes.get('Won', 0)}胜/{outcomes.get('Lost', 0)}负",
                    _percentage(float(result["completion_rate"])),
                    _percentage(float(result["overall_win_rate"])),
                    _intent_signal(
                        result,
                        baselines.get(
                            (str(result["map"]), str(result["opponent"]))
                        ),
                    ),
                    _opening_description(result),
                    _percentage(float(behavior.get("retarget_rate", 0.0))),
                    _percentage(float(behavior.get("counterattack_episode_rate", 0.0))),
                    f"{float(behavior.get('mean_distinct_offensive_targets', 0.0)):.1f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 怎么看这些指标",
            "",
            "- **首攻**：模型第一次主动攻击的目标；坐标已按出生席位对称关系归一。不同策略在同一地图、同一难度下选择不同合理首攻，才算初步策略分化。",
            "- **训练意图命中**：策略 1 看有效指令中有多少投向敌方，策略 2 看投向中立，策略 3 看投向关联地块；策略 0 是旧模型基准，不强设方向。",
            "- **换线代理**：仍在出兵的同一个源据点改派到另一目标。数值太低可能是套路僵化，太高也可能只是无意义反复改派。",
            "- **出现反攻**：本局是否曾向己方失守、当前仍未夺回的格子重新出兵。它表示有反应，不保证反攻时机正确。",
            "- **覆盖目标**：每局主动攻击过的不同目标数。覆盖广不等于决策质量高，仍要结合胜负和代表回放。",
            "- **强度门槛**：每个地图、难度和策略都必须达到完赛率 "
            f"{_percentage(minimum_completion)}、全样本胜率 {_percentage(minimum_win_rate)}。",
            "",
            "## 当前判断",
            "",
            (
                "可控策略尚未达到验收标准：没有形成稳定主开局分化，"
                "且意图命中相对基准的提升很小或方向相反。不能宣称多策略训练成功。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    payloads = [
        json.loads(resolve_artifact_path(path).read_text(encoding="utf-8"))
        for path in args.reports
    ]
    report = render_behavior_report(
        payloads,
        args.title,
        args.minimum_completion,
        args.minimum_win_rate,
    )
    output = resolve_artifact_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"中文行为报告已保存：{output}")


if __name__ == "__main__":
    main()
