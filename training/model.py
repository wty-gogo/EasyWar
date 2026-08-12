from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.distributions import Categorical


def temperature_scaled_logits(
    logits: Tensor, action_mask: Tensor, temperature: float
) -> Tensor:
    """只缩放合法动作，避免极小掩码值在低温下溢出。"""

    if temperature <= 0.0:
        raise ValueError("策略采样温度必须大于 0")
    scaled = torch.where(action_mask, logits, torch.zeros_like(logits)) / temperature
    return scaled.masked_fill(~action_mask, torch.finfo(logits.dtype).min)


def deterministic_strategy_codes(
    strategy_count: int,
    hidden: int,
    scale: float = 0.05,
) -> Tensor:
    """保留策略 0 的旧模型行为，并用微小确定性编码打破新增策略的对称。"""

    codes = torch.zeros((strategy_count, hidden))
    for strategy_id in range(1, strategy_count):
        positive = (2 * strategy_id - 2) % hidden
        negative = (2 * strategy_id - 1) % hidden
        codes[strategy_id, positive] = scale
        codes[strategy_id, negative] = -scale
    return codes


class EasyWarActorCritic(nn.Module):
    """用空间特征组合源据点与目标格，避免巨大无结构全连接动作头。"""

    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        max_bases: int,
        strategy_count: int = 1,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.max_bases = max_bases
        self.observation_channels = channels
        self.strategy_count = strategy_count
        self.cells = height * width
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.source_projection = nn.Linear(hidden, hidden)
        self.target_projection = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.source_context_projection = nn.Linear(hidden, hidden)
        self.target_context_projection = nn.Linear(hidden, hidden)
        self.stop_head = nn.Linear(hidden, 1)
        self.no_op_head = nn.Linear(hidden, 1)
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.strategy_embedding = nn.Embedding(strategy_count, hidden)
        with torch.no_grad():
            self.strategy_embedding.weight.copy_(
                deterministic_strategy_codes(strategy_count, hidden)
            )
            # 从旧模型升级时先保持原动作不变，再由新课程学习全局局势修正量。
            self.source_context_projection.weight.zero_()
            self.source_context_projection.bias.zero_()
            self.target_context_projection.weight.zero_()
            self.target_context_projection.bias.zero_()

    def logits_and_value(
        self,
        observation: Tensor,
        base_cells: Tensor,
        action_mask: Tensor,
        strategy_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        features = self.encoder(observation)
        resolved_strategies = (
            torch.zeros(
                observation.shape[0], dtype=torch.long, device=observation.device
            )
            if strategy_ids is None
            else strategy_ids
        )
        if resolved_strategies.shape != (observation.shape[0],):
            raise ValueError("策略编号必须与批量观察一一对应")
        style = self.strategy_embedding(resolved_strategies)
        features = features + style.unsqueeze(-1).unsqueeze(-1)
        pooled = features.mean(dim=(2, 3))
        flat = features.flatten(2).transpose(1, 2)
        safe_cells = base_cells.clamp(min=0)
        base_features = flat.gather(
            1,
            safe_cells.unsqueeze(-1).expand(-1, -1, flat.shape[-1]),
        )
        source = self.source_projection(base_features)
        source = source + self.source_context_projection(pooled).unsqueeze(1)
        target = self.target_projection(features).flatten(2).transpose(1, 2)
        target = target + self.target_context_projection(pooled).unsqueeze(1)
        pair_logits = torch.einsum("bsh,bth->bst", source, target) / math.sqrt(
            source.shape[-1]
        )
        stop_logits = self.stop_head(base_features).squeeze(-1)
        no_op_logits = self.no_op_head(pooled)
        logits = torch.cat(
            [no_op_logits, pair_logits.flatten(1), stop_logits], dim=1
        )
        logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        value = self.value_head(pooled).squeeze(-1)
        return logits, value

    def act(
        self,
        observation: Tensor,
        base_cells: Tensor,
        action_mask: Tensor,
        strategy_ids: Tensor | None = None,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        logits, value = self.logits_and_value(
            observation, base_cells, action_mask, strategy_ids
        )
        distribution = Categorical(
            logits=temperature_scaled_logits(logits, action_mask, temperature)
        )
        action = logits.argmax(dim=1) if deterministic else distribution.sample()
        return action, distribution.log_prob(action), value

    def evaluate_actions(
        self,
        observation: Tensor,
        base_cells: Tensor,
        action_mask: Tensor,
        action: Tensor,
        strategy_ids: Tensor | None = None,
        temperature: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        logits, value = self.logits_and_value(
            observation, base_cells, action_mask, strategy_ids
        )
        distribution = Categorical(
            logits=temperature_scaled_logits(logits, action_mask, temperature)
        )
        return distribution.log_prob(action), distribution.entropy(), value, logits
