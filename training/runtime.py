from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import easywar_rl
import numpy as np
import torch
from torch import Tensor

from model import EasyWarActorCritic


BASE_OBSERVATION_CHANNELS = easywar_rl.OBSERVATION_CHANNELS
TEMPORAL_OBSERVATION_CHANNELS = BASE_OBSERVATION_CHANNELS * 2


@dataclass(frozen=True)
class TensorObservation:
    values: Tensor
    masks: Tensor
    bases: Tensor

    def select(self, indices: Tensor) -> TensorObservation:
        return TensorObservation(
            values=self.values.index_select(0, indices),
            masks=self.masks.index_select(0, indices),
            bases=self.bases.index_select(0, indices),
        )


def to_model_tensors(
    batch: object,
    device: torch.device,
    observation_channels: int,
    previous_values: Tensor | None = None,
    reset_indices: list[int] | tuple[int, ...] = (),
) -> tuple[TensorObservation, Tensor]:
    """把当前局面与逐环境变化量组合成统一策略输入。"""

    current = to_tensors(batch, device)
    if 0 < observation_channels <= BASE_OBSERVATION_CHANNELS:
        return (
            TensorObservation(
                values=current.values[:, :observation_channels],
                masks=current.masks,
                bases=current.bases,
            ),
            current.values,
        )
    if observation_channels != TEMPORAL_OBSERVATION_CHANNELS:
        raise ValueError(f"不支持的模型观察通道数：{observation_channels}")
    delta = (
        torch.zeros_like(current.values)
        if previous_values is None
        else current.values - previous_values
    )
    if reset_indices:
        indices = torch.as_tensor(reset_indices, dtype=torch.long, device=device)
        delta = delta.clone()
        delta.index_fill_(0, indices, 0.0)
    return (
        TensorObservation(
            values=torch.cat((current.values, delta), dim=1),
            masks=current.masks,
            bases=current.bases,
        ),
        current.values,
    )


def model_input_values(model: EasyWarActorCritic, values: Tensor) -> Tensor:
    """让旧观察锚点读取新模型输入中的兼容通道前缀。"""

    expected = model.observation_channels
    actual = values.shape[1]
    if actual == expected:
        return values
    if actual > expected:
        return values[:, :expected]
    if expected == actual * 2:
        return torch.cat((values, torch.zeros_like(values)), dim=1)
    raise ValueError(f"观察通道不兼容：模型需要 {expected}，实际 {actual}")


class HistoricalOpponentPool:
    """把多个冻结检查点稳定分配到批量环境，组成历史对手池。"""

    def __init__(
        self,
        paths: list[Path],
        device: torch.device,
        environment_count: int,
        temperature: float = 0.0,
    ) -> None:
        if not paths:
            raise ValueError("历史模型对手池不能为空")
        if temperature < 0.0:
            raise ValueError("历史对手采样温度不能小于 0")
        self.models = [self._load_model(path, device) for path in paths]
        self.temperature = temperature
        self.assignments = torch.arange(environment_count, device=device) % len(
            self.models
        )

    @staticmethod
    def _load_model(path: Path, device: torch.device) -> EasyWarActorCritic:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = build_model(
            device,
            checkpoint_strategy_count(checkpoint),
            checkpoint_observation_channels(checkpoint),
        )
        load_model_weights(model, checkpoint)
        model.eval()
        model.requires_grad_(False)
        return model

    def actions(self, observation: TensorObservation) -> Tensor:
        actions = torch.zeros(
            observation.values.shape[0], dtype=torch.long, device=observation.values.device
        )
        with torch.no_grad():
            for model_index, model in enumerate(self.models):
                indices = torch.nonzero(
                    self.assignments == model_index, as_tuple=False
                ).flatten()
                if indices.numel() == 0:
                    continue
                selected = observation.select(indices)
                selected_actions, _, _ = model.act(
                    model_input_values(model, selected.values),
                    selected.bases,
                    selected.masks,
                    deterministic=self.temperature == 0.0,
                    temperature=self.temperature or 1.0,
                )
                actions.index_copy_(0, indices, selected_actions)
        return actions


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


def training_directory() -> Path:
    return Path(__file__).resolve().parent


def resolve_artifact_path(path: Path) -> Path:
    return path if path.is_absolute() else training_directory() / path


def training_map_names(phase: str) -> list[str]:
    if phase != "main":
        raise ValueError("H 图已退出训练课程；当前只允许 main 主策略课程")
    return [
        "dual_ladder_1v1.toml",
        "braided_rings_1v1.toml",
    ]


def training_maps(phase: str) -> list[str]:
    maps = repository_root() / "assets" / "maps"
    return [str(maps / name) for name in training_map_names(phase)]


def map_transform(map_name: str) -> str:
    if map_name == "h_1v1.toml":
        raise ValueError("H 图只允许用于环境接口夹具，不得进入训练或评测课程")
    return "vertical"


def training_transforms(phase: str) -> list[str]:
    return [map_transform(name) for name in training_map_names(phase)]


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def to_tensors(batch: object, device: torch.device) -> TensorObservation:
    values = torch.as_tensor(
        np.asarray(batch.observations, dtype=np.float32), device=device
    ).reshape(
        -1,
        easywar_rl.OBSERVATION_CHANNELS,
        easywar_rl.MAX_HEIGHT,
        easywar_rl.MAX_WIDTH,
    )
    masks = torch.as_tensor(
        np.asarray(batch.action_masks, dtype=np.bool_), device=device
    )
    bases = torch.as_tensor(
        np.asarray(batch.base_cells, dtype=np.int64), device=device
    )
    return TensorObservation(values=values, masks=masks, bases=bases)


def build_model(
    device: torch.device,
    strategy_count: int = 1,
    observation_channels: int = BASE_OBSERVATION_CHANNELS,
) -> EasyWarActorCritic:
    return EasyWarActorCritic(
        observation_channels,
        easywar_rl.MAX_HEIGHT,
        easywar_rl.MAX_WIDTH,
        easywar_rl.MAX_BASES,
        strategy_count=strategy_count,
    ).to(device)


def checkpoint_observation_channels(checkpoint: dict[str, object]) -> int:
    state = checkpoint.get("training_state")
    if isinstance(state, dict) and "observation_channels" in state:
        return int(state["observation_channels"])
    model_state = checkpoint.get("model")
    if isinstance(model_state, dict):
        first_layer = model_state.get("encoder.0.weight")
        if isinstance(first_layer, Tensor):
            return int(first_layer.shape[1])
    return BASE_OBSERVATION_CHANNELS


def checkpoint_strategy_count(checkpoint: dict[str, object]) -> int:
    state = checkpoint.get("training_state")
    if isinstance(state, dict) and "strategy_count" in state:
        return int(state["strategy_count"])
    model_state = checkpoint.get("model")
    if isinstance(model_state, dict):
        embedding = model_state.get("strategy_embedding.weight")
        if isinstance(embedding, Tensor):
            return int(embedding.shape[0])
    return 1


def load_model_weights(
    model: EasyWarActorCritic,
    checkpoint: dict[str, object],
) -> None:
    source = dict(checkpoint["model"])
    target = model.state_dict()
    embedding_key = "strategy_embedding.weight"
    if embedding_key in source and source[embedding_key].shape != target[embedding_key].shape:
        expanded = target[embedding_key].clone()
        shared = min(source[embedding_key].shape[0], expanded.shape[0])
        expanded[:shared] = source[embedding_key][:shared]
        source[embedding_key] = expanded
    first_layer_key = "encoder.0.weight"
    if (
        first_layer_key in source
        and source[first_layer_key].shape != target[first_layer_key].shape
    ):
        old = source[first_layer_key]
        expanded = target[first_layer_key].clone()
        if (
            old.shape[0] != expanded.shape[0]
            or old.shape[2:] != expanded.shape[2:]
            or old.shape[1] > expanded.shape[1]
        ):
            raise ValueError(
                f"首层卷积权重不兼容：来源 {tuple(old.shape)}，目标 {tuple(expanded.shape)}"
            )
        expanded[:, : old.shape[1]] = old
        expanded[:, old.shape[1] :] = 0.0
        source[first_layer_key] = expanded
    missing, unexpected = model.load_state_dict(source, strict=False)
    allowed_missing = {
        embedding_key,
        "source_context_projection.weight",
        "source_context_projection.bias",
        "target_context_projection.weight",
        "target_context_projection.bias",
    }
    if set(missing) - allowed_missing or unexpected:
        raise ValueError(
            f"模型权重不兼容：缺少 {missing}，多出 {unexpected}"
        )


def environment_signature() -> dict[str, int]:
    return {
        "observation_channels": easywar_rl.OBSERVATION_CHANNELS,
        "max_width": easywar_rl.MAX_WIDTH,
        "max_height": easywar_rl.MAX_HEIGHT,
        "max_bases": easywar_rl.MAX_BASES,
        "action_count": easywar_rl.ACTION_COUNT,
    }
