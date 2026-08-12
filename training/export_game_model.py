from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path

import torch
from torch import Tensor

from runtime import resolve_artifact_path


MAGIC = b"EWNNv2\0\0"
MODEL_SPATIAL_SHAPE = (13, 17, 16, 64)
ACTOR_TENSORS = (
    "encoder.0.weight",
    "encoder.0.bias",
    "encoder.2.weight",
    "encoder.2.bias",
    "source_projection.weight",
    "source_projection.bias",
    "target_projection.weight",
    "target_projection.bias",
    "source_context_projection.weight",
    "source_context_projection.bias",
    "target_context_projection.weight",
    "target_context_projection.bias",
    "stop_head.weight",
    "stop_head.bias",
    "no_op_head.weight",
    "no_op_head.bias",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出游戏内神经模型权重")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategy-id", type=int, default=0)
    return parser.parse_args()


def _tensor_bytes(tensor: Tensor) -> bytes:
    values = tensor.detach().cpu().to(torch.float32).contiguous().flatten()
    return struct.pack(f"<{values.numel()}f", *values.tolist())


def export_actor(
    checkpoint_path: Path,
    output_path: Path,
    strategy_id: int,
) -> str:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["model"]
    strategies = state["strategy_embedding.weight"]
    if not 0 <= strategy_id < strategies.shape[0]:
        raise ValueError(f"策略编号必须位于 0..{strategies.shape[0] - 1}")
    tensors = [state[name] for name in ACTOR_TENSORS] + [strategies[strategy_id]]
    input_channels = int(state["encoder.0.weight"].shape[1])
    payload = bytearray(MAGIC)
    payload.extend(struct.pack("<5I", input_channels, *MODEL_SPATIAL_SHAPE))
    for tensor in tensors:
        values = _tensor_bytes(tensor)
        payload.extend(struct.pack("<I", len(values) // 4))
        payload.extend(values)
    digest = hashlib.sha256(payload).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    return digest


def main() -> None:
    args = parse_args()
    checkpoint = resolve_artifact_path(args.checkpoint)
    output = (
        args.output
        if args.output.is_absolute()
        else resolve_artifact_path(args.output)
    )
    digest = export_actor(checkpoint, output, args.strategy_id)
    print(f"游戏内神经模型权重已导出：{output}")
    print(f"策略：{args.strategy_id} | SHA-256：{digest}")


if __name__ == "__main__":
    main()
