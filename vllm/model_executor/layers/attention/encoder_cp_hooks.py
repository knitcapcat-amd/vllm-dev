# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Encoder Context Parallelism hooks.

Provides non-intrusive sequence sharding/gathering for vision encoder
context parallelism, using standard PyTorch forward hooks.  This
approach avoids modifying individual model forward() methods.

The hook pair works on the **encoder** submodule (e.g. ``CLIPEncoder``,
``Qwen2_5_VisionTransformer.blocks``):

- **pre-forward hook**: shards ``hidden_states`` along the sequence dim
  so each CP rank processes 1/CP of the tokens.
- **post-forward hook**: all-gathers the output along the sequence dim
  to restore the full sequence for downstream consumption (projector,
  LLM embeddings, etc.).

Usage::

    from vllm.model_executor.layers.attention.encoder_cp_hooks import (
        apply_encoder_cp_hooks,
    )
    # After model loading, apply hooks to the encoder submodule:
    apply_encoder_cp_hooks(vision_model.encoder, seq_dim=0)
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.distributed as dist
import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)

_ENCODER_CP_HOOK_KEY = "_encoder_cp_hooks"

# Runtime context: set by the shard hook, read by MMEncoderAttention
_active_encoder_cp_group: dist.ProcessGroup | None = None


def get_active_encoder_cp_group() -> dist.ProcessGroup | None:
    """Return the encoder CP group if we are inside a CP-hooked forward."""
    return _active_encoder_cp_group


def _shard_along_dim(
    tensor: torch.Tensor,
    dim: int,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """Split a tensor into world_size chunks and return the rank-th chunk."""
    size = tensor.size(dim)
    if size % world_size != 0:
        # Pad to make divisible
        pad_size = world_size - (size % world_size)
        pad_shape = list(tensor.shape)
        pad_shape[dim] = pad_size
        padding = torch.zeros(pad_shape, dtype=tensor.dtype,
                              device=tensor.device)
        tensor = torch.cat([tensor, padding], dim=dim)
        size = tensor.size(dim)

    chunk_size = size // world_size
    return tensor.narrow(dim, rank * chunk_size, chunk_size).contiguous()


def _gather_along_dim(
    tensor: torch.Tensor,
    dim: int,
    group: dist.ProcessGroup,
    world_size: int,
    original_size: int | None = None,
) -> torch.Tensor:
    """All-gather tensor along the given dim across the CP group."""
    tensor = tensor.contiguous()
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor, group=group)
    output = torch.cat(gathered, dim=dim)
    if original_size is not None and output.size(dim) > original_size:
        output = output.narrow(dim, 0, original_size)
    return output


class EncoderCPShardHook:
    """Pre-forward hook that shards hidden_states along the sequence dim."""

    def __init__(
        self,
        cp_group: dist.ProcessGroup,
        seq_dim: int = 0,
    ):
        self.cp_group = cp_group
        self.cp_rank = dist.get_rank(cp_group)
        self.cp_world_size = dist.get_world_size(cp_group)
        self.seq_dim = seq_dim
        self._original_sizes: dict[int, int] = {}

    def __call__(
        self,
        module: nn.Module,
        args: tuple,
        kwargs: dict,
    ) -> tuple[tuple, dict]:
        global _active_encoder_cp_group

        tensor, source = self._extract_hidden_states(args, kwargs)
        if tensor is None:
            return args, kwargs

        module._encoder_cp_original_seq_len = tensor.size(self.seq_dim)

        # Activate CP context so MMEncoderAttention routes to Ring Attention
        _active_encoder_cp_group = self.cp_group

        sharded = _shard_along_dim(
            tensor, self.seq_dim, self.cp_rank, self.cp_world_size)

        return self._replace_hidden_states(args, kwargs, sharded, source)

    def _extract_hidden_states(
        self,
        args: tuple,
        kwargs: dict,
    ) -> tuple[torch.Tensor | None, str]:
        """Find the hidden_states tensor from args/kwargs."""
        # Check common kwarg names
        for name in ("inputs_embeds", "hidden_states", "x"):
            if name in kwargs and isinstance(kwargs[name], torch.Tensor):
                return kwargs[name], f"kwarg:{name}"

        # Fall back to first positional arg
        if args and isinstance(args[0], torch.Tensor):
            return args[0], "arg:0"

        return None, ""

    def _replace_hidden_states(
        self,
        args: tuple,
        kwargs: dict,
        new_tensor: torch.Tensor,
        source: str,
    ) -> tuple[tuple, dict]:
        """Replace the hidden_states tensor in args/kwargs."""
        if source.startswith("kwarg:"):
            name = source.split(":")[1]
            kwargs = {**kwargs, name: new_tensor}
        elif source == "arg:0":
            args = (new_tensor,) + args[1:]
        return args, kwargs


class EncoderCPGatherHook:
    """Post-forward hook that gathers output along the sequence dim."""

    def __init__(
        self,
        cp_group: dist.ProcessGroup,
        seq_dim: int = 0,
    ):
        self.cp_group = cp_group
        self.cp_world_size = dist.get_world_size(cp_group)
        self.seq_dim = seq_dim

    def __call__(
        self,
        module: nn.Module,
        args: tuple,
        output: torch.Tensor | list[torch.Tensor],
    ) -> torch.Tensor | list[torch.Tensor]:
        global _active_encoder_cp_group

        original_seq_len = getattr(
            module, "_encoder_cp_original_seq_len", None)

        if isinstance(output, torch.Tensor):
            result = _gather_along_dim(
                output, self.seq_dim, self.cp_group,
                self.cp_world_size, original_seq_len)
        elif isinstance(output, (list, tuple)):
            gathered = []
            for item in output:
                if isinstance(item, torch.Tensor):
                    gathered.append(_gather_along_dim(
                        item, self.seq_dim, self.cp_group,
                        self.cp_world_size, original_seq_len))
                else:
                    gathered.append(item)
            result = type(output)(gathered)
        else:
            result = output

        # Deactivate CP context
        _active_encoder_cp_group = None
        return result


def apply_encoder_cp_hooks(
    module: nn.Module,
    cp_group: dist.ProcessGroup,
    seq_dim: int = 0,
) -> None:
    """Attach shard/gather hooks to an encoder module for context parallelism.

    The pre-forward hook shards the input along ``seq_dim``, and the
    post-forward hook gathers the output along the same dimension.
    All transformer blocks inside the module will operate on the
    sharded sequence, with ``MMEncoderAttention`` handling cross-rank
    KV communication via Ring Attention.

    Args:
        module: The encoder submodule (e.g. ``CLIPEncoder`` or the
            module containing the transformer blocks).
        cp_group: Process group for context parallelism.
        seq_dim: Sequence dimension to shard (default: 0).  For models
            where hidden_states is ``[seq_len, batch, hidden]`` this
            should be 0; for ``[batch, seq_len, hidden]`` use 1.
    """
    if hasattr(module, _ENCODER_CP_HOOK_KEY):
        logger.warning("Encoder CP hooks already applied to %s, skipping.",
                       module.__class__.__name__)
        return

    shard_hook = EncoderCPShardHook(cp_group, seq_dim)
    gather_hook = EncoderCPGatherHook(cp_group, seq_dim)

    handle_pre = module.register_forward_pre_hook(
        shard_hook, with_kwargs=True)
    handle_post = module.register_forward_hook(gather_hook)

    # Store handles for potential removal
    setattr(module, _ENCODER_CP_HOOK_KEY, (handle_pre, handle_post))

    cp_rank = dist.get_rank(cp_group)
    cp_size = dist.get_world_size(cp_group)
    logger.info(
        "Applied encoder CP hooks to %s (rank=%d, cp_size=%d, seq_dim=%d)",
        module.__class__.__name__, cp_rank, cp_size, seq_dim)


def remove_encoder_cp_hooks(module: nn.Module) -> None:
    """Remove previously applied encoder CP hooks."""
    hooks = getattr(module, _ENCODER_CP_HOOK_KEY, None)
    if hooks is not None:
        for handle in hooks:
            handle.remove()
        delattr(module, _ENCODER_CP_HOOK_KEY)


def find_and_apply_encoder_cp_hooks(
    model: nn.Module,
    cp_group: dist.ProcessGroup,
) -> int:
    """Walk the model tree and apply CP hooks to encoder modules.

    Looks for modules that contain ``MMEncoderAttention`` layers (meaning
    they are vision/audio encoder blocks) and applies shard/gather hooks
    to the **encoder container** — the module whose ``forward()`` iterates
    over the encoder layers.

    Detection heuristic: if a module's **direct children** include
    ``MMEncoderAttention`` instances (e.g. inside a
    ``CLIPEncoderLayer``), then the **grandparent** (the encoder that
    holds the layer list) is the hook target.

    For simpler models, we look for any ``nn.ModuleList`` whose elements
    contain ``MMEncoderAttention`` submodules.

    Args:
        model: The top-level model.
        cp_group: CP process group.

    Returns:
        Number of encoder modules that got hooks applied.
    """
    from vllm.model_executor.layers.attention.mm_encoder_attention import (
        MMEncoderAttention,
    )

    # Strategy: find modules that CONTAIN a ModuleList of encoder layers
    # (layers with MMEncoderAttention), then hook the PARENT module —
    # since the parent's forward() iterates the layers.
    targets: set[int] = set()
    target_modules: list[tuple[str, nn.Module, int]] = []

    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.ModuleList):
                has_encoder_attn = any(
                    isinstance(m, MMEncoderAttention)
                    for m in child.modules()
                )
                if has_encoder_attn and id(module) not in targets:
                    targets.add(id(module))
                    seq_dim = _infer_seq_dim(child_name, name)
                    target_modules.append((name, module, seq_dim))

    for path, module, seq_dim in target_modules:
        apply_encoder_cp_hooks(module, cp_group, seq_dim=seq_dim)
        logger.info("Encoder CP hooks applied to %s (seq_dim=%d)", path, seq_dim)

    return len(target_modules)


def _infer_seq_dim(child_name: str, parent_name: str) -> int:
    """Infer the sequence dimension for a given encoder module.

    Most CLIP/SigLIP encoders use [batch, seq, hidden] → seq_dim=1.
    Some models like Qwen2.5-VL use [seq, batch, hidden] → seq_dim=0.

    This is a heuristic; models with unusual layouts may need explicit
    configuration.
    """
    # Qwen2.5-VL and similar: blocks input is [S, B, H]
    if "block" in child_name.lower() or "block" in parent_name.lower():
        return 0
    # Most ViTs: encoder.layers input is [B, S, H]
    return 1
