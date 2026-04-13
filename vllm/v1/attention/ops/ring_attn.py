# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Ring Attention for Context Parallelism.

Implements Ring Attention where Q stays local while K/V circulate through
a ring of CP ranks.  Each step computes a partial attention block and
incrementally merges the result using online softmax correction.

The online merge approach (from vllm-omni / long-context-attention) updates
the running output in-place at each step using sigmoid/logsigmoid, which
is faster and more memory-efficient than the two-phase collect-then-rescale
approach.

Two interfaces are provided:

* :func:`ring_flash_attn_func` — 4-D batched tensors ``[B, S, H, D]``,
  used by multimodal encoder attention (``MMEncoderAttention``).
* :func:`ring_flash_attn_varlen_func` — packed variable-length tensors
  ``[total_tokens, H, D]`` with ``cu_seqlens``, reserved for future
  AR prefill context parallelism.

References:
    - vllm-omni diffusion/attention/backends/ring_flash_attn.py
    - Fang et al., long-context-attention (yunchang)
    - Liu et al., "Ring Attention with Blockwise Transformers" (2023)
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from vllm.distributed.ring_comm import RingComm


# ---------------------------------------------------------------------------
# Online softmax merge
# ---------------------------------------------------------------------------


def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Incrementally merge a new attention block into the running output.

    Uses the online softmax correction formula:

        out = out - sigmoid(block_lse - lse) * (out - block_out)
        lse = lse - logsigmoid(lse - block_lse)

    All computation is done in float32 for numerical stability.

    Args:
        out: Running merged output ``[B, S, H, D]`` (float32).
        lse: Running log-sum-exp ``[B, S, H, 1]`` (float32).
        block_out: New block output ``[B, S, H, D]``.
        block_lse: New block LSE ``[B, H, S]`` (float32 from FA).

    Returns:
        Updated (out, lse), both float32.
    """
    block_out = block_out.to(torch.float32)
    # block_lse from FA is [B, H, S] → convert to [B, S, H, 1]
    block_lse = block_lse.transpose(1, 2).unsqueeze(-1)

    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)
    return out, lse


def _init_out_and_lse(
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize the running output and LSE from the first block.

    Args:
        block_out: First block output ``[B, S, H, D]``.
        block_lse: First block LSE ``[B, H, S]`` (float32 from FA).

    Returns:
        (out, lse) in float32.  lse shape is ``[B, S, H, 1]``.
    """
    out = block_out.to(torch.float32)
    lse = block_lse.transpose(1, 2).unsqueeze(-1)
    return out, lse


# ---------------------------------------------------------------------------
# Ring Flash Attention — batched (4D) interface for encoder
# ---------------------------------------------------------------------------


def ring_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cp_group: dist.ProcessGroup,
    cp_stream: torch.cuda.Stream | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Ring Attention for 4-D batched Q/K/V tensors.

    This is the primary interface for multimodal encoder attention.
    Q stays local; K and V are circulated through the ring.

    Args:
        q: Query ``[B, S_local, H, D]``.
        k: Key   ``[B, S_local, H, D]`` (this rank's chunk).
        v: Value ``[B, S_local, H, D]`` (this rank's chunk).
        cp_group: Process group for context parallelism.
        cp_stream: Dedicated CUDA stream for P2P comm.  If *None*,
            a new stream is created.
        softmax_scale: Attention scale.  Defaults to ``1/sqrt(D)``.
        causal: Whether to apply causal masking.  For encoders this
            should be *False*.

    Returns:
        Output tensor ``[B, S_local, H, D]`` in the original dtype.
    """
    from flash_attn import flash_attn_func as _flash_attn_func

    comm = RingComm(cp_group, cp_stream)

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    out: torch.Tensor | None = None
    lse: torch.Tensor | None = None

    for step in range(comm.world_size):
        if step + 1 < comm.world_size:
            next_k = comm.send_recv(k)
            next_v = comm.send_recv(v)
            comm.commit()

        if not causal or step <= comm.rank:
            block_out, block_lse, _ = _flash_attn_func(
                q, k, v,
                softmax_scale=softmax_scale,
                causal=causal and step == 0,
                return_attn_probs=True,
            )

            if out is None:
                out, lse = _init_out_and_lse(block_out, block_lse)
            else:
                out, lse = _update_out_and_lse(
                    out, lse, block_out, block_lse)

        if step + 1 < comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    assert out is not None
    return out.to(q.dtype)


# ---------------------------------------------------------------------------
# Ring Flash Attention — varlen interface (for future AR PCP)
# ---------------------------------------------------------------------------


def ring_flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    cp_group: dist.ProcessGroup,
    cp_stream: torch.cuda.Stream | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Ring Attention for packed variable-length sequences.

    Reserved for future autoregressive prefill context parallelism.

    Args:
        q: Query ``[total_q_tokens, H, D]``.
        k: Key   ``[total_k_tokens, H, D]`` (this rank's chunk).
        v: Value ``[total_k_tokens, H, D]`` (this rank's chunk).
        cu_seqlens_q: Cumulative query sequence lengths ``[B+1]``.
        cu_seqlens_k: Cumulative key sequence lengths ``[B+1]``.
        max_seqlen_q: Maximum query sequence length in batch.
        max_seqlen_k: Maximum key sequence length in batch.
        cp_group: Process group for context parallelism.
        cp_stream: Dedicated CUDA stream for P2P comm.
        softmax_scale: Attention scale.
        causal: Whether to apply causal masking.

    Returns:
        Output tensor ``[total_q_tokens, H, D]`` in the original dtype.
    """
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func

    comm = RingComm(cp_group, cp_stream)

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    out: torch.Tensor | None = None
    lse: torch.Tensor | None = None

    for step in range(comm.world_size):
        if step + 1 < comm.world_size:
            next_k = comm.send_recv(k)
            next_v = comm.send_recv(v)
            comm.commit()

        if not causal or step <= comm.rank:
            block_out, block_lse, _ = _flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=causal and step == 0,
                return_attn_probs=True,
            )

            # varlen FA: block_out [N, H, D], block_lse [H, N]
            # Add batch dim for _init/_update which expect [B, S, H, D]
            bo = block_out.unsqueeze(0)
            bl = block_lse.unsqueeze(0)  # [1, H, N]

            if out is None:
                out, lse = _init_out_and_lse(bo, bl)
            else:
                out, lse = _update_out_and_lse(out, lse, bo, bl)

        if step + 1 < comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    assert out is not None
    return out.squeeze(0).to(q.dtype)
