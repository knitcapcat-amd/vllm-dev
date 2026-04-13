# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Ring Attention for Context Parallelism.

Implements Ring Attention where Q stays local while K/V circulate through
a ring of CP ranks.  Each step computes a partial attention block and
incrementally merges the result using numerically-stable online softmax
(log-sum-exp correction).

Two interfaces are provided:

* :func:`ring_flash_attn_func` — 4-D batched tensors ``[B, S, H, D]``,
  used by multimodal encoder attention (``MMEncoderAttention``).
* :func:`ring_flash_attn_varlen_func` — packed variable-length tensors
  ``[total_tokens, H, D]`` with ``cu_seqlens``, reserved for future
  AR prefill context parallelism.

References:
    - TransformerEngine context_parallel.py (NVIDIA, 2024-2026)
    - vllm-omni diffusion/attention/backends/ring_flash_attn.py
    - Liu et al., "Ring Attention with Blockwise Transformers" (2023)
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from vllm.distributed.ring_comm import RingComm


# ---------------------------------------------------------------------------
# Online softmax merge (numerically stable)
# ---------------------------------------------------------------------------
# TE uses:  max_scale + log1p(exp(min_scale - max_scale))
# vllm-omni uses:  out - sigmoid(block_lse - lse) * (out - block_out)
#
# We follow TE's two-phase approach (merge LSE first, then correct output)
# because it decouples the LSE accumulation from the output correction and
# is easier to reason about numerically.  The final output correction is
# done once after all ring steps.
# ---------------------------------------------------------------------------


def _merge_lse(
    lse: torch.Tensor,
    lse_new: torch.Tensor,
) -> torch.Tensor:
    """Merge two log-sum-exp values: log(exp(a) + exp(b)).

    Uses the numerically stable formula:
        max(a, b) + log1p(exp(min(a, b) - max(a, b)))

    Args:
        lse: Running accumulated LSE.
        lse_new: LSE from the current ring step.

    Returns:
        Merged LSE (same shape, float32).
    """
    max_scale = torch.maximum(lse, lse_new)
    min_scale = torch.minimum(lse, lse_new)
    return max_scale + torch.log1p(torch.exp(min_scale - max_scale))


def _rescale_out(
    out: torch.Tensor,
    lse_cur: torch.Tensor,
    lse_merged: torch.Tensor,
    seq_dim: int,
) -> torch.Tensor:
    """Rescale an output block by exp(lse_cur - lse_merged).

    ``lse_*`` tensors have shape ``[B, H, S]`` while ``out`` has shape
    ``[B, S, H, D]``.  We move the H dim to align for broadcasting.

    Args:
        out: Output tensor ``[B, S, H, D]``.
        lse_cur: LSE that was used to produce *out*, ``[B, H, S]``.
        lse_merged: Global merged LSE, ``[B, H, S]``.
        seq_dim: Sequence dimension in *out* (typically 1).

    Returns:
        Rescaled output (float32).
    """
    # [B, H, S] → [B, S, H] → [B, S, H, 1]
    scale = torch.exp(lse_cur - lse_merged)
    scale = scale.movedim(-1, seq_dim).unsqueeze(-1)
    return out * scale


def _correct_outputs(
    out_per_step: list[torch.Tensor],
    lse_per_step: list[torch.Tensor],
    seq_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Correct and merge partial outputs from all ring steps.

    Phase 1 — accumulate global LSE:
        global_lse = log(sum_i exp(lse_i))

    Phase 2 — rescale each step's output and sum:
        out = sum_i  out_i * exp(lse_i - global_lse)

    This follows TransformerEngine's approach.

    Args:
        out_per_step: List of partial output tensors ``[B, S, H, D]``.
        lse_per_step: List of LSE tensors ``[B, H, S]`` (float32).
        seq_dim: Sequence dimension in the output tensors.

    Returns:
        (merged_out, global_lse) where merged_out is in the original
        dtype and global_lse is ``[B, H, S]`` in float32.
    """
    assert len(out_per_step) == len(lse_per_step) > 0

    # Phase 1: accumulate global LSE
    global_lse = lse_per_step[0].clone()
    for lse_i in lse_per_step[1:]:
        global_lse = _merge_lse(global_lse, lse_i)

    # Phase 2: rescale and accumulate outputs
    out_dtype = out_per_step[0].dtype
    merged_out = _rescale_out(
        out_per_step[0].float(), lse_per_step[0], global_lse, seq_dim)
    for out_i, lse_i in zip(out_per_step[1:], lse_per_step[1:]):
        merged_out = merged_out + _rescale_out(
            out_i.float(), lse_i, global_lse, seq_dim)

    return merged_out.to(out_dtype), global_lse


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

    out_per_step: list[torch.Tensor] = []
    lse_per_step: list[torch.Tensor] = []

    for step in range(comm.world_size):
        # --- async P2P: send current KV, receive next KV ---
        if step + 1 < comm.world_size:
            next_k = comm.send_recv(k)
            next_v = comm.send_recv(v)
            comm.commit()

        # --- compute attention for this step ---
        # For causal: only attend to KV from ranks 0..self.rank
        if not causal or step <= comm.rank:
            block_out, block_lse, _ = _flash_attn_func(
                q, k, v,
                softmax_scale=softmax_scale,
                causal=causal and step == 0,
                return_attn_probs=True,
            )
            # block_out: [B, S, H, D],  block_lse: [B, H, S] (float32)
            out_per_step.append(block_out)
            lse_per_step.append(block_lse)

        # --- wait for P2P, rotate KV ---
        if step + 1 < comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    # --- merge all partial outputs ---
    output, _ = _correct_outputs(out_per_step, lse_per_step, seq_dim=1)
    return output


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

    out_per_step: list[torch.Tensor] = []
    lse_per_step: list[torch.Tensor] = []

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
            out_per_step.append(block_out)
            lse_per_step.append(block_lse)

        if step + 1 < comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    # varlen: out is [total_tokens, H, D], lse is [H, total_tokens]
    # Adapt _correct_outputs for 3D by adding a batch dim, then squeeze
    out_stacked = [o.unsqueeze(0) for o in out_per_step]
    # lse from varlen FA is [H, total_tokens]; make it [1, H, total_tokens]
    lse_stacked = [l.unsqueeze(0) for l in lse_per_step]
    output, _ = _correct_outputs(out_stacked, lse_stacked, seq_dim=1)
    return output.squeeze(0)
