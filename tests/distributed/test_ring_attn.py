# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for Ring Attention correctness.

Verifies that Ring Attention (Context Parallelism) produces the same
output as standard single-GPU attention across various configurations.

Two run modes:
  1. pytest (uses ray for multi-GPU):
       pytest tests/distributed/test_ring_attn.py -v
  2. torchrun (standalone, no ray/pytest):
       torchrun --nproc_per_node=2 tests/distributed/test_ring_attn.py
"""

from __future__ import annotations

import os
import sys

import torch

# -----------------------------------------------------------------------
# Test configurations
# -----------------------------------------------------------------------

ATOL = 1e-2
RTOL = 1e-2


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Standard Flash Attention on full Q/K/V (single GPU baseline)."""
    from flash_attn import flash_attn_func
    return flash_attn_func(q, k, v, causal=causal)


# =====================================================================
# Standalone torchrun mode
# =====================================================================

def _run_standalone():
    """Quick smoke test with torchrun (no ray, no pytest).

    Usage:
        torchrun --nproc_per_node=2 tests/distributed/test_ring_attn.py
        torchrun --nproc_per_node=4 tests/distributed/test_ring_attn.py
    """
    import torch.distributed as dist

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    from vllm.v1.attention.ops.ring_attn import ring_flash_attn_func

    configs = [
        # (batch, seq_len, heads, head_dim, causal)
        (1, 256, 16, 80, False),
        (2, 512, 16, 80, False),
        (1, 1024, 12, 64, False),
        (1, 2048, 16, 80, False),
        (1, 256, 16, 80, True),
        (1, 512, 16, 80, True),
    ]

    all_pass = True
    for batch_size, seq_len, num_heads, head_dim, causal in configs:
        torch.manual_seed(42)
        q_full = torch.randn(
            batch_size, seq_len, num_heads, head_dim,
            dtype=torch.bfloat16, device=device)
        k_full = torch.randn(
            batch_size, seq_len, num_heads, head_dim,
            dtype=torch.bfloat16, device=device)
        v_full = torch.randn(
            batch_size, seq_len, num_heads, head_dim,
            dtype=torch.bfloat16, device=device)

        ref_out = reference_attention(q_full, k_full, v_full, causal=causal)

        chunk = seq_len // world_size
        q_local = q_full[:, rank * chunk:(rank + 1) * chunk].contiguous()
        k_local = k_full[:, rank * chunk:(rank + 1) * chunk].contiguous()
        v_local = v_full[:, rank * chunk:(rank + 1) * chunk].contiguous()

        ring_out = ring_flash_attn_func(
            q_local, k_local, v_local,
            cp_group=dist.group.WORLD,
            causal=causal,
        )

        ref_chunk = ref_out[:, rank * chunk:(rank + 1) * chunk]
        max_diff = (ring_out.float() - ref_chunk.float()).abs().max().item()
        is_close = torch.allclose(
            ring_out.float(), ref_chunk.float(), atol=ATOL, rtol=RTOL)

        if not is_close:
            all_pass = False

        tag = "causal" if causal else "bidir"
        status = "PASS" if is_close else "FAIL"
        if rank == 0:
            print(f"[{status}] {tag} B={batch_size} S={seq_len} "
                  f"H={num_heads} D={head_dim} CP={world_size} "
                  f"max_diff={max_diff:.6f}")

    dist.destroy_process_group()
    if rank == 0:
        print(f"\n{'All tests passed!' if all_pass else 'SOME TESTS FAILED!'}")
    sys.exit(0 if all_pass else 1)


# =====================================================================
# pytest + ray mode
# =====================================================================

if __name__ == "__main__":
    _run_standalone()
else:
    import pytest
    import ray

    from tests.utils import (
        init_test_distributed_environment,
        multi_process_parallel,
    )

    CP_SIZES = [2, 4]
    SEQ_LENS = [128, 512, 2048]
    NUM_HEADS_CONFIGS = [
        (16, 80),
        (12, 64),
    ]
    BATCH_SIZES = [1, 2]
    DTYPES = [torch.bfloat16]

    @ray.remote(num_gpus=1, max_calls=1)
    def ring_attn_correctness_worker(
        monkeypatch: pytest.MonkeyPatch,
        tp_size: int,
        pp_size: int,
        rank: int,
        distributed_init_port: str,
    ) -> None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(
            tp_size, pp_size, rank, distributed_init_port)

        from vllm.distributed.parallel_state import get_tp_group
        from vllm.v1.attention.ops.ring_attn import ring_flash_attn_func

        seq_len = int(os.environ["TEST_SEQ_LEN"])
        num_heads = int(os.environ["TEST_NUM_HEADS"])
        head_dim = int(os.environ["TEST_HEAD_DIM"])
        batch_size = int(os.environ["TEST_BATCH_SIZE"])
        dtype_str = os.environ["TEST_DTYPE"]
        causal = os.environ.get("TEST_CAUSAL", "0") == "1"
        dtype = getattr(torch, dtype_str)

        cp_size = tp_size
        cp_group = get_tp_group().device_group

        torch.manual_seed(42)
        q_full = torch.randn(
            batch_size, seq_len, num_heads, head_dim,
            dtype=dtype, device=device)
        k_full = torch.randn(
            batch_size, seq_len, num_heads, head_dim,
            dtype=dtype, device=device)
        v_full = torch.randn(
            batch_size, seq_len, num_heads, head_dim,
            dtype=dtype, device=device)

        ref_out = reference_attention(
            q_full, k_full, v_full, causal=causal)

        chunk_size = seq_len // cp_size
        q_local = q_full[
            :, rank * chunk_size:(rank + 1) * chunk_size].contiguous()
        k_local = k_full[
            :, rank * chunk_size:(rank + 1) * chunk_size].contiguous()
        v_local = v_full[
            :, rank * chunk_size:(rank + 1) * chunk_size].contiguous()

        ring_out = ring_flash_attn_func(
            q_local, k_local, v_local,
            cp_group=cp_group,
            causal=causal,
        )

        ref_chunk = ref_out[
            :, rank * chunk_size:(rank + 1) * chunk_size]

        torch.testing.assert_close(
            ring_out.float(), ref_chunk.float(),
            atol=ATOL, rtol=RTOL,
            msg=f"Ring Attention output mismatch on rank {rank}")

    @pytest.mark.parametrize("cp_size", CP_SIZES)
    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    @pytest.mark.parametrize("num_heads,head_dim", NUM_HEADS_CONFIGS)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_ring_attn_bidirectional(
        monkeypatch: pytest.MonkeyPatch,
        cp_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        batch_size: int,
        dtype: torch.dtype,
    ):
        """Ring Attention (causal=False) matches standard attention."""
        monkeypatch.setenv("TEST_SEQ_LEN", str(seq_len))
        monkeypatch.setenv("TEST_NUM_HEADS", str(num_heads))
        monkeypatch.setenv("TEST_HEAD_DIM", str(head_dim))
        monkeypatch.setenv("TEST_BATCH_SIZE", str(batch_size))
        monkeypatch.setenv("TEST_DTYPE", str(dtype).split(".")[-1])
        monkeypatch.setenv("TEST_CAUSAL", "0")

        multi_process_parallel(
            monkeypatch,
            tp_size=cp_size,
            pp_size=1,
            test_target=ring_attn_correctness_worker,
        )

    @pytest.mark.parametrize("cp_size", [2])
    @pytest.mark.parametrize("seq_len", [128, 512])
    @pytest.mark.parametrize("num_heads,head_dim", [(16, 80)])
    @pytest.mark.parametrize("batch_size", [1])
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_ring_attn_causal(
        monkeypatch: pytest.MonkeyPatch,
        cp_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        batch_size: int,
        dtype: torch.dtype,
    ):
        """Ring Attention (causal=True) matches standard causal attention."""
        monkeypatch.setenv("TEST_SEQ_LEN", str(seq_len))
        monkeypatch.setenv("TEST_NUM_HEADS", str(num_heads))
        monkeypatch.setenv("TEST_HEAD_DIM", str(head_dim))
        monkeypatch.setenv("TEST_BATCH_SIZE", str(batch_size))
        monkeypatch.setenv("TEST_DTYPE", str(dtype).split(".")[-1])
        monkeypatch.setenv("TEST_CAUSAL", "1")

        multi_process_parallel(
            monkeypatch,
            tp_size=cp_size,
            pp_size=1,
            test_target=ring_attn_correctness_worker,
        )
