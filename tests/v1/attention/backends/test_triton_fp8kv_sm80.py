# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80+ FP8-KV serving on the Triton attention backend (Gemma-4 class).

FlashInfer does not support large-head multimodal attention and
FLASH_ATTN rejects FP8 KV below SM90, so TRITON_ATTN is the only
backend that can serve image support for large-head fp8-KV checkpoints
on pre-SM100 GPUs — and stock vLLM gates its FP8 KV path to SM89+
because triton cannot compile fp8e4nv below SM89. These tests pin the
fork's SM80+ enablement: the backend gate, the e5m2 KV flavor selection
below SM89, torch-side round-to-nearest-even quantization in the
cache-store wrapper, and the SM80/86 shared-memory staging for
large-head prefill tiles.
"""

from unittest.mock import patch

import torch

from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    _is_supported_kv_cache_dtype,
)


class _FakePlatform:
    def __init__(self, cap: int):
        self._cap = cap

    def is_cuda(self):
        return True

    def is_xpu(self):
        return False

    def has_device_capability(self, major):
        return self._cap >= major

    def is_device_capability_family(self, family):
        return False

    def fp8_dtype(self):
        return torch.float8_e4m3fn


# --- gate: FP8 KV supported from SM80 on CUDA ---


def test_fp8_kv_supported_on_sm80():
    with patch(
        "vllm.v1.attention.ops.triton_reshape_and_cache_flash.current_platform",
        _FakePlatform(80),
    ):
        assert _is_supported_kv_cache_dtype("fp8")


def test_fp8_kv_still_rejected_below_sm80():
    with patch(
        "vllm.v1.attention.ops.triton_reshape_and_cache_flash.current_platform",
        _FakePlatform(75),
    ):
        assert not _is_supported_kv_cache_dtype("fp8")


# --- e5m2 quantization helper: torch-side RNE below SM89 ---


def _quant(key, value, k_scale=None, v_scale=None):
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        quantize_kv_e5m2_sm80,
    )

    return quantize_kv_e5m2_sm80(key, value, k_scale, v_scale)


def test_quantize_kv_e5m2_produces_e5m2_tensors_round_nearest_even():
    key = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float32)
    value = torch.tensor([4.0], dtype=torch.float32)
    qk, qv = _quant(key, value)
    assert qk.dtype == torch.float8_e5m2
    assert qv.dtype == torch.float8_e5m2
    # e5m2 representable values: RNE of 1.0, 2.0, 0.5 is exact.
    assert qk.to(torch.float32).tolist() == [1.0, 2.0, 0.5]
    assert qv.to(torch.float32).tolist() == [4.0]


def test_quantize_kv_e5m2_applies_scales_before_quantization():
    key = torch.tensor([2.0], dtype=torch.float32)
    value = torch.tensor([8.0], dtype=torch.float32)
    k_scale = torch.tensor(2.0)
    v_scale = torch.tensor(4.0)
    qk, qv = _quant(key, value, k_scale, v_scale)
    assert qk.to(torch.float32).tolist() == [1.0]
    assert qv.to(torch.float32).tolist() == [2.0]


def test_quantize_kv_e5m2_rounds_ties_to_even():
    # In e5m2's exp=-1 band the representable values are 0.5, 0.625,
    # 0.75, 0.875 (step 0.125). The tie point between 0.5 (mantissa 00,
    # even) and 0.625 (mantissa 01, odd) is 0.5625: RNE selects the
    # even-mantissa neighbor (0.5), ties-away-from-even selects 0.625.
    key = torch.tensor([0.5625], dtype=torch.float32)
    value = torch.zeros(1, dtype=torch.float32)
    qk, _ = _quant(key, value)
    assert qk.to(torch.float32).item() == 0.5


def test_quantize_kv_e5m2_rounds_non_tie_to_nearest():
    # 0.55 is strictly between 0.5 and 0.5625 — closer to 0.5 than to
    # 0.625 under any rounding scheme, so it must round to 0.5.
    key = torch.tensor([0.55], dtype=torch.float32)
    value = torch.zeros(1, dtype=torch.float32)
    qk, _ = _quant(key, value)
    assert qk.to(torch.float32).item() == 0.5


# --- backend gate + flavor selection ---


def test_triton_backend_gate_message_says_sm80():
    # The gate's failure message names SM80+ as the requirement, not
    # SM89+ (which would misreport a stock restriction the fork lifts).
    import inspect

    from vllm.v1.attention.backends import triton_attn

    src = inspect.getsource(triton_attn)
    assert "SM80+" in src or "requires SM80" in src or "kernel dequant" in src


def test_triton_backend_selects_e5m2_below_sm89():
    from vllm.v1.attention.backends.triton_attn import kv_fp8_dtype_for_platform

    # Below SM89 on CUDA: e5m2 replaces the default e4m3 — triton
    # cannot compile fp8e4nv there.
    assert (
        kv_fp8_dtype_for_platform(torch.float8_e4m3fn, platform=_FakePlatform(86))
        is torch.float8_e5m2
    )


def test_triton_backend_keeps_default_dtype_on_sm89_plus():
    from vllm.v1.attention.backends.triton_attn import kv_fp8_dtype_for_platform

    for cap in (89, 90, 100):
        assert (
            kv_fp8_dtype_for_platform(torch.float8_e4m3fn, platform=_FakePlatform(cap))
            is torch.float8_e4m3fn
        )


# --- unified attention staging ---


class _KernelCapture:
    """Records the kwargs of a mocked triton kernel launch."""

    def __init__(self):
        self.kwargs: dict | None = None

    def __getitem__(self, grid):
        def launch(**kwargs):
            self.kwargs = kwargs

        return launch


def _launch_prefill_below_sm90(kv_dtype: torch.dtype, head_size: int) -> dict:
    """Run one prefill through unified_attention on a mocked sub-SM90 CUDA
    platform and return the kwargs the kernel launch received."""
    from vllm.v1.attention.ops import triton_unified_attention as tua

    block_size, num_blocks, num_tokens = 16, 8, 64
    q = torch.zeros(num_tokens, 1, head_size, dtype=torch.bfloat16)
    k_cache = torch.zeros(num_blocks, block_size, 1, head_size, dtype=kv_dtype)
    v_cache = torch.zeros_like(k_cache)
    out = torch.zeros_like(q)
    cu_seqlens_q = torch.tensor([0, num_tokens], dtype=torch.int32)
    seqused_k = torch.tensor([64], dtype=torch.int32)
    block_table = torch.zeros(1, num_blocks, dtype=torch.int32)

    capture = _KernelCapture()
    with (
        patch.object(tua, "current_platform", _FakePlatform(80)),
        patch.object(tua, "kernel_unified_attention", capture),
    ):
        tua.unified_attention(
            q,
            k_cache,
            v_cache,
            out,
            cu_seqlens_q,
            num_tokens,  # max_seqlen_q > 1: prefill, 2D kernel
            seqused_k,
            64,
            1.0,
            True,
            (-1, -1),
            block_table,
            0.0,
            None,
            None,
            None,
        )
    assert capture.kwargs is not None
    return capture.kwargs


def test_non_fp8_prefill_below_sm90_keeps_default_launch_stages():
    """The stage cap and tile halving are the fp8-KV large-head shared-memory
    workaround; a bf16 sub-SM90 prefill must stay on stock launch behavior
    (Triton's default num_stages, default prefill tile)."""
    kwargs = _launch_prefill_below_sm90(torch.bfloat16, head_size=128)
    assert "num_stages" not in kwargs
    assert kwargs["TILE_SIZE"] == 32

    kwargs = _launch_prefill_below_sm90(torch.bfloat16, head_size=512)
    assert "num_stages" not in kwargs
    assert kwargs["TILE_SIZE"] == 32


def test_fp8_large_head_prefill_below_sm90_keeps_stage_cap():
    """The workaround itself stays: fp8 KV with a large head on sub-SM90
    launches with num_stages=1 and the halved prefill tile."""
    kwargs = _launch_prefill_below_sm90(torch.float8_e5m2, head_size=512)
    assert kwargs.get("num_stages") == 1
    assert kwargs["TILE_SIZE"] == 16


def test_unified_attention_caps_stages_for_large_head_below_sm90():
    import inspect

    from vllm.v1.attention.ops import triton_unified_attention as tua

    src = inspect.getsource(tua)
    assert "TILE_SIZE_PREFILL = 16" in src
    assert "launch_num_stages = 1" in src
