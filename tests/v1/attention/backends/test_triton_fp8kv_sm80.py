# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8-KV serving on the Triton attention backend for SM80/SM86 GPUs.

Stock gates the Triton FP8 KV path on SM89+ because triton cannot compile
fp8e4nv below SM89. Below SM89 the cache is instead stored as e5m2 (native
in triton on SM80+), quantized in torch with round-to-nearest-even by the
cache-store wrapper, and dequantized in-kernel at decode time. These tests
pin that behavior: the store-path dtype gate, the e5m2 flavor selection,
the RNE quantization helper, the backend gate, the query-quant exclusion,
the calibrated-scheme relaxation, and the SM80/86 shared-memory staging
for large-head prefill tiles.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    _is_supported_kv_cache_dtype,
)

_IS_SUB_SM89_CUDA = (
    current_platform.is_cuda() and not current_platform.has_device_capability(89)
)
_IS_SUB_SM90_CUDA = (
    current_platform.is_cuda() and not current_platform.has_device_capability(90)
)


class _FakePlatform:
    def __init__(self, cap: int):
        self._cap = cap

    def is_cuda(self):
        return True

    def is_rocm(self):
        return False

    def is_xpu(self):
        return False

    def has_device_capability(self, major):
        return self._cap >= major

    def is_device_capability(self, cap):
        return self._cap == cap[0] * 10 + cap[1]

    def is_device_capability_family(self, family):
        return False

    def get_device_capability(self):
        return DeviceCapability(major=self._cap // 10, minor=self._cap % 10)

    def get_device_name(self):
        return f"FakeSM{self._cap}"

    def num_compute_units(self):
        return 128

    def fp8_dtype(self):
        return torch.float8_e4m3fn


# --- store-path dtype gate: FP8 KV supported from SM80 on CUDA ---


def test_fp8_kv_supported_on_sm80():
    with patch(
        "vllm.v1.attention.ops.triton_reshape_and_cache_flash.current_platform",
        _FakePlatform(80),
    ):
        assert _is_supported_kv_cache_dtype("fp8")
        assert _is_supported_kv_cache_dtype("fp8_e5m2")


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


# --- store-branch dispatch: _quantize_kv_e5m2_for_sm80 ---


def _store_branch(kv_cache_dtype, key_dtype, cap):
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        _quantize_kv_e5m2_for_sm80,
    )

    key = torch.randn(4, 2, 8).to(key_dtype)
    value = torch.randn(4, 2, 8).to(key_dtype)
    caches = (torch.zeros(4, 16, 2, 8, dtype=torch.uint8),)
    ctx = (
        patch(
            "vllm.v1.attention.ops.triton_reshape_and_cache_flash.current_platform",
            _FakePlatform(cap),
        )
        if cap is not None
        else None
    )
    if ctx is None:
        return _quantize_kv_e5m2_for_sm80(
            kv_cache_dtype, torch.uint8, key, value, None, None, *caches
        )
    with ctx:
        return _quantize_kv_e5m2_for_sm80(
            kv_cache_dtype, torch.uint8, key, value, None, None, *caches
        )


@pytest.mark.skipif(not _IS_SUB_SM89_CUDA, reason="needs a CUDA device below SM89")
def test_store_branch_fires_below_sm89():
    # The branch quantizes bf16 KV to e5m2 and re-views the uint8 cache.
    key, value, caches = _store_branch("fp8", torch.bfloat16, cap=None)
    assert key.dtype == torch.float8_e5m2
    assert value.dtype == torch.float8_e5m2
    assert caches[0].dtype == torch.float8_e5m2


def test_store_branch_noop_on_sm89_plus():
    # Above SM89 the platform fp8 dtype (e4m3) stays; no re-quantization.
    key, value, caches = _store_branch("fp8", torch.bfloat16, cap=89)
    assert key.dtype == torch.bfloat16
    assert value.dtype == torch.bfloat16
    assert caches[0].dtype == torch.uint8


def test_store_branch_noop_on_prequantized_input():
    # KV already quantized to an fp8 dtype flows through untouched.
    key, value, caches = _store_branch("fp8", torch.float8_e4m3fn, cap=80)
    assert key.dtype == torch.float8_e4m3fn
    assert caches[0].dtype == torch.uint8


def test_store_branch_noop_for_unquantized_cache_dtype():
    # A bf16 cache never triggers the e5m2 branch.
    key, value, caches = _store_branch("auto", torch.bfloat16, cap=80)
    assert key.dtype == torch.bfloat16
    assert caches[0].dtype == torch.uint8


# --- backend gate + flavor selection ---


def _construct_impl(kv_cache_dtype, cap):
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    def build():
        return TritonAttentionImpl(
            num_heads=2,
            head_size=128,
            scale=0.1,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype=kv_cache_dtype,
        )

    if cap is None:
        return build()
    with patch(
        "vllm.v1.attention.backends.triton_attn.current_platform", _FakePlatform(cap)
    ):
        return build()


@pytest.mark.skipif(not _IS_SUB_SM89_CUDA, reason="needs a CUDA device below SM89")
def test_triton_backend_allows_fp8_kv_below_sm89():
    impl = _construct_impl("fp8", cap=None)
    assert impl.fp8_dtype is torch.float8_e5m2


@pytest.mark.skipif(not _IS_SUB_SM89_CUDA, reason="needs a CUDA device below SM89")
def test_triton_backend_allows_explicit_e5m2_below_sm89():
    impl = _construct_impl("fp8_e5m2", cap=None)
    assert impl.fp8_dtype is torch.float8_e5m2


def test_triton_backend_rejects_fp8_kv_below_sm80_with_sm80_message():
    with pytest.raises(ValueError, match="e5m2 fallback requires SM80+"):
        _construct_impl("fp8", cap=75)


def test_triton_backend_keeps_e4m3_on_sm89_plus():
    impl = _construct_impl("fp8", cap=89)
    assert impl.fp8_dtype is torch.float8_e4m3fn


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


# --- Attention layer: e5m2 KV keeps bf16 queries ---


def _make_attention_layer(cache_dtype):
    from tests.v1.attention.utils import create_vllm_config
    from vllm.config import CacheConfig
    from vllm.config.vllm import set_current_vllm_config
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

    with set_current_vllm_config(create_vllm_config()):
        return Attention(
            num_heads=2,
            head_size=128,
            scale=0.1,
            num_kv_heads=1,
            cache_config=CacheConfig(block_size=16, cache_dtype=cache_dtype),
            attn_backend=TritonAttentionBackend,
            prefix=f"layer_{cache_dtype}",
        )


@pytest.mark.skipif(not _IS_SUB_SM89_CUDA, reason="needs a CUDA device below SM89")
def test_e5m2_kv_skips_query_quant_setup():
    # Query quantization targets e4m3 kernels; e5m2 KV runs with bf16
    # queries and in-kernel KV dequant.
    layer = _make_attention_layer("fp8_e5m2")
    assert layer.query_quant is None


@pytest.mark.parametrize(
    "cap,cache_dtype,armed",
    [
        # Below SM89 the "fp8" string serves e5m2 KV (see
        # kv_fp8_dtype_for_platform), so queries must stay bf16.
        (86, "fp8", False),
        (86, "fp8_e5m2", False),
        # SM89+ keeps e4m3 KV and arms query quantization.
        (90, "fp8", True),
    ],
)
def test_query_quant_arming_follows_effective_kv_dtype(cap, cache_dtype, armed):
    with patch(
        "vllm.v1.attention.backends.triton_attn.current_platform",
        _FakePlatform(cap),
    ):
        layer = _make_attention_layer(cache_dtype)
    assert (layer.query_quant is not None) == armed


# --- calibrated fp8 checkpoints: e5m2 allowed below SM89 ---


def _calibrated_ct_method():
    from vllm.model_executor.layers.quantization.compressed_tensors import (  # noqa: E501
        compressed_tensors as ct_mod,
    )

    method = ct_mod.CompressedTensorsKVCacheMethod.__new__(
        ct_mod.CompressedTensorsKVCacheMethod
    )
    method.quant_config = SimpleNamespace(kv_cache_scheme={"dtype": "float8_e4m3fn"})
    return method


def _run_kv_cache_quant_init(cap):
    from torch import nn

    from vllm.model_executor.layers.attention import attention as attn_mod

    layer = nn.Module()
    layer.kv_cache_dtype = "fp8_e5m2"
    method = _calibrated_ct_method()
    method.create_weights = lambda layer: None
    with (
        patch.object(attn_mod, "resolve_quant_method", return_value=method),
        patch.object(attn_mod, "should_load_quant_weights", return_value=True),
    ):
        if cap is None:
            attn_mod._init_kv_cache_quant(layer, SimpleNamespace(), "layer.0")
        else:
            with patch("vllm.platforms.current_platform", _FakePlatform(cap)):
                attn_mod._init_kv_cache_quant(layer, SimpleNamespace(), "layer.0")


@pytest.mark.skipif(not _IS_SUB_SM89_CUDA, reason="needs a CUDA device below SM89")
def test_calibrated_scheme_allowed_with_e5m2_below_sm89():
    # Static scales calibrated for e4m3 semantics still serve below SM89:
    # e5m2 storage adds mantissa noise but no range risk, and e5m2 is the
    # only fp8 KV flavor triton compiles there.
    _run_kv_cache_quant_init(cap=None)


def test_calibrated_scheme_still_rejected_on_sm89_plus():
    # Where e4m3 is available the original restriction stands.
    with pytest.raises(ValueError, match="fp8_e5m2 kv-cache is not supported"):
        _run_kv_cache_quant_init(cap=89)


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


def test_fp8_prefill_below_sm90_caps_launch_stages():
    """Small-head fp8 KV fits smem with the stock tile, but still launches
    with num_stages=1 below SM90."""
    kwargs = _launch_prefill_below_sm90(torch.float8_e5m2, head_size=128)
    assert kwargs.get("num_stages") == 1
    assert kwargs["TILE_SIZE"] == 32


def test_fp8_large_head_prefill_below_sm90_keeps_stage_cap():
    """fp8 KV with a large head on sub-SM90 launches with num_stages=1 and
    the halved prefill tile: the stock tile/stage combination needs
    114944 B of shared memory against the 101376 B SM80/86 limit."""
    kwargs = _launch_prefill_below_sm90(torch.float8_e5m2, head_size=512)
    assert kwargs.get("num_stages") == 1
    assert kwargs["TILE_SIZE"] == 16


# --- end-to-end on a real sub-SM89 GPU: store + prefill as e5m2 ---


@pytest.mark.skipif(not _IS_SUB_SM90_CUDA, reason="needs a CUDA device below SM90")
@pytest.mark.parametrize("head_size", [128, 256, 512])
def test_store_and_attend_e5m2_end_to_end(head_size):
    """Store bf16 KV through the cache-store wrapper into a uint8 cache on a
    sub-SM89 GPU, read it back through unified_attention, and match a
    reference computed from the stored e5m2 bytes. The head-512 leg also
    guards the SM80/86 shared-memory stage cap: without it the prefill
    launch dies with out-of-resource shared memory."""
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash,
    )
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    from vllm.v1.kv_cache_interface import KVQuantMode

    dev = "cuda"
    block_size, num_blocks = 16, 16
    num_tokens = 64
    num_kv_heads = 1
    torch.manual_seed(0)

    key = torch.randn(num_tokens, num_kv_heads, head_size, device=dev).to(
        torch.bfloat16
    )
    value = torch.randn(num_tokens, num_kv_heads, head_size, device=dev).to(
        torch.bfloat16
    )
    key_cache = torch.zeros(
        num_blocks, block_size, num_kv_heads, head_size, dtype=torch.uint8, device=dev
    )
    value_cache = torch.zeros_like(key_cache)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=dev)
    scale = head_size**-0.5
    ones = torch.ones(1, dtype=torch.float32, device=dev)

    triton_reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        "fp8",
        ones,
        ones,
    )
    torch.cuda.synchronize()

    # Reference from the stored bytes: the wrapper quantizes (KV / scale)
    # to e5m2 with round-to-nearest-even in torch.
    k_e5m2 = key_cache.view(torch.float8_e5m2)
    v_e5m2 = value_cache.view(torch.float8_e5m2)
    stored_k = k_e5m2.view(num_blocks * block_size, num_kv_heads, head_size)[
        :num_tokens, 0
    ].to(torch.float32)
    stored_v = v_e5m2.view(num_blocks * block_size, num_kv_heads, head_size)[
        :num_tokens, 0
    ].to(torch.float32)

    q = torch.randn(num_tokens, 1, head_size, dtype=torch.bfloat16, device=dev)
    out = torch.zeros_like(q)
    cu_seqlens_q = torch.tensor([0, num_tokens], dtype=torch.int32, device=dev)
    seqused_k = torch.tensor([num_tokens], dtype=torch.int32, device=dev)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=dev).view(1, -1)
    k_descale = torch.ones(1, num_kv_heads, dtype=torch.float32, device=dev)
    v_descale = torch.ones(1, num_kv_heads, dtype=torch.float32, device=dev)

    unified_attention(
        q,
        k_e5m2,
        v_e5m2,
        out,
        cu_seqlens_q,
        num_tokens,
        seqused_k,
        num_tokens,
        scale,
        True,
        (-1, -1),
        block_table,
        0.0,
        None,
        k_descale,
        v_descale,
        kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
    )
    torch.cuda.synchronize()

    qf = q[:, 0].to(torch.float32)
    scores = torch.einsum("qh,th->qt", qf, stored_k) * scale
    mask = torch.triu(
        torch.ones(num_tokens, num_tokens, dtype=torch.bool, device=dev), diagonal=1
    )
    probs = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
    ref = torch.einsum("qt,th->qh", probs, stored_v)
    got = out[:, 0].to(torch.float32)
    rel = (got - ref).abs().max() / ref.abs().max().clamp_min(1e-6)
    assert rel < 2e-2, f"max relative error {rel}"


@pytest.mark.skipif(not _IS_SUB_SM90_CUDA, reason="needs a CUDA device below SM90")
def test_unified_attention_rejects_null_q_scale_for_fp8_query():
    """A platform-fp8 query with per-tensor fp8 KV and no q_descale must
    raise before launch: USE_FP8_Q_DESCALE compiles true for such a query,
    so the kernel would read the null q_scale pointer instead of failing
    with a diagnosable error."""
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    from vllm.v1.kv_cache_interface import KVQuantMode

    dev = "cuda"
    head_size, num_tokens = 128, 8
    block_size, num_blocks = 16, 4
    torch.manual_seed(0)

    k_e5m2 = torch.randn(
        num_blocks, block_size, 1, head_size, dtype=torch.bfloat16, device=dev
    ).to(torch.float8_e5m2)
    v_e5m2 = torch.randn_like(k_e5m2, dtype=torch.bfloat16).to(torch.float8_e5m2)
    q = torch.randn(num_tokens, 1, head_size, device=dev)
    q = q.to(current_platform.fp8_dtype())
    out = torch.zeros(num_tokens, 1, head_size, dtype=torch.float32, device=dev)
    cu_seqlens_q = torch.tensor([0, num_tokens], dtype=torch.int32, device=dev)
    seqused_k = torch.tensor([num_tokens], dtype=torch.int32, device=dev)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=dev).view(1, -1)
    descale = torch.ones(1, 1, dtype=torch.float32, device=dev)

    with pytest.raises(ValueError, match="q_descale"):
        unified_attention(
            q,
            k_e5m2,
            v_e5m2,
            out,
            cu_seqlens_q,
            num_tokens,
            seqused_k,
            num_tokens,
            head_size**-0.5,
            True,
            (-1, -1),
            block_table,
            0.0,
            None,
            descale,
            descale,
            kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
        )
