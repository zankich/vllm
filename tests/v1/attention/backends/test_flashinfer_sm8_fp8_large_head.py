# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM8 large-head fp8-KV opt-in for FlashInfer attention modules.

FlashInfer gates all one-byte-KV large-head (head_dim > 256) FA2
modules to SM100+; the only SM8 opt-in,
`allow_nvfp4_sm8_large_head`, is not recognized for fp8 dtypes, so
large-head models under fp8 KV fail JIT on SM8x ("No supported CUDA
architectures found for major versions [10, 11, 12]"). The fork widens
the opt-in to fp8 at the vLLM flashinfer-backend boundary.

These tests pin the wrapper's contract: fp8 large-head prefill gains
the SM8-inclusive arch list, everything else (nvfp4 semantics,
non-opted-in paths, 16-bit KV) keeps flashinfer's own behavior, the
install is idempotent, and a drifted flashinfer layout raises instead
of silently serving the SM100+ gate.
"""

import pytest
import torch

import vllm.utils.flashinfer as vllm_fi_utils

fi_modules = pytest.importorskip("flashinfer.jit.attention.modules")


@pytest.fixture(autouse=True)
def _installed():
    vllm_fi_utils.install_sm8_fp8_large_head_optin()
    yield


def _flags_for(head_dim, dtype, allow):
    return fi_modules._fa2_head_dim_nvcc_flags(
        head_dim, head_dim, dtype, allow_nvfp4_sm8_large_head=allow
    )


def _majors(flags):
    # e.g. "-gencode=arch=compute_86,code=sm_86" -> 86
    out = set()
    for f in flags or []:
        if "compute_" in f:
            out.add(int(f.split("compute_")[1].split(",")[0].split("}")[0]))
    return out


def test_fp8_large_head_prefill_gains_sm8_arches():
    # The opted-in prefill path with fp8 KV must compile for SM8x
    # instead of refusing with the SM100+ list.
    flags = _flags_for(512, torch.float8_e4m3fn, allow=True)
    majors = _majors(flags)
    assert any(80 <= m < 90 for m in majors), flags
    flags = _flags_for(512, torch.float8_e5m2, allow=True)
    assert any(80 <= m < 90 for m in _majors(flags)), flags


def test_fp8_large_head_without_optin_keeps_sm100_gate():
    # The decode path passes no opt-in: flashinfer's own restriction
    # stands — on an SM8x host that materializes as the JIT arch
    # refusal.
    with pytest.raises(RuntimeError, match="No supported CUDA"):
        _flags_for(512, torch.float8_e4m3fn, allow=False)


def test_sixteen_bit_large_head_unchanged():
    # bf16 large-head already uses the Ampere+ path with or without
    # the wrapper.
    for allow in (True, False):
        majors = _majors(_flags_for(512, torch.bfloat16, allow=allow))
        assert any(80 <= m < 90 for m in majors), (allow, majors)


def test_small_head_returns_none():
    assert _flags_for(128, torch.float8_e4m3fn, allow=True) is None


def test_install_is_idempotent():
    wrapped = fi_modules._fa2_head_dim_nvcc_flags
    vllm_fi_utils.install_sm8_fp8_large_head_optin()
    assert fi_modules._fa2_head_dim_nvcc_flags is wrapped


def test_drifted_layout_raises():
    original = fi_modules._fa2_head_dim_nvcc_flags
    saved = vllm_fi_utils._ORIG_FA2_HEAD_DIM_NVCC_FLAGS
    del fi_modules._fa2_head_dim_nvcc_flags
    vllm_fi_utils._ORIG_FA2_HEAD_DIM_NVCC_FLAGS = None
    try:
        with pytest.raises(RuntimeError, match="flashinfer layout"):
            vllm_fi_utils.install_sm8_fp8_large_head_optin()
    finally:
        fi_modules._fa2_head_dim_nvcc_flags = original
        vllm_fi_utils._ORIG_FA2_HEAD_DIM_NVCC_FLAGS = saved
