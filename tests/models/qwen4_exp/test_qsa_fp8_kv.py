# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E4M3 KV reader for sm_86: decode exactness and kernel equivalence.

The dispatch helper `_qsa_kv_mode` is unit-tested CPU-side by mocking
`torch.cuda.get_device_capability`. The kernel-bit-exactness and direct
path / split-K path / warmup-compiles tests require CUDA and run as
GPU-OWED on the production fleet.
"""

import json

import pytest
import torch

# Production import order: model.py imports .qsa, never the reverse entry.
# Importing qsa first trips a pre-existing partial-init cycle.
import vllm.models.qwen4_exp.nvidia.model  # noqa: F401
from vllm.models.qwen4_exp.nvidia import qsa as qsa_mod
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.triton_utils import tl, triton

# --- _qsa_kv_mode dispatch (CPU-side, mocks torch.cuda capability) ----------


def test_qsa_kv_mode_dispatch_pins_all_modes(monkeypatch):
    """The wrapper and the warmup both read the mode from this helper; the
    four-way mapping is the single dispatch decision. Mock the capability
    probe so the test runs CPU-side."""
    bf16 = torch.bfloat16
    e5m2 = torch.float8_e5m2
    e4m3 = torch.float8_e4m3fn
    u8 = torch.uint8
    dev = torch.device("cuda:0")

    def with_cap(cap):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: cap)

    with_cap((8, 6))
    assert qsa_ops._qsa_kv_mode(bf16, dev) == 0
    assert qsa_ops._qsa_kv_mode(e5m2, dev) == 1
    assert qsa_ops._qsa_kv_mode(e4m3, dev) == 2
    assert qsa_ops._qsa_kv_mode(u8, dev) == 2

    with_cap((8, 9))
    assert qsa_ops._qsa_kv_mode(bf16, dev) == 0
    assert qsa_ops._qsa_kv_mode(e5m2, dev) == 1
    assert qsa_ops._qsa_kv_mode(e4m3, dev) == 3
    assert qsa_ops._qsa_kv_mode(u8, dev) == 3

    with_cap((9, 0))
    assert qsa_ops._qsa_kv_mode(e4m3, dev) == 3

    with_cap((8, 6))
    with pytest.raises(ValueError, match="does not support KV dtype"):
        qsa_ops._qsa_kv_mode(torch.float16, dev)
    with pytest.raises(ValueError, match="does not support KV dtype"):
        qsa_ops._qsa_kv_mode(torch.float32, dev)


# --- _decode_e4m3_to_bf16: bit-exact over all 254 finite codepoints ----------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_decode_e4m3_bit_exact_on_all_finite_codepoints():
    """The shift-mul decode must equal torch's fp8->bf16 on all 254 finite
    codepoints; 0x7F/0xFF are NaN in e4m3fn and out of contract."""

    @triton.jit
    def _decode_probe(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        b = tl.load(src_ptr + offs, mask=m, other=0)
        tl.store(dst_ptr + offs, qsa_ops._decode_e4m3_to_bf16(b), mask=m)

    codepoints = torch.tensor(
        [b for b in range(256) if b not in (0x7F, 0xFF)], dtype=torch.uint8
    ).to("cuda")
    out = torch.empty(codepoints.shape[0], dtype=torch.bfloat16, device="cuda")
    n = codepoints.shape[0]
    _decode_probe[(triton.cdiv(n, 128),)](codepoints, out, n, BLOCK=128)
    reference = codepoints.view(torch.float8_e4m3fn).to(torch.bfloat16)
    assert torch.equal(out, reference)


# --- kernel equivalence over dequantized rows (direct and split-K paths) -----


def _make_paged_case(selection_width=6, device="cuda", seed=0):
    """Synthetic paged QSA case with in-range fp8 quantizable values.

    Scales are HOST FLOATS, matching the upstream wrapper signature. The
    kernel receives them folded into softmax_scale and output_scale, with
    no device scale buffers in the signature.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    num_rows, num_q_heads, num_kv_heads, head_dim = 8, 4, 2, 64
    page_size = 4
    num_pages = max(16, selection_width + 1)
    k_scale = 0.02
    v_scale = 0.03

    q = (
        (torch.randn(num_rows, num_q_heads, head_dim, generator=g) * 0.5)
        .to(torch.bfloat16)
        .to(device)
    )
    # |k| max ~ 3*sigma = 6 << 448 * 0.02 = 8.96: no saturation, cast is exact
    k = (
        (torch.randn(num_pages * page_size, num_kv_heads, head_dim, generator=g) * 2.0)
        .to(torch.bfloat16)
        .to(device)
    )
    v = (
        (torch.randn(num_pages * page_size, num_kv_heads, head_dim, generator=g) * 2.0)
        .to(torch.bfloat16)
        .to(device)
    )
    indices = torch.zeros(
        num_rows, selection_width + 1, dtype=torch.int32, device=device
    )
    for r in range(num_rows):
        count = int(torch.randint(2, selection_width + 1, (1,), generator=g))
        idx = torch.randperm(num_pages * page_size, generator=g)[:count]
        indices[r, :count] = idx.to(torch.int32)
        indices[r, selection_width] = count
    block_table = (
        torch.arange(num_pages, dtype=torch.int32, device=device)
        .unsqueeze(0)
        .repeat(2, 1)
    )
    token_to_req = torch.zeros(num_rows, dtype=torch.int32, device=device)

    def _page(rows):
        return rows.view(num_pages, page_size, num_kv_heads, head_dim)

    return q, k, v, _page, indices, block_table, token_to_req, k_scale, v_scale


def _ungated(q):
    # The sparse primitive applies sigmoid(output_gate) to its output on this
    # branch. sigmoid(20) is exactly 1.0 in fp32, so a +20 gate leaves the
    # ungated attention values these comparisons are written against.
    return torch.full_like(q, 20.0)


def _fp8_outputs(case):
    q, k, v, page, indices, block_table, token_to_req, k_scale, v_scale = case
    kq = (k.float() / k_scale).to(torch.float8_e4m3fn)
    vq = (v.float() / v_scale).to(torch.float8_e4m3fn)
    k_deq = (kq.to(torch.bfloat16).float() * k_scale).to(torch.bfloat16).to(q.device)
    v_deq = (vq.to(torch.bfloat16).float() * v_scale).to(torch.bfloat16).to(q.device)
    # bf16 reference: the same rows the fp8 path will decode to
    ref = qsa_ops.qsa_sparse_paged_attention(
        q,
        page(k_deq),
        page(v_deq),
        indices,
        block_table,
        token_to_req,
        False,
        output_gate=_ungated(q),
    )
    fp8 = qsa_ops.qsa_sparse_paged_attention(
        q,
        page(kq).to(q.device),
        page(vq).to(q.device),
        indices,
        block_table,
        token_to_req,
        False,
        output_gate=_ungated(q),
        k_scale=k_scale,
        v_scale=v_scale,
    )
    fp8_bytes = qsa_ops.qsa_sparse_paged_attention(
        q,
        page(kq).to(q.device).view(torch.uint8),
        page(vq).to(q.device).view(torch.uint8),
        indices,
        block_table,
        token_to_req,
        False,
        output_gate=_ungated(q),
        k_scale=k_scale,
        v_scale=v_scale,
    )
    return ref, fp8, fp8_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp8_kv_direct_path_matches_bf16_over_dequantized_rows():
    # 6-wide selection: single-split direct-store profile. The bf16 reference
    # cache is itself a bf16-rounded product (decoded * scale), one extra
    # rounding the fp8 path avoids by folding the scale in fp32, so the
    # comparison bound is magnitude-scaled, not elementwise-exact.
    ref, fp8, fp8_bytes = _fp8_outputs(_make_paged_case(selection_width=6))
    diff = (fp8.float() - ref.float()).abs()
    assert diff.max().item() <= 0.01 * ref.float().abs().max().item()
    torch.testing.assert_close(fp8.float(), ref.float(), rtol=1.5e-2, atol=5e-2)
    assert torch.equal(fp8, fp8_bytes)  # uint8 view == fp8 dtype view


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp8_kv_splitk_path_matches_bf16_over_dequantized_rows():
    # 100-wide selection: num_tiles=4, splits=4 -> partials + merge kernel,
    # exercising the v_scale fold through the linear LSE merge
    ref, fp8, fp8_bytes = _fp8_outputs(_make_paged_case(selection_width=100))
    diff = (fp8.float() - ref.float()).abs()
    assert diff.max().item() <= 0.01 * ref.float().abs().max().item()
    torch.testing.assert_close(fp8.float(), ref.float(), rtol=1.5e-2, atol=5e-2)
    assert torch.equal(fp8, fp8_bytes)


# --- warmup: below SM89 a hardcoded mode-3 compiles fp8e4nv at engine boot ---


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_warmup_compiles_fp8_cache_on_this_device():
    """Engine boot warms a uint8 (fp8) cache through the same mode helper
    as dispatch; below SM89 a native e4m3 mode fails to compile fp8e4nv."""
    from vllm.models.qwen4_exp.nvidia.ops.qsa import warmup_qsa_sparse_paged_attention

    head_dim = 128
    kv_cache = torch.zeros((4, 1, 64, 2 * head_dim), dtype=torch.uint8, device="cuda")
    block_table = torch.zeros((1, 4), dtype=torch.int32, device="cuda")
    profiles = warmup_qsa_sparse_paged_attention(
        kv_cache, block_table, num_query_heads=8, selection_width=64
    )
    assert profiles


# --- static sidecar loader and dtype gates -----------------------------------


def _fake_qsa_layer(name, kv_cache_dtype="fp8_e4m3"):
    layer = qsa_mod.Qwen4ExpQSAAttention.__new__(qsa_mod.Qwen4ExpQSAAttention)
    layer.layer_name = name
    layer.kv_cache_dtype = kv_cache_dtype
    layer._k_scale = torch.ones(1)
    layer._v_scale = torch.ones(1)
    return layer


def _sidecar(tmp_path, entries):
    p = tmp_path / "scales.json"
    p.write_text(json.dumps(entries), encoding="utf-8")
    return p


def test_loader_strict_applies_scales(tmp_path):
    layers = {
        "model.layers.3.self_attn.attn": _fake_qsa_layer(
            "model.layers.3.self_attn.attn"
        ),
        "model.layers.7.self_attn.attn": _fake_qsa_layer(
            "model.layers.7.self_attn.attn"
        ),
    }
    p = _sidecar(
        tmp_path,
        {
            "model.layers.3.self_attn.attn": {"k_scale": 0.02, "v_scale": 0.04},
            "model.layers.7.self_attn.attn": {"k_scale": 0.03, "v_scale": 0.05},
        },
    )
    applied = qsa_mod.load_qsa_static_kv_scales(layers, p, strict=True)
    assert applied == [
        "model.layers.3.self_attn.attn",
        "model.layers.7.self_attn.attn",
    ]
    # _k_scale is float32: 0.02 is not exactly representable
    assert layers["model.layers.3.self_attn.attn"]._k_scale.item() == pytest.approx(
        0.02, rel=1e-6
    )
    assert layers["model.layers.7.self_attn.attn"]._v_scale.item() == pytest.approx(
        0.05, rel=1e-6
    )


def test_loader_strict_rejects_partial_and_unknown(tmp_path):
    layers = {"a.attn": _fake_qsa_layer("a.attn"), "b.attn": _fake_qsa_layer("b.attn")}
    with pytest.raises(ValueError, match="Unknown QSA layer names"):
        qsa_mod.load_qsa_static_kv_scales(
            layers,
            _sidecar(
                tmp_path,
                {
                    "a.attn": {"k_scale": 1.0, "v_scale": 1.0},
                    "ghost.attn": {"k_scale": 1.0, "v_scale": 1.0},
                },
            ),
        )
    with pytest.raises(ValueError, match="Missing QSA layer scales"):
        qsa_mod.load_qsa_static_kv_scales(
            layers,
            _sidecar(tmp_path, {"a.attn": {"k_scale": 1.0, "v_scale": 1.0}}),
            strict=True,
        )
    with pytest.raises(ValueError, match="Missing QSA layer scales"):
        qsa_mod.load_qsa_static_kv_scales(layers, _sidecar(tmp_path, {}), strict=True)


def test_loader_rejects_bad_values_and_non_fp8_layers(tmp_path):
    layers = {"a.attn": _fake_qsa_layer("a.attn")}
    with pytest.raises(ValueError, match="finite and positive"):
        qsa_mod.load_qsa_static_kv_scales(
            layers,
            _sidecar(tmp_path, {"a.attn": {"k_scale": 0.0, "v_scale": 1.0}}),
        )
    with pytest.raises(ValueError, match="exactly k_scale and v_scale"):
        qsa_mod.load_qsa_static_kv_scales(
            layers, _sidecar(tmp_path, {"a.attn": {"k_scale": 1.0}})
        )
    bf16_layer = {"a.attn": _fake_qsa_layer("a.attn", kv_cache_dtype="auto")}
    with pytest.raises(ValueError, match="non-FP8 layers"):
        qsa_mod.load_qsa_static_kv_scales(
            bf16_layer,
            _sidecar(tmp_path, {"a.attn": {"k_scale": 1.0, "v_scale": 1.0}}),
        )


def test_qsa_kv_mode_for_e4m3_pins_capability(monkeypatch):
    """After pick #25 retired the SM86-only validation gate, the kernel
    serves e4m3 on every capability. The dispatch helper drives the mode
    decision: mode 2 on SM<89 (software decode) and mode 3 on SM89+
    (native cast)."""
    dev = torch.device("cuda:0")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 6))
    assert qsa_ops._qsa_kv_mode(torch.float8_e4m3fn, dev) == 2
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))
    assert qsa_ops._qsa_kv_mode(torch.float8_e4m3fn, dev) == 3
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    assert qsa_ops._qsa_kv_mode(torch.float8_e4m3fn, dev) == 3


def test_backend_supports_kv_cache_dtype_bypasses_fa_hardware_check():
    # The inherited FlashAttentionBackend.supports_kv_cache_dtype defers
    # quantized dtypes to flash_attn_supports_kv_cache_dtype, which is False
    # for fp8 on sm_86; QSA never dispatches to an FA kernel, so the backend
    # must answer from its own supported_kv_cache_dtypes.
    assert qsa_mod.Qwen4ExpQSAFlashAttentionBackend.supports_kv_cache_dtype("fp8_e4m3")
    assert qsa_mod.Qwen4ExpQSAFlashAttentionBackend.supports_kv_cache_dtype("fp8")
    assert qsa_mod.Qwen4ExpQSAFlashAttentionBackend.supports_kv_cache_dtype(None)
    assert not qsa_mod.Qwen4ExpQSAFlashAttentionBackend.supports_kv_cache_dtype(
        "fp8_e5m2"
    )


def test_sidecar_scales_reach_the_backend_reader(tmp_path):
    """The CUDA writer quantizes with layer._k_scale while the upstream
    backend call passes layer._k_scale_float to the kernel; a sidecar that
    fills only one form decodes with scale 1.0 against a calibrated writer."""
    layer = _fake_qsa_layer("model.layers.0.self_attn.attn")
    sidecar = _sidecar(
        tmp_path,
        {layer.layer_name: {"k_scale": 0.25, "v_scale": 0.5}},
    )
    qsa_mod.load_qsa_static_kv_scales({layer.layer_name: layer}, sidecar)
    assert layer._k_scale.item() == pytest.approx(0.25)
    assert layer._v_scale.item() == pytest.approx(0.5)
    assert layer._k_scale_float == pytest.approx(0.25)
    assert layer._v_scale_float == pytest.approx(0.5)


def test_maybe_load_env_gate(tmp_path, monkeypatch, caplog):
    import logging
    from types import SimpleNamespace

    from vllm.models.qwen4_exp.nvidia import model as model_mod

    sidecar = _sidecar(tmp_path, {"a.attn": {"k_scale": 0.02, "v_scale": 0.04}})
    fake = SimpleNamespace(modules=lambda: iter(()))

    monkeypatch.delenv("VLLM_QSA_KV_SCALES", raising=False)
    model_mod._maybe_load_qsa_static_kv_scales(fake)  # no-op, no raise

    # patch the name model.py bound, not the qsa module attribute
    monkeypatch.setattr(
        model_mod, "load_qsa_static_kv_scales", lambda model, path, strict: ["a.attn"]
    )
    monkeypatch.setenv("VLLM_QSA_KV_SCALES", str(sidecar))
    with caplog.at_level(logging.INFO):
        model_mod._maybe_load_qsa_static_kv_scales(fake)
    assert any("applied static K/V scales" in r.message for r in caplog.records)
