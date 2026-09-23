# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E4M3 KV reader for sm_86: decode exactness and kernel equivalence."""

import json
import math
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl

# Production import order: model.py imports .qsa, never the reverse entry.
# Importing qsa first trips a pre-existing partial-init cycle.
import vllm.models.qwen4_exp.nvidia.model  # noqa: F401
from vllm.models.qwen4_exp.nvidia import qsa as qsa_mod
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.platforms import current_platform


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


def _make_paged_case(selection_width=6, device="cuda", seed=0):
    """Synthetic paged QSA case with in-range fp8 quantizable values."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    num_rows, num_q_heads, num_kv_heads, head_dim = 8, 4, 2, 64
    page_size = 4
    num_pages = max(16, selection_width + 1)
    k_scale = torch.tensor(0.02, dtype=torch.float32, device=device)
    v_scale = torch.tensor(0.03, dtype=torch.float32, device=device)

    q = (torch.randn(num_rows, num_q_heads, head_dim, generator=g) * 0.5).to(
        torch.bfloat16
    ).to(device)
    # |k| max ~ 3*sigma = 6 << 448 * 0.02 = 8.96: no saturation, cast is exact
    k = (
        torch.randn(num_pages * page_size, num_kv_heads, head_dim, generator=g)
        * 2.0
    ).to(torch.bfloat16).to(device)
    v = (
        torch.randn(num_pages * page_size, num_kv_heads, head_dim, generator=g)
        * 2.0
    ).to(torch.bfloat16).to(device)
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


def _fp8_outputs(case):
    q, k, v, page, indices, block_table, token_to_req, k_scale, v_scale = case
    kq = (k.float() / k_scale.cpu()).to(torch.float8_e4m3fn)
    vq = (v.float() / v_scale.cpu()).to(torch.float8_e4m3fn)
    k_deq = (kq.to(torch.bfloat16).float() * k_scale.cpu()).to(torch.bfloat16).to(
        q.device
    )
    v_deq = (vq.to(torch.bfloat16).float() * v_scale.cpu()).to(torch.bfloat16).to(
        q.device
    )
    # bf16 reference: the same rows the fp8 path will decode to
    ref = qsa_ops.qsa_sparse_paged_attention(
        q, page(k_deq), page(v_deq), indices, block_table, token_to_req, False
    )
    fp8 = qsa_ops.qsa_sparse_paged_attention(
        q,
        page(kq).to(q.device),
        page(vq).to(q.device),
        indices,
        block_table,
        token_to_req,
        False,
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
        k_scale=k_scale,
        v_scale=v_scale,
    )
    return ref, fp8, fp8_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp8_kv_direct_path_matches_bf16_over_dequantized_rows():
    # 6-wide selection: single-split direct-store profile. The bf16 reference
    # cache is itself a bf16-rounded product (decoded * scale), one extra
    # rounding the fp8 path avoids by folding the scale in fp32 — so the
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp8_kv_rejects_bad_scales_and_mixed_dtypes():
    (
        q,
        k,
        v,
        page,
        indices,
        block_table,
        token_to_req,
        k_scale,
        v_scale,
    ) = _make_paged_case()
    kq = (k.float() / k_scale.cpu()).to(torch.float8_e4m3fn)
    vq = (v.float() / v_scale.cpu()).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="scalar float32"):
        qsa_ops.qsa_sparse_paged_attention(
            q,
            page(kq).to(q.device),
            page(vq).to(q.device),
            indices,
            block_table,
            token_to_req,
            False,
            k_scale=k_scale.double(),
            v_scale=v_scale,
        )
    with pytest.raises(ValueError, match="matching K/V dtypes"):
        qsa_ops.qsa_sparse_paged_attention(
            q,
            page(kq).to(q.device),
            page(v).to(q.device),
            indices,
            block_table,
            token_to_req,
            False,
        )
    with pytest.raises(ValueError, match="BF16 queries"):
        qsa_ops.qsa_sparse_paged_attention(
            q.float(),
            page(kq).to(q.device),
            page(vq).to(q.device),
            indices,
            block_table,
            token_to_req,
            False,
        )


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


def test_fp8_dtype_validation_gates_on_sm86(monkeypatch):

    class _Cap:
        def to_int(self):
            return 86

        def as_version_str(self):
            return "8.6"

    class _Plat:
        @staticmethod
        def is_cuda():
            return True

        @staticmethod
        def get_device_capability():
            return _Cap()

        @staticmethod
        def get_device_name():
            return "fake"

        @staticmethod
        def fp8_dtype():
            return torch.float8_e4m3fn

    monkeypatch.setattr(qsa_mod, "current_platform", _Plat)
    assert qsa_mod._validated_qsa_fp8_dtype("fp8_e4m3") == torch.float8_e4m3fn
    assert qsa_mod._validated_qsa_fp8_dtype("bfloat16") is None

    class _Cap89(_Cap):
        def to_int(self):
            return 89

    monkeypatch.setattr(_Plat, "get_device_capability", staticmethod(lambda: _Cap89()))
    with pytest.raises(ValueError, match="validated only on SM86"):
        qsa_mod._validated_qsa_fp8_dtype("fp8")


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
