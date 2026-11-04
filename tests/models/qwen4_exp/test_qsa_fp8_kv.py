# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QSA fp8 KV surface: decode selection, kernel numerics and the scale sidecar.

The fp8-e4m3 cache reader, the fused-prepare capability conjunct and the
capability-aware warmup are upstream mechanisms; the tests in the first two
sections pin them so the sidecar loader and the calibration collectors that
build on them cannot silently detach from them. The static sidecar loader
(``load_qsa_static_kv_scales``) and the ``VLLM_QSA_KV_SCALES`` call site are
fork additions re-authored onto the upstream weight-load seam.
"""

import json
import math
from types import SimpleNamespace

import pytest
import torch

# Production import order: model.py imports .qsa, never the reverse entry.
# Importing qsa first trips a pre-existing partial-init cycle.
import vllm.models.qwen4_exp.nvidia.model  # noqa: F401
from vllm.models.qwen4_exp.nvidia import qsa as qsa_mod
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

# --- kernel equivalence over dequantized rows (upstream reader numerics) -----


def _make_paged_case(selection_width=6, device="cuda", seed=0):
    """Synthetic paged QSA case with in-range fp8 quantizable values.

    Scales are HOST FLOATS, matching the kernel signature: they are folded
    into softmax_scale and output_scale, with no device scale buffers.
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
    # The kernel accepts the storage form the reader hands it per capability:
    # the raw uint8 bytes below SM89 (a native fp8e4nv pointer does not
    # compile there), the native e4m3 view at or above.
    if qsa_mod.current_platform.has_device_capability(89):
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
    else:
        fp8 = None
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
    diff = (fp8_bytes.float() - ref.float()).abs()
    assert diff.max().item() <= 0.01 * ref.float().abs().max().item()
    torch.testing.assert_close(fp8_bytes.float(), ref.float(), rtol=1.5e-2, atol=5e-2)
    if fp8 is not None:  # SM89+: both storage forms decode identically
        assert torch.equal(fp8, fp8_bytes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp8_kv_splitk_path_matches_bf16_over_dequantized_rows():
    # 100-wide selection: num_tiles=4, splits=4 -> partials + merge kernel,
    # exercising the v_scale fold through the linear LSE merge
    ref, fp8, fp8_bytes = _fp8_outputs(_make_paged_case(selection_width=100))
    diff = (fp8_bytes.float() - ref.float()).abs()
    assert diff.max().item() <= 0.01 * ref.float().abs().max().item()
    torch.testing.assert_close(fp8_bytes.float(), ref.float(), rtol=1.5e-2, atol=5e-2)
    if fp8 is not None:
        assert torch.equal(fp8, fp8_bytes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_warmup_compiles_fp8_cache_on_this_device():
    """Engine boot warms a uint8 (fp8) cache through the same capability
    check as dispatch; below SM89 the warmed dtype is the raw uint8 bytes,
    so boot must not compile a native fp8e4nv kernel."""
    from vllm.models.qwen4_exp.nvidia.ops.qsa import warmup_qsa_sparse_paged_attention

    head_dim = 128
    kv_cache = torch.zeros((4, 1, 64, 2 * head_dim), dtype=torch.uint8, device="cuda")
    block_table = torch.zeros((1, 4), dtype=torch.int32, device="cuda")
    profiles = warmup_qsa_sparse_paged_attention(
        kv_cache, block_table, num_query_heads=8, selection_width=64
    )
    assert profiles


# --- reader selection: byte-decode below SM89, e4m3 view at or above ---------


def _mock_capability(monkeypatch, cap_int: int):
    """Patch the platform singleton so `has_device_capability(<cap>)` answers
    for the given SM integer (e.g. 86, 89) regardless of the real device."""

    def _has(capability, device_id=0):
        cap = (
            capability
            if isinstance(capability, int)
            else capability[0] * 10 + capability[1]
        )
        return cap_int >= cap

    monkeypatch.setattr(qsa_mod.current_platform, "has_device_capability", _has)


def _spy_reader(monkeypatch, seen: dict) -> None:
    def _spy(q, k_cache, v_cache, *args, **kwargs):
        seen["k_dtype"] = k_cache.dtype
        seen["v_dtype"] = v_cache.dtype
        seen["k_scale"] = kwargs.get("k_scale")
        seen["v_scale"] = kwargs.get("v_scale")

    # forward_qsa imports the op from the ops module at call time, so the
    # ops-module attribute is the interception point.
    monkeypatch.setattr(
        "vllm.models.qwen4_exp.nvidia.ops.qsa.qsa_sparse_paged_attention", _spy
    )


def test_reader_selects_byte_decode_below_sm89(monkeypatch):
    """The sm_86 leg of the fp8 reader: forward_qsa must hand the kernel the
    raw uint8 storage (byte-decode) with the layer's host-float scales, and
    the e4m3 view at or above SM89. Mocked both ways because only one
    capability is real on any given host."""
    impl = qsa_mod.Qwen4ExpQSAFlashAttentionImpl.__new__(
        qsa_mod.Qwen4ExpQSAFlashAttentionImpl
    )
    impl.alibi_slopes = None
    impl.sinks = None
    impl.sliding_window = (-1, -1)
    impl.head_size = 64
    impl.kv_cache_dtype = "fp8_e4m3"

    num_tokens, num_kv_heads = 4, 2
    layer = SimpleNamespace(
        topk_indices_buffer=torch.zeros(num_tokens, 7, dtype=torch.int32),
        _k_scale_float=0.25,
        _v_scale_float=0.5,
    )
    kv_cache = torch.zeros(2, num_kv_heads, 8, 2 * 64, dtype=torch.uint8)
    metadata = SimpleNamespace(
        num_actual_tokens=num_tokens,
        block_table=torch.zeros(1, 2, dtype=torch.int32),
    )
    query = torch.zeros(num_tokens, 4, 64, dtype=torch.bfloat16)
    output = torch.zeros(num_tokens, 4, 64, dtype=torch.bfloat16)

    for cap, expected in ((86, torch.uint8), (89, torch.float8_e4m3fn)):
        seen: dict = {}
        _mock_capability(monkeypatch, cap)
        _spy_reader(monkeypatch, seen)
        impl.forward_qsa(
            layer,
            query,
            query,
            query,
            kv_cache,
            metadata,
            output,
            token_to_req=torch.zeros(num_tokens, dtype=torch.int32),
            use_prefill_config=False,
            output_gate=torch.full_like(query, 20.0),
        )
        assert seen["k_dtype"] is expected, f"capability {cap}"
        assert seen["v_dtype"] is expected, f"capability {cap}"
        assert seen["k_scale"] == 0.25
        assert seen["v_scale"] == 0.5


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


# --- static sidecar loader ----------------------------------------------------


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
    # A typo'd local name is skipped as non-local and its layer surfaces
    # through the strict missing check instead.
    with pytest.raises(ValueError, match="Missing QSA layer scales"):
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


def test_loader_skips_non_local_names_under_pipeline_parallel(tmp_path):
    """Under pipeline parallelism each rank's module holds only its stage's
    layers, so a complete sidecar names layers this rank does not own. Those
    names must be skipped, not rejected: strict semantics still hold for the
    local names, whose scales must all be present."""
    local = {
        "model.layers.3.self_attn.attn": _fake_qsa_layer(
            "model.layers.3.self_attn.attn"
        )
    }
    p = _sidecar(
        tmp_path,
        {
            "model.layers.3.self_attn.attn": {"k_scale": 0.02, "v_scale": 0.04},
            "model.layers.7.self_attn.attn": {"k_scale": 0.03, "v_scale": 0.05},
        },
    )
    applied = qsa_mod.load_qsa_static_kv_scales(local, p, strict=True)
    assert applied == ["model.layers.3.self_attn.attn"]
    layer = local["model.layers.3.self_attn.attn"]
    assert layer._k_scale.item() == pytest.approx(0.02, rel=1e-6)
    assert layer._v_scale.item() == pytest.approx(0.04, rel=1e-6)
    assert layer._k_scale_float == pytest.approx(0.02)
    assert layer._v_scale_float == pytest.approx(0.04)


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


def test_sidecar_scales_reach_the_backend_reader(tmp_path):
    """The CUDA writer quantizes with layer._k_scale while the backend call
    passes layer._k_scale_float to the kernel; a sidecar that fills only one
    form decodes with scale 1.0 against a calibrated writer."""
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


# --- VLLM_QSA_KV_SCALES call site --------------------------------------------


def test_maybe_load_env_gate(tmp_path, monkeypatch, caplog):
    import logging

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


# --- fused-prepare wiring and the calibration writer switch -------------------


@pytest.fixture
def tp1_gloo(tmp_path):
    """Single-rank gloo group so the layer's real __init__ can size its
    TP-sharded projections. Gloo never touches CUDA."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"file://{tmp_path / 'dist'}",
            backend="gloo",
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
    yield
    destroy_model_parallel()
    destroy_distributed_environment()


def _build_real_qsa_layer(cache_dtype: str, layer_id: int):
    """Construct the production layer far enough that __init__ itself
    evaluates the fused-prepare conjunct against the mocked device."""
    from transformers import Qwen4ExpTextConfig

    from vllm.config import CacheConfig, VllmConfig, set_current_vllm_config

    hf_cfg = Qwen4ExpTextConfig(
        vocab_size=128,
        eos_token_id=0,
        hidden_size=256,
        num_hidden_layers=2,
        layer_types=["qwen_sparse_attention"] * 2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=4,
        indexer_compress_ratio=4,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
    )
    vc = VllmConfig(cache_config=CacheConfig(cache_dtype=cache_dtype))
    vc.model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        hf_text_config=hf_cfg,
        multimodal_config=None,
        uses_mrope=False,
    )
    # The engine builds models under the model dtype as the default dtype;
    # the rope gate requires a bf16 rope.
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with set_current_vllm_config(vc):
            return qsa_mod.Qwen4ExpQSAAttention(
                vllm_config=vc,
                config=hf_cfg,
                layer_id=layer_id,
                prefix=f"model.layers.{layer_id}.self_attn",
            )
    finally:
        torch.set_default_dtype(prev)


def test_use_fused_qsa_prepare_wiring_pins_sm86_fp8_to_unfused_writer(
    monkeypatch, tp1_gloo
):
    """Drive the real constructor with the capability mocked both ways, so the
    conjunct's inputs flow from the config. No upstream test evaluates the
    wired value: on SM86 an e4m3 cache must resolve to the unfused writer
    (the fused launch's fp8 cast does not compile below SM89), and on SM89 a
    bf16 cache to the fused writer. The calibration switch must not perturb
    either leg while it is unarmed."""
    monkeypatch.setattr(qsa_mod.current_platform, "is_cuda", lambda: True)

    _mock_capability(monkeypatch, 86)
    low = _build_real_qsa_layer("fp8_e4m3", 0)
    assert low.kv_cache_dtype == "fp8_e4m3"
    assert low.use_fused_qk_norm_rope_gate
    assert low.indexer.use_fused_pre_indexer
    assert low.use_fused_qsa_prepare is False

    _mock_capability(monkeypatch, 89)
    high = _build_real_qsa_layer("auto", 1)
    assert high.kv_cache_dtype == "auto"
    assert high.use_fused_qk_norm_rope_gate
    assert high.indexer.use_fused_pre_indexer
    assert high.use_fused_qsa_prepare is True


def test_collect_dir_forces_the_unfused_writer(monkeypatch, tp1_gloo):
    """The absmax collector reads the pre-quantization BF16 K/V that only the
    unfused writer materializes; with it armed, the fused prepare must be off
    even where every upstream conjunct holds. Only the armed flag matters at
    wiring time; no dump happens in this test."""
    _mock_capability(monkeypatch, 89)
    monkeypatch.setattr(qsa_mod.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(qsa_mod, "_qsa_collect_dir", "/tmp/qsa-collect")
    monkeypatch.setattr(qsa_mod, "_qsa_clip_on", False)
    layer = _build_real_qsa_layer("auto", 0)
    assert layer.use_fused_qk_norm_rope_gate
    assert layer.indexer.use_fused_pre_indexer
    assert layer.use_fused_qsa_prepare is False


def test_clip_on_forces_the_unfused_writer(monkeypatch, tp1_gloo):
    """The clip counter reads the same BF16 K/V on the unfused path; with it
    armed, the fused prepare must be off even where every upstream conjunct
    holds."""
    _mock_capability(monkeypatch, 89)
    monkeypatch.setattr(qsa_mod.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(qsa_mod, "_qsa_collect_dir", "")
    monkeypatch.setattr(qsa_mod, "_qsa_clip_on", True)
    layer = _build_real_qsa_layer("auto", 1)
    assert layer.use_fused_qk_norm_rope_gate
    assert layer.indexer.use_fused_pre_indexer
    assert layer.use_fused_qsa_prepare is False


# --- QSA K/V absmax calibration collector -------------------------------------


def test_collector_running_max_and_dump(tmp_path, monkeypatch):
    monkeypatch.setattr(qsa_mod, "_qsa_collect_dir", str(tmp_path))
    monkeypatch.setattr(qsa_mod, "_qsa_collect_state", {})
    monkeypatch.setattr(qsa_mod, "_qsa_collect_calls", 0)
    qsa_mod._qsa_collect_absmax("l0", torch.tensor([1.0, -3.0]), torch.tensor([0.5]))
    qsa_mod._qsa_collect_absmax("l0", torch.tensor([2.0, -1.0]), torch.tensor([4.0]))
    qsa_mod._qsa_collect_absmax("l1", torch.tensor([0.1]), torch.tensor([0.2]))
    st = qsa_mod._qsa_collect_state
    assert st["l0"][0].item() == 3.0 and st["l0"][1].item() == 4.0
    assert st["l1"][0].item() == pytest.approx(0.1)
    assert st["l1"][1].item() == pytest.approx(0.2)
    qsa_mod._qsa_collect_dump()
    # outside a TP group the rank falls back to the pid: any rank file proves
    # the dump contract
    dumped = next(iter(tmp_path.glob("qsa_absmax_rank*.json")))
    data = json.loads(dumped.read_text())
    assert data["l0"] == {"k_absmax": 3.0, "v_absmax": 4.0}


def test_collector_caches_rank_across_distributed_teardown(tmp_path, monkeypatch):
    """Atexit dumps must keep the TP rank name even after the distributed
    group is gone: resolve once, cache, never fall back to pid mid-run."""
    import vllm.distributed as dist_mod

    monkeypatch.setattr(qsa_mod, "_qsa_collect_dir", str(tmp_path))
    monkeypatch.setattr(qsa_mod, "_qsa_collect_state", {})
    monkeypatch.setattr(qsa_mod, "_qsa_collect_calls", 0)
    monkeypatch.setattr(qsa_mod, "_qsa_collect_rank", None)
    monkeypatch.setattr(dist_mod, "get_tensor_model_parallel_rank", lambda: 3)
    qsa_mod._qsa_collect_absmax("l0", torch.tensor([1.0]), torch.tensor([1.0]))
    qsa_mod._qsa_collect_dump()
    assert (tmp_path / "qsa_absmax_rank3.json").is_file()

    def _boom():
        raise RuntimeError("distributed already destroyed")

    monkeypatch.setattr(dist_mod, "get_tensor_model_parallel_rank", _boom)
    qsa_mod._qsa_collect_dump()
    # still the rank name, no pid-named duplicate
    assert (tmp_path / "qsa_absmax_rank3.json").is_file()
    assert len(list(tmp_path.glob("qsa_absmax_rank*.json"))) == 1


# --- QSA K/V clip counter -----------------------------------------------------


def test_clip_env_fail_closed(monkeypatch):
    import importlib

    monkeypatch.delenv("VLLM_QSA_KV_CLIP_COUNT", raising=False)
    monkeypatch.delenv("VLLM_QSA_KV_CLIP_READER", raising=False)
    monkeypatch.setenv("VLLM_QSA_KV_CLIP_READER", "not-a-number")
    with pytest.raises(ValueError, match="an interval in seconds"):
        importlib.reload(qsa_mod)
    monkeypatch.setenv("VLLM_QSA_KV_CLIP_READER", "0")
    with pytest.raises(ValueError, match="finite positive"):
        importlib.reload(qsa_mod)
    monkeypatch.delenv("VLLM_QSA_KV_CLIP_COUNT", raising=False)
    monkeypatch.delenv("VLLM_QSA_KV_CLIP_READER", raising=False)
    importlib.reload(qsa_mod)  # restore


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_clip_drain_reads_counters_armed_under_inference_mode(monkeypatch):
    """Production arms inside a forward running under torch.inference_mode.
    The drain runs later on the engine thread at shutdown; it must read those
    counters without inference-mode or concurrency hazards."""
    monkeypatch.setattr(qsa_mod, "_qsa_clip_counters", {})
    monkeypatch.setattr(qsa_mod, "_qsa_clip_on", True)
    monkeypatch.setattr(qsa_mod, "_qsa_clip_reader_started", True)  # no thread

    layer = _fake_qsa_layer("l0")
    # scale 0.001 -> ceiling 448*0.001 = 0.448: every 5.0 element clips
    layer._k_scale = torch.full((1,), 0.001, device="cuda")
    layer._v_scale = torch.full((1,), 0.001, device="cuda")
    key = torch.full((1, 2, 8), 5.0, device="cuda", dtype=torch.bfloat16)
    value = torch.full((1, 2, 8), 5.0, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        qsa_mod._qsa_clip_count(layer, key, value)
    totals = qsa_mod._qsa_clip_drain()
    assert totals == {"l0": (16, 16)}  # all 16 elements above the ceiling


def test_clip_count_env_enables_drain_reader_env_separate(monkeypatch):
    import importlib

    monkeypatch.setenv("VLLM_QSA_KV_CLIP_COUNT", "1")
    monkeypatch.delenv("VLLM_QSA_KV_CLIP_READER", raising=False)
    importlib.reload(qsa_mod)
    assert qsa_mod._qsa_clip_counting_enabled()
    assert qsa_mod._qsa_clip_interval == 0.0  # no live reader

    monkeypatch.setenv("VLLM_QSA_KV_CLIP_READER", "5")
    importlib.reload(qsa_mod)
    assert qsa_mod._qsa_clip_interval == 5.0

    monkeypatch.delenv("VLLM_QSA_KV_CLIP_COUNT")
    monkeypatch.delenv("VLLM_QSA_KV_CLIP_READER")
    importlib.reload(qsa_mod)  # restore
    assert not qsa_mod._qsa_clip_counting_enabled()
