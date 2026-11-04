# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Synthetic gates for ple_int4 (no serve, no full checkpoint).

Run inside the pinned venv:
    python -m pytest ple-int4/tests/test_ple_int4.py -q
or directly:
    python ple-int4/tests/test_ple_int4.py
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ple_int4.kernel import lookup_ple_int4_from_pinned  # noqa: E402
from ple_int4.pack import (  # noqa: E402
    decode_int4,
    pack_table_to_int4,
)

GROUP_SIZE = 32
DIM = 160  # production head width


def _reference_rows(
    table: torch.Tensor, ids: torch.Tensor, group_size: int
) -> torch.Tensor:
    from ple_int4.pack import pack_table_to_int4 as pack

    words, scales = pack(table, group_size)
    rows = decode_int4(words[ids], scales[ids], group_size)
    return rows.to(torch.bfloat16)


def test_pack_round_trip_bound():
    torch.manual_seed(0)
    table = torch.randn(513, DIM) * 1e-3
    words, scales = pack_table_to_int4(table, GROUP_SIZE)
    ref = table
    dec = decode_int4(words, scales, GROUP_SIZE)
    s = scales.to(torch.float32).repeat_interleave(GROUP_SIZE, dim=-1)
    residual = (ref - dec).abs()
    assert (residual <= s * (0.5 + 1e-3)).all()


def test_pack_zero_rows_exact():
    torch.manual_seed(0)
    table = torch.randn(64, DIM)
    table[7] = 0
    table[8] = 0
    words, scales = pack_table_to_int4(table, GROUP_SIZE)
    dec = decode_int4(words, scales, GROUP_SIZE)
    assert (dec[7] == 0).all() and (dec[8] == 0).all()
    assert (scales[7] == 1.0).all()  # sentinel


def test_kernel_bit_exact_vs_reference():
    """The Triton kernel must reproduce the torch decode bit-for-bit (bf16
    nearest-even), with exact +0.0 for out-of-shard rows."""
    if not torch.cuda.is_available():
        print("SKIP (no cuda)")
        return
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    torch.manual_seed(1)
    rows = 4096
    table = torch.randn(rows, DIM) * 0.05
    words, scales = pack_table_to_int4(table, GROUP_SIZE)

    pinned_w = words.to("cpu").pin_memory()
    pinned_s = scales.to("cpu").pin_memory()
    uva_w = get_accelerator_view_from_cpu_tensor(pinned_w)
    uva_s = get_accelerator_view_from_cpu_tensor(pinned_s)

    ids = torch.randint(0, rows, (2048,), dtype=torch.int64, device="cuda")
    out = torch.empty(2048, DIM, dtype=torch.bfloat16, device="cuda")
    lookup_ple_int4_from_pinned(
        uva_w,
        uva_s,
        ids,
        out,
        vocab_start=0,
        vocab_end=rows,
        group_size=GROUP_SIZE,
    )
    ref = _reference_rows(table, ids.cpu(), GROUP_SIZE).cuda()
    assert torch.equal(out, ref), (
        f"max abs diff {(out.float() - ref.float()).abs().max()}"
    )

    # Out-of-range ids -> exact +0.0 (not -0.0)
    ids2 = torch.tensor(
        [rows + 5, rows + 6, 123456789], dtype=torch.int64, device="cuda"
    )
    out2 = torch.empty(3, DIM, dtype=torch.bfloat16, device="cuda")
    lookup_ple_int4_from_pinned(
        uva_w,
        uva_s,
        ids2,
        out2,
        vocab_start=0,
        vocab_end=rows,
        group_size=GROUP_SIZE,
    )
    assert not out2.any() and torch.equal(out2, torch.zeros_like(out2))


def test_kernel_shard_partitioning_exact_zeros():
    """Simulate 2 ETP ranks: disjoint [0, rows/2) and [rows/2, rows) shards;
    after SUM the combined result equals the full-table lookup bitwise."""
    if not torch.cuda.is_available():
        print("SKIP (no cuda)")
        return
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    torch.manual_seed(2)
    rows = 8192
    table = torch.randn(rows, DIM) * 0.05
    words, scales = pack_table_to_int4(table, GROUP_SIZE)
    half = rows // 2

    outs = []
    ids = torch.randint(0, rows, (1024,), dtype=torch.int64, device="cuda")
    for start, end in ((0, half), (half, rows)):
        pinned_w = words[start:end].pin_memory()
        pinned_s = scales[start:end].pin_memory()
        uva_w = get_accelerator_view_from_cpu_tensor(pinned_w)
        uva_s = get_accelerator_view_from_cpu_tensor(pinned_s)
        out = torch.empty(1024, DIM, dtype=torch.bfloat16, device="cuda")
        lookup_ple_int4_from_pinned(
            uva_w,
            uva_s,
            ids,
            out,
            vocab_start=start,
            vocab_end=end,
            group_size=GROUP_SIZE,
        )
        outs.append(out)
    combined = outs[0] + outs[1]  # the ETP SUM over single-owner partials
    ref = _reference_rows(table, ids.cpu(), GROUP_SIZE).cuda()
    assert torch.equal(combined, ref)


def test_fp16_and_int32_uva_views():
    """Burn down the untested fp16 UVA surface empirically."""
    if not torch.cuda.is_available():
        print("SKIP (no cuda)")
        return
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    t32 = torch.zeros(1024, 20, dtype=torch.int32).pin_memory()
    t16 = torch.zeros(1024, 5, dtype=torch.float16).pin_memory()
    t32[:, 3] = 0x01234567
    t16[:, 2] = 1.5
    v32 = get_accelerator_view_from_cpu_tensor(t32)
    v16 = get_accelerator_view_from_cpu_tensor(t16)
    assert v32.device.type == "cuda" and v16.device.type == "cuda"
    assert (v16[:, 2] == 1.5).all().item(), "fp16 UVA readback mismatch"
    assert (v32[:, 3] == 0x01234567).all().item(), "int32 UVA readback mismatch"


def test_dispatch_marker_and_env():
    """The wrapped from_quant_config returns the int4 method on the
    marker, delegates every stock dtype to stock, and unknown markers
    still raise stock's ValueError."""
    from ple_int4 import install, uninstall

    import vllm.models.qwen4_exp.nvidia.ngram_embedding as ng

    uninstall()
    try:
        install()
        m = ng.Qwen4ExpPLEEmbeddingMethod.from_quant_config(None, "prefix", "int4")
        from ple_int4.method import Qwen4ExpPLEInt4EmbeddingMethod

        assert isinstance(m, Qwen4ExpPLEInt4EmbeddingMethod)
        # fp8 marker still routes to stock
        m8 = ng.Qwen4ExpPLEEmbeddingMethod.from_quant_config(
            None, "prefix", "float8_e4m3fn"
        )
        assert type(m8).__name__ == "Qwen4ExpPLEFp8EmbeddingMethod"
        # nvfp4 marker routes to stock's packed-table method
        m4 = ng.Qwen4ExpPLEEmbeddingMethod.from_quant_config(None, "prefix", "nvfp4")
        assert type(m4).__name__ == "Qwen4ExpPLENvFp4EmbeddingMethod"
        # unknown marker with a None quant_config routes to stock unquantized
        mu = ng.Qwen4ExpPLEEmbeddingMethod.from_quant_config(None, "prefix", None)
        assert type(mu).__name__ == "Qwen4ExpPLEUnquantizedEmbeddingMethod"
        # binding follows the last-seen dtype (config-driven): stock after mu
        from ple_int4.method import Qwen4ExpPLEPinnedHostInt4Embedding

        assert (
            ng.Qwen4ExpPLEPinnedHostEmbedding is not Qwen4ExpPLEPinnedHostInt4Embedding
        )
        # and flips to the int4 backend on the int4 marker
        ng.Qwen4ExpPLEEmbeddingMethod.from_quant_config(None, "prefix", "int4")
        assert ng.Qwen4ExpPLEPinnedHostEmbedding is Qwen4ExpPLEPinnedHostInt4Embedding
    finally:
        uninstall()
    # stock behavior restored: the int4 marker is not a stock dtype
    with pytest.raises(ValueError, match="ple_embedding_dtype"):
        ng.Qwen4ExpPLEEmbeddingMethod.from_quant_config(None, "prefix", "int4")


def test_plugin_never_filters_offload_groups():
    """The in-tree exclusion owns group filtering; the plugin must
    not install its own group filter on the offloading config module."""
    from ple_int4 import install, uninstall

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading import config as oc

    uninstall()
    original = oc.get_offloading_group_ids
    install()
    try:
        assert oc.get_offloading_group_ids is original
    finally:
        uninstall()


@pytest.fixture(autouse=True, scope="module")
def _single_process_group():
    """ModelWeightParameter construction reads the TP group; initialize a
    single-process gloo group once for the whole module."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import get_tp_group

    try:
        get_tp_group()
    except AssertionError:
        from vllm.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file:///tmp/ple_int4_dist_{os.getpid()}",
                backend="gloo",
            )
            initialize_model_parallel(tensor_model_parallel_size=1)


def _make_int4_layer():
    """Build the int4 pinned-host layer, with the smallest int32-packed
    weight and per-group scales __init__ accepts."""
    from ple_int4.method import (
        Qwen4ExpPLEInt4EmbeddingMethod,
        Qwen4ExpPLEPinnedHostInt4Embedding,
    )

    rows = 8
    torch.manual_seed(0)
    words, scales = pack_table_to_int4(torch.randn(rows, DIM) * 0.05, GROUP_SIZE)
    layer = Qwen4ExpPLEPinnedHostInt4Embedding(
        rows,
        DIM,
        params_dtype=torch.bfloat16,
        padding_size=rows,
        prefix="ngram_embedding.shard_0",
        embedding_method=Qwen4ExpPLEInt4EmbeddingMethod(),
        num_ngram_heads=1,
        max_total_tokens=8,
    )
    layer.weight.data.copy_(words)
    layer.weight_scale.data.copy_(scales)
    return layer


def test_int4_layer_prefetch_round_trip_through_seam():
    """The int4 layer runs entirely on the stock pinned-host machinery:
    no eager prefetch buffer at construction, a buffer allocated lazily at
    the method's lookup dtype on first prefetch, and decoded rows equal to
    the reference decode with exact +0.0 for out-of-shard ids."""
    if not torch.cuda.is_available():
        print("SKIP (no cuda)")
        return
    layer = _make_int4_layer()

    from ple_int4.method import Qwen4ExpPLEPinnedHostInt4Embedding

    assert isinstance(layer, Qwen4ExpPLEPinnedHostInt4Embedding)
    assert layer.embedding_method.lookup_dtype(layer) == torch.bfloat16
    # D4: no eager prefetch buffer; the base class allocates lazily.
    assert layer._prefetch_buffer is None

    ids = torch.tensor([[0], [1], [2], [3], [4], [5], [6], [99]], device="cuda")
    layer.start_prefetch(ids)
    buffer = layer._prefetch_buffer
    assert buffer is not None
    assert buffer.dtype == torch.bfloat16
    assert buffer.shape == (8, 1, DIM)

    out = layer.finalize_prefetch(total_tokens=8)
    assert out.shape == (8, DIM)
    assert out.dtype == torch.bfloat16

    words = layer.weight.data.cpu()
    scales = layer.weight_scale.data.cpu()
    in_range = ids[:, 0].cpu()
    ref = decode_int4(words[in_range[:7]], scales[in_range[:7]], GROUP_SIZE)
    expected = torch.cat(
        [ref.to(torch.bfloat16), torch.zeros(1, DIM, dtype=torch.bfloat16)]
    ).cuda()
    assert torch.equal(out, expected), (
        f"max abs diff {(out.float() - expected.float()).abs().max()}"
    )


def test_int4_method_refuses_stock_pinned_host_layer():
    """The int4 PLE method must only see layers that are themselves the int4
    pinned-host backend. The stock pinned-host class is resolved through
    module globals the plugin does not rebind (the shared AMD path, and the
    resident-device backend when engram cpu_offload is off), so the wrapper
    can hand the int4 method to a stock layer whose byte-copy lookup would
    then index int32-packed rows with the full row stride and read out of
    bounds. The guard must raise before any allocation."""
    from ple_int4.method import Qwen4ExpPLEInt4EmbeddingMethod

    from vllm.models.qwen4_exp.common.ngram_embedding import (
        Qwen4ExpPLEPinnedHostEmbedding,
    )

    class _StockLikeLayer(Qwen4ExpPLEPinnedHostEmbedding):
        """A stock pinned-host-shaped layer that is NOT the int4 subclass."""

        def __init__(self) -> None:  # noqa: D401 - bypass the real ctor
            # nn.Module setup only; the guard must trigger before
            # allocate_embedding_weight or any UVA handle.
            torch.nn.Module.__init__(self)

        def allocate_embedding_weight(self, num_embeddings, embedding_dim, dtype):
            return torch.empty(num_embeddings, embedding_dim, dtype=dtype, device="cpu")

    method = Qwen4ExpPLEInt4EmbeddingMethod()
    layer = _StockLikeLayer()
    with pytest.raises(RuntimeError, match="int4 pinned-host"):
        method.create_weights(
            layer,
            input_size_per_partition=DIM,
            output_partition_sizes=[128],
            input_size=DIM,
            output_size=128,
            params_dtype=torch.bfloat16,
        )


class _RecordingLoader:
    """Stands in for the embedding's weight_loader; records shard routing and
    copies into the checkpoint-overlapping rows like the real loader."""

    def __init__(self, parameter) -> None:
        self.parameter = parameter
        self.calls: list[tuple[tuple[int, ...], int | None]] = []

    def __call__(self, parameter, loaded_weight, checkpoint_start=None):
        assert parameter is self.parameter
        self.calls.append((tuple(loaded_weight.shape), checkpoint_start))
        with torch.no_grad():
            target = self.parameter.narrow(
                0, checkpoint_start or 0, loaded_weight.shape[0]
            )
            target.copy_(loaded_weight.to(dtype=target.dtype))


class _FakePleEmbedding(torch.nn.Module):
    def __init__(self, rows: int):
        super().__init__()
        from ple_int4.method import Qwen4ExpPLEInt4EmbeddingMethod

        from vllm.model_executor.parameter import ModelWeightParameter

        self.org_vocab_size = rows
        self.embedding_method = Qwen4ExpPLEInt4EmbeddingMethod()
        self.weight = ModelWeightParameter(
            data=torch.zeros(rows, DIM // 8, dtype=torch.int32),
            input_dim=1,
            output_dim=0,
            weight_loader=None,
        )
        self.weight_scale = ModelWeightParameter(
            data=torch.zeros(rows, DIM // GROUP_SIZE, dtype=torch.float16),
            input_dim=1,
            output_dim=0,
            weight_loader=None,
        )
        self.weight.weight_loader = _RecordingLoader(self.weight)
        self.weight_scale.weight_loader = _RecordingLoader(self.weight_scale)


class _FakeNGramEmbedding(torch.nn.Module):
    def __init__(self, embedding: _FakePleEmbedding, split_ngram_parts: int):
        super().__init__()
        self.ngram_embedding = embedding
        self.split_ngram_parts = split_ngram_parts
        self.register_buffer(
            "layer_multipliers", torch.zeros(2, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "ngram_heads_offsets", torch.zeros(4, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "ngram_heads_vocab_sizes", torch.zeros(4, dtype=torch.long), persistent=True
        )


def test_loader_routes_int4_scale_shards_through_ple_loader():
    """The int4 table's per-group scales are checkpoint-sharded like its
    rows; shard_{i}.weight_scale must take the PLE shard loader with the
    destination-derived column count, not the generic AutoWeightsLoader."""
    from ple_int4.method import patched_load_weights

    rows = 8
    embedding = _FakePleEmbedding(rows)
    fake = _FakeNGramEmbedding(embedding, split_ngram_parts=2)
    shard_size = rows // 2
    words = torch.zeros(shard_size, DIM // 8, dtype=torch.int32)
    scales = torch.zeros(shard_size, DIM // GROUP_SIZE, dtype=torch.float16)

    loaded = patched_load_weights(
        fake,
        [
            ("ngram_embedding.shard_0.weight", words),
            ("ngram_embedding.shard_0.weight_scale", scales),
        ],
    )
    assert loaded == {"ngram_embedding.weight", "ngram_embedding.weight_scale"}
    assert embedding.weight.weight_loader.calls == [((shard_size, DIM // 8), 0)]
    assert embedding.weight_scale.weight_loader.calls == [
        ((shard_size, DIM // GROUP_SIZE), 0)
    ]


def test_loader_refuses_unsharded_int4_table():
    """Policy: a packed int4 table arriving unsharded is refused loudly,
    mirroring the stock NVFP4 behavior, instead of silently falling through
    to the generic loader."""
    from ple_int4.method import patched_load_weights

    embedding = _FakePleEmbedding(8)
    fake = _FakeNGramEmbedding(embedding, split_ngram_parts=2)
    unsharded = torch.zeros(8, DIM // 8, dtype=torch.int32)
    with pytest.raises(ValueError, match="unsharded"):
        patched_load_weights(fake, [("ngram_embedding.weight", unsharded)])


def test_loader_refuses_cast_packed_int4_shard():
    """A dtype mismatch on packed int4 words would be silently reinterpreted
    by a cast; the loader must refuse it."""
    from ple_int4.method import patched_load_weights

    embedding = _FakePleEmbedding(8)
    fake = _FakeNGramEmbedding(embedding, split_ngram_parts=2)
    wrong_dtype = torch.zeros(4, DIM // 8, dtype=torch.int8)
    with pytest.raises(ValueError, match="requires"):
        patched_load_weights(fake, [("ngram_embedding.shard_0.weight", wrong_dtype)])


def test_loader_refuses_wrong_shard_shape():
    from ple_int4.method import patched_load_weights

    embedding = _FakePleEmbedding(8)
    fake = _FakeNGramEmbedding(embedding, split_ngram_parts=2)
    wrong_cols = torch.zeros(4, DIM // 8 - 1, dtype=torch.int32)
    with pytest.raises(ValueError, match="Shape mismatch"):
        patched_load_weights(fake, [("ngram_embedding.shard_0.weight", wrong_cols)])


def test_install_refuses_on_common_ngram_drift(monkeypatch):
    """The PLE base classes and the lookup seam live in common/; drift
    there must refuse install like drift in nvidia/."""
    import ple_int4

    monkeypatch.setattr(ple_int4, "PINNED_COMMON_NGRAM_SHA256", "0" * 64)
    ple_int4.uninstall()
    with pytest.raises(RuntimeError, match="common/ngram_embedding.py"):
        ple_int4.install()


def test_install_refuses_on_ngram_drift(monkeypatch):
    import ple_int4

    monkeypatch.setattr(ple_int4, "PINNED_NGRAM_SHA256", "0" * 64)
    ple_int4.uninstall()
    with pytest.raises(RuntimeError, match="ngram_embedding.py"):
        ple_int4.install()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    raise SystemExit(1 if failed else 0)
