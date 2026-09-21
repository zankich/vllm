"""INT4 PLE lookup through pinned host memory (UVA).

One Triton program per output row. Reads packed int4 words (8 little-endian
nibbles per int32, element i of each 8-group in bits 4i, matching
vllm/model_executor/layers/quantization/utils/quant_utils.py pack order) and
per-group fp16 scales through CUDA views of pinned host tensors, dequantizes
inline, and stores bf16 rows.

Rows outside this rank's ETP vocab shard store exact +0.0 so the ETP combine
(single-owner SUM over bf16 partials) is bit-exact: only the owning rank
contributes a nonzero value, and +0.0 + x = x exactly for finite x. Scales are
non-negative and stored codes decode as (code - 8) * scale, so an owner value
of -0.0 is unreachable (0 codes decode to +0.0).

Storage layout produced by ple_int4.pack:
    weight      int32 [rows, embedding_dim // 8]   packed symmetric int4
    weight_scale fp16 [rows, embedding_dim // group_size]
Dequant: value = (code - 8) * scale, with the FP8 global scale already folded
into the group scales at pack time (so the runtime dequant is an identity
multiply and Qwen4ExpPLEInt4EmbeddingMethod.dequantize is a plain cast).
"""
import triton
import triton.language as tl

from ple_int4.probe import _maybe_build

# None unless PLE_INT4_PROBE=1; see probe.py.
_PROBE = _maybe_build()


@triton.jit
def _lookup_ple_int4_from_pinned_kernel(
    weight_ptr,      # int32 [*, words_per_row], device pointer over pinned host
    scale_ptr,       # fp16 [*, n_groups], device pointer over pinned host
    ids_ptr,         # int64 [n] global row ids
    output_ptr,      # bf16 [n, embedding_dim]
    embedding_dim,   # per-head row width in elements (160 for Flash-Next)
    words_per_row,   # embedding_dim // 8
    n_groups,        # embedding_dim // group_size
    group_size,      # values per scale group (32)
    vocab_start,     # shard_indices.org_vocab_start_index
    vocab_end,       # shard_indices.org_vocab_end_index
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    global_idx = tl.load(ids_ptr + row)
    in_range = (global_idx >= vocab_start) & (global_idx < vocab_end)
    local_idx = tl.where(in_range, global_idx - vocab_start, 0)

    offs = tl.arange(0, BLOCK_D)
    store_mask = offs < embedding_dim
    load_mask = store_mask & in_range

    # Duplicate-index word loads: lanes sharing a word hit L1 (idiom from
    # vllm's compressed-tensors embedding kernel; no tl.interleave needed).
    words = tl.load(
        weight_ptr + local_idx * words_per_row + offs // 8,
        mask=load_mask,
        other=0,
    )
    code = ((words >> ((offs % 8) * 4)) & 0xF).to(tl.float32)

    scale = tl.load(
        scale_ptr + local_idx * n_groups + offs // group_size,
        mask=load_mask,
        other=0.0,
    ).to(tl.float32)

    val = tl.where(in_range, (code - 8.0) * scale, 0.0)
    tl.store(
        output_ptr + row * embedding_dim + offs,
        val.to(output_ptr.dtype.element_ty),
        mask=store_mask,
    )


def lookup_ple_int4_from_pinned(
    uva_weight,       # device view over pinned int32 [rows, dim // 8]
    uva_scale,        # device view over pinned fp16 [rows, n_groups]
    flat_ids,         # int64 cuda tensor [n]
    output,           # bf16 cuda tensor [n, embedding_dim]
    *,
    vocab_start: int,
    vocab_end: int,
    group_size: int,
) -> None:
    """Launch the int4 pinned-host lookup into `output` (bf16)."""
    rows, embedding_dim_times8 = uva_weight.shape
    del rows
    embedding_dim = embedding_dim_times8 * 8
    n_groups = uva_scale.shape[1]
    assert n_groups * group_size == embedding_dim, (n_groups, group_size, embedding_dim)
    block_d = triton.next_power_of_2(embedding_dim)
    numel = flat_ids.numel()
    if numel:
        if _PROBE is not None:
            _PROBE.record(flat_ids, vocab_start, vocab_end)
        _lookup_ple_int4_from_pinned_kernel[(numel,)](
            uva_weight,
            uva_scale,
            flat_ids,
            output,
            embedding_dim,
            embedding_dim // 8,
            n_groups,
            group_size,
            vocab_start,
            vocab_end,
            BLOCK_D=block_d,
        )
