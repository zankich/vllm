# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Translate vLLM KV cache metadata for native offloading backends."""

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.utils.math_utils import round_up
from vllm.v1.core.kv_cache_utils import (
    resolve_dcp_kv_block_size,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheGroupRole,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
    iter_layer_specs,
)
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec

logger = init_logger(__name__)


def _get_role_selected_group_ids(kv_cache_config: "KVCacheConfig") -> tuple[int, ...]:
    if kv_cache_config.hisparse_host_num_blocks is None:
        return tuple(range(len(kv_cache_config.kv_cache_groups)))
    return tuple(
        group_id
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        if group.role is KVCacheGroupRole.HISPARSE_INDEXER
    )


def get_misaligned_offloading_group_ids(
    kv_cache_config: "KVCacheConfig", vllm_config: "VllmConfig"
) -> tuple[int, ...]:
    """Role-selected groups whose blocks cannot be chunk-hashed for offload.

    Hybrid + MTP configs can form such groups (e.g. the QSA indexer's
    8-token raw_key_cache against an 800-token hash). Their KV is
    request-lifetime and worthless to offload; callers keep it GPU-resident.
    """
    dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    _, tokens_per_hash = resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)
    return tuple(
        group_id
        for group_id in _get_role_selected_group_ids(kv_cache_config)
        if resolve_dcp_kv_block_size(
            kv_cache_config.kv_cache_groups[group_id].kv_cache_spec, dcp_size
        )
        % tokens_per_hash
        != 0
    )


def get_offloading_group_ids(
    kv_cache_config: "KVCacheConfig", vllm_config: "VllmConfig"
) -> tuple[int, ...]:
    """Group ids that participate in offloading.

    Role selection minus misaligned groups: a group whose tokens_per_block
    does not divide the cross-group tokens_per_hash cannot be chunk-hashed,
    so it neither registers layers nor transfers.
    """
    misaligned_ids = frozenset(
        get_misaligned_offloading_group_ids(kv_cache_config, vllm_config)
    )
    return tuple(
        group_id
        for group_id in _get_role_selected_group_ids(kv_cache_config)
        if group_id not in misaligned_ids
    )


def _group_kv_bytes_per_block(group: "KVCacheGroupSpec") -> int:
    """Return the physical bytes occupied by one block of a cache group.

    Worker configs may retain ``UniformTypeKVCacheSpecs`` while scheduler
    configs flatten that wrapper to one representative per-layer spec.  The
    result is invariant across those two representations only when every
    member has the same page size; mixed-size wrappers must size through
    ``_selected_kv_bytes_per_block_from_tensors`` instead.
    """
    spec = group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return spec.page_size_bytes
    return spec.page_size_bytes * len(group.layer_names)


def _selected_kv_bytes_per_block_from_tensors(
    kv_cache_config: "KVCacheConfig",
    selected_group_ids: tuple[int, ...],
) -> int | None:
    """Per-block bytes of the selected groups, from the tensor layout.

    ``generate_scheduler_kv_cache_config`` flattens every
    ``UniformTypeKVCacheSpecs`` group to one arbitrary representative layer
    spec, so a spec-derived sum is not invariant across the worker and
    scheduler representations when members differ in size: the scheduler
    sizes the offload region by the representative instead of the member
    sum and the two processes build differently sized mmaps over the same
    deterministic path.  The tensor layout is deep-copied unchanged by
    that flattening and buckets layers by spec, so both representations
    yield the same exact sum here.

    Returns None when the tensors cannot serve as the source: when no
    tensors are present; on HiSparse layouts (their hot/resident pages
    use per-block strides); for any selected layer no single-group
    bucket covers exactly; and on the generic block-outer layout, whose
    mixed-page buckets carry a per-block L-axis stride
    (``page_size_bytes``, not ``page * num_blocks``) so the
    divisibility check bails them -- mixed-size wrappers on that path
    still size through the spec-derived sum and can diverge across the
    scheduler flattening.  The caller falls back to the spec-derived
    sum.
    """
    if kv_cache_config.hisparse_host_num_blocks is not None:
        return None
    layer_to_group = {
        layer_name: group_id
        for group_id in selected_group_ids
        for layer_name in kv_cache_config.kv_cache_groups[group_id].layer_names
    }
    num_blocks = kv_cache_config.num_blocks
    covered: set[str] = set()
    total = 0
    for tensor in kv_cache_config.kv_cache_tensors:
        bucket = set(tensor.layers)
        if not bucket or not bucket <= layer_to_group.keys():
            continue
        if len({layer_to_group[name] for name in bucket}) != 1:
            return None
        if tensor.layer_stride % num_blocks != 0:
            group = kv_cache_config.kv_cache_groups[layer_to_group[next(iter(bucket))]]
            logger.debug(
                "offloading: sizing group %s through group specs: tensor "
                "layer_stride=%d does not scale with num_blocks=%d",
                group.layer_names[0],
                tensor.layer_stride,
                num_blocks,
            )
            return None
        total += len(bucket) * (tensor.layer_stride // num_blocks)
        covered |= bucket
    if covered != set(layer_to_group):
        return None
    return total


def build_offloading_config(
    vllm_config: "VllmConfig",
    kv_cache_config: "KVCacheConfig",
) -> OffloadingConfig:
    """Translate vLLM configuration into the native offloading boundary."""
    kv_transfer_config = vllm_config.kv_transfer_config
    assert kv_transfer_config is not None
    extra_config = kv_transfer_config.kv_connector_extra_config
    assert kv_transfer_config.engine_id is not None
    engine_id = kv_transfer_config.engine_id

    parallel_config = vllm_config.parallel_config
    _, tokens_per_hash = resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)
    misaligned_ids = frozenset(
        get_misaligned_offloading_group_ids(kv_cache_config, vllm_config)
    )
    selected_group_ids = get_offloading_group_ids(kv_cache_config, vllm_config)
    if not selected_group_ids:
        if misaligned_ids:
            raise ValueError(
                f"no KV cache group has tokens_per_block divisible by "
                f"tokens_per_hash={tokens_per_hash}; offloading cannot proceed"
            )
        raise ValueError("KV offloading found no eligible cache groups.")
    selected_groups = tuple(
        (group_id, kv_cache_config.kv_cache_groups[group_id])
        for group_id in selected_group_ids
    )
    # Misaligned groups keep a positional entry with no layers: the worker
    # spec and the scheduler index groups in parallel, so dropping the entry
    # would shift every later group. Such a group registers nothing on the
    # worker, never stores or loads, and its KV stays GPU-resident.
    for group_id in sorted(misaligned_ids):
        group = kv_cache_config.kv_cache_groups[group_id]
        logger.warning(
            "offloading: group %s has tokens_per_block=%d not divisible "
            "by tokens_per_hash=%d; keeping it GPU-resident, no layers "
            "registered for offload",
            group.layer_names[0],
            resolve_dcp_kv_block_size(
                group.kv_cache_spec, parallel_config.decode_context_parallel_size
            ),
            tokens_per_hash,
        )
    groups = tuple(
        OffloadingGroupConfig(
            group_id=group_id,
            tokens_per_block=resolve_dcp_kv_block_size(
                kv_cache_config.kv_cache_groups[group_id].kv_cache_spec,
                parallel_config.decode_context_parallel_size,
            ),
            layer_names=(
                ()
                if group_id in misaligned_ids
                else tuple(kv_cache_config.kv_cache_groups[group_id].layer_names)
            ),
        )
        for group_id in sorted(selected_group_ids + tuple(misaligned_ids))
    )

    blocks_per_chunk = 1
    blocks_per_chunk_config = extra_config.get("blocks_per_chunk")
    tokens_per_chunk = extra_config.get("block_size")

    if blocks_per_chunk_config is not None and tokens_per_chunk is not None:
        raise ValueError(
            "Specify only one of 'block_size' or 'blocks_per_chunk' "
            "in kv_connector_extra_config."
        )

    if blocks_per_chunk_config is not None:
        blocks_per_chunk = int(blocks_per_chunk_config)

        if blocks_per_chunk <= 0:
            raise ValueError("'blocks_per_chunk' must be greater than 0.")

    elif tokens_per_chunk is not None:
        tokens_per_chunk_int = int(tokens_per_chunk)

        # Only groups that offload constrain the chunk size; misaligned
        # groups never chunk-hash at any size.
        unique_tokens_per_block = {
            group.tokens_per_block for group in groups if group.layer_names
        }

        assert len(unique_tokens_per_block) == 1, (
            "If 'block_size' is specified in kv_connector_extra_config, "
            "there must be at least one KV cache group, "
            "and all groups must have the same block size."
        )

        tokens_per_block = unique_tokens_per_block.pop()
        if tokens_per_chunk_int % tokens_per_block == 0:
            blocks_per_chunk = tokens_per_chunk_int // tokens_per_block
        else:
            raise ValueError(
                f"'block_size'={tokens_per_chunk_int} in kv_connector_extra_config "
                f"must be a multiple of the GPU KV cache block size "
                f"({tokens_per_block} tokens). Use "
                f"{round_up(tokens_per_chunk_int, tokens_per_block)} instead, or set "
                f"'blocks_per_chunk' to express the chunk size in blocks."
            )

    worker_kv_bytes_per_block = 0
    all_groups_selected = len(selected_groups) == len(kv_cache_config.kv_cache_groups)
    if (
        all_groups_selected
        and kv_cache_config.num_blocks > 0
        and kv_cache_config.kv_cache_tensors
    ):
        # Every KVCacheTensor describes placement within the same backing allocation,
        # so its size is the total, not a per-tensor share.
        total_gpu_kv_bytes = kv_cache_config.kv_cache_tensors[0].size
        worker_kv_bytes_per_block = total_gpu_kv_bytes // kv_cache_config.num_blocks
    elif kv_cache_config.num_blocks > 0:
        selected_bytes = _selected_kv_bytes_per_block_from_tensors(
            kv_cache_config, selected_group_ids
        )
        if selected_bytes is None:
            selected_bytes = sum(
                _group_kv_bytes_per_block(group) for _, group in selected_groups
            )
        worker_kv_bytes_per_block = selected_bytes

    single_group_spec = (
        kv_cache_config.kv_cache_groups[0].kv_cache_spec
        if len(kv_cache_config.kv_cache_groups) == 1
        else None
    )
    replicated_layout = (
        vllm_config.model_config.use_mla
        # Exact type: fail closed on wrappers and sliding-window variants.
        and type(single_group_spec) is MLAAttentionSpec
        # Page accounting: one MLA page per layer, no packed/mixed rows.
        and worker_kv_bytes_per_block > 0
        and worker_kv_bytes_per_block
        == single_group_spec.page_size_bytes
        * len(kv_cache_config.kv_cache_groups[0].layer_names)
        # Safe MVP boundary: TP-only, no other parallel axes.
        and parallel_config.tensor_parallel_size > 1
        and parallel_config.pipeline_parallel_size == 1
        and parallel_config.prefill_context_parallel_size == 1
        and parallel_config.decode_context_parallel_size == 1
        and parallel_config.world_size == parallel_config.tensor_parallel_size
        # Shared /dev/shm mmap layout is single-node mp only.
        and parallel_config.distributed_executor_backend == "mp"
        and parallel_config.nnodes_within_dp == 1
    )

    canonical_layout = bool(extra_config.get("canonical_layout", False))

    # Only a single non-MLA full-attention group with genuinely head-sharded
    # pages is parallelism-invariant: replicated latent or GQA heads,
    # per-token-head scales, CP token sharding, and the V2 model runner's
    # layout are all excluded.
    is_parallelism_agnostic = (
        not vllm_config.use_v2_model_runner
        and single_group_spec is not None
        and isinstance(single_group_spec, FullAttentionSpec)
        and not isinstance(single_group_spec, MLAAttentionSpec)
        and single_group_spec.num_kv_heads * parallel_config.tensor_parallel_size
        == vllm_config.model_config.get_total_num_kv_heads()
        and not single_group_spec.kv_quant_mode.is_per_token_head
        and parallel_config.decode_context_parallel_size == 1
        and parallel_config.prefill_context_parallel_size == 1
    )
    # Canonical pages are topology-free, so the gate widens to every config
    # whose mappings derive portable, group by group; certification happens
    # per layer at registration and create_worker fails closed on this flag.
    if canonical_layout and not is_parallelism_agnostic:
        tp_size = parallel_config.tensor_parallel_size
        total_kv_heads = vllm_config.model_config.get_total_num_kv_heads()

        def spec_certifiable(spec: KVCacheSpec) -> bool:
            """Conservative static mirror of _layer_mapping's per-layer checks."""
            if not isinstance(spec, AttentionSpec):
                return False
            if spec.kv_quant_mode.is_per_token_head:
                return False
            if type(spec) is MLAAttentionSpec:
                return (
                    spec.tokens_per_state == 1
                    and spec.real_page_size_bytes % spec.block_size == 0
                )
            if isinstance(spec, (SlidingWindowMLASpec, MLAAttentionSpec)):
                return False
            if not isinstance(spec, (FullAttentionSpec, SlidingWindowSpec)):
                return False
            return (
                total_kv_heads % tp_size == 0 or tp_size % total_kv_heads == 0
            ) and spec.num_kv_heads == max(1, total_kv_heads // tp_size)

        # UniformTypeKVCacheSpecs groups (e.g. MLA plus its DSA indexer) hold
        # one spec per layer; certify per layer, as the mapping derivation does.
        layer_specs = [
            spec
            for group in kv_cache_config.kv_cache_groups
            for spec in iter_layer_specs(group.kv_cache_spec)
        ]
        is_parallelism_agnostic = (
            len(layer_specs) > 0
            and all(spec_certifiable(spec) for spec in layer_specs)
            and parallel_config.decode_context_parallel_size == 1
            and parallel_config.prefill_context_parallel_size == 1
            and parallel_config.world_size == tp_size
        )

    kv_events_config = vllm_config.kv_events_config
    cache_dtype = (
        vllm_config.model_config.dtype
        if vllm_config.cache_config.cache_dtype == "auto"
        else vllm_config.cache_config.cache_dtype
    )

    return OffloadingConfig(
        groups=groups,
        worker_kv_bytes_per_block=worker_kv_bytes_per_block,
        enable_kv_cache_events=(
            kv_events_config is not None and kv_events_config.enable_kv_cache_events
        ),
        extra_config=extra_config,
        engine_id=engine_id,
        model=OffloadingModelConfig(
            name=vllm_config.model_config.model,
            dtype=str(cache_dtype).removeprefix("torch."),
        ),
        cache=OffloadingCacheConfig(
            tokens_per_hash=tokens_per_hash,
            blocks_per_chunk=blocks_per_chunk,
        ),
        parallel=OffloadingParallelConfig(
            rank=parallel_config.rank,
            world_size=parallel_config.world_size,
            tp_size=parallel_config.tensor_parallel_size,
            pp_size=parallel_config.pipeline_parallel_size,
            pcp_size=parallel_config.prefill_context_parallel_size,
            dcp_size=parallel_config.decode_context_parallel_size,
            data_parallel_index=parallel_config.data_parallel_index,
            data_parallel_size=parallel_config.data_parallel_size,
            data_parallel_rank_local=parallel_config.data_parallel_rank_local,
            is_parallelism_agnostic=is_parallelism_agnostic,
        ),
        replicated_layout=replicated_layout,
        canonical_layout=canonical_layout,
        kv_cache_layout=vllm_config.cache_config.kv_cache_layout,
    )
