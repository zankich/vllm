# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A vanished pinned load at the scheduler-plus-worker pairing.

Fork-local: when a corruption rejection evicts a chunk that two requests'
confirming lookups pinned, the confirmer's load can no longer be served.
The offloading scheduler builds no load job for it and every worker rank
reports the request as a failed receive, so the request never reaches
`transfer_async` with a spec shorter than its GPU block list.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from tests.v1.kv_offload.cpu.test_manager import (
    _make_integrity_region,
    make_req_context,
    make_tiering_cpu_manager,
)
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadingKVEventsConfig,
)
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

BLOCK_SIZE = 16
NUM_TOKENS = 2 * BLOCK_SIZE
NUM_GPU_BLOCKS = 8
NUM_CPU_CHUNKS = 4
NUM_WORKERS = 2


def _make_request(sched: OffloadingConnectorScheduler, req_id: str) -> MagicMock:
    request = MagicMock()
    request.request_id = req_id
    request.kv_transfer_params = None
    request.kv_hints = None
    request.num_prompt_tokens = NUM_TOKENS
    request.num_tokens = NUM_TOKENS
    request.block_hashes = [BlockHash(f"h{i}".encode()) for i in range(2)]
    request.all_token_ids = list(range(NUM_TOKENS))
    request.lora_request = None
    request.skip_reading_prefix_cache = False
    request.is_finished.return_value = False
    sched.on_new_request(request)
    return request


def _make_scheduler(manager) -> OffloadingConnectorScheduler:
    groups = [
        KVCacheGroupSpec(
            ["layer"],
            FullAttentionSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
            ),
        )
    ]
    spec = SimpleNamespace(
        config=SimpleNamespace(
            groups=(SimpleNamespace(group_id=0, tokens_per_block=BLOCK_SIZE),)
        ),
        tokens_per_block=(BLOCK_SIZE,),
        blocks_per_chunk=1,
        tokens_per_hash=BLOCK_SIZE,
        offload_prompt_only=True,
        kv_events_config=OffloadingKVEventsConfig(
            enable_kv_cache_events=False, self_describing_kv_events=False
        ),
        get_manager=lambda: manager,
    )
    config = SimpleNamespace(
        speculative_config=None,
        cache_config=SimpleNamespace(
            enable_prefix_caching=True, prefix_cache_retention_interval=None
        ),
        parallel_config=SimpleNamespace(world_size=1, decode_context_parallel_size=1),
    )
    return OffloadingConnectorScheduler(
        spec,
        config,
        KVCacheConfig(
            num_blocks=NUM_GPU_BLOCKS, kv_cache_tensors=[], kv_cache_groups=groups
        ),
    )


def _make_connector_worker(rank: int) -> OffloadingConnectorWorker:
    """A real connector worker over the real CPU offloading worker, whose
    direction handlers are mocks: the test asserts nothing reaches them."""
    spec = SimpleNamespace(
        replicated_layout=False,
        config=SimpleNamespace(
            canonical_layout=False, parallel=SimpleNamespace(rank=rank)
        ),
    )
    connector_worker = OffloadingConnectorWorker(spec, MagicMock(), MagicMock())
    cpu_worker = CPUOffloadingWorker.__new__(CPUOffloadingWorker)
    cpu_worker._load_handler = MagicMock()
    cpu_worker._load_handler.get_finished.return_value = []
    cpu_worker._store_handler = MagicMock()
    cpu_worker._store_handler.get_finished.return_value = []
    connector_worker.worker = cpu_worker
    return connector_worker


def test_vanished_pinned_load_is_reported_failed_by_every_worker() -> None:
    """A load whose pinned key was rejected mid-flight is never issued: the
    scheduler builds no job and every worker rank reports the request in
    both finished_recving and failed_recving, which is the precondition for
    the output aggregator to surface the failure to the core scheduler.

    Trigger chain: requests A and B confirm the same two chunks (two lookup
    pin shares each); a corruption rejection evicts the first chunk and
    drops its pin; A's prepare_load then finds that key gone. Before the
    fix the manager degraded it to a miss and the short spec, paired with A's
    full GPU block list, tripped the worker's transfer-planning assert."""
    region, write_slot, tamper_slot = _make_integrity_region(
        num_chunks=NUM_CPU_CHUNKS, chunk_size=64
    )
    manager = make_tiering_cpu_manager(num_chunks=NUM_CPU_CHUNKS, mmap_region=region)
    sched = _make_scheduler(manager)

    request_a = _make_request(sched, "req-a")
    request_b = _make_request(sched, "req-b")

    # Seed both chunks clean, through the manager's real store path.
    state_a = sched._req_status["req-a"]
    state_a.update_offload_keys()
    keys = state_a.group_states[0].offload_keys
    assert len(keys) == 2
    seed_ctx = make_req_context(req_id="seed")
    store = manager.prepare_store(keys, seed_ctx)
    assert store is not None and not store.evicted_keys
    chunk_of = dict(zip(store.keys_to_store, store.store_spec.chunk_ids))
    for key, chunk_id in chunk_of.items():
        write_slot(int(chunk_id), int.from_bytes(key[:4], "big") % 1000)
    manager.complete_store(store.keys_to_store, seed_ctx)

    # Two confirmers pin both chunks; the first chunk ends up with two shares.
    external_a, _ = sched.get_num_new_matched_tokens(request_a, 0)
    assert external_a == NUM_TOKENS
    external_b, _ = sched.get_num_new_matched_tokens(request_b, 0)
    assert external_b == NUM_TOKENS
    assert manager._lookup_pin_refs[keys[0]] == 2

    # Between A's lookup and A's load, a third request's integrity check
    # rejects the first chunk as corrupt and evicts it.
    tamper_slot(int(chunk_of[keys[0]]))
    assert manager.lookup(keys[0], make_req_context(req_id="rejector")) is (
        LookupResult.MISS
    )

    # A's allocation reaches the load pairing exactly as production does.
    pool = BlockPool(NUM_GPU_BLOCKS, enable_caching=True, hash_block_size=BLOCK_SIZE)
    full_attn = FullAttentionManager(
        kv_cache_spec=FullAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
        ),
        block_pool=pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=BLOCK_SIZE,
    )
    full_attn.add_local_computed_blocks("req-a", [], 0, external_a)
    full_attn.allocate_external_computed_blocks("req-a", 0, external_a)
    blocks = KVCacheBlocks((full_attn.req_to_blocks["req-a"],))
    sched.update_state_after_alloc(request_a, blocks, external_a)

    # No job, nothing consumed: the surviving key keeps both lookup shares
    # and A's pins stay in place for request finish to release.
    assert not sched._current_batch_load_jobs
    assert not sched._jobs
    assert not sched._req_status["req-a"].transfer_jobs
    assert manager._lookup_pin_refs[keys[1]] == 2
    assert manager._policy.get(keys[1]).ref_cnt == 0
    assert set(manager._get_load_pin_state(state_a.req_context).pins) == set(keys)

    meta = sched.build_connector_meta(SchedulerOutput.make_empty())
    assert not meta.load_jobs
    assert meta.failed_loads == {"req-a"}

    outputs = []
    workers = [_make_connector_worker(rank) for rank in range(NUM_WORKERS)]
    for worker in workers:
        worker.start_kv_transfers(meta)
        results = worker.get_transfer_results(set())
        assert results.failed_recving == {"req-a"}
        assert results.finished_recving == {"req-a"}
        outputs.append(
            ModelRunnerOutput.with_kv_conn_output_only(
                KVConnectorOutput(
                    finished_recving=results.finished_recving,
                    failed_recving=results.failed_recving,
                )
            )
        )
        worker.worker._load_handler.transfer_async.assert_not_called()

    aggregated = KVOutputAggregator(NUM_WORKERS).aggregate(outputs)
    assert aggregated is not None
    kv_output = aggregated.kv_connector_output
    assert kv_output is not None
    assert kv_output.finished_recving == {"req-a"}
    assert kv_output.failed_recving == {"req-a"}

    # Reported once: the next step carries no stale failure.
    assert workers[0].get_transfer_results(set()).failed_recving == set()
