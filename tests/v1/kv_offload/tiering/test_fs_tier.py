# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for FileSystemTierManager.

These tests use real disk I/O to verify the filesystem tier implementation.
The tier manager writes KV cache blocks to disk and reads them back, verifying
data integrity throughout the process.
"""

import mmap
import os
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadingKVEventsConfig,
    OffloadKey,
    ReqContext,
    ScheduleEndContext,
    make_offload_key,
)
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.tiering.base import TransferJob
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm.v1.kv_offload.tiering.fs.manager import (
    FileSystemTierManager,
)
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NUM_BLOCKS = 8
_BLOCK_ELEMENTS = 128 * mmap.PAGESIZE  # 2MB per block for pagesize 4096.
_DTYPE: torch.dtype = torch.float32
_CTX = ReqContext(req_id="test")


def _make_offloading_spec(
    enable_kv_cache_events: bool = False,
    *,
    tp_size: int = 1,
    rank: int = 0,
    world_size: int | None = None,
    replicated_layout: bool = False,
    is_parallelism_agnostic: bool = False,
) -> MagicMock:
    """Mock spec with an explicit global KV events flag."""
    if world_size is None:
        world_size = tp_size
    spec = MagicMock()
    spec.config = OffloadingConfig(
        groups=(),
        worker_kv_bytes_per_block=0,
        enable_kv_cache_events=enable_kv_cache_events,
        extra_config={},
        engine_id="test-engine",
        model=OffloadingModelConfig(name="test-model", dtype="float32"),
        cache=OffloadingCacheConfig(tokens_per_hash=16, blocks_per_chunk=1),
        parallel=OffloadingParallelConfig(
            rank=rank,
            world_size=world_size,
            tp_size=tp_size,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            data_parallel_size=1,
            data_parallel_rank_local=None,
            is_parallelism_agnostic=is_parallelism_agnostic,
        ),
        replicated_layout=replicated_layout,
    )
    spec.blocks_per_chunk = 1
    spec.kv_events_config = OffloadingKVEventsConfig(
        enable_kv_cache_events=enable_kv_cache_events,
        self_describing_kv_events=False,
    )
    return spec


_MOCK_OFFLOADING_SPEC = _make_offloading_spec(enable_kv_cache_events=False)


def key(n: int) -> OffloadKey:
    return make_offload_key(n.to_bytes(8, "big"), 0)


def make_job(
    job_id: int,
    keys: list[OffloadKey],
    block_ids: list[int] | None = None,
    is_promotion: bool = False,
) -> TransferJob:
    if block_ids is None:
        block_ids = list(range(len(keys)))
    return TransferJob(
        job_id=job_id,
        keys=keys,
        block_ids=np.array(block_ids, dtype=np.int64),
        is_promotion=is_promotion,
        req_context=_CTX,
    )


def drain(tier: FileSystemTierManager) -> list:
    """Block until all in-flight jobs finish, then collect results."""
    tier.drain_jobs()
    return list(tier.get_finished_jobs())


def lookup_and_wait(
    tier: FileSystemTierManager,
    keys: list[OffloadKey],
    ctx: ReqContext = _CTX,
    timeout: float = 1.0,
) -> list[LookupResult]:
    """Perform a full async lookup cycle and return resolved results."""
    for k in keys:
        tier.lookup(k, ctx)
    tier.on_schedule_end(ScheduleEndContext(new_req_ids=[], preempted_req_ids=()))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not tier._lookup_manager._pending_results.empty():
            break
        time.sleep(0.01)
    return [tier.lookup(k, ctx) for k in keys]


def _page_aligned_zero_tensor(
    num_blocks: int, block_elements: int, dtype: torch.dtype = _DTYPE
) -> torch.Tensor:
    page_size = mmap.PAGESIZE
    dtype_num_bytes = torch.tensor([], dtype=dtype).element_size()

    num_bytes = num_blocks * block_elements * dtype_num_bytes
    num_bytes_aligned = num_bytes + page_size
    t = torch.zeros(num_bytes_aligned, dtype=torch.uint8)

    ptr = t.data_ptr()
    alignment_offset = ptr % page_size
    # Move tensor to next page regardless.
    shift = page_size - alignment_offset
    t = t[shift : shift + num_bytes]
    return t.view(dtype).view(num_blocks, block_elements)


def _page_aligned_rand_tensor(
    num_blocks: int, block_elements: int, dtype: torch.dtype = _DTYPE
) -> torch.Tensor:
    rand_tensor = _page_aligned_zero_tensor(num_blocks, block_elements)
    rand_tensor[:] = torch.rand(num_blocks, block_elements, dtype=dtype)
    return rand_tensor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fs_tier(tmp_path):
    tensor = _page_aligned_zero_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)
    mock_view = memoryview(tensor.numpy())
    tier = FileSystemTierManager(
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        primary_kv_view=mock_view,
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=4,
        n_write_threads=4,
    )
    yield tier, tensor
    tier.shutdown()


@pytest.fixture
def fs_tier_with_events(tmp_path):
    tensor = _page_aligned_zero_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)
    mock_view = memoryview(tensor.numpy())
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(enable_kv_cache_events=True),
        primary_kv_view=mock_view,
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=4,
        n_write_threads=4,
        enable_kv_events=True,
        locality="LOCAL",
    )
    yield tier
    tier.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_lookup_empty_tier(fs_tier):
    tier, _ = fs_tier
    results = lookup_and_wait(tier, [key(1), key(2)])
    assert results == [LookupResult.MISS, LookupResult.MISS]


def test_store_creates_file_and_lookup_succeeds(fs_tier):
    tier, _ = fs_tier
    job = make_job(1, [key(1)], [0])
    tier.submit_store(job)
    results = drain(tier)
    assert len(results) == 1
    assert results[0].success
    assert lookup_and_wait(tier, [key(1)]) == [LookupResult.HIT]
    dest = tier.file_mapper.get_file_name(key(1))
    assert os.path.exists(dest), f"Expected file at {dest}"


def test_store_then_load_roundtrip(fs_tier):
    tier, _ = fs_tier
    job_s = make_job(1, [key(1), key(2)], [0, 1])
    tier.submit_store(job_s)
    store_results = drain(tier)
    assert all(r.success for r in store_results)

    assert lookup_and_wait(tier, [key(1), key(2)]) == [
        LookupResult.HIT,
        LookupResult.HIT,
    ]

    job_l = make_job(2, [key(1), key(2)], [2, 3], is_promotion=True)
    tier.submit_load(job_l)
    load_results = drain(tier)
    assert all(r.success for r in load_results)
    # A successful load must NOT touch the file: the delete path fires only on
    # a provable short read, so a good block stays on disk (guards against an
    # over-eager delete regressing to upstream's delete-on-any-error).
    for k in (key(1), key(2)):
        assert os.path.exists(tier.file_mapper.get_file_name(k))
    # Blocks stay on disk after load
    assert lookup_and_wait(tier, [key(1), key(2)]) == [
        LookupResult.HIT,
        LookupResult.HIT,
    ]


def test_invalid_path_raises_at_construction():
    """Construction must fail immediately when the config file cannot be written."""
    tensor = _page_aligned_zero_tensor(32, _BLOCK_ELEMENTS)
    mock_view = memoryview(tensor.numpy())

    with pytest.raises(OSError):
        FileSystemTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=mock_view,
            tier_type="fs",
            root_dir="/dev/null/invalid_path",
        )


@pytest.mark.parametrize("locality", ["local", ""])
def test_invalid_locality_raises_at_construction(tmp_path, locality):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)

    with pytest.raises(ValueError, match="Locality"):
        FileSystemTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=memoryview(tensor.numpy()),
            tier_type="fs",
            root_dir=str(tmp_path),
            locality=locality,
        )


def test_factory_forwards_locality_to_fs_tier(tmp_path):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    tier = SecondaryTierFactory.create_secondary_tier(
        {
            "type": "fs",
            "root_dir": str(tmp_path),
            "n_read_threads": 1,
            "n_write_threads": 1,
            "locality": "LOCAL",
        },
        memoryview(tensor.numpy()),
        _MOCK_OFFLOADING_SPEC,
    )
    try:
        assert isinstance(tier, FileSystemTierManager)
        assert tier.locality is Locality.LOCAL
    finally:
        tier.shutdown()


def test_failed_load_missing_file(fs_tier):
    """Test that loading a block whose file does not exist results in a failed job."""
    tier, _ = fs_tier
    job = make_job(1, [key(99)], [0], is_promotion=True)
    tier.submit_load(job)
    results = drain(tier)
    assert len(results) == 1
    assert not results[0].success


def test_multiple_jobs_tracked_independently(fs_tier):
    tier, _ = fs_tier
    job1 = make_job(1, [key(1)], [0])
    job2 = make_job(2, [key(2)], [1])
    tier.submit_store(job1)
    tier.submit_store(job2)
    results = drain(tier)
    job_ids = {r.job_id for r in results}
    assert job_ids == {1, 2}
    assert lookup_and_wait(tier, [key(1), key(2)]) == [
        LookupResult.HIT,
        LookupResult.HIT,
    ]


def test_multi_block_job_partial_failure(fs_tier):
    """A load job where one block file is missing yields a single failed JobResult."""
    tier, _ = fs_tier
    # Store two of three keys
    tier.submit_store(make_job(1, [key(10), key(11)], [0, 1]))
    assert all(r.success for r in drain(tier))

    # Load all three — key(99) was never stored
    tier.submit_load(
        make_job(2, [key(10), key(11), key(99)], [0, 1, 2], is_promotion=True)
    )
    results = drain(tier)

    assert len(results) == 1
    assert results[0].job_id == 2
    assert not results[0].success


def test_shutdown_discards_pending_tasks(fs_tier):
    """Shutdown clears both queues and stops all worker threads without draining."""
    tier, _ = fs_tier
    # Submit many tasks to ensure some remain pending
    for i in range(10):
        tier.submit_store(make_job(i, [key(i)], [i % 4]))

    # Shutdown immediately without draining
    tier.shutdown()

    # Verify queues are cleared and threads stopped
    assert len(tier._pool._load_q) == 0
    assert len(tier._pool._store_q) == 0
    assert all(not t.is_alive() for t in tier._pool._threads)


@pytest.mark.parametrize("batch_size", [0, 1, 2, 5])
@pytest.mark.parametrize("use_c_ext", [True, False])
def test_store_load_data_integrity(fs_tier, monkeypatch, use_c_ext, batch_size):
    """Data written by store must be exactly recovered by load, for batches
    of any size -- including the empty batch."""
    import vllm.v1.kv_offload.tiering.fs.io as io_mod

    if use_c_ext and not io_mod._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", use_c_ext)

    tier, tensor = fs_tier
    # Populate tensor with random data
    tensor[:] = _page_aligned_rand_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)

    keys = [key(i) for i in range(batch_size)]
    store_block_ids = list(range(batch_size))
    load_block_ids = list(range(_NUM_BLOCKS - batch_size, _NUM_BLOCKS))
    expected = tensor[:batch_size].clone()

    tier.submit_store(make_job(1, keys, store_block_ids))
    store_results = drain(tier)
    assert len(store_results) == 1
    assert store_results[0].success
    assert all(os.path.exists(tier.file_mapper.get_file_name(k)) for k in keys)

    # reset tensor to prove data is read from disk
    tensor[:] = 0.0

    # Load into a range disjoint by index from the store ids, to also
    # exercise loading a block into a different id than it was stored from.
    tier.submit_load(make_job(2, keys, load_block_ids, is_promotion=True))
    load_results = drain(tier)
    assert len(load_results) == 1
    assert load_results[0].success

    for i, bid in enumerate(load_block_ids):
        assert torch.allclose(tensor[bid], expected[i]), (
            f"Block {bid} data mismatch after store+load"
        )


def test_store_load_roundtrip_without_o_direct(tmp_path, monkeypatch):
    """Buffered fallback must round-trip data when O_DIRECT is unsupported.

    Simulates filesystems (e.g. overlayfs, some NFS) that reject O_DIRECT by
    forcing the capability probe to report it unavailable.
    """
    monkeypatch.setattr(
        "vllm.v1.kv_offload.tiering.fs.manager.probe_o_direct",
        lambda _dir: False,
    )
    tensor = _page_aligned_rand_tensor(4, _BLOCK_ELEMENTS)
    tier = FileSystemTierManager(
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=4,
        n_write_threads=4,
    )
    try:
        assert tier._use_o_direct is False

        keys = [key(0), key(1)]
        expected = tensor[:2].clone()
        tier.submit_store(make_job(1, keys, [0, 1]))
        assert all(r.success for r in drain(tier))

        tensor[:2] = 0.0
        tier.submit_load(make_job(2, keys, [2, 3], is_promotion=True))
        assert all(r.success for r in drain(tier))

        for i, bid in enumerate([2, 3]):
            assert torch.allclose(tensor[bid], expected[i])
    finally:
        tier.shutdown()


def test_wait_idle_blocks_until_tasks_complete():
    """wait_idle must not return while a task is still in flight."""
    pool = DualQueueThreadPool(n_read_threads=1, n_write_threads=1)
    gate = threading.Event()
    pool.enqueue_store(job_id=1, n_tasks=1, tasks=[lambda: gate.wait(timeout=5.0)])

    waiter = threading.Thread(target=pool.wait_idle)
    waiter.start()
    try:
        waiter.join(timeout=0.2)
        assert waiter.is_alive(), "wait_idle returned before task completed"
        gate.set()
        waiter.join(timeout=5.0)
        assert not waiter.is_alive(), "wait_idle did not unblock"
    finally:
        gate.set()
        pool.shutdown(wait=True)
        waiter.join(timeout=5.0)


def test_batch_lookup_c_extension(tmp_path):
    """Validates batch_lookup_C: empty, single, all-existing, all-missing,
    mixed ordering, and input type validation."""
    try:
        from vllm.fs_io_C import batch_lookup as batch_lookup_C
    except ImportError:
        pytest.skip("fs_io_C extension not built")

    # Setup
    all_exist = [str(tmp_path / f"e{i}.bin") for i in range(3)]
    for p in all_exist:
        open(p, "w").close()
    all_missing = [str(tmp_path / f"m{i}.bin") for i in range(3)]

    # Empty list
    assert batch_lookup_C([]) == []

    # Single existing / missing
    assert batch_lookup_C([all_exist[0]]) == [True]
    assert batch_lookup_C([all_missing[0]]) == [False]

    # All existing / all missing
    assert batch_lookup_C(all_exist) == [True, True, True]
    assert batch_lookup_C(all_missing) == [False, False, False]

    # Mixed — verifies index ordering is preserved
    paths = [val for pair in zip(all_exist, all_missing) for val in pair]
    assert batch_lookup_C(paths) == [True, False, True, False, True, False]

    # Input validation: non-list argument
    with pytest.raises(TypeError):
        batch_lookup_C(("/tmp/foo",))
    with pytest.raises(TypeError):
        batch_lookup_C(None)

    # Input validation: non-str elements in list
    with pytest.raises(TypeError):
        batch_lookup_C([None])
    with pytest.raises(TypeError):
        batch_lookup_C([b"/tmp/foo"])
    with pytest.raises(TypeError):
        batch_lookup_C([42])
    with pytest.raises(TypeError):
        batch_lookup_C([all_exist[0], None])  # valid first, invalid mid-list


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_batch_lookup_dispatch(fs_tier, monkeypatch, use_c_ext):
    import vllm.v1.kv_offload.tiering.fs.manager as mgr_mod

    if use_c_ext and not mgr_mod._HAS_BATCH_LOOKUP_C:
        pytest.skip("fs_io_C extension not built")

    monkeypatch.setattr(mgr_mod, "_HAS_BATCH_LOOKUP_C", use_c_ext)

    tier, _ = fs_tier
    tier.submit_store(make_job(1, [key(1)], [0]))
    assert all(r.success for r in drain(tier))

    results = lookup_and_wait(tier, [key(1), key(2)])
    assert results == [LookupResult.HIT, LookupResult.MISS]


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_out_of_bounds_block_id_smoke(fs_tier, monkeypatch, use_c_ext):
    """Smoke test: a block id beyond the primary tensor's block count must
    fail the job, for both the C extension and the Python fallback."""
    import vllm.v1.kv_offload.tiering.fs.io as io_mod

    if use_c_ext and not io_mod._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", use_c_ext)

    tier, tensor = fs_tier
    out_of_bounds_bid = tensor.shape[0]  # one past the last valid block

    tier.submit_store(make_job(1, [key(1)], [out_of_bounds_bid]))
    store_results = drain(tier)
    assert len(store_results) == 1
    assert not store_results[0].success

    tier.submit_load(make_job(2, [key(1)], [out_of_bounds_bid], is_promotion=True))
    load_results = drain(tier)
    assert len(load_results) == 1
    assert not load_results[0].success


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_failed_load_corrects_verdict_and_removes_corrupt_file(
    fs_tier, monkeypatch, use_c_ext
):
    """Failed-load livelock regression, covering the whole contract.

    A successful promotion leaves the cached HIT and the on-disk block intact.
    A promotion that short-reads a truncated (corrupt) block fails, and in
    get_finished_jobs() the tier removes the corrupt file (stores are atomic,
    so a too-short file is genuine corruption) and marks the cached verdict
    False. The SAME request's next lookup is then a MISS served from cache with
    NO re-probe, so the scheduler cannot re-issue the doomed promotion.
    """
    import vllm.v1.kv_offload.tiering.fs.io as io_mod

    if use_c_ext and not io_mod._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", use_c_ext)

    tier, _ = fs_tier
    tier.submit_store(make_job(1, [key(1)], [0]))
    assert all(r.success for r in drain(tier))
    path = tier.file_mapper.get_file_name(key(1))

    ctx = ReqContext(req_id="livelock-req")
    assert lookup_and_wait(tier, [key(1)], ctx=ctx) == [LookupResult.HIT]

    # A successful promotion must NOT touch the verdict or the file.
    tier.submit_load(make_job(2, [key(1)], [0], is_promotion=True))
    results = drain(tier)
    assert len(results) == 1 and results[0].success
    assert tier.lookup(key(1), ctx) == LookupResult.HIT
    assert os.path.exists(path)

    # Truncate below block_size so the next promotion short-reads.
    with open(path, "wb") as f:
        f.write(b"x" * 10)
    tier.submit_load(make_job(3, [key(1)], [0], is_promotion=True))
    results = drain(tier)  # get_finished_jobs() marks the verdict False here
    assert len(results) == 1 and not results[0].success

    # Corrupt file removed; the SAME request now misses from cache, no re-probe.
    assert not os.path.exists(path)
    lm = tier._lookup_manager
    assert tier.lookup(key(1), ctx) == LookupResult.MISS
    assert lm._lookup_batch == []

    # A FRESH request re-probes the tier (no cached verdict) and misses too,
    # since the corrupt file is gone -- the real batch_lookup re-probe path.
    fresh = ReqContext(req_id="fresh-after-short-read")
    assert lookup_and_wait(tier, [key(1)], ctx=fresh) == [LookupResult.MISS]


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_batched_partial_load_failure_keeps_loaded_blocks(
    fs_tier, monkeypatch, use_c_ext
):
    """A batched promotion stops at the first bad block and reports how many
    loaded before it (#50321). Corrupt the LAST block: the earlier blocks load
    fine, so the job reports successful_keys for them and marks only the failed
    tail a miss. The earlier keys stay HIT — including for the same request —
    while the corrupt block stays a MISS (its file was removed)."""
    import vllm.v1.kv_offload.tiering.fs.io as io_mod

    if use_c_ext and not io_mod._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", use_c_ext)

    tier, _ = fs_tier
    keys = [key(1), key(2), key(3)]  # last one is the "bad" block
    tier.submit_store(make_job(1, keys, [0, 1, 2]))
    assert all(r.success for r in drain(tier))
    bad_path = tier.file_mapper.get_file_name(key(3))

    ctx = ReqContext(req_id="batch-req")
    assert lookup_and_wait(tier, keys, ctx=ctx) == [LookupResult.HIT] * 3

    # Corrupt only the last block, then load the whole batch as one job.
    with open(bad_path, "wb") as f:
        f.write(b"x" * 10)
    tier.submit_load(make_job(2, keys, [0, 1, 2], is_promotion=True))
    results = drain(tier)
    # (a) the job fails but reports the two blocks that loaded before the bad one.
    assert len(results) == 1 and not results[0].success
    assert tuple(results[0].successful_keys) == (key(1), key(2))

    # (b) Only the failed tail is a miss; the loaded blocks stay HIT on the same
    # request, and nothing was re-probed.
    lm = tier._lookup_manager
    assert [tier.lookup(k, ctx) for k in keys] == [
        LookupResult.HIT,
        LookupResult.HIT,
        LookupResult.MISS,
    ]
    assert lm._lookup_batch == []

    # (c) A fresh request re-probes: the loaded blocks are still on disk (HIT),
    # only the corrupt block was removed (MISS).
    tier.on_request_finished(ctx)
    fresh = ReqContext(req_id="fresh-batch-req")
    assert lookup_and_wait(tier, keys, ctx=fresh) == [
        LookupResult.HIT,
        LookupResult.HIT,
        LookupResult.MISS,
    ]


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_batched_load_first_block_fails_marks_whole_batch(
    fs_tier, monkeypatch, use_c_ext
):
    """When the FIRST block fails, nothing loaded before it: the job reports no
    successful_keys (None) and the whole batch is marked a miss for the
    request."""
    import vllm.v1.kv_offload.tiering.fs.io as io_mod

    if use_c_ext and not io_mod._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", use_c_ext)

    tier, _ = fs_tier
    keys = [key(1), key(2), key(3)]  # first one is the "bad" block
    tier.submit_store(make_job(1, keys, [0, 1, 2]))
    assert all(r.success for r in drain(tier))

    ctx = ReqContext(req_id="batch-first-fail")
    assert lookup_and_wait(tier, keys, ctx=ctx) == [LookupResult.HIT] * 3

    with open(tier.file_mapper.get_file_name(key(1)), "wb") as f:
        f.write(b"x" * 10)
    tier.submit_load(make_job(2, keys, [0, 1, 2], is_promotion=True))
    results = drain(tier)
    assert len(results) == 1 and not results[0].success
    # Nothing loaded before the failure -> no partial success reported.
    assert results[0].successful_keys is None
    # The whole batch is a miss for this request.
    assert [tier.lookup(k, ctx) for k in keys] == [LookupResult.MISS] * 3


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_transient_load_failure_leaves_file(fs_tier, monkeypatch, use_c_ext):
    """A transient host error (here ELOOP on open) is NOT a short read: the job
    fails but the block file must survive untouched, on both the C and Python
    paths. Deleting on a transient error would turn a passing hiccup into
    permanent data loss."""
    import vllm.v1.kv_offload.tiering.fs.io as io_mod

    if use_c_ext and not io_mod._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", use_c_ext)

    tier, _ = fs_tier
    tier.submit_store(make_job(1, [key(1)], [0]))
    assert all(r.success for r in drain(tier))
    path = tier.file_mapper.get_file_name(key(1))
    with open(path, "rb") as f:
        original = f.read()

    # Make open() fail with ELOOP (fd < 0) without truncating the block. Not
    # chmod 000: CI runs as root, which bypasses permission bits, so open()
    # would succeed and the load would not fail at all.
    saved = path + ".saved"
    loop = path + ".loop"
    os.rename(path, saved)
    os.symlink(loop, path)
    os.symlink(path, loop)

    tier.submit_load(make_job(2, [key(1)], [0], is_promotion=True))
    results = drain(tier)
    assert len(results) == 1 and not results[0].success

    # The path is left alone: a non-short-read error must not unlink.
    assert os.path.lexists(path)

    os.unlink(path)
    os.unlink(loop)
    os.rename(saved, path)
    with open(path, "rb") as f:
        assert f.read() == original


# ---------------------------------------------------------------------------
# KV events
# ---------------------------------------------------------------------------


def test_successful_store_emits_stored_event(fs_tier_with_events):
    """A completed store job emits one stored event with the job's keys."""
    tier = fs_tier_with_events
    keys = [key(1), key(2)]
    tier.submit_store(make_job(1, keys, [0, 1]))
    assert all(r.success for r in drain(tier))

    events = list(tier.take_events())
    assert len(events) == 1
    assert events[0].keys == keys
    assert events[0].medium == Medium.STORAGE
    assert events[0].locality is Locality.LOCAL
    assert not events[0].removed
    # take_events drains the buffer.
    assert list(tier.take_events()) == []


@pytest.mark.parametrize(
    ("locality", "expected"),
    [(None, None), ("REMOTE", Locality.REMOTE)],
)
def test_store_event_uses_configured_locality(tmp_path, locality, expected):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    locality_config = {} if locality is None else {"locality": locality}
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(enable_kv_cache_events=True),
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        enable_kv_events=True,
        **locality_config,
    )
    try:
        tier.submit_store(make_job(1, [key(1)], [0]))
        assert all(r.success for r in drain(tier))

        events = list(tier.take_events())
        assert len(events) == 1
        assert events[0].locality is expected
    finally:
        tier.shutdown()


def test_load_job_emits_no_event(fs_tier_with_events):
    tier = fs_tier_with_events
    tier.submit_store(make_job(1, [key(1)], [0]))
    results = drain(tier)
    assert len(results) == 1
    assert results[0].success
    list(tier.take_events())

    tier.submit_load(make_job(2, [key(1)], [1], is_promotion=True))
    results = drain(tier)
    assert len(results) == 1
    assert results[0].success
    assert list(tier.take_events()) == []


def test_mixed_job_results_emit_event_only_for_successful_job(
    fs_tier_with_events, monkeypatch
):
    """With a failed and a successful store job in flight, exactly one event
    is emitted and its keys belong to the successful job."""
    import vllm.v1.kv_offload.tiering.fs.manager as mgr_mod

    tier = fs_tier_with_events
    failing_path = tier.file_mapper.get_file_name(key(1))
    original_batch_store_block = mgr_mod.batch_store_block

    def flaky_batch_store_block(paths, *args, **kwargs):
        if failing_path in paths:
            raise OSError("injected store failure")
        return original_batch_store_block(paths, *args, **kwargs)

    monkeypatch.setattr(mgr_mod, "batch_store_block", flaky_batch_store_block)

    tier.submit_store(make_job(1, [key(1)], [0]))
    tier.submit_store(make_job(2, [key(2)], [1]))
    results = drain(tier)
    assert len(results) == 2
    by_id = {r.job_id: r for r in results}
    assert not by_id[1].success
    assert by_id[2].success

    events = list(tier.take_events())
    assert len(events) == 1
    assert events[0].keys == [key(2)]


def test_partially_failed_store_emits_no_event(fs_tier_with_events, monkeypatch):
    """A store job with any failed block emits no event for the whole job."""
    import vllm.v1.kv_offload.tiering.fs.manager as mgr_mod

    tier = fs_tier_with_events
    failing_path = tier.file_mapper.get_file_name(key(2))
    original_batch_store_block = mgr_mod.batch_store_block

    def flaky_batch_store_block(paths, *args, **kwargs):
        if failing_path in paths:
            raise OSError("injected store failure")
        return original_batch_store_block(paths, *args, **kwargs)

    monkeypatch.setattr(mgr_mod, "batch_store_block", flaky_batch_store_block)

    tier.submit_store(make_job(1, [key(1), key(2)], [0, 1]))
    results = drain(tier)
    assert len(results) == 1
    assert not results[0].success
    assert list(tier.take_events()) == []
    assert tier._store_job_keys == {}


def test_events_disabled_by_default(fs_tier):
    tier, _ = fs_tier
    tier.submit_store(make_job(1, [key(1)], [0]))
    results = drain(tier)
    assert len(results) == 1
    assert results[0].success
    assert tier.events is None
    assert tier._store_job_keys == {}
    assert list(tier.take_events()) == []


def test_events_require_global_kv_events_flag(tmp_path):
    """Tier-level opt-in alone is not enough; the global flag gates events."""
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(enable_kv_cache_events=False),
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        enable_kv_events=True,
    )
    try:
        assert tier.events is None
        tier.submit_store(make_job(1, [key(1)], [0]))
        results = drain(tier)
        assert len(results) == 1
        assert results[0].success
        assert list(tier.take_events()) == []
        assert tier._store_job_keys == {}
    finally:
        tier.shutdown()


def test_cascade_store_emits_fs_event_through_tiering_manager(tmp_path):
    """A GPU->CPU->fs cascade surfaces the tier-owned FS stored event via the
    TieringOffloadingManager's aggregated take_events()."""
    from vllm.v1.kv_offload.tiering.manager import (
        CPUPrimaryTierOffloadingManager,
        TieringOffloadingManager,
    )

    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    view = memoryview(tensor.numpy())
    mock_region = MagicMock()
    mock_region.create_kv_memoryview.return_value = view
    primary = CPUPrimaryTierOffloadingManager(num_blocks=4, mmap_region=mock_region)
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(enable_kv_cache_events=True),
        primary_kv_view=primary.get_kv_memoryview(),
        tier_type="fs",
        root_dir=str(tmp_path),
        enable_kv_events=True,
    )
    manager = TieringOffloadingManager(primary_tier=primary, secondary_tiers=[tier])
    try:
        keys = [key(1), key(2)]
        manager.on_new_request(_CTX)
        assert manager.prepare_store(keys, _CTX) is not None
        manager.complete_store(keys, _CTX)  # cascades to the fs tier

        events: list[OffloadingEvent] = []
        ctx = ScheduleEndContext(new_req_ids=[], preempted_req_ids=())
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not events:
            manager.on_schedule_end(ctx)
            events.extend(manager.take_events())
            time.sleep(0.01)

        fs_events = [e for e in events if e.medium == Medium.STORAGE]
        assert len(fs_events) == 1
        assert set(fs_events[0].keys) == set(keys)
        assert not fs_events[0].removed
    finally:
        tier.shutdown()


def test_fs_tier_cross_tp_round_trip(tmp_path):
    """TP=2 replicated writer and TP=4 reader share namespace and bytes."""
    root = str(tmp_path)
    writer_tensor = _page_aligned_rand_tensor(4, _BLOCK_ELEMENTS)
    expected = writer_tensor[0].clone()
    writer = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(
            tp_size=2, world_size=2, rank=0, replicated_layout=True
        ),
        primary_kv_view=memoryview(writer_tensor.numpy()),
        tier_type="fs",
        root_dir=root,
        n_read_threads=2,
        n_write_threads=2,
    )
    try:
        writer.submit_store(make_job(1, [key(7)], [0]))
        assert all(r.success for r in drain(writer))
        writer_base = writer.file_mapper.base_path
        writer_path = writer.file_mapper.get_file_name(key(7))
    finally:
        writer.shutdown()

    reader_tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    reader = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(
            tp_size=4, world_size=4, rank=3, replicated_layout=True
        ),
        primary_kv_view=memoryview(reader_tensor.numpy()),
        tier_type="fs",
        root_dir=root,
        n_read_threads=2,
        n_write_threads=2,
    )
    try:
        assert reader.file_mapper.base_path == writer_base
        assert reader.file_mapper.get_file_name(key(7)) == writer_path
        assert lookup_and_wait(reader, [key(7)]) == [LookupResult.HIT]
        reader.submit_load(make_job(2, [key(7)], [1], is_promotion=True))
        assert all(r.success for r in drain(reader))
        assert torch.allclose(reader_tensor[1], expected)
    finally:
        reader.shutdown()


# ---------------------------------------------------------------------------
# Integrity: silently-wrong tier content must degrade to a clean miss.
# Fork-local (2026-09-17): the corruption RCA showed full-length but
# wrong-for-key block bytes restoring without any error. Every test below
# corrupts stored content in a way today's load path cannot see; each load
# must fail, remove the bad file, and flip the cached verdict to MISS so the
# scheduler recomputes instead of attending to garbage.
# ---------------------------------------------------------------------------

_SIDECAR_SUFFIX = ".meta"  # legacy carrier, kept for the orphan-inertness test
_XATTR = "user.vllm_kv_integrity"


def _store_blocks(tier, tensor, seeds: list[tuple[int, float]]):
    """Store one block per (block_id, fill value) and wait for completion."""
    for block_id, fill in seeds:
        tensor[block_id] = fill
    job = make_job(1, [key(i) for i in range(len(seeds))], [b for b, _ in seeds])
    tier.submit_store(job)
    assert all(r.success for r in drain(tier))


def _strip_record(path: str) -> None:
    import contextlib

    with contextlib.suppress(OSError):
        os.removexattr(path, _XATTR)


def test_load_rejects_payload_tampered_in_place(fs_tier):
    """Same-length in-place byte rewrite must fail the load, not serve it."""
    tier, tensor = fs_tier
    _store_blocks(tier, tensor, [(0, 0.25)])
    path = tier.file_mapper.get_file_name(key(0))
    with open(path, "r+b") as f:
        f.write(b"\x01" * os.path.getsize(path))
    ctx = ReqContext(req_id="tamper-req")
    assert lookup_and_wait(tier, [key(0)], ctx=ctx) == [LookupResult.HIT]
    tier.submit_load(make_job(2, [key(0)], [1], is_promotion=True))
    results = drain(tier)
    assert not results[0].success, "tampered payload must fail the load"
    assert not os.path.exists(path), "tampered payload must be removed"
    assert tier.lookup(key(0), ctx) in (LookupResult.MISS, LookupResult.RETRY)


def test_load_rejects_block_written_under_a_different_key(fs_tier):
    """Index-confusion shape: key(1)'s payload file overwritten with key(2)'s
    bytes. The integrity record must not verify content against the wrong key."""
    tier, tensor = fs_tier
    _store_blocks(tier, tensor, [(0, 0.25), (1, 0.75)])
    path0 = tier.file_mapper.get_file_name(key(0))
    path1 = tier.file_mapper.get_file_name(key(1))
    import shutil

    shutil.copyfile(path1, path0)
    ctx = ReqContext(req_id="crosskey-req")
    assert lookup_and_wait(tier, [key(0)], ctx=ctx) == [LookupResult.HIT]
    tier.submit_load(make_job(2, [key(0)], [2], is_promotion=True))
    results = drain(tier)
    assert not results[0].success, "wrong-for-key payload must fail the load"
    assert not os.path.exists(path0), "rejected payload must be removed"
    assert tier.lookup(key(0), ctx) in (LookupResult.MISS, LookupResult.RETRY)


def test_load_rejects_transplanted_integrity_record(fs_tier):
    """Even a fully consistent payload+record pair transplanted from another
    key must be rejected: the record names its key, and the load is for a
    different one."""
    import struct

    tier, tensor = fs_tier
    _store_blocks(tier, tensor, [(0, 0.25), (1, 0.75)])
    path0 = tier.file_mapper.get_file_name(key(0))
    path1 = tier.file_mapper.get_file_name(key(1))
    import shutil

    shutil.copyfile(path1, path0)
    # Hand-pack a format-correct record naming key(1) and plant it on key(0)'s
    # payload; the checksum field is irrelevant to the key-binding rejection.
    k1 = bytes(key(1))
    record = struct.pack("<4sBH16s", b"KVMI", 1, len(k1), b"\x00" * 16) + k1
    os.setxattr(path0, _XATTR, record)
    ctx = ReqContext(req_id="transplant-req")
    assert lookup_and_wait(tier, [key(0)], ctx=ctx) == [LookupResult.HIT]
    tier.submit_load(make_job(2, [key(0)], [2], is_promotion=True))
    results = drain(tier)
    assert not results[0].success, "transplanted record must fail the load"
    assert not os.path.exists(path0), "rejected payload must be removed"


def test_load_without_integrity_record_is_a_miss(fs_tier):
    """A payload whose record is stripped (legacy or tampered) cannot be
    verified, so it must not be trusted: fail, remove, recompute."""
    tier, tensor = fs_tier
    _store_blocks(tier, tensor, [(0, 0.25)])
    path = tier.file_mapper.get_file_name(key(0))
    assert _has_record(path), "store must write an integrity record xattr"
    _strip_record(path)
    ctx = ReqContext(req_id="nosidecar-req")
    assert lookup_and_wait(tier, [key(0)], ctx=ctx) == [LookupResult.HIT]
    tier.submit_load(make_job(2, [key(0)], [1], is_promotion=True))
    results = drain(tier)
    assert not results[0].success, "unverifiable payload must fail the load"
    assert not os.path.exists(path), "unverifiable payload must be removed"


def _has_record(path: str) -> bool:
    import contextlib

    with contextlib.suppress(OSError):
        os.getxattr(path, _XATTR)
        return True
    return False


def test_legacy_sidecar_files_are_inert(fs_tier):
    """Pre-carrier-migration .meta sidecars may sit beside payloads forever;
    they must never be consulted — a store under the xattr carrier with a
    poisoned legacy .meta next to it still loads clean."""
    tier, tensor = fs_tier
    _store_blocks(tier, tensor, [(0, 0.25)])
    path = tier.file_mapper.get_file_name(key(0))
    with open(path + _SIDECAR_SUFFIX, "wb") as f:
        f.write(b"garbage that would poison any sidecar-reading load")
    ctx = ReqContext(req_id="legacy-req")
    assert lookup_and_wait(tier, [key(0)], ctx=ctx) == [LookupResult.HIT]
    tier.submit_load(make_job(2, [key(0)], [1], is_promotion=True))
    results = drain(tier)
    assert results[0].success, "legacy .meta must not affect xattr-verified loads"
    assert torch.all(tensor[1] == 0.25)


def test_tier_construction_fails_loud_without_xattr_support(tmp_path, monkeypatch):
    """On a filesystem without user.* xattr support the tier must refuse to
    start, not silently run as a 100% cache miss."""

    def no_xattr(*args, **kwargs):
        raise OSError(95, "Operation not supported")

    monkeypatch.setattr(os, "setxattr", no_xattr)
    tensor = _page_aligned_zero_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)
    with pytest.raises((OSError, ValueError)):
        FileSystemTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=memoryview(tensor.numpy()),
            tier_type="fs",
            root_dir=str(tmp_path),
            n_read_threads=2,
            n_write_threads=2,
        )


def test_storage_replaced_under_live_tier_degrades_to_miss(fs_tier):
    """Wipe the tier contents under a live manager and drop a foreign file at
    a known path (a live-wipe remediation shape). The load must fail, and
    the tier must keep working: a fresh store/load roundtrip afterwards
    succeeds."""
    tier, tensor = fs_tier
    _store_blocks(tier, tensor, [(0, 0.25)])
    path = tier.file_mapper.get_file_name(key(0))
    import shutil

    for entry in os.listdir(os.path.dirname(path)):
        full = os.path.join(os.path.dirname(path), entry)
        if os.path.isfile(full):
            os.remove(full)
        else:
            shutil.rmtree(full)
    with open(path, "wb") as f:
        f.write(b"\xff" * (tier._block_size if hasattr(tier, "_block_size") else 4096))
    ctx = ReqContext(req_id="wipe-req")
    assert lookup_and_wait(tier, [key(0)], ctx=ctx) == [LookupResult.HIT]
    tier.submit_load(make_job(2, [key(0)], [1], is_promotion=True))
    results = drain(tier)
    assert not results[0].success, "replaced storage must not be trusted"
    # Production shape: the request that saw the rejection finishes (dropping
    # its cached miss verdict), a later request re-probes fresh.
    tier.on_request_finished(ctx)
    # The tier remains serviceable: re-store and load clean.
    tensor[0] = 0.5
    tier.submit_store(make_job(3, [key(0)], [0]))
    assert all(r.success for r in drain(tier))
    assert lookup_and_wait(tier, [key(0)], ctx=ReqContext(req_id="wipe-req2")) == [
        LookupResult.HIT
    ]
    tier.submit_load(make_job(4, [key(0)], [2], is_promotion=True))
    assert all(r.success for r in drain(tier))
    assert torch.all(tensor[2] == 0.5)


# ---------------------------------------------------------------------------
# Store-side record minting and removal signaling (2026-09-28 incident)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_store_of_existing_key_keeps_first_writers_record(
    fs_tier, monkeypatch, use_c_ext
):
    """Poison regression: the payload write skips files that already exist,
    so a later store of the same key must not re-mint the integrity record
    from its own view bytes either. Under FP8 KV a recompute quantizes
    differently, and the re-minted record described bytes that never reached
    the file: every later load then failed the checksum deterministically,
    was removed, re-stored, and re-poisoned."""
    import vllm.v1.kv_offload.tiering.fs.io as io_mod

    if use_c_ext and not io_mod._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", use_c_ext)

    tier, tensor = fs_tier
    _store_blocks(tier, tensor, [(0, 0.25)])
    path = tier.file_mapper.get_file_name(key(0))
    with open(path, "rb") as f:
        first = f.read()

    # Same key, drifted bytes in the primary view (recompute under FP8).
    tensor[0] = 0.75
    tier.submit_store(make_job(2, [key(0)], [0]))
    assert all(r.success for r in drain(tier))

    with open(path, "rb") as f:
        assert f.read() == first, "skip-existing payload write keeps first bytes"

    ctx = ReqContext(req_id="keep-record")
    assert lookup_and_wait(tier, [key(0)], ctx=ctx) == [LookupResult.HIT]
    tier.submit_load(make_job(3, [key(0)], [1], is_promotion=True))
    results = drain(tier)
    assert results[0].success, "kept payload+record pair must still verify"
    assert os.path.exists(path)
    assert torch.all(tensor[1] == 0.25), "restores the first writer's bytes"


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_store_does_not_mint_over_recordless_foreign_file(
    fs_tier, monkeypatch, use_c_ext
):
    """A file present before the store with no record (crash corner, planted
    foreign bytes) must not be minted over: the record would bind bytes that
    may not be the file's. The load path owns rejecting and removing it; the
    next store after that removal rewrites clean."""
    import vllm.v1.kv_offload.tiering.fs.io as io_mod

    if use_c_ext and not io_mod._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", use_c_ext)

    from vllm.v1.kv_offload.tiering.fs import integrity

    tier, tensor = fs_tier
    _store_blocks(tier, tensor, [(0, 0.25)])
    path = tier.file_mapper.get_file_name(key(0))

    # Replace the payload with foreign bytes and strip the record.
    with open(path, "wb") as f:
        f.write(b"\xff" * tier._block_size)
    os.removexattr(path, integrity.XATTR_NAME)

    tensor[0] = 0.75
    tier.submit_store(make_job(2, [key(0)], [0]))
    assert all(r.success for r in drain(tier))
    assert integrity.read_record(path) is None, "no mint over a foreign file"

    # The load rejects the record-less file and removes it.
    ctx = ReqContext(req_id="foreign-req")
    assert lookup_and_wait(tier, [key(0)], ctx=ctx) == [LookupResult.HIT]
    tier.submit_load(make_job(3, [key(0)], [1], is_promotion=True))
    assert not drain(tier)[0].success
    assert not os.path.exists(path)

    # Next store rewrites clean and the roundtrip works again.
    tier.submit_store(make_job(4, [key(0)], [0]))
    assert all(r.success for r in drain(tier))
    tier.submit_load(make_job(5, [key(0)], [2], is_promotion=True))
    assert all(r.success for r in drain(tier))
    assert torch.all(tensor[2] == 0.75)


def test_postcheck_failure_keeps_verified_prefix(fs_tier):
    """A post-check mismatch carries num_succeeded up to the failing block:
    blocks whose checksum verified keep their HIT verdict and show up as
    successful_keys, per the partial-keep contract."""
    tier, tensor = fs_tier
    _store_blocks(tier, tensor, [(0, 0.25), (1, 0.75)])
    path1 = tier.file_mapper.get_file_name(key(1))
    with open(path1, "r+b") as f:
        f.write(b"\x01" * os.path.getsize(path1))

    ctx = ReqContext(req_id="postcheck-partial")
    assert lookup_and_wait(tier, [key(0), key(1)], ctx=ctx) == [
        LookupResult.HIT,
        LookupResult.HIT,
    ]
    tier.submit_load(make_job(2, [key(0), key(1)], [2, 3], is_promotion=True))
    results = drain(tier)
    assert not results[0].success
    assert results[0].successful_keys == (key(0),)
    assert not os.path.exists(path1)
    assert os.path.exists(tier.file_mapper.get_file_name(key(0)))
    assert tier.lookup(key(0), ctx) == LookupResult.HIT
    assert tier.lookup(key(1), ctx) == LookupResult.MISS


def test_integrity_removal_emits_removed_event(fs_tier_with_events, caplog):
    """The pruner feed must learn that a mismatch-removed key died: one
    OffloadingEvent(removed=True) per failed load job carrying the removed
    keys, plus a distinct engine log line distinguishing 'removed once, gone'
    from 'removed and re-minted'."""
    import logging

    tier = fs_tier_with_events
    tier.submit_store(make_job(1, [key(0)], [0]))
    assert all(r.success for r in drain(tier))
    assert [ev.removed for ev in tier.take_events()] == [False]
    path = tier.file_mapper.get_file_name(key(0))
    with open(path, "r+b") as f:
        f.write(b"\x01" * os.path.getsize(path))

    ctx = ReqContext(req_id="removed-evt")
    assert lookup_and_wait(tier, [key(0)], ctx=ctx) == [LookupResult.HIT]
    with caplog.at_level(logging.WARNING):
        tier.submit_load(make_job(2, [key(0)], [0], is_promotion=True))
        assert not drain(tier)[0].success
    assert not os.path.exists(path)
    assert any(
        "integrity-mismatch removed" in r.message and path in r.message
        for r in caplog.records
    ), f"expected a distinct removal line in {caplog.records!r}"

    events = list(tier.take_events())
    assert len(events) == 1
    assert events[0].removed is True
    assert list(events[0].keys) == [key(0)]
    assert events[0].medium == Medium.STORAGE
    assert not events[0].removal_expected


def test_transient_record_read_failure_emits_no_removed_event(
    fs_tier_with_events,
):
    """Contract pin: a transient record-read errno must not tell the
    pruner feed a key died. See
    ``test_integrity_record_branch_emits_removed_event`` for the
    regression guard on the recorded-removal path."""
    tier = fs_tier_with_events
    tier.submit_store(make_job(1, [key(0)], [0]))
    assert all(r.success for r in drain(tier))
    store_events = list(tier.take_events())
    assert len(store_events) == 1 and not store_events[0].removed
    path = tier.file_mapper.get_file_name(key(0))

    # The cached HIT verdict is taken before the loop exists, so the failure
    # surfaces in the load task itself (read_record runs before any file
    # open). chmod-000 would not work: CI runs as root.
    ctx = ReqContext(req_id="xattr-err")
    assert lookup_and_wait(tier, [key(0)], ctx=ctx) == [LookupResult.HIT]

    saved = path + ".saved"
    loop = path + ".loop"
    os.rename(path, saved)
    os.symlink(loop, path)
    os.symlink(path, loop)

    tier.submit_load(make_job(2, [key(0)], [0], is_promotion=True))
    assert not drain(tier)[0].success

    assert os.path.lexists(path), "transient record-read errno keeps the file"
    assert list(tier.take_events()) == [], "no removed event for a transient"

    os.unlink(path)
    os.unlink(loop)
    os.rename(saved, path)


def test_integrity_record_branch_emits_removed_event(fs_tier_with_events, caplog):
    """Regression guard for the ``record`` reason branch: a payload
    whose integrity record is missing or foreign must be removed, the
    pruner feed must learn the key died
    (``OffloadingEvent(removed=True)`` on ``medium=STORAGE``,
    ``removal_expected=False``), and the engine log must carry
    ``integrity-record removed <path>``.

    Pre-fix the load removed the file silently, emitted no removed
    event, and logged nothing: this test fails on both counts.
    Pair: ``test_integrity_removal_emits_removed_event`` covers the
    ``mismatch`` reason branch.
    """
    import logging

    tier = fs_tier_with_events
    tier.submit_store(make_job(1, [key(0)], [0]))
    assert all(r.success for r in drain(tier))
    assert [ev.removed for ev in tier.take_events()] == [False]
    path = tier.file_mapper.get_file_name(key(0))
    _strip_record(path)

    ctx = ReqContext(req_id="record-branch")
    assert lookup_and_wait(tier, [key(0)], ctx=ctx) == [LookupResult.HIT]
    with caplog.at_level(logging.WARNING):
        tier.submit_load(make_job(2, [key(0)], [0], is_promotion=True))
        assert not drain(tier)[0].success
    assert not os.path.exists(path)
    assert any(
        "integrity-record removed" in r.message and path in r.message
        for r in caplog.records
    ), f"expected an integrity-record removal line in {caplog.records!r}"

    events = list(tier.take_events())
    assert len(events) == 1
    assert events[0].removed is True
    assert list(events[0].keys) == [key(0)]
    assert events[0].medium == Medium.STORAGE
    assert not events[0].removal_expected


def test_integrity_storage_id_branch_emits_removed_event(fs_tier_with_events, caplog):
    """Regression guard for the ``storage-id`` reason branch: a payload
    whose storage identity changed under a live tier must be removed,
    the pruner feed must learn the key died
    (``OffloadingEvent(removed=True)`` on ``medium=STORAGE``,
    ``removal_expected=False``), and the engine log must carry
    ``integrity-storage-id removed <path>``.

    Pre-fix the load removed the file silently, emitted no removed
    event, and logged nothing: this test fails on both counts.
    """
    import logging
    import shutil

    tier = fs_tier_with_events
    tier.submit_store(make_job(1, [key(0)], [0]))
    assert all(r.success for r in drain(tier))
    assert [ev.removed for ev in tier.take_events()] == [False]
    path = tier.file_mapper.get_file_name(key(0))

    # Replace the file in-place so the path stays valid but the inode
    # changes: the integrity record rides along with the rename so the
    # storage-id check fires before the record check rejects the payload.
    # ext4 may reuse an inode just freed in the same directory, so the
    # file is staged through a sibling path before being renamed back to
    # ``path``: the rename does not change the inode, but the round
    # trip through ``<path>.diff`` evicts the original from the FS's
    # free-inode cache.
    saved = path + ".saved"
    shutil.copy2(path, saved)
    os.unlink(path)
    tmp_diff = path + ".diff"
    os.rename(saved, tmp_diff)
    os.rename(tmp_diff, path)
    assert os.path.exists(path)

    ctx = ReqContext(req_id="storage-id-branch")
    assert lookup_and_wait(tier, [key(0)], ctx=ctx) == [LookupResult.HIT]
    with caplog.at_level(logging.WARNING):
        tier.submit_load(make_job(2, [key(0)], [0], is_promotion=True))
        assert not drain(tier)[0].success
    assert not os.path.exists(path)
    assert any(
        "integrity-storage-id removed" in r.message and path in r.message
        for r in caplog.records
    ), f"expected an integrity-storage-id removal line in {caplog.records!r}"

    events = list(tier.take_events())
    assert len(events) == 1
    assert events[0].removed is True
    assert list(events[0].keys) == [key(0)]
    assert events[0].medium == Medium.STORAGE
    assert not events[0].removal_expected


def test_mismatch_removal_spares_sibling_group_files(fs_tier):
    """Contract pin: removal is path-scoped. See
    ``test_integrity_record_branch_emits_removed_event`` and the
    storage-id sibling test for the regression guards on the actual
    removal paths."""
    import shutil

    from vllm.v1.kv_offload.tiering.fs import integrity

    tier, tensor = fs_tier
    _store_blocks(tier, tensor, [(0, 0.25)])
    path0 = tier.file_mapper.get_file_name(key(0))
    assert "_g0/" in path0

    sibling = path0.replace("_g0/", "_g1/")
    os.makedirs(os.path.dirname(sibling), exist_ok=True)
    shutil.copyfile(path0, sibling)
    with open(sibling, "rb") as f:
        integrity.write_record(sibling, bytes(key(0)), f.read())

    with open(path0, "r+b") as f:
        f.write(b"\x01" * os.path.getsize(path0))

    ctx = ReqContext(req_id="sibling-req")
    assert lookup_and_wait(tier, [key(0)], ctx=ctx) == [LookupResult.HIT]
    tier.submit_load(make_job(2, [key(0)], [1], is_promotion=True))
    assert not drain(tier)[0].success

    assert not os.path.exists(path0)
    assert os.path.exists(sibling), "sibling group files must be untouched"


def test_store_refuses_block_on_cross_check_mismatch(fs_tier, caplog):
    """The cascade cross-check: the fs tier must refuse to persist bytes
    that no longer match the CPU tier's recorded checksum for the key.
    The bytes drifted between the CPU store and the fs write — the
    store-side mislabel class that load-side integrity verifies as clean
    forever. A refusal costs a miss and recompute; a silent write costs
    permanent poison."""
    import logging

    tier, tensor = fs_tier
    tier._store_cross_check = lambda key, payload: False
    tier.submit_store(make_job(1, [key(0)], [0]))
    with caplog.at_level(logging.WARNING):
        results = drain(tier)
    assert all(r.success for r in results)
    path = tier.file_mapper.get_file_name(key(0))

    assert not os.path.exists(path), "mismatched bytes must not persist"
    assert any(
        "cascade cross-check rejected" in r.message and path in r.message
        for r in caplog.records
    ), f"expected a rejection line in {caplog.records!r}"
    del tensor


def test_store_proceeds_when_cross_check_matches(fs_tier):
    """Contract pin: a passing cross-check (True) does not impede the
    store."""
    tier, _ = fs_tier
    tier._store_cross_check = lambda key, payload: True
    tier.submit_store(make_job(1, [key(0)], [0]))
    assert all(r.success for r in drain(tier))
    assert os.path.exists(tier.file_mapper.get_file_name(key(0)))


def test_store_cross_check_none_is_inert(fs_tier):
    """No CPU record (None) means the check cannot judge: proceed, the
    load-side integrity still guards what lands."""
    tier, _ = fs_tier
    tier._store_cross_check = lambda key, payload: None
    tier.submit_store(make_job(1, [key(0)], [0]))
    assert all(r.success for r in drain(tier))
    assert os.path.exists(tier.file_mapper.get_file_name(key(0)))


def test_store_cross_check_partitions_mixed_batch(fs_tier, caplog):
    """One job, two keys, verdicts split True/False: the passing key
    persists, the failing key is refused, and the job still succeeds."""
    import logging

    tier, _ = fs_tier
    tier._store_cross_check = lambda k, payload: k != key(1)
    tier.submit_store(make_job(1, [key(0), key(1)], [0, 1]))
    with caplog.at_level(logging.WARNING):
        results = drain(tier)

    assert all(r.success for r in results)
    assert os.path.exists(tier.file_mapper.get_file_name(key(0)))
    assert not os.path.exists(tier.file_mapper.get_file_name(key(1)))
    assert any("cascade cross-check rejected" in r.message for r in caplog.records)
