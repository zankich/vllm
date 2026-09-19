# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from collections.abc import Collection, Iterable

import logging

from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
)
from vllm.v1.kv_offload.cpu.common import (
    CPULoadStoreSpec,
    CPUOffloadingMetrics,
)
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy
from vllm.v1.kv_offload.cpu.policies.factory import CachePolicyFactory

logger = logging.getLogger(__name__)


class CPUOffloadingManager(OffloadingManager):
    """
    An OffloadingManager with a pluggable CachePolicy, resolved by name via
    CachePolicyFactory (built in: "lru", "arc"; external policies can either
    register their own or be loaded out-of-tree via cache_policy_module_path).

    The manager owns all shared logic: ref-counting, event emission,
    block pool management, and the prepare_store/complete_store skeletons.
    Policy-specific block organization and eviction decisions are delegated
    to the CachePolicy implementation.
    """

    def __init__(
        self,
        num_blocks: int,
        cache_policy: str = "lru",
        cache_policy_module_path: str | None = None,
        enable_events: bool = False,
        store_threshold: int = 1,
        max_tracker_size: int = 64_000,
        kv_memoryview: memoryview | None = None,
    ):
        self.medium: Medium = Medium.CPU
        self._num_blocks: int = num_blocks
        self._num_allocated_blocks: int = 0
        self._free_list: list[int] = []
        self.events: list[OffloadingEvent] | None = [] if enable_events else None
        policy_cls = CachePolicyFactory.get_cache_policy_cls(
            cache_policy, cache_policy_module_path
        )
        self._policy: CachePolicy = policy_cls(cache_capacity=num_blocks)
        # Track the number of blocks in the cache that are evictable. i.e. ref_cnt 0.
        self._num_evictable_cache_blocks: int = 0
        # Track blocks with an in-flight store (ref_cnt -1, not yet completed).
        self._num_write_pending_blocks: int = 0
        # Keys pinned by a confirming lookup, awaiting their load's ref_cnt
        # handoff or the request-finish release (fork, 2026-09-18).
        self._lookup_pinned: set[OffloadKey] = set()

        # Fork-local integrity (2026-09-17): with a view of the shm tier's
        # bytes, every completed store records sha256(key, slot) and every
        # lookup re-verifies; post-recording corruption answers MISS and the
        # block is evicted instead of serving wrong bytes silently. None
        # disables checking entirely (unit construction, legacy behavior).
        # Corruption between the GPU->CPU copy and the recording lands
        # recorded-torn and is not detectable here — same store-time limit
        # as the fs tier's xattr carrier.
        self._kv_bytes: memoryview | None = (
            kv_memoryview.cast("B") if kv_memoryview is not None else None
        )
        self._integrity: dict[OffloadKey, bytes] | None = (
            {} if self._kv_bytes is not None else None
        )

        self.store_threshold: int = store_threshold
        self.max_tracker_size: int = max_tracker_size
        self.stores_skipped_in_current_batch: int = 0
        self.allocation_sizes_in_current_batch: list[int] = []

        # Number of block references. It is ordered so can evict the LRU entry in O(1).
        self.counts: OrderedDict[OffloadKey, int] | None = (
            OrderedDict() if store_threshold >= 2 else None
        )

    def _slot_checksum(self, key: OffloadKey, block_id: int) -> bytes:
        from vllm.v1.kv_offload.tiering.fs.integrity import block_checksum

        size = len(self._kv_bytes) // self._num_blocks
        slot = self._kv_bytes[block_id * size : (block_id + 1) * size]
        return block_checksum(key, slot)

    def _reject_corrupt_block(self, key: OffloadKey, block: BlockStatus) -> None:
        """Evict a block whose bytes no longer match its recorded checksum."""
        logger.warning(
            "CPU offload tier: slot %d for key %.16s failed its integrity "
            "check; evicting and answering MISS (corruption class: "
            "post-store clobber, aliasing, or torn writers)",
            block.block_id,
            key,
        )
        # The block was ready and unreferenced (callers gate on ref_cnt),
        # so it is counted as evictable — unless a lookup pin already
        # moved it out of the evictable count.
        if key in self._lookup_pinned:
            self._lookup_pinned.discard(key)
        else:
            self._num_evictable_cache_blocks -= 1
            assert self._num_evictable_cache_blocks >= 0
        self._policy.remove(key)
        self._free_block(block)
        self._integrity.pop(key, None)
        if self.events is not None:
            self.events.append(
                OffloadingEvent(keys=[key], medium=self.medium, removed=True)
            )

    # --- block pool ---

    def _get_num_free_blocks(self) -> int:
        return len(self._free_list) + self._num_blocks - self._num_allocated_blocks

    def _allocate_blocks(self, keys: list[OffloadKey]) -> list[BlockStatus]:
        num_fresh = min(len(keys), self._num_blocks - self._num_allocated_blocks)
        num_reused = len(keys) - num_fresh
        assert len(self._free_list) >= num_reused

        # allocate fresh blocks
        blocks: list[BlockStatus] = []
        for _ in range(num_fresh):
            blocks.append(BlockStatus(self._num_allocated_blocks))
            self._num_allocated_blocks += 1

        # allocate reused blocks
        for _ in range(num_reused):
            blocks.append(BlockStatus(self._free_list.pop()))
        return blocks

    def _free_block(self, block: BlockStatus) -> None:
        self._free_list.append(block.block_id)

    def _get_load_store_spec(
        self,
        keys: Iterable[OffloadKey],
        blocks: Iterable[BlockStatus],
    ) -> CPULoadStoreSpec:
        return CPULoadStoreSpec([block.block_id for block in blocks])

    def _record_access(self, key: OffloadKey) -> None:
        """Count one observation of ``key`` for store admission."""
        assert self.counts is not None
        if key in self.counts:
            self.counts.move_to_end(key)
            self.counts[key] += 1
        else:
            if len(self.counts) >= self.max_tracker_size:
                self.counts.popitem(last=False)
            self.counts[key] = 1

    # --- OffloadingManager interface ---

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def on_request_finished(self, req_context: ReqContext) -> None:
        # Release lookup pins for keys the request never loaded (scanned
        # hits beyond the converged boundary, eagle-popped chunks):
        # without this, every confirmed-but-unloaded hit would stay
        # non-evictable forever.
        pins: list[OffloadKey] | None = getattr(req_context, "_load_pins", None)
        if pins:
            for key in pins:
                if key not in self._lookup_pinned:
                    continue
                self._lookup_pinned.discard(key)
                block = self._policy.get(key)
                if block is not None and block.ref_cnt == 0:
                    self._policy.mark_evictable(key)
                    self._num_evictable_cache_blocks += 1
            pins.clear()
        super().on_request_finished(req_context)

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        block = self._policy.get(key)
        if block is None:
            return LookupResult.MISS
        if not block.is_ready:
            return LookupResult.HIT_PENDING
        if self._integrity is not None and block.ref_cnt == 0:
            # Verify each key once per request: the scheduler thread hashes
            # the slot on the request's first lookup of the key and trusts
            # the verdict for the request's lifetime. Unmitigated, a
            # restore-heavy request re-hashing 10k+ MB-scale slots per step
            # would dominate TTFT.
            verified: set[OffloadKey] | None = getattr(
                req_context, "_integrity_verified", None
            )
            if verified is None:
                verified = set()
                req_context._integrity_verified = verified  # type: ignore[attr-defined]
            if key not in verified:
                recorded = self._integrity.get(key)
                if (
                    recorded is None
                    or self._slot_checksum(key, block.block_id) != recorded
                ):
                    # Never serve silently-wrong bytes: the corrupt block
                    # leaves the cache and the caller recomputes. Blocks
                    # referenced by an in-flight load (ref_cnt > 0) are left
                    # alone here; the copy already in flight completes, and
                    # the next request's first lookup rejects.
                    self._reject_corrupt_block(key, block)
                    return LookupResult.MISS
                verified.add(key)
        if block.ref_cnt == 0 and key not in self._lookup_pinned:
            # Pin confirmed hits (fork, 2026-09-18): store completions —
            # and their LRU evictions — run on transfer threads
            # asynchronously from the scheduler thread, so an unpinned
            # confirmed hit could vanish between this lookup and
            # prepare_load (fatal `Block ... not found in cache` under restore-heavy load).
            # The pin hands off to the load's ref_cnt in prepare_load;
            # never-loaded pins release at request finish. The context's
            # pin list is insertion-ordered so a release re-marks keys at
            # MRU in lookup-scan order, deterministic for the LRU.
            self._lookup_pinned.add(key)
            pins: list[OffloadKey] | None = getattr(req_context, "_load_pins", None)
            if pins is None:
                pins = []
                req_context._load_pins = pins  # type: ignore[attr-defined]
            pins.append(key)
            self._policy.mark_non_evictable(key)
            self._num_evictable_cache_blocks -= 1
            assert self._num_evictable_cache_blocks >= 0
        return LookupResult.HIT

    @override
    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        blocks = []
        pins: list[OffloadKey] | None = getattr(req_context, "_load_pins", None)
        for key in keys:
            block = self._policy.get(key)
            assert block is not None, f"Block {key!r} not found in cache"
            assert block.is_ready, f"Block {key!r} is not ready for reading"
            if pins is not None and key in self._lookup_pinned:
                # Already pinned (and counted non-evictable) by the
                # confirming lookup; the load's ref_cnt takes over.
                if key in pins:
                    pins.remove(key)
                self._lookup_pinned.discard(key)
            elif block.ref_cnt == 0:
                self._policy.mark_non_evictable(key)
                self._num_evictable_cache_blocks -= 1  # ref_cnt 0 -> 1
                assert self._num_evictable_cache_blocks >= 0
            block.ref_cnt += 1
            blocks.append(block)
        return self._get_load_store_spec(keys, blocks)

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        self._policy.touch(keys, req_context)

    @override
    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        for key in keys:
            block = self._policy.get(key)
            assert block is not None, f"Block {key!r} not found"
            assert block.ref_cnt > 0, f"Block {key!r} ref_cnt is already 0"
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                self._num_evictable_cache_blocks += 1  # ref_cnt 1 -> 0
                self._policy.mark_evictable(key)

    @override
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        if self.counts is not None:
            num_keys = len(keys)
            for key in keys:
                self._record_access(key)
            keys = [k for k in keys if self.counts.get(k, 0) >= self.store_threshold]
            self.stores_skipped_in_current_batch += num_keys - len(keys)
        # filter out blocks that are already stored
        keys_to_store = [k for k in keys if self._policy.get(k) is None]

        if not keys_to_store:
            return PrepareStoreOutput(
                keys_to_store=[],
                store_spec=self._get_load_store_spec([], []),
                evicted_keys=[],
            )

        self.allocation_sizes_in_current_batch.append(len(keys_to_store))
        num_blocks_to_evict = len(keys_to_store) - self._get_num_free_blocks()

        to_evict: list[OffloadKey] = []
        if num_blocks_to_evict > 0:
            if num_blocks_to_evict > self._num_evictable_cache_blocks:
                # Eviction will fail.
                return None
            # There is a still a chance for eviction failure as some of the
            # idle blocks might be in the protected list.

            # Blocks from the original input are excluded from eviction candidates:
            # a block that was already stored must remain in the cache after this call.
            protected = set(keys)
            evicted = self._policy.evict(num_blocks_to_evict, protected)
            if evicted is None:
                return None

            # cache-policy removes only idle blocks.
            self._num_evictable_cache_blocks -= len(evicted)
            assert self._num_evictable_cache_blocks >= 0

            for key, block in evicted:
                self._free_block(block)
                to_evict.append(key)
                if self._integrity is not None:
                    self._integrity.pop(key, None)

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=to_evict,
                    medium=self.medium,
                    removed=True,
                )
            )

        blocks = self._allocate_blocks(keys_to_store)
        assert len(blocks) == len(keys_to_store), (
            "Block pool did not allocate the expected number of blocks"
        )

        for key, block in zip(keys_to_store, blocks):
            self._policy.insert(key, block)
        self._num_write_pending_blocks += len(keys_to_store)

        # build store specs for allocated blocks
        store_spec = self._get_load_store_spec(keys_to_store, blocks)

        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=store_spec,
            evicted_keys=to_evict,
        )

    @override
    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        stored_keys: list[OffloadKey] = []

        if success:
            for key in keys:
                block = self._policy.get(key)
                if block is not None and not block.is_ready:
                    block.ref_cnt = 0
                    self._num_write_pending_blocks -= 1
                    self._num_evictable_cache_blocks += 1
                    self._policy.mark_evictable(key)
                    stored_keys.append(key)
                    if self._integrity is not None:
                        self._integrity[key] = self._slot_checksum(key, block.block_id)
        else:
            for key in keys:
                block = self._policy.get(key)
                if block is not None and not block.is_ready:
                    self._num_write_pending_blocks -= 1
                    self._policy.remove(key)
                    self._free_block(block)

        if stored_keys and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=stored_keys,
                    medium=self.medium,
                    removed=False,
                )
            )

    @override
    def reset_cache(self) -> None:
        # Clear ALL blocks unconditionally. The scheduler's _stale_job_threshold
        # guarantees that complete_load / complete_store are never called for
        # pre-reset jobs, so no lazy cleanup is needed. The scheduler also
        # flushes in-flight load job IDs to the workers before any new stores
        # can begin, preventing a cross-direction data race on reused offload block IDs.
        self._policy.clear()
        self._num_evictable_cache_blocks = 0
        self._num_write_pending_blocks = 0
        self._lookup_pinned.clear()
        if self._integrity is not None:
            self._integrity.clear()

        self._free_list.clear()
        self._num_allocated_blocks = 0

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = OffloadingConnectorStats()

        # Compute cache usage.
        num_used = (
            self._num_allocated_blocks
            - len(self._free_list)
            - self._num_evictable_cache_blocks
        )
        usage = num_used / self._num_blocks if self._num_blocks > 0 else 0.0
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC, usage)

        for allocation_size in self.allocation_sizes_in_current_batch:
            stats.observe_histogram(
                CPUOffloadingMetrics.CPU_ALLOCATION_SIZE, allocation_size
            )
        self.allocation_sizes_in_current_batch.clear()

        write_usage = (
            self._num_write_pending_blocks / self._num_blocks
            if self._num_blocks > 0
            else 0.0
        )
        read_usage = max(usage - write_usage, 0.0)
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC, write_usage)
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC, read_usage)

        if self.store_threshold >= 2:
            stats.increase_counter(
                CPUOffloadingMetrics.STORES_SKIPPED,
                self.stores_skipped_in_current_batch,
            )
            self.stores_skipped_in_current_batch = 0

        return stats
