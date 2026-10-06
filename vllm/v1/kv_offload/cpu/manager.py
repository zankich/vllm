# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field

from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.logger import init_logger
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
    get_offload_group_idx,
)
from vllm.v1.kv_offload.cpu.common import (
    CPULoadStoreSpec,
    CPUOffloadingMetrics,
)
from vllm.v1.kv_offload.cpu.policies.base import CachePolicy, ChunkStatus
from vllm.v1.kv_offload.cpu.policies.factory import CachePolicyFactory

logger = init_logger(__name__)


class _IntegrityVerified:
    """ReqContext state: keys whose slot checksum was verified this request."""

    __slots__ = ("keys",)

    def __init__(self) -> None:
        self.keys: set[OffloadKey] = set()


class _LoadPinState:
    """ReqContext state: keys this request's confirming lookups pinned
    (and not yet consumed by prepare_load). Released on request finish,
    ahead of the cache-access state guards, so a stale or foreign
    _RequestCacheAccess does not strand a pin's refcount share
    (fork, 2026-10-06)."""

    __slots__ = ("pins",)

    def __init__(self) -> None:
        self.pins: list[OffloadKey] = []


@dataclass(slots=True)
class _RequestCacheAccess:
    """Cache keys observed by one request, grouped in prefix order."""

    owner: object
    cache_generation: int
    key_groups: dict[int, list[OffloadKey]] = field(default_factory=dict)
    seen_keys: set[OffloadKey] = field(default_factory=set)
    inserted_keys: set[OffloadKey] = field(default_factory=set)
    reused_keys: set[OffloadKey] = field(default_factory=set)
    store_miss_keys: set[OffloadKey] = field(default_factory=set)
    finished: bool = False


class CPUOffloadingManager(OffloadingManager):
    """An OffloadingManager with a pluggable CachePolicy, resolved by name via
    CachePolicyFactory (built in: "lru", "arc"; external policies can either
    register their own or be loaded out-of-tree via cache_policy_module_path).

    The manager owns all shared logic: ref-counting, event emission,
    chunk pool management, and the prepare_store/complete_store skeletons.
    Policy-specific chunk organization and eviction decisions are delegated
    to the CachePolicy implementation.
    """

    def __init__(
        self,
        num_chunks: int,
        cache_policy: str = "lru",
        cache_policy_module_path: str | None = None,
        enable_events: bool = False,
        store_threshold: int = 1,
        max_tracker_size: int = 64_000,
    ):
        self.medium: Medium = Medium.CPU
        self._num_chunks: int = num_chunks
        self._num_allocated_chunks: int = 0
        self._free_list: list[int] = []
        self.events: list[OffloadingEvent] | None = [] if enable_events else None
        policy_cls = CachePolicyFactory.get_cache_policy_cls(
            cache_policy, cache_policy_module_path
        )
        self._policy: CachePolicy = policy_cls(cache_capacity=num_chunks)
        # Track the number of chunks in the cache that are evictable. i.e. ref_cnt 0.
        self._num_evictable_cache_chunks: int = 0
        # Track chunks with an in-flight store (ref_cnt -1, not yet completed).
        self._num_write_pending_chunks: int = 0
        # Keys pinned by a confirming lookup, awaiting their load's ref_cnt
        # handoff or the request-finish release (fork, 2026-09-18).
        # Refcounted because a second request's confirming lookup also holds
        # a share of the pin; the first finisher's release must not pull
        # the key out from under a still-pending loader
        # (fork, 2026-10-05).
        self._lookup_pinned: set[OffloadKey] = set()
        self._lookup_pin_refs: dict[OffloadKey, int] = {}

        # Fork-local integrity (2026-09-17): when armed with a view of the
        # shm tier's bytes (CPUPrimaryTierOffloadingManager), every completed
        # store records sha256(key, slot) and every lookup re-verifies;
        # post-recording corruption answers MISS and the chunk is evicted
        # instead of serving wrong bytes silently. None disables checking
        # entirely (plain-spec construction has no buffer to checksum).
        # Corruption between the GPU->CPU copy and the recording lands
        # recorded-torn and is not detectable here — same store-time limit
        # as the fs tier's xattr carrier.
        self._kv_bytes: memoryview | None = None
        self._integrity: dict[OffloadKey, bytes] | None = None

        self.store_threshold: int = store_threshold
        self.max_tracker_size: int = max_tracker_size
        self.stores_skipped_in_current_batch: int = 0
        self.allocation_sizes_in_current_batch: list[int] = []
        self._cache_generation = 0

        # Number of chunk references. It is ordered so can evict the LRU entry in O(1).
        self.counts: OrderedDict[OffloadKey, int] | None = (
            OrderedDict() if store_threshold >= 2 else None
        )

    def _slot_checksum(self, key: OffloadKey, chunk_id: int) -> bytes:
        # Imported lazily: the integrity helpers live under the tiering
        # package, which imports this module.
        from vllm.v1.kv_offload.tiering.fs.integrity import block_checksum

        assert self._kv_bytes is not None, "checksums require armed kv bytes"
        size = len(self._kv_bytes) // self._num_chunks
        slot = self._kv_bytes[chunk_id * size : (chunk_id + 1) * size]
        return block_checksum(key, slot)

    def verify_stored_slot(
        self, key: OffloadKey, payload: bytes | memoryview
    ) -> bool | None:
        """Whether *payload* matches the checksum recorded at this key's
        CPU-store completion. The cascade cross-check source for
        secondary tiers: None when slot checksums are off or the key has
        no record, so callers treat it as "cannot judge"."""
        if self._integrity is None:
            return None
        from vllm.v1.kv_offload.tiering.fs.integrity import block_checksum

        recorded = self._integrity.get(key)
        if recorded is None:
            return None
        return block_checksum(key, payload) == recorded

    def _reject_corrupt_chunk(self, key: OffloadKey, chunk: ChunkStatus) -> None:
        """Evict a chunk whose bytes no longer match its recorded checksum."""
        assert self._integrity is not None, "rejection requires armed integrity"
        logger.warning(
            "CPU offload tier: slot %d for key %.16s failed its integrity "
            "check; evicting and answering MISS (corruption class: "
            "post-store clobber, aliasing, or torn writers)",
            chunk.chunk_id,
            key,
        )
        # The chunk was ready and unreferenced (callers gate on ref_cnt),
        # so it is counted as evictable — unless a lookup pin already
        # moved it out of the evictable count.
        if key in self._lookup_pinned:
            # Another request's share may survive; their prepare_load
            # will see the chunk is gone and degrade to a miss.
            # Decrement this rejector's implicit share (it never pinned
            # because integrity-check failures short-circuit before the
            # pin branch). Refcounted: only discard from the set when
            # the last share goes.
            self._lookup_pin_refs[key] -= 1
            assert self._lookup_pin_refs[key] >= 0
            if self._lookup_pin_refs[key] == 0:
                del self._lookup_pin_refs[key]
                self._lookup_pinned.discard(key)
        else:
            self._num_evictable_cache_chunks -= 1
            assert self._num_evictable_cache_chunks >= 0
        self._policy.remove(key)
        self._free_chunk(chunk)
        self._integrity.pop(key, None)
        if self.events is not None:
            self.events.append(
                OffloadingEvent(keys=[key], medium=self.medium, removed=True)
            )

    # --- chunk pool ---

    def _get_num_free_chunks(self) -> int:
        return len(self._free_list) + self._num_chunks - self._num_allocated_chunks

    def _allocate_chunks(self, keys: list[OffloadKey]) -> list[ChunkStatus]:
        num_fresh = min(len(keys), self._num_chunks - self._num_allocated_chunks)
        num_reused = len(keys) - num_fresh
        assert len(self._free_list) >= num_reused

        # allocate fresh chunks
        chunks: list[ChunkStatus] = []
        for _ in range(num_fresh):
            chunks.append(ChunkStatus(self._num_allocated_chunks))
            self._num_allocated_chunks += 1

        # allocate reused chunks
        for _ in range(num_reused):
            chunks.append(ChunkStatus(self._free_list.pop()))
        return chunks

    def _free_chunk(self, chunk: ChunkStatus) -> None:
        self._free_list.append(chunk.chunk_id)

    def _get_load_store_spec(
        self,
        keys: Iterable[OffloadKey],
        chunks: Iterable[ChunkStatus],
    ) -> CPULoadStoreSpec:
        return CPULoadStoreSpec([chunk.chunk_id for chunk in chunks])

    def _record_accesses(self, keys: Collection[OffloadKey]) -> None:
        """Record an offer without evicting its tracked candidates."""
        assert self.counts is not None
        protected: set[OffloadKey] = set()
        for key in keys:
            if key in self.counts:
                self.counts.move_to_end(key)
                self.counts[key] += 1
                protected.add(key)

        num_unprotected = len(self.counts) - len(protected)
        for key in keys:
            if key in self.counts:
                continue
            if len(self.counts) >= self.max_tracker_size:
                if num_unprotected == 0:
                    continue
                self.counts.popitem(last=False)
                num_unprotected -= 1
            self.counts[key] = 1

    def _get_request_cache_access(self, req_context: ReqContext) -> _RequestCacheAccess:
        state = req_context.get_state(_RequestCacheAccess)
        if (
            state is None
            or state.owner is not self
            or state.cache_generation != self._cache_generation
        ):
            state = _RequestCacheAccess(
                owner=self, cache_generation=self._cache_generation
            )
            req_context.set_state(state)
        return state

    def _get_load_pin_state(self, req_context: ReqContext) -> _LoadPinState:
        state = req_context.get_state(_LoadPinState)
        if state is None:
            state = _LoadPinState()
            req_context.set_state(state)
        return state

    def _record_request_cache_access(
        self,
        keys: Iterable[OffloadKey],
        req_context: ReqContext,
        inserted_keys: Iterable[OffloadKey] = (),
        reused_keys: Iterable[OffloadKey] = (),
    ) -> None:
        state = self._get_request_cache_access(req_context)
        for key in keys:
            if key in state.seen_keys:
                continue
            assert not state.finished, (
                "New cache keys observed after request finalization"
            )
            group_idx = get_offload_group_idx(key)
            state.key_groups.setdefault(group_idx, []).append(key)
            state.seen_keys.add(key)
        state.inserted_keys.update(inserted_keys)
        # Re-reading a chunk inserted by this request is an internal transfer
        # (for example, a tiering cascade), not a second cache access.
        state.reused_keys.update(
            key for key in reused_keys if key not in state.inserted_keys
        )

    # --- OffloadingManager interface ---

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        self._get_request_cache_access(req_context)
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        chunk = self._policy.get(key)
        if chunk is None:
            return LookupResult.MISS
        if not chunk.is_ready:
            return LookupResult.HIT_PENDING
        if self._integrity is not None and chunk.ref_cnt == 0:
            # Verify each key once per request: the scheduler thread hashes
            # the slot on the request's first lookup of the key and trusts
            # the verdict for the request's lifetime. Unmitigated, a
            # restore-heavy request re-hashing 10k+ MB-scale slots per step
            # would dominate TTFT.
            verified = req_context.get_state(_IntegrityVerified)
            if verified is None:
                verified = _IntegrityVerified()
                req_context.set_state(verified)
            if key not in verified.keys:
                recorded = self._integrity.get(key)
                if (
                    recorded is None
                    or self._slot_checksum(key, chunk.chunk_id) != recorded
                ):
                    # Never serve silently-wrong bytes: the corrupt chunk
                    # leaves the cache and the caller recomputes. Chunks
                    # referenced by an in-flight load (ref_cnt > 0) are left
                    # alone here; the copy already in flight completes, and
                    # the next request's first lookup rejects.
                    self._reject_corrupt_chunk(key, chunk)
                    return LookupResult.MISS
                verified.keys.add(key)
        if chunk.ref_cnt == 0 and key not in self._lookup_pinned:
            # Pin confirmed hits (fork, 2026-09-18): store completions —
            # and their LRU evictions — run on transfer threads
            # asynchronously from the scheduler thread, so an unpinned
            # confirmed hit could vanish between this lookup and
            # prepare_load (fatal `Block ... not found in cache` under
            # restore-heavy load). The pin hands off to the load's ref_cnt
            # in prepare_load; never-loaded pins release at request finish.
            # The context's pin list is insertion-ordered, but the LRU
            # policy applies recency at request finish rather than on
            # the per-key release path: mark_evictable re-pushes the
            # key's existing rank and the policy then ranks the request's
            # loaded keys on finish.
            self._lookup_pinned.add(key)
            self._lookup_pin_refs[key] = 1
            self._get_load_pin_state(req_context).pins.append(key)
            self._policy.mark_non_evictable(key)
            self._num_evictable_cache_chunks -= 1
            assert self._num_evictable_cache_chunks >= 0
        elif chunk.ref_cnt == 0 and self._lookup_pin_refs.get(key, 0) > 0:
            # A previous confirmer already pinned this key. Hold a share:
            # the first finisher's release must not pull the key out
            # from under our pending load (fork, 2026-10-05). The
            # non-evictable count is unchanged because the previous
            # pinner already moved it out of the evictable pool.
            self._lookup_pin_refs[key] += 1
            self._get_load_pin_state(req_context).pins.append(key)
        return LookupResult.HIT

    @override
    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        return self._prepare_load(keys, req_context, record_access=True)

    def _prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        *,
        record_access: bool,
    ) -> LoadStoreSpec:
        chunks = []
        pin_state = req_context.get_state(_LoadPinState)
        pins = pin_state.pins if pin_state is not None else None
        for key in keys:
            chunk = self._policy.get(key)
            if chunk is None and pins is not None and key in pins:
                # This request's confirming lookup pinned the key, but a
                # later integrity rejection evicted it in between (pins do
                # not refcount, so the rejector's ref_cnt==0 gate passes).
                # Degrade to a miss for this key: drop the stale pin (the
                # rejection already settled its non-evictable count) and
                # skip, instead of asserting mid-batch. The corrupt bytes
                # were evicted by the reject path and stay evicted.
                logger.warning(
                    "CPU offload tier: pinned key %.16s vanished before its "
                    "load (rejected by another request's integrity check); "
                    "degrading to a miss for this key",
                    key,
                )
                pins.remove(key)
                continue
            assert chunk is not None, f"Chunk {key!r} not found in cache"
            assert chunk.is_ready, f"Chunk {key!r} is not ready for reading"
            if key in self._lookup_pinned:
                # Already pinned (and counted non-evictable) by a
                # confirming lookup — this request's or another's; whoever
                # loads takes its share of the pin over. The set entry
                # survives if other confirmers still hold a share; the
                # chunk's ref_cnt protection takes over for this load.
                if pins is not None and key in pins:
                    pins.remove(key)
                self._lookup_pin_refs[key] -= 1
                assert self._lookup_pin_refs[key] >= 0
                if self._lookup_pin_refs[key] == 0:
                    del self._lookup_pin_refs[key]
                    self._lookup_pinned.discard(key)
            elif chunk.ref_cnt == 0:
                self._policy.mark_non_evictable(key)
                self._num_evictable_cache_chunks -= 1  # ref_cnt 0 -> 1
                assert self._num_evictable_cache_chunks >= 0
            chunk.ref_cnt += 1
            chunks.append(chunk)
        if record_access:
            self._record_request_cache_access(keys, req_context, reused_keys=keys)
        return self._get_load_store_spec(keys, chunks)

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        self._policy.touch(keys, req_context)

    @override
    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        for key in keys:
            chunk = self._policy.get(key)
            assert chunk is not None, f"Chunk {key!r} not found"
            assert chunk.ref_cnt > 0, f"Chunk {key!r} ref_cnt is already 0"
            chunk.ref_cnt -= 1
            if chunk.ref_cnt == 0:
                self._num_evictable_cache_chunks += 1  # ref_cnt 1 -> 0
                self._policy.mark_evictable(key)

    @override
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        keys = list(keys)
        if self.counts is not None:
            self._record_accesses(keys)
        # Partition the original offer before admission filtering. Ready
        # resident chunks are request reuses even if their tracker entry was
        # aged out; the threshold applies only to new store candidates.
        keys_to_store: list[OffloadKey] = []
        ready_existing_keys: list[OffloadKey] = []
        for key in keys:
            chunk = self._policy.get(key)
            if chunk is None:
                keys_to_store.append(key)
            else:
                if chunk.is_ready:
                    ready_existing_keys.append(key)

        if self.counts is not None:
            num_store_candidates = len(keys_to_store)
            keys_to_store = [
                key
                for key in keys_to_store
                if self.counts.get(key, 0) >= self.store_threshold
            ]
            self.stores_skipped_in_current_batch += num_store_candidates - len(
                keys_to_store
            )

        state = self._get_request_cache_access(req_context)
        new_store_misses = [
            key for key in keys_to_store if key not in state.store_miss_keys
        ]
        if new_store_misses:
            # ARC learns from B1/B2 before insert() removes the ghost entry.
            # Deduplication makes this one policy observation per request.
            self._policy.on_store_miss(new_store_misses, req_context)
            state.store_miss_keys.update(new_store_misses)

        recorded_keys = set(keys_to_store)
        recorded_keys.update(ready_existing_keys)
        # Record once before any allocation-related early return. Store
        # candidates are classified as insertions only after allocation succeeds.
        self._record_request_cache_access(
            (key for key in keys if key in recorded_keys),
            req_context,
            reused_keys=ready_existing_keys,
        )

        if not keys_to_store:
            return PrepareStoreOutput(
                keys_to_store=[],
                store_spec=self._get_load_store_spec([], []),
                evicted_keys=[],
            )

        self.allocation_sizes_in_current_batch.append(len(keys_to_store))
        num_chunks_to_evict = len(keys_to_store) - self._get_num_free_chunks()

        to_evict: list[OffloadKey] = []
        if num_chunks_to_evict > 0:
            if num_chunks_to_evict > self._num_evictable_cache_chunks:
                # Eviction will fail.
                return None
            # There is a still a chance for eviction failure as some of the
            # idle chunks might be in the protected list.

            # Chunks from the original input are excluded from eviction candidates:
            # a chunk that was already stored must remain in the cache after this call.
            protected = set(keys)
            evicted = self._policy.evict(num_chunks_to_evict, protected)
            if evicted is None:
                return None

            # cache-policy removes only idle chunks.
            self._num_evictable_cache_chunks -= len(evicted)
            assert self._num_evictable_cache_chunks >= 0

            for key, chunk in evicted:
                self._free_chunk(chunk)
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

        chunks = self._allocate_chunks(keys_to_store)
        assert len(chunks) == len(keys_to_store), (
            "Chunk pool did not allocate the expected number of chunks"
        )

        for key, chunk in zip(keys_to_store, chunks):
            self._policy.insert(key, chunk)
        self._num_write_pending_chunks += len(keys_to_store)
        state.inserted_keys.update(keys_to_store)

        # build store specs for allocated chunks
        store_spec = self._get_load_store_spec(keys_to_store, chunks)

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
                chunk = self._policy.get(key)
                if chunk is not None and not chunk.is_ready:
                    # Record before flipping readiness: the bytes are final
                    # once the transfer completes, and a ready chunk with no
                    # recorded checksum reads as corruption in lookup.
                    if self._integrity is not None:
                        self._integrity[key] = self._slot_checksum(key, chunk.chunk_id)
                    chunk.ref_cnt = 0
                    self._num_write_pending_chunks -= 1
                    self._num_evictable_cache_chunks += 1
                    self._policy.mark_evictable(key)
                    stored_keys.append(key)
        else:
            for key in keys:
                chunk = self._policy.get(key)
                if chunk is not None and not chunk.is_ready:
                    self._num_write_pending_chunks -= 1
                    self._policy.remove(key)
                    self._free_chunk(chunk)

        if stored_keys and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=stored_keys,
                    medium=self.medium,
                    removed=False,
                )
            )

    @override
    def on_request_finished(self, req_context: ReqContext) -> None:
        # Release lookup pins for keys the request never loaded (scanned
        # hits beyond the converged boundary, eagle-popped chunks):
        # without this, every confirmed-but-unloaded hit would stay
        # non-evictable forever. Refcounted release: each request's
        # share decrements, and the key only leaves the set when the
        # last share goes (fork, 2026-10-05). The release path runs
        # ahead of the cache-access state guards below, so a stale or
        # foreign _RequestCacheAccess does not strand a share
        # (fork, 2026-10-06).
        pin_state = req_context.get_state(_LoadPinState)
        pins = pin_state.pins if pin_state is not None else None
        if pins:
            released: set[OffloadKey] = set()
            for key in pins:
                if key in released:
                    continue
                released.add(key)
                if key not in self._lookup_pinned:
                    # Stale pin: the chunk was already removed (e.g. by a
                    # corruption rejector that decremented the refcount).
                    continue
                self._lookup_pin_refs[key] -= 1
                assert self._lookup_pin_refs[key] >= 0
                if self._lookup_pin_refs[key] == 0:
                    del self._lookup_pin_refs[key]
                    self._lookup_pinned.discard(key)
                    chunk = self._policy.get(key)
                    if chunk is not None and chunk.ref_cnt == 0:
                        self._policy.mark_evictable(key)
                        self._num_evictable_cache_chunks += 1
            pins.clear()

        state = req_context.get_state(_RequestCacheAccess)
        if (
            state is None
            or state.owner is not self
            or state.cache_generation != self._cache_generation
            or state.finished
        ):
            return
        state.finished = True
        key_groups = []
        for group_idx in sorted(state.key_groups):
            keys = state.key_groups[group_idx]
            positions = {
                key: position
                for key in keys
                if (position := req_context.get_offload_key_position(key)) is not None
            }
            if len(positions) == len(keys):
                keys = sorted(keys, key=positions.__getitem__)
            key_groups.append(tuple(keys))
        self._policy.on_request_finished(
            tuple(key_groups),
            state.inserted_keys - state.reused_keys,
            state.reused_keys,
            req_context,
        )

    @override
    def reset_cache(self) -> None:
        # Clear ALL chunks unconditionally. The scheduler's _stale_job_threshold
        # guarantees that complete_load / complete_store are never called for
        # pre-reset jobs, so no lazy cleanup is needed. The scheduler also
        # flushes in-flight load job IDs to the workers before any new stores
        # can begin, preventing a cross-direction data race on reused offload chunk IDs.
        self._policy.clear()
        self._cache_generation += 1
        self._num_evictable_cache_chunks = 0
        self._num_write_pending_chunks = 0
        self._lookup_pinned.clear()
        self._lookup_pin_refs.clear()
        if self._integrity is not None:
            self._integrity.clear()

        self._free_list.clear()
        self._num_allocated_chunks = 0

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = OffloadingConnectorStats()

        # Compute cache usage.
        num_used = (
            self._num_allocated_chunks
            - len(self._free_list)
            - self._num_evictable_cache_chunks
        )
        usage = num_used / self._num_chunks if self._num_chunks > 0 else 0.0
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC, usage)

        for allocation_size in self.allocation_sizes_in_current_batch:
            stats.observe_histogram(
                CPUOffloadingMetrics.CPU_ALLOCATION_SIZE, allocation_size
            )
        self.allocation_sizes_in_current_batch.clear()

        write_usage = (
            self._num_write_pending_chunks / self._num_chunks
            if self._num_chunks > 0
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
