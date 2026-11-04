# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import ctypes
import errno
import fcntl
import functools
import glob
import mmap
import os
import time
from collections.abc import Callable, Iterator
from ctypes import c_int, c_long, c_size_t, c_void_p

import numpy as np
import torch

from vllm.distributed.device_communicators.shm_broadcast import (
    check_shm_free_space,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.kv_offload.cpu.host_register import host_unregister

logger = init_logger(__name__)

# MADV_POPULATE_WRITE was added in Linux 5.14 (value 23).
_MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)
_MAP_FIXED = 0x10
# Page-cache inserts are serialized per shm file, capping pre-faulting of one
# file near 10 GiB/s; smaller segments are not worth a file.
_MIN_SEGMENT_BYTES = 1 << 30

_REGLOCK_SUFFIX = ".reglock"


def _hold_shared_lock(fd: int, path: str) -> None:
    """Mark this region as in use for as long as the process lives.

    The lock is advisory and shared, so every worker of the same engine can
    hold it at once; the kernel releases it when the process exits, however
    it exits. ``_reclaim_orphaned_regions`` reads it to tell a live region
    from an orphan. A filesystem that does not support ``flock`` only costs
    us that signal, so failure here is logged and ignored.
    """
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except OSError:
        logger.debug("Could not flock %s; orphan reclaim disabled for it", path)


def _reglock_matches_path(fd: int, path: str) -> bool:
    """Whether the locked fd is still the file the path names.

    A remover that unlinks the reglock under the lock leaves a rank that
    opened the path earlier holding a lock on an inode the path no longer
    names, and the next rank to open the path creates a second inode. The
    two ranks then hold different inodes and their critical sections
    overlap. A rank whose fd names a stale inode retries on the file the
    path names now, and a remover whose fd names a stale inode must not
    unlink the path: the file the path names now belongs to someone else.
    """
    st = os.fstat(fd)
    if st.st_nlink == 0:
        return False
    try:
        path_st = os.stat(path)
    except FileNotFoundError:
        return False
    return (st.st_dev, st.st_ino) == (path_st.st_dev, path_st.st_ino)


def _remove_reglock(path: str) -> bool:
    """Unlink a registration lock file unless a rank holds its lock.

    The lock is the only thing that separates a live registration from a
    leftover file, and it is held for the whole registration critical
    section, so taking it before the unlink is what keeps the removal
    from splitting that section: a rank that opens the path after the
    unlink gets a new inode and would no longer exclude a rank that
    already holds the old one.

    The lock is not enough on its own: two removers can open the same
    file, and the second one's lock is on an inode the first has already
    unlinked, while a rank may have created a new file under the same
    name. The file is unlinked only when it is still the file the path
    names, so no remover takes away a file it does not own.

    Returns whether the file was removed.
    """
    try:
        # /dev/shm is world-writable, so refuse to follow a symlink a local
        # user may have planted under this name; O_NOFOLLOW makes os.open
        # raise ELOOP, which the handler below catches.
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    except FileNotFoundError:
        return False
    except OSError:
        logger.debug("Could not open reglock %s", path, exc_info=True)
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # A rank is inside the registration critical section.
            logger.debug("Reglock %s is held; leaving it in place", path)
            return False
        if not _reglock_matches_path(fd, path):
            # The path names a different file now: another remover took
            # this one, and a rank registered on a new file under the
            # same name. Unlinking would take that rank's file away.
            logger.debug("Reglock %s was replaced; leaving it in place", path)
            return False
        try:
            os.unlink(path)
        except FileNotFoundError:
            return False
        except OSError:
            logger.debug("Failed to unlink reglock %s", path, exc_info=True)
            return False
        return True
    finally:
        # Closing drops the lock, on the unlinked inode if we got that far.
        os.close(fd)


_MMAP_GLOB = "/dev/shm/vllm_offload_*.mmap"
_REGLOCK_GLOB = "/dev/shm/vllm_offload_*.reglock"

# A registration lock file younger than this may still belong to an engine
# that is registering right now. The file is created moments before the
# registration that uses it and is never written to again, so its mtime is
# its creation, and the region name it belongs to is already gone by then:
# nothing else in the sweep tells a booting engine's lock from residue.
# The window is comfortably above the gap between a rank finishing its own
# page populate after the barrier and opening the reglock, so the ranks of
# an engine that is still coming up all reach the lock inside the window.
_MIN_ORPHAN_AGE_S = 90.0


def _reclaim_orphaned_regions() -> None:
    """Remove offload regions left behind by an engine that died hard.

    ``cleanup()`` unlinks the region on a graceful shutdown, but a SIGKILL,
    an OOM kill or a segfault skips it. Where ``/dev/shm`` outlives the
    process -- a container restart with a tmpfs-backed ``/dev/shm``, for
    instance -- the orphan keeps consuming the tmpfs budget, and the next
    start fails to allocate its own region.

    Every participant holds a shared ``flock`` on its region for as long
    as it lives, and the kernel drops that lock when the process dies. So
    an orphan is exactly a region no one can be holding: if this process
    can take the exclusive lock, nothing else is using the file and it is
    safe to unlink. That keeps a second engine sharing this host's
    /dev/shm untouched, and covers this engine's own name too: a prior
    incarnation that died mid-creation leaves its files under the same
    engine id, and only this sweep could reclaim them, so skipping the
    own name would strand a stalled creation and wedge the reboot on the
    joiner's size wait. A live participant of the same engine blocks the
    sweep's exclusive probe with its shared lock.

    Only the first segment file carries the engine's liveness mark, so it
    is the probe for the whole region: once it proves orphaned, the
    numbered segment files mapped back to back beneath the region are
    unlinked with it, and the reglock sibling is reclaimed through the
    same locked removal, so a rank holding the lock (a reused engine id)
    keeps its file for the orphan sweep to take later.

    Zero-length files are a region mid-creation, not orphans: a base is
    empty only between its creator's ``O_EXCL`` create and the ``ftruncate``
    that sizes it last, after the numbered segments are already at full
    size. Inside the orphan window they are skipped, since the creator may
    still be running; past it, nothing live can own the files, and the
    uncontended probe below reclaims the stalled creation together with
    its full-size segments, which no glob reaches on their own.

    The unlink re-verifies that the locked fd still names the path: a
    remover can unlink an orphan between this sweep's open and its lock
    while a new engine creates a fresh region under the same name, and
    unlinking the path then would take the new region's file, whose lock
    this fd does not hold.
    """
    now = time.time()
    for path in glob.glob(_MMAP_GLOB):
        try:
            if os.path.getsize(path) == 0:
                try:
                    age = now - os.path.getmtime(path)
                except OSError:
                    continue
                if age < _MIN_ORPHAN_AGE_S:
                    continue
            # /dev/shm is world-writable, so refuse to follow a symlink a
            # local user may have planted under our glob; O_NOFOLLOW makes
            # os.open raise ELOOP, which the enclosing handler catches.
            fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
            with os.fdopen(fd, "r+b") as f:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    continue  # still held: a live engine owns this region
                if not _reglock_matches_path(f.fileno(), path):
                    # The path names a different file now: another remover
                    # took this orphan and a new region lives at the name.
                    # Unlinking would delete that region's file.
                    continue
                os.unlink(path)
                segment = 1
                while True:
                    if os.path.exists(path):
                        # A creator won the base name back between the
                        # base unlink and this segment unlink; the
                        # remaining segments belong to its fresh region.
                        break
                    try:
                        os.unlink(f"{path}.{segment}")
                    except FileNotFoundError:
                        break
                    segment += 1
            _remove_reglock(path + _REGLOCK_SUFFIX)
        except OSError:
            logger.debug("Skipped reclaiming %s", path, exc_info=True)
        else:
            logger.info("Reclaimed orphaned mmap file %s", path)

    _reclaim_orphaned_reglocks()


def _reclaim_orphaned_reglocks() -> None:
    """Remove registration lock files whose region name is gone.

    The barrier flow unlinks the region file at construction, so the
    reglock is the only name a booting engine leaves in /dev/shm and
    nothing else matches it: one zero-byte inode per engine boot, kept
    until the process that created it exits.

    A reglock is only taken when the region it belongs to is gone -- a
    region file that is still there is reclaimed together with its lock
    by the region sweep above -- and when nothing holds its lock, which
    is the case only when no rank is inside the registration critical
    section. Its age says the rest: the lock file is created moments
    before the registration that uses it and is never written to again,
    so a young one belongs to an engine that is still starting up.
    """
    now = time.time()
    for path in glob.glob(_REGLOCK_GLOB):
        if os.path.exists(path[: -len(_REGLOCK_SUFFIX)]):
            # The region file is still there: the region sweep above
            # reclaims the pair together.
            continue
        try:
            age = now - os.path.getmtime(path)
        except OSError:
            continue
        if age < _MIN_ORPHAN_AGE_S:
            continue
        if _remove_reglock(path):
            logger.info("Reclaimed orphaned reglock file %s", path)


def _wait_for_file_size(fd: int, expected_size: int, timeout: float = 30.0) -> None:
    """Spin-wait until the file reaches expected_size (creator truncated it)."""
    deadline = time.monotonic() + timeout
    while True:
        if os.fstat(fd).st_size >= expected_size:
            return
        if os.fstat(fd).st_nlink == 0:
            raise RuntimeError(
                "Shared offload region creator failed during initialization."
            )
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for mmap file to reach {expected_size} bytes"
            )
        time.sleep(0.005)


@functools.cache
def _libc_mmap() -> Callable[..., int | None]:
    mmap_fn = ctypes.CDLL(None, use_errno=True).mmap
    mmap_fn.argtypes = [c_void_p, c_size_t, c_int, c_int, c_int, c_long]
    mmap_fn.restype = c_void_p
    return mmap_fn


def _madvise_populate_write(mmap_obj: mmap.mmap, offset: int, length: int) -> None:
    mmap_obj.madvise(_MADV_POPULATE_WRITE, offset, length)


def _fallback_populate_write(mmap_obj: mmap.mmap, offset: int, length: int) -> None:
    # Touch one byte per page via a read-modify-write so existing bytes are
    # preserved — a peer worker may have already written KV data into this
    # shared mmap by the time we run on a kernel without MADV_POPULATE_WRITE.
    arr = np.frombuffer(mmap_obj, dtype=np.uint8)
    arr[offset : offset + length : mmap.PAGESIZE] |= 0


def _get_populate_write_fn(
    mmap_obj: mmap.mmap,
) -> Callable[[mmap.mmap, int, int], None]:
    """Select the pre-faulting method once for this mmap."""
    try:
        _madvise_populate_write(mmap_obj, 0, mmap.PAGESIZE)
    except OSError as e:
        if e.errno != errno.EINVAL:
            raise
        logger.warning(
            "MADV_POPULATE_WRITE is not supported; falling back to per-page "
            "writes for mmap pre-population. Startup may be slower."
        )
        return _fallback_populate_write
    return _madvise_populate_write


class SharedOffloadRegion:
    """Single mmap-backed memory region shared across all workers for a
    vLLM instance.  Workers coordinate via the filesystem: the first worker
    to open the file with O_EXCL becomes the creator and calls ftruncate;
    the rest open the existing file and wait until it reaches the expected
    size.  Each worker then mmap()s the full file.

    File path: /dev/shm/vllm_offload_{engine_id}.mmap.  When a barrier is
    given, the path is unlinked once every worker has mapped the file, so
    the kernel reclaims the memory when the last worker exits, no matter
    how it exits; mappings taken before the unlink stay valid.

    ``populate_shard=(index, count)`` pre-faults only this worker's contiguous
    share, for layouts where every worker maps the same bytes. The barrier
    also waits for all shards to finish pre-faulting before returning.
    ``num_segments`` splits backing files beneath the contiguous mapping.
    """

    BLOCK_SIZE_ALIGNMENT: int = mmap.PAGESIZE

    def __init__(
        self,
        engine_id: str,
        num_chunks: int,
        rank: int | None,
        kv_bytes_per_chunk: int,
        cpu_page_size: int,
        barrier: Callable[[], None] | None = None,
        *,
        creator_memory_check: Callable[[int], None] | None = None,
        populate_shard: tuple[int, int] | None = None,
        num_segments: int = 1,
    ) -> None:
        self.page_size = mmap.PAGESIZE
        assert kv_bytes_per_chunk % self.page_size == 0

        self.num_chunks = num_chunks
        self._row_stride = kv_bytes_per_chunk
        self.total_size_bytes = self.num_chunks * self._row_stride
        self.cpu_page_size = cpu_page_size

        self.mmap_path = f"/dev/shm/vllm_offload_{engine_id}.mmap"
        num_segments = min(num_segments, self.total_size_bytes // _MIN_SEGMENT_BYTES)
        rows = max(1, cdiv(num_chunks, max(1, num_segments)))
        # (path, offset, length) per file.
        self._segments = [
            (
                f"{self.mmap_path}.{i}" if i else self.mmap_path,
                row * self._row_stride,
                min(rows, num_chunks - row) * self._row_stride,
            )
            for i, row in enumerate(range(0, max(num_chunks, 1), rows))
        ]
        self._creator = False  # set True only if this worker creates the file
        self.rank = rank
        if rank is not None:
            # byte offset to this worker's first slot within each chunk row
            self._worker_offset = rank * cpu_page_size
            # exclusive upper bound for this worker's area within each row
            self._worker_area_end = (rank + 1) * cpu_page_size
        barriers_remaining = 2 if populate_shard is not None else 1
        # Reclaim orphans before the space check, so a crashed engine's
        # region frees the tmpfs budget this allocation draws on.
        _reclaim_orphaned_regions()
        try:
            try:
                self.fd: int | None = os.open(
                    self.mmap_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
                )
            except FileExistsError:
                # Joiner path — another worker won O_EXCL. Reopen and wait
                # for the file to reach expected size.
                self.fd = os.open(self.mmap_path, os.O_RDWR)
                _hold_shared_lock(self.fd, self.mmap_path)
                _wait_for_file_size(self.fd, self._segments[0][2])
                logger.info("Opened existing mmap file %s", self.mmap_path)
            else:
                # Creator path. We won O_EXCL, so we own the file: any
                # failure here must clean up so concurrent joiners don't
                # land on a 0-byte stub and spin in _wait_for_file_size
                # for the full 30 s timeout.
                self._creator = True
                # Lock before anything fallible: the lock is the liveness
                # mark a sibling rank's orphan sweep probes, and the window
                # between the create and this lock is the one window where
                # the sweep could mistake this fresh region for a dead
                # prior incarnation under the same engine id.
                _hold_shared_lock(self.fd, self.mmap_path)
                if creator_memory_check is not None:
                    creator_memory_check(self.total_size_bytes)
                check_shm_free_space(
                    self.total_size_bytes,
                    allocation_name="CPU KV offload shared region in /dev/shm",
                )
                for path, _, length in self._segments[1:]:
                    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                    os.ftruncate(fd, length)
                    os.close(fd)
                # Joiners wait for the first file, so size it last.
                os.ftruncate(self.fd, self._segments[0][2])
                logger.info(
                    "Created mmap file %s (%.2f GB)",
                    self.mmap_path,
                    self.total_size_bytes / 1e9,
                )

            single = len(self._segments) == 1
            self.mmap_obj: mmap.mmap | None = mmap.mmap(
                self.fd if single else -1,
                self.total_size_bytes,
                flags=mmap.MAP_SHARED if single else mmap.MAP_PRIVATE,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
            if not single:
                self._map_segments()
            if barrier is not None:
                barriers_remaining -= 1
                barrier()
                if self._creator:
                    self._unlink()
                    self._creator = False
                    logger.info("Unlinked mmap file %s", self.mmap_path)
            if populate_shard is not None:
                populate_write_fn = _get_populate_write_fn(self.mmap_obj)
                index, count = populate_shard
                share = round_up(cdiv(self.total_size_bytes, count), self.page_size)
                start = min(index * share, self.total_size_bytes)
                if start < self.total_size_bytes:
                    length = min(share, self.total_size_bytes - start)
                    populate_write_fn(self.mmap_obj, start, length)
                if barrier is not None:
                    barriers_remaining -= 1
                    barrier()
        except Exception:
            if self._creator:
                self._unlink()
                self._creator = False
            if hasattr(self, "mmap_obj") and self.mmap_obj is not None:
                # The prefault traceback may retain an exported buffer.
                with contextlib.suppress(BufferError):
                    self.mmap_obj.close()
                self.mmap_obj = None
            if hasattr(self, "fd") and self.fd is not None:
                os.close(self.fd)
                self.fd = None
            # Join remaining barriers so peers do not wait until the collective
            # times out. A barrier failure must not replace the setup error.
            if barrier is not None:
                for _ in range(barriers_remaining):
                    try:
                        barrier()
                    except Exception:
                        logger.warning(
                            "Failed to release peers waiting at the mmap barrier",
                            exc_info=True,
                        )
            raise

        self._base = torch.frombuffer(memoryview(self.mmap_obj), dtype=torch.int8)
        self._views: list[torch.Tensor] = []
        self._canonical_offset = 0
        self.is_pinned: bool = False
        self.pinned_addresses: list[int] = []

        if populate_shard is not None:
            return
        populate_write_fn = _get_populate_write_fn(self.mmap_obj)

        if rank is not None:
            # Populate only this worker's pages (one slot per chunk row).
            _t0 = time.perf_counter()
            for offset, length in self.iter_worker_slot_ranges():
                populate_write_fn(self.mmap_obj, offset, length)
            logger.debug(
                "MADV_POPULATE_WRITE loop: %d chunks in %.3f s",
                num_chunks,
                time.perf_counter() - _t0,
            )
        else:
            # No rank — populate the entire shared region in one call.
            _t0 = time.perf_counter()
            populate_write_fn(self.mmap_obj, 0, self.total_size_bytes)
            logger.debug(
                "MADV_POPULATE_WRITE entire region: %.3f s", time.perf_counter() - _t0
            )

    def _map_segments(self) -> None:
        """Replace the anonymous mapping with the segment files, back to back."""
        assert self.mmap_obj is not None and self.fd is not None
        base = ctypes.addressof(ctypes.c_char.from_buffer(self.mmap_obj))
        for path, offset, length in self._segments:
            fd = self.fd if offset == 0 else os.open(path, os.O_RDWR)
            address = _libc_mmap()(
                base + offset,
                length,
                mmap.PROT_READ | mmap.PROT_WRITE,
                mmap.MAP_SHARED | _MAP_FIXED,
                fd,
                0,
            )
            if fd != self.fd:
                os.close(fd)
            if address != base + offset:
                raise OSError(ctypes.get_errno(), f"Failed to map {path}")

    def _unlink(self) -> None:
        for path, _, _ in self._segments:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)

    def iter_worker_slot_ranges(self) -> Iterator[tuple[int, int]]:
        """Yield this worker's slots as page-aligned (offset, length) ranges."""
        assert self.rank is not None
        worker_offset = self.rank * self.cpu_page_size
        first = 0
        if len(self._segments) > 1:
            # Start each worker on a different file.
            first = (
                self.rank * self.num_chunks // (self._row_stride // self.cpu_page_size)
            )
        for i in range(self.num_chunks):
            row = (first + i) % self.num_chunks
            start = row * self._row_stride + worker_offset
            aligned = start - start % self.page_size
            end = round_up(start + self.cpu_page_size, self.page_size)
            yield aligned, end - aligned

    @property
    def base_tensor(self) -> torch.Tensor:
        if self._base is None:
            raise RuntimeError("Shared offload region has been released.")
        return self._base

    def create_next_worker_view(self, tensor_page_size: int) -> torch.Tensor:
        """Allocate a strided int8 view for this worker, one canonical tensor.

        Must be called once per canonical tensor. The full mmap layout is:

            worker0_chunk0 | worker1_chunk0 | ... | worker{M-1}_chunk0
            worker0_chunk1 | worker1_chunk1 | ... | worker{M-1}_chunk1
            ...

        Each worker_chunk cell is cpu_page_size bytes and holds all canonical
        tensors for that worker and chunk concatenated:
            [ tensor0_data | tensor1_data | ... | tensor{L-1}_data ]

        Consecutive rows are separated by row_stride = cpu_page_size * M.

        Returns an int8 tensor of shape (num_chunks, tensor_page_size) with stride
        (row_stride, 1).  Using int8 keeps stride == bytes, so swap_blocks
        address arithmetic works without any dtype conversion.

        Args:
            tensor_page_size: Bytes per chunk for this tensor.

        """
        assert self.rank is not None
        new_offset = self._worker_offset + tensor_page_size
        assert new_offset <= self._worker_area_end, (
            f"Worker offset {new_offset} exceeds worker area end "
            f"{self._worker_area_end} (overflowed by "
            f"{new_offset - self._worker_area_end} bytes)"
        )
        worker_layer_view = torch.as_strided(
            self._base,
            size=(self.num_chunks, tensor_page_size),
            stride=(self._row_stride, 1),
            storage_offset=self._worker_offset,
        )
        self._worker_offset = new_offset
        self._views.append(worker_layer_view)
        return worker_layer_view

    def create_next_canonical_view(self, tensor_page_size: int) -> torch.Tensor:
        """Allocate a strided int8 view shared by all workers for one
        canonical tensor (canonical layout).

        Must be called once per canonical tensor, instead of
        create_next_worker_view. The full mmap layout is:

            |<-------- canonical area ------->|<-------- unused ------->|
            |  all workers share this area    |                         |
            |                                 |                         |
            | [ canonical_t0 | canonical_t1 ] |                         |
            | [ canonical_t0 | canonical_t1 ] |                         |
            | [ canonical_t0 | canonical_t1 ] |                         |
            ^                ^
            _canonical_offset=0, then advances by each tensor's size

        Each canonical_t{i} cell is that tensor's canonical page for the
        chunk. Canonical areas are carved consecutively from the start of
        each chunk row; consecutive rows are separated by row_stride. Every
        worker gets the identical byte ranges and writes only its disjoint
        bytes within them, as described by its canonical mappings — unlike
        create_next_worker_view, which gives each worker a private
        cpu_page_size slot per row.

        The trailing unused bytes exist only when the canonical pages sum to
        less than row_stride: page-alignment padding of the row, or
        deduplication of KV replicated across workers (e.g. the MLA latent),
        where one canonical copy replaces world_size worker copies.

        Args:
            tensor_page_size: Canonical bytes per chunk for this tensor.

        """
        new_offset = self._canonical_offset + tensor_page_size
        assert new_offset <= self._row_stride
        view = torch.as_strided(
            self._base,
            size=(self.num_chunks, tensor_page_size),
            stride=(self._row_stride, 1),
            storage_offset=self._canonical_offset,
        )
        self._canonical_offset = new_offset
        self._views.append(view)
        return view

    def create_kv_memoryview(self) -> memoryview:
        """Return a zero-copy memoryview over the entire KV buffer.

        Shape: (num_chunks, row_stride_bytes). Secondary tiers address
        chunk *b* as ``view[b]``.
        """
        kv_tensor = self._base.view(self.num_chunks, self._row_stride)
        np_arr = kv_tensor.numpy()
        assert np_arr.ctypes.data == self._base.data_ptr(), (
            "view()/numpy() created a copy instead of sharing the mmap buffer; "
            "secondary tiers require zero-copy access to primary KV data"
        )
        return memoryview(np_arr)

    def cleanup(self) -> None:
        if self.is_pinned and self._base is not None:
            base_ptr = self._base.data_ptr()
            addresses = self.pinned_addresses or [base_ptr]
            for address in reversed(addresses):
                host_unregister(address)
            self.pinned_addresses.clear()
            self.is_pinned = False
        # Release views before _base: each view holds a _base reference and a
        # direct StorageImpl reference.  Freeing views first lets both refcounts
        # drop so the storage (which holds the mmap_obj buffer export) is freed
        # before mmap_obj.close() is called below.
        if self._views is not None:
            self._views.clear()
        self._base = None
        if self.mmap_obj:
            retained = False
            try:
                self.mmap_obj.close()
            except BufferError:
                # A native user still exports this mapping, so unmapping would
                # let the NIC write into reused address space. Keep it mapped;
                # dropping the fd and the pathname below is still safe.
                retained = True
                logger.warning(
                    "Keeping %s mapped: buffers are still exported to native users",
                    self.mmap_path,
                )
            except Exception:
                logger.warning("Failed to close mmap_obj", exc_info=True)
            if not retained:
                self.mmap_obj = None
        if self.fd is not None:
            try:
                os.close(self.fd)
            except Exception:
                logger.warning("Failed to close fd %s", self.fd, exc_info=True)
            self.fd = None
        if self._creator and getattr(self, "mmap_path", None):
            try:
                self._unlink()
                logger.info("Removed mmap file %s", self.mmap_path)
            except Exception:
                logger.warning(
                    "Failed to unlink path %s", self.mmap_path, exc_info=True
                )
            self._creator = False
        # pin_mmap_region creates <mmap_path>.reglock lazily and closes its fd
        # when done, so the lock file outlives the registration. Remove it
        # whoever created the region: the barrier flow drops the region name
        # at construction, which leaves the reglock as the only name this
        # engine owns in /dev/shm. The lock itself guards the removal, so a
        # rank still inside the registration keeps the file and the orphan
        # sweep takes it later.
        mmap_path = getattr(self, "mmap_path", None)
        if mmap_path:
            reglock_path = mmap_path + _REGLOCK_SUFFIX
            if _remove_reglock(reglock_path):
                logger.debug("Removed reglock file %s", reglock_path)
