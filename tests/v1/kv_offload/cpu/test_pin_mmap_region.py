# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""pin_mmap_region takes an exclusive flock on the region fd across its
whole chunked registration, so concurrent ranks on the same shared mmap
serialize against one another and the CudaRTLibrary mock receives the
full sequence under the lock.

The multi-process test pins the flock serialization: two ranks racing
the registration path on the same region file must not overlap their
cudaHostRegister sections (the observed 3090 Ti TP2 failure mode).
"""

import contextlib
import fcntl
import os
import threading
import time
import uuid
from unittest.mock import MagicMock

import pytest

from vllm.v1.kv_offload.cpu import gpu_worker

REGISTER_HOLD_SECONDS = 0.5


def _make_region(fd=None):
    region = MagicMock()
    region.rank = 0
    region.total_size_bytes = 4096
    region.fd = fd
    region._row_stride = 4096
    base = MagicMock()
    base.data_ptr.return_value = 0x7F0000000000
    region._base = base
    region.is_pinned = False
    return region


def test_registration_takes_flock_on_reglock_fd(monkeypatch, tmp_path):
    """Registration takes its exclusive flock on a dedicated .reglock file,
    not on the region fd whose LOCK_SH liveness mark is held for the engine's
    lifetime. The orphan reclaimer's Liveness check on LOCK_SH's probe is what
    the dedicated file isolates from the registration critical section."""
    cudart = MagicMock()
    cudart.cudaHostRegister.return_value = 0
    cudart.cudaHostUnregister.return_value = 0
    monkeypatch.setattr(gpu_worker, "CudaRTLibrary", lambda: cudart)
    monkeypatch.setattr(gpu_worker.current_platform, "is_cuda_alike", lambda: True)

    region_path = tmp_path / "region"
    region_path.write_bytes(b"\0" * 4096)
    reglock_path = tmp_path / "region.reglock"

    region_fd = os.open(region_path, os.O_RDWR)
    observer_fd = os.open(region_path, os.O_RDWR)
    try:
        # Mimic the boot-time LOCK_SH that _hold_shared_lock establishes
        # when the region is opened. The registration context must leave
        # this shared mark alone, so the orphan reclaimer can keep reading
        # it on a separate open file description.
        fcntl.flock(region_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)

        region = _make_region(fd=region_fd)
        region.mmap_path = str(region_path)

        flock_calls: list[tuple[int, int]] = []
        real_flock = fcntl.flock

        def tracked_flock(f, op):
            flock_calls.append((int(f), op))
            return real_flock(f, op)

        monkeypatch.setattr(fcntl, "flock", tracked_flock)

        gpu_worker.pin_mmap_region(region)

        assert region.is_pinned

        # The reglock file must have been created lazily inside the
        # registration context manager.
        assert reglock_path.exists(), (
            "reglock file should be created at registration time"
        )

        reglock_fd = os.open(reglock_path, os.O_RDWR)
        try:
            # The exclusive lock must have been taken and released on the
            # reglock fd, never on the region fd.
            assert any(
                call[0] == reglock_fd and call[1] == fcntl.LOCK_EX
                for call in flock_calls
            ), (
                f"LOCK_EX must be taken on the reglock fd {reglock_fd}; "
                f"observed {flock_calls}"
            )
            assert any(
                call[0] == reglock_fd and call[1] == fcntl.LOCK_UN
                for call in flock_calls
            ), (
                f"LOCK_UN must be taken on the reglock fd {reglock_fd}; "
                f"observed {flock_calls}"
            )
            # The region fd must NEVER have been touched by registration:
            # no LOCK_EX, no LOCK_UN, no re-LOCK_SH on the region fd. The
            # boot-time LOCK_SH is the only one we want.
            for f, op in flock_calls:
                if f == region_fd:
                    pytest.fail(
                        f"region fd {region_fd} must not be touched by "
                        f"registration; observed flock({f}, {op})"
                    )
        finally:
            os.close(reglock_fd)

        # The region's LOCK_SH liveness mark is still held: an observer
        # trying to take LOCK_EX on a separate open file description of the
        # same inode must fail (EWOULDBLOCK), proving the LOCK_SH was never
        # released during registration.
        try:
            fcntl.flock(observer_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            pass  # expected: the shared liveness mark blocks the exclusive
        else:
            fcntl.flock(observer_fd, fcntl.LOCK_UN)
            fcntl.flock(region_fd, fcntl.LOCK_UN)
            pytest.fail(
                "region fd was left unlocked after _region_registration_lock "
                "exited; a concurrent reclaim sweep could unlink the path"
            )
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(observer_fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            fcntl.flock(region_fd, fcntl.LOCK_UN)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(reglock_path)
        os.close(observer_fd)
        os.close(region_fd)


def _pin_region_child(region_path: str, rank: int, events, barrier) -> None:
    """Child rank: acquire the boot-time LOCK_SH on the region fd (the
    liveness mark _hold_shared_lock sets), then race pin_mmap_region.

    The mocked cudaHostRegister records a timestamped (kind, rank, t) entry
    before and after a hold interval. The parent's monotonic clock sees
    the events in the order they were appended within each process, and
    the cross-process intervals reconstruct the lock's critical section."""
    import vllm.v1.kv_offload.cpu.gpu_worker as gpu_worker

    cudart = MagicMock()

    def register_with_hold(ptr, size):
        events.append(("enter", rank, time.monotonic()))
        time.sleep(REGISTER_HOLD_SECONDS)
        events.append(("exit", rank, time.monotonic()))
        return 0

    cudart.cudaHostRegister = register_with_hold
    gpu_worker.CudaRTLibrary = lambda: cudart
    gpu_worker.current_platform.is_cuda_alike = lambda: True

    fd = os.open(region_path, os.O_RDWR)
    try:
        # The boot-time LOCK_SH that _hold_shared_lock establishes. With
        # this mark in place, the OLD code's LOCK_EX on the region fd
        # deadlocks against the other rank's LOCK_SH on the same inode;
        # the FIX routes registration to a dedicated reglock fd so the
        # two ranks serialize without touching each other's LOCK_SH.
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        region = _make_region(fd=fd)
        region.rank = rank
        region.mmap_path = region_path
        base = MagicMock()
        base.data_ptr.return_value = 0x7F0000000000 + rank
        region._base = base
        barrier.wait(timeout=30)
        gpu_worker.pin_mmap_region(region)
        assert region.is_pinned
    finally:
        os.close(fd)


@pytest.fixture(autouse=True)
def _set_spawn_method(monkeypatch):
    # Keep the multiprocessing start method explicit (see
    # test_shared_offload_region for the WSL/NVML rationale).
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def test_registration_serializes_two_ranks_on_shared_region(tmp_path):
    """Two ranks racing pin_mmap_region on the same region file must hold
    their cudaHostRegister sections exclusively: each rank's [enter, exit]
    interval is non-overlapping with the other's. Without the dedicated
    reglock file the barrier-synchronized ranks deadlock, because each
    rank's boot-time LOCK_SH on the region inode blocks the other's
    LOCK_EX on the same inode. The FIX routes registration to a sibling
    reglock file so the two ranks serialize without touching each
    other's LOCK_SH."""
    from vllm.utils.system_utils import get_mp_context

    ctx = get_mp_context()
    mgr = ctx.Manager()
    events = mgr.list()
    barrier = ctx.Barrier(2)
    region_path = tmp_path / "region"
    region_path.write_bytes(b"\0" * 4096)
    reglock_path = tmp_path / "region.reglock"

    children = [
        ctx.Process(
            target=_pin_region_child, args=(str(region_path), rank, events, barrier)
        )
        for rank in (0, 1)
    ]
    for child in children:
        child.start()
    try:
        for child in children:
            child.join(timeout=30)
        assert all(child.exitcode == 0 for child in children), (
            "ranks did not serialize; one or more deadlocked against the "
            "other's LOCK_SH on the shared region inode"
        )

        seen = list(events)
        # Group the four events into the two ranks' [enter, exit] intervals.
        intervals: dict[int, tuple[float, float]] = {}
        for kind, rank, t in seen:
            if kind == "enter":
                intervals.setdefault(rank, (t, t))
                intervals[rank] = (t, intervals[rank][1])
            else:
                intervals[rank] = (intervals[rank][0], t)
        (a_enter, a_exit), (b_enter, b_exit) = intervals.values()
        # Non-overlap: one rank's exit precedes the other's enter.
        assert a_exit < b_enter or b_exit < a_enter, (
            f"critical sections overlapped: {intervals}"
        )
        # Registration went through a dedicated reglock file, not via
        # the region fd's own flock (which the LOCK_SH holders share).
        assert reglock_path.exists(), (
            "reglock file must be created by registration on each rank"
        )
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(timeout=10)
        mgr.shutdown()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(reglock_path)


def test_region_registration_lock_keeps_shared_liveness_mark(monkeypatch, tmp_path):
    """The registration context must leave the boot-time LOCK_SH alone.

    Every participant holds a LOCK_SH on the region fd for the engine's
    lifetime, and `_reclaim_orphaned_regions` probes that lock to tell a
    live region from an orphan. The barrier flow unlinks the path on
    barrier release, but the tiering flow does not, so the boot-time
    LOCK_SH is the only thing protecting the full-size on-disk region
    from a concurrent engine's reclaim sweep.

    Registration serializes through a dedicated reglock file, so the
    region fd is untouched across the cudaHostRegister critical section.
    The boot-time LOCK_SH is what stays visible to a reclaim sweep."""
    cudart = MagicMock()
    cudart.cudaHostRegister.return_value = 0
    cudart.cudaHostUnregister.return_value = 0
    monkeypatch.setattr(gpu_worker, "CudaRTLibrary", lambda: cudart)
    monkeypatch.setattr(gpu_worker.current_platform, "is_cuda_alike", lambda: True)

    region_path = tmp_path / "region"
    region_path.write_bytes(b"\0" * 4096)
    # Two independent fds to the same file: region.fd for registration,
    # observer_fd to probe the lock state from the side.
    region_fd = os.open(region_path, os.O_RDWR)
    observer_fd = os.open(region_path, os.O_RDWR)
    try:
        # Mimic the boot-time LOCK_SH that _hold_shared_lock establishes
        # when the region is opened. fcntl.flock on the region fd must
        # hold for the engine's lifetime.
        fcntl.flock(region_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)

        region = _make_region(fd=region_fd)
        region.mmap_path = str(region_path)
        gpu_worker.pin_mmap_region(region)
        assert region.is_pinned

        # After registration the region fd must still hold its shared
        # liveness mark: an observer trying to take LOCK_EX on a
        # separate open file description of the same file must fail.
        try:
            fcntl.flock(observer_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            pass  # expected: the shared liveness mark blocks the exclusive
        else:
            # The exclusive probe succeeded, which means the region fd
            # was left unlocked after registration exited. Release the
            # observer's lock and the region's (no-op) lock before
            # letting the finally block close both fds and reporting
            # the failure.
            fcntl.flock(observer_fd, fcntl.LOCK_UN)
            fcntl.flock(region_fd, fcntl.LOCK_UN)
            pytest.fail(
                "region fd was left unlocked after _region_registration_lock "
                "exited; a concurrent reclaim sweep could unlink the path"
            )

        # The shared liveness mark is intact: the region fd holds a
        # shared lock and the observer's exclusive probe above raised.
        # Release both before the finally block closes the fds.
        fcntl.flock(region_fd, fcntl.LOCK_UN)
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(observer_fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            fcntl.flock(region_fd, fcntl.LOCK_UN)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(region_path.with_name(region_path.name + ".reglock"))
        os.close(observer_fd)
        os.close(region_fd)


def _hold_region_child(engine_id: str, release, held) -> None:
    """Sibling rank: hold a SharedOffloadRegion on `engine_id` open with its
    boot-time LOCK_SH until the parent signals release. The LOCK_SH is the
    blocker that the OLD code's LOCK_EX on the region fd can never get
    past; the FIX's reglock file ignores it. Signals `held` once the LOCK_SH
    is in place so the parent only runs after the sibling is a real blocker."""
    import mmap

    from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

    page_size = mmap.PAGESIZE
    region = SharedOffloadRegion(
        engine_id=engine_id,
        num_chunks=4,
        rank=1,
        kv_bytes_per_chunk=2 * page_size,
        cpu_page_size=page_size,
    )
    held.set()
    try:
        release.wait(timeout=30)
    finally:
        region.cleanup()


def test_registration_completes_when_sibling_region_holds_lock(monkeypatch):
    """CPU-runnable stand-in for the cross_topology_roundtrip gate.

    Two regions on the same engine_id both hold LOCK_SH on the region
    inode (the boot-time mark `_hold_shared_lock` sets). The OLD code's
    LOCK_EX on the region fd would deadlock: any second region's LOCK_SH
    on the same inode blocks the exclusive conversion, and no participant
    is willing to release until registration completes. The FIX routes
    registration to a dedicated `.reglock` file, so the sibling's LOCK_SH
    on the region inode no longer conflicts and pin_mmap_region runs to
    completion.

    This is the same kernel-level mechanism that hangs the four stock
    `test_cross_topology_roundtrip` params: writer_tp ranks open the
    region first and hold their LOCK_SH through their cudaHostRegister,
    reader_tp ranks join and their pin_mmap_region locks against the
    writers' LOCK_SH on the same inode.
    """
    import mmap

    from vllm.utils.system_utils import get_mp_context
    from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

    cudart = MagicMock()
    cudart.cudaHostRegister.return_value = 0
    monkeypatch.setattr(gpu_worker, "CudaRTLibrary", lambda: cudart)
    monkeypatch.setattr(gpu_worker.current_platform, "is_cuda_alike", lambda: True)

    iid = str(uuid.uuid4())

    ctx = get_mp_context()
    release = ctx.Event()
    held = ctx.Event()
    own_region = None
    child = ctx.Process(target=_hold_region_child, args=(iid, release, held))
    child.start()
    try:
        # We construct after the child starts, so the child races us for
        # O_EXCL. With spawn and our quick start we usually win; either
        # way both regions land on the same inode and both fds end up
        # holding LOCK_SH.
        own_region = SharedOffloadRegion(
            engine_id=iid,
            num_chunks=4,
            rank=0,
            kv_bytes_per_chunk=2 * mmap.PAGESIZE,
            cpu_page_size=mmap.PAGESIZE,
        )

        # Wait for the sibling child to acquire LOCK_SH before running
        # pin_mmap_region; otherwise the child is still warming up its
        # Python interpreter and is not a real blocker yet.
        held.wait(timeout=30)
        assert held.is_set(), "sibling child did not signal LOCK_SH"

        # Run pin_mmap_region in a thread so the test framework can
        # observe a hang without blocking the process from cleanup.
        done = threading.Event()
        pin_result: list = []

        def attempt() -> None:
            try:
                gpu_worker.pin_mmap_region(own_region)
                pin_result.append("ok")
            except Exception as e:
                pin_result.append(e)
            finally:
                done.set()

        t = threading.Thread(target=attempt)
        t.start()
        completed = done.wait(timeout=5.0)
        if not completed:
            # The OLD code is hung on LOCK_EX against the region fd,
            # blocked by the sibling's LOCK_SH on the same inode. The
            # FIX routes registration through a dedicated reglock file,
            # so this branch is unreachable on the fixed code.
            release.set()
            t.join(timeout=10)
            pytest.fail(
                "pin_mmap_region deadlocked: a sibling region's LOCK_SH "
                "on the shared region inode blocked the registration "
                "exclusive lock. Registration must use a dedicated "
                ".reglock file."
            )

        assert pin_result == ["ok"], f"pin_mmap_region failed: {pin_result[0]}"
        assert own_region.is_pinned
    finally:
        release.set()
        if child.is_alive():
            child.join(timeout=10)
            if child.is_alive():
                child.terminate()
                child.join(timeout=5)
        if own_region is not None:
            own_region.cleanup()
            with contextlib.suppress(FileNotFoundError):
                os.unlink(own_region.mmap_path + ".reglock")
