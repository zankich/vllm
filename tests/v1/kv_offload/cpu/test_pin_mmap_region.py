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
import time
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


def test_registration_takes_flock_on_region_fd(monkeypatch, tmp_path):
    cudart = MagicMock()
    cudart.cudaHostRegister.return_value = 0
    cudart.cudaHostUnregister.return_value = 0
    monkeypatch.setattr(gpu_worker, "CudaRTLibrary", lambda: cudart)
    monkeypatch.setattr(gpu_worker.current_platform, "is_cuda_alike", lambda: True)

    lock_path = tmp_path / "region"
    lock_path.write_bytes(b"\0" * 4096)
    fd = os.open(lock_path, os.O_RDWR)
    try:
        region = _make_region(fd=fd)

        flock_calls: list[tuple[int, int]] = []
        real_flock = fcntl.flock

        def tracked_flock(f, op):
            flock_calls.append((int(f), op))
            return real_flock(f, op)

        monkeypatch.setattr(fcntl, "flock", tracked_flock)

        gpu_worker.pin_mmap_region(region)

        assert region.is_pinned
        # The region fd must have been taken with LOCK_EX during registration
        # and released by the time the function returns.
        assert any(call[0] == fd and call[1] == fcntl.LOCK_EX for call in flock_calls)
        assert any(call[0] == fd and call[1] == fcntl.LOCK_UN for call in flock_calls)
    finally:
        os.close(fd)


def _pin_region_child(region_path: str, rank: int, events, barrier) -> None:
    """Child rank: mock the registration, then race pin_mmap_region.

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
        region = _make_region(fd=fd)
        region.rank = rank
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
    interval is non-overlapping with the other's. Without the flock the
    barrier-synchronized ranks overlap their sections."""
    from vllm.utils.system_utils import get_mp_context

    ctx = get_mp_context()
    mgr = ctx.Manager()
    events = mgr.list()
    barrier = ctx.Barrier(2)
    region_path = tmp_path / "region"
    region_path.write_bytes(b"\0" * 4096)

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
        assert all(child.exitcode == 0 for child in children)

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
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(timeout=10)
        mgr.shutdown()


def test_region_registration_lock_keeps_shared_liveness_mark(monkeypatch, tmp_path):
    """The registration context must re-acquire LOCK_SH on exit.

    Every participant holds a LOCK_SH on the region fd for the engine's
    lifetime, and `_reclaim_orphaned_regions` probes that lock to tell a
    live region from an orphan. The barrier flow unlinks the path on
    barrier release, but the tiering flow does not, so the boot-time
    LOCK_SH is the only thing protecting the full-size on-disk region
    from a concurrent engine's reclaim sweep.

    The registration context upgrades to LOCK_EX for the cudaHostRegister
    critical section. If it leaves the fd unlocked on exit, a reclaim
    sweep that wins the race between the last worker's release and the
    scheduler's LOCK_SH can unlink the path while workers still hold
    mappings. The fix re-acquires LOCK_SH on exit, mirroring
    `_hold_shared_lock` in shared_offload_region.
    """
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
        os.close(observer_fd)
        os.close(region_fd)
