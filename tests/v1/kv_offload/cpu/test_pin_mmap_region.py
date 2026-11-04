# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""pin_mmap_region holds an exclusive flock on the dedicated
``<mmap_path>.reglock`` file across its whole chunked registration, so
concurrent ranks on the same shared mmap serialize against one another
and the region fd's LOCK_SH liveness mark stays untouched for the
orphan-reclaim sweep.

The multi-process test pins the serialization on both pin paths: two
ranks racing the registration on the same region file must not overlap
their host_register sections, the observed failure mode where concurrent
cudaHostRegister over the same pages intermittently poisons the CUDA
context.
"""

import contextlib
import errno
import fcntl
import mmap
import os
import threading
import time
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.utils.system_utils import get_mp_context
from vllm.v1.kv_offload.cpu import gpu_worker
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

PAGE_SIZE = mmap.PAGESIZE
REGISTER_HOLD_SECONDS = 0.5


@pytest.fixture(autouse=True)
def _set_spawn_method(monkeypatch):
    # Keep the multiprocessing start method explicit (see
    # test_shared_offload_region for the WSL/NVML rationale).
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def _registration_region(mmap_path: str, fd: int, rank: int = 0) -> SharedOffloadRegion:
    """Describe a region for pin_mmap_region without mapping it."""
    region = object.__new__(SharedOffloadRegion)
    region.rank = rank
    region.num_chunks = 4
    region.page_size = PAGE_SIZE
    region.cpu_page_size = PAGE_SIZE
    region._row_stride = 2 * PAGE_SIZE
    region.total_size_bytes = region.num_chunks * region._row_stride
    region._segments = [(mmap_path, 0, region.total_size_bytes)]
    region._base = SimpleNamespace(data_ptr=lambda: 0x7F0000000000 + rank)
    region.pinned_addresses = []
    region.is_pinned = False
    region.fd = fd
    region.mmap_path = mmap_path
    return region


def test_registration_takes_flock_on_reglock_fd(monkeypatch, tmp_path):
    """Registration takes its exclusive flock on a dedicated .reglock file,
    never on the region fd, whose LOCK_SH liveness mark is held for the
    engine's lifetime and is what the orphan-reclaim sweep probes."""
    monkeypatch.setattr(gpu_worker, "host_register", MagicMock(return_value=True))
    monkeypatch.setattr(gpu_worker, "host_unregister", MagicMock())
    monkeypatch.setattr(gpu_worker, "MIN_HOST_REGISTER_SAVED_BYTES", PAGE_SIZE)

    region_path = tmp_path / "region"
    region_path.write_bytes(b"\0" * PAGE_SIZE)
    reglock_path = str(region_path) + ".reglock"

    region_fd = os.open(region_path, os.O_RDWR)
    observer_fd = os.open(region_path, os.O_RDWR)
    try:
        # The boot-time LOCK_SH a live region holds on its own fd.
        fcntl.flock(region_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)

        flock_by_ino: list[tuple[int, int]] = []
        real_flock = fcntl.flock

        def tracked_flock(fd, op):
            flock_by_ino.append((os.fstat(fd).st_ino, op))
            return real_flock(fd, op)

        monkeypatch.setattr(fcntl, "flock", tracked_flock)

        region = _registration_region(str(region_path), region_fd)
        gpu_worker.pin_mmap_region(region)

        assert region.is_pinned
        assert os.path.exists(reglock_path), (
            "reglock file should be created at registration time"
        )
        reglock_ino = os.stat(reglock_path).st_ino
        region_ino = os.fstat(region_fd).st_ino
        assert (reglock_ino, fcntl.LOCK_EX) in flock_by_ino, (
            f"LOCK_EX must be taken on the reglock inode; observed {flock_by_ino}"
        )
        assert (reglock_ino, fcntl.LOCK_UN) in flock_by_ino, (
            f"LOCK_UN must release the reglock inode; observed {flock_by_ino}"
        )
        for ino, op in flock_by_ino:
            assert ino != region_ino, (
                f"registration must not flock the region fd; "
                f"observed flock(ino={ino}, op={op})"
            )

        # The region's LOCK_SH liveness mark survives registration: an
        # observer's exclusive probe on a separate open of the same inode
        # must still fail.
        try:
            fcntl.flock(observer_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            pass
        else:
            fcntl.flock(observer_fd, fcntl.LOCK_UN)
            pytest.fail(
                "the region fd was left unlocked after registration; a "
                "concurrent reclaim sweep could unlink the path"
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


def _pin_region_child(region_path, rank, events, barrier, worker_slot_only) -> None:
    """Child rank: hold the boot-time LOCK_SH on the region fd, then race
    pin_mmap_region on the given path.

    The mocked host_register records (kind, rank, timestamp) before and
    after a hold interval, so the parent reconstructs each rank's
    critical section from the shared event list."""
    import vllm.v1.kv_offload.cpu.gpu_worker as gpu_worker

    def register_with_hold(ptr, size):
        events.append(("enter", rank, time.monotonic()))
        time.sleep(REGISTER_HOLD_SECONDS)
        events.append(("exit", rank, time.monotonic()))
        return True

    gpu_worker.host_register = register_with_hold
    gpu_worker.host_unregister = lambda ptr: None
    gpu_worker.MIN_HOST_REGISTER_SAVED_BYTES = PAGE_SIZE

    fd = os.open(region_path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        region = _registration_region(region_path, fd, rank=rank)
        barrier.wait(timeout=30)
        gpu_worker.pin_mmap_region(region, worker_slot_only=worker_slot_only)
        assert region.is_pinned
    finally:
        os.close(fd)


@pytest.mark.parametrize("worker_slot_only", [False, True])
def test_registration_serializes_two_ranks_on_shared_region(tmp_path, worker_slot_only):
    """Two ranks racing pin_mmap_region on the same region file must hold
    their host_register sections exclusively, on both pin paths: the
    full-region registration the canonical layout takes and the
    worker-slot registration the private-slot layout takes.

    Each rank holds a boot-time LOCK_SH on the region inode, so the
    serialization must run through a dedicated reglock file rather than
    the region fd, where each rank's LOCK_SH would block the other's
    LOCK_EX and deadlock the boot."""
    ctx = get_mp_context()
    mgr = ctx.Manager()
    events = mgr.list()
    barrier = ctx.Barrier(2)
    region_path = tmp_path / "region"
    region_path.write_bytes(b"\0" * PAGE_SIZE)
    reglock_path = str(region_path) + ".reglock"

    children = [
        ctx.Process(
            target=_pin_region_child,
            args=(str(region_path), rank, events, barrier, worker_slot_only),
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
        intervals: dict[int, tuple[float, float]] = {}
        for kind, rank, t in seen:
            if kind == "enter":
                intervals.setdefault(rank, (t, t))
                intervals[rank] = (t, intervals[rank][1])
            else:
                intervals[rank] = (intervals[rank][0], t)
        (a_enter, a_exit), (b_enter, b_exit) = intervals.values()
        assert a_exit < b_enter or b_exit < a_enter, (
            f"critical sections overlapped: {intervals}"
        )
        assert os.path.exists(reglock_path), (
            "registration must serialize through a dedicated reglock file"
        )
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(timeout=10)
        mgr.shutdown()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(reglock_path)


def test_registration_lock_keeps_shared_liveness_mark(monkeypatch, tmp_path):
    """The registration must leave the region fd's boot-time LOCK_SH alone.

    Every participant holds that shared mark for the engine's lifetime
    and the reclaim sweep probes it to tell a live region from an orphan.
    Registration serializes through a dedicated reglock file, so the mark
    stays visible to a concurrent sweep across the whole cudaHostRegister
    critical section."""
    monkeypatch.setattr(gpu_worker, "host_register", MagicMock(return_value=True))
    monkeypatch.setattr(gpu_worker, "host_unregister", MagicMock())
    monkeypatch.setattr(gpu_worker, "MIN_HOST_REGISTER_SAVED_BYTES", PAGE_SIZE)

    region_path = tmp_path / "region"
    region_path.write_bytes(b"\0" * PAGE_SIZE)
    region_fd = os.open(region_path, os.O_RDWR)
    observer_fd = os.open(region_path, os.O_RDWR)
    try:
        fcntl.flock(region_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)

        region = _registration_region(str(region_path), region_fd)
        gpu_worker.pin_mmap_region(region)
        assert region.is_pinned

        # An observer's exclusive probe on a separate open must fail: the
        # shared liveness mark is still held on the region inode.
        try:
            fcntl.flock(observer_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            pass
        else:
            fcntl.flock(observer_fd, fcntl.LOCK_UN)
            pytest.fail(
                "the region fd was left unlocked after registration; a "
                "concurrent reclaim sweep could unlink the path"
            )
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(observer_fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            fcntl.flock(region_fd, fcntl.LOCK_UN)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(str(region_path) + ".reglock")
        os.close(observer_fd)
        os.close(region_fd)


def test_registration_lock_reverifies_inode_after_removal(tmp_path, monkeypatch):
    """A removal of the reglock must not split the registration.

    The lock only excludes ranks while they all hold it on the same inode.
    A remover that unlinks the reglock between a rank's open and its lock
    leaves that rank locking an orphaned inode while the next rank's
    O_CREAT makes a second one, and the two critical sections run at once.
    A rank whose fd names a stale inode must retry on the file the path
    names now."""
    region_path = tmp_path / "region"
    region_path.write_bytes(b"\0" * PAGE_SIZE)
    reglock_path = str(region_path) + ".reglock"

    region_fd = os.open(region_path, os.O_RDWR)
    region = _registration_region(str(region_path), region_fd)

    real_flock = fcntl.flock
    remover_ran = threading.Event()
    b_lock_fds: list[int] = []
    b_thread: threading.Thread | None = None

    def tracked_flock(fd, operation):
        is_rank_b = b_thread is not None and threading.current_thread() is b_thread
        if is_rank_b and operation == fcntl.LOCK_EX:
            b_lock_fds.append(fd)
            if not remover_ran.is_set():
                # Take the lock on the inode the rank has open and unlink
                # the path under it, the way the orphan sweep and a failing
                # rank's cleanup do. The rank has not locked yet, so the
                # removal succeeds.
                remover_ran.set()
                remover_fd = os.open(reglock_path, os.O_RDWR | os.O_NOFOLLOW)
                try:
                    real_flock(remover_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    os.unlink(reglock_path)
                finally:
                    os.close(remover_fd)
        return real_flock(fd, operation)

    monkeypatch.setattr(fcntl, "flock", tracked_flock)

    errors: list[BaseException] = []
    inside = 0
    overlap: list[str] = []
    state_lock = threading.Lock()

    def enter(rank: str) -> None:
        nonlocal inside
        with state_lock:
            inside += 1
            if inside > 1:
                overlap.append(rank)

    def leave() -> None:
        nonlocal inside
        with state_lock:
            inside -= 1

    b_inside = threading.Event()
    release_b = threading.Event()
    d_go = threading.Event()
    d_inside = threading.Event()
    release_d = threading.Event()

    def run_b() -> None:
        try:
            with gpu_worker._region_registration_lock(region):
                enter("B")
                try:
                    b_inside.set()
                    if not release_b.wait(timeout=30):
                        errors.append(AssertionError("rank B was never released"))
                finally:
                    leave()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def run_d() -> None:
        try:
            if not d_go.wait(timeout=30):
                errors.append(AssertionError("rank D was never started"))
            with gpu_worker._region_registration_lock(region):
                enter("D")
                try:
                    d_inside.set()
                    if not release_d.wait(timeout=30):
                        errors.append(AssertionError("rank D was never released"))
                finally:
                    leave()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    b_thread = threading.Thread(target=run_b, name="rank-b")
    d_thread = threading.Thread(target=run_d, name="rank-d")
    b_thread.start()
    d_thread.start()
    try:
        assert b_inside.wait(timeout=30), "rank B never entered the critical section"

        # The rank must hold the lock on the file the path names: an
        # unlinked inode excludes nobody, and a replacement inode the path
        # names later is a different lock.
        b_st = os.fstat(b_lock_fds[-1])
        try:
            path_st = os.stat(reglock_path)
        except FileNotFoundError:
            path_st = None
        assert path_st is not None, (
            "the reglock path was removed while a rank held its lock; the "
            "registration now excludes nobody"
        )
        assert (b_st.st_dev, b_st.st_ino) == (path_st.st_dev, path_st.st_ino), (
            f"the registration lock is on inode {b_st.st_ino} but the path "
            f"names inode {path_st.st_ino}"
        )
        assert b_st.st_nlink > 0, "the registration lock is on an unlinked inode"

        # A late rank that opens the path now must not be able to take the
        # lock: it would register concurrently with rank B.
        late_fd = os.open(reglock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(late_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            late_rank_blocked = True
        else:
            late_rank_blocked = False
            fcntl.flock(late_fd, fcntl.LOCK_UN)
        finally:
            os.close(late_fd)
        assert late_rank_blocked, (
            "a late rank could take the registration lock while rank B held "
            "it: the two registrations would run concurrently"
        )

        # Rank D goes through the real lock: it must wait for rank B.
        d_go.set()
        release_b.set()
        assert d_inside.wait(timeout=30), (
            "the late rank never entered the critical section"
        )
        release_d.set()
    finally:
        release_b.set()
        release_d.set()
        d_go.set()
        b_thread.join(timeout=30)
        d_thread.join(timeout=30)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(reglock_path)
        os.close(region_fd)

    assert not overlap, f"two ranks were inside the critical section: {overlap}"
    assert not errors, errors


def _hold_region_child(engine_id: str, release, held) -> None:
    """Sibling rank: hold a SharedOffloadRegion with its boot-time LOCK_SH
    until the parent signals release. The LOCK_SH is the mark a same-inode
    exclusive lock can never get past, so the parent's registration must
    route around it. Signals ``held`` once the mark is in place."""
    from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

    region = SharedOffloadRegion(
        engine_id=engine_id,
        num_chunks=4,
        rank=1,
        kv_bytes_per_chunk=2 * PAGE_SIZE,
        cpu_page_size=PAGE_SIZE,
    )
    held.set()
    try:
        release.wait(timeout=30)
    finally:
        region.cleanup()


def test_registration_completes_when_sibling_region_holds_lock(monkeypatch):
    """A sibling region's LOCK_SH on the shared inode must not block
    registration.

    Both regions open the same engine's file and each holds the boot-time
    shared mark. Registration must serialize through the dedicated
    .reglock file, leaving the region inode's marks alone, so pin_mmap_region
    runs to completion instead of deadlocking."""
    monkeypatch.setattr(gpu_worker, "host_register", MagicMock(return_value=True))
    monkeypatch.setattr(gpu_worker, "host_unregister", MagicMock())
    monkeypatch.setattr(
        "vllm.v1.kv_offload.cpu.shared_offload_region.host_unregister",
        MagicMock(),
    )

    iid = str(uuid.uuid4())

    ctx = get_mp_context()
    release = ctx.Event()
    held = ctx.Event()
    own_region = None
    child = ctx.Process(target=_hold_region_child, args=(iid, release, held))
    child.start()
    try:
        # Construct after the child starts, so the child races us for
        # O_EXCL. Either way both regions land on the same inode and both
        # fds hold the boot-time LOCK_SH.
        own_region = SharedOffloadRegion(
            engine_id=iid,
            num_chunks=4,
            rank=0,
            kv_bytes_per_chunk=2 * PAGE_SIZE,
            cpu_page_size=PAGE_SIZE,
        )

        held.wait(timeout=30)
        assert held.is_set(), "sibling child did not signal its liveness mark"

        # Run pin_mmap_region in a thread so the test framework can observe
        # a hang without blocking the process from cleanup.
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
            release.set()
            t.join(timeout=10)
            pytest.fail(
                "pin_mmap_region deadlocked: a sibling region's LOCK_SH on "
                "the shared region inode blocked the registration lock. "
                "Registration must use a dedicated .reglock file."
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
            with contextlib.suppress(FileNotFoundError):
                os.unlink(own_region.mmap_path)


def test_registration_refuses_symlinked_reglock(monkeypatch, tmp_path):
    """/dev/shm is world-writable, so a local user may plant a symlink
    under the reglock name. The registration opener must refuse it rather
    than flocking the target inode, and the refusal must propagate out of
    the retry loop instead of spinning."""
    rec_register = MagicMock(return_value=True)
    monkeypatch.setattr(gpu_worker, "host_register", rec_register)
    monkeypatch.setattr(gpu_worker, "host_unregister", MagicMock())

    region_path = tmp_path / "region"
    region_path.write_bytes(b"\0" * PAGE_SIZE)
    decoy = tmp_path / "decoy"
    decoy.write_bytes(b"")
    reglock_path = tmp_path / "region.reglock"
    os.symlink(decoy, reglock_path)

    region_fd = os.open(region_path, os.O_RDWR)
    try:
        region = _registration_region(str(region_path), region_fd)

        deadline = time.monotonic() + 10.0
        with pytest.raises(OSError) as exc_info:
            gpu_worker.pin_mmap_region(region)
        assert time.monotonic() < deadline, (
            "pin_mmap_region spun on the symlinked reglock instead of raising"
        )
        assert exc_info.value.errno == errno.ELOOP
        # The refusal happens before any registration work, so the region
        # stays unpinned and the decoy was never locked through.
        rec_register.assert_not_called()
        assert not region.is_pinned
    finally:
        os.close(region_fd)


def test_registration_lock_releases_fd_when_recheck_raises(monkeypatch, tmp_path):
    """The fd must close, releasing its lock, if the inode re-check raises.

    The re-check runs after the exclusive lock is taken. If its stat
    raises something other than FileNotFoundError, the exception must not
    leave the fd open: an open fd keeps the lock, and every later rank
    blocks until the leaking process exits."""
    reglock_path = tmp_path / "region.reglock"
    reglock_path.write_bytes(b"")
    path = str(reglock_path)

    real_stat = os.stat

    def stat_raises_on_reglock(p, *args, **kwargs):
        if os.fspath(p) == path:
            raise PermissionError(13, "Permission denied", os.fspath(p))
        return real_stat(p, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stat_raises_on_reglock)

    with pytest.raises(PermissionError):
        gpu_worker._open_and_lock_reglock(path)

    # The fd was closed before the exception propagated, so the exclusive
    # lock was released and a later rank takes it instead of blocking.
    probe = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(probe)
