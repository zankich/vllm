# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""pin_mmap_region must fail the boot on registration failure.

A failed cudaHostRegister poisons the CUDA context on affected drivers:
the next CUDA operation aborts with cudaErrorInvalidValue (observed
2026-09-08: rank 0's registration failed, the warning fired, and
the subsequent torch.arange in kernel warmup killed the worker). The
warn-and-continue fallback is therefore not survivable.

The multi-process test pins the flock serialization: two ranks racing
the registration path on the same region file must not overlap their
cudaHostRegister sections (the observed 3090 Ti TP2 failure mode).
"""

import os
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.kv_offload.cpu.gpu_worker import pin_mmap_region

REGISTER_HOLD_SECONDS = 0.5


def _make_region(fd=None):
    region = MagicMock()
    region.rank = 0
    region.total_size_bytes = 4096
    region.fd = fd
    base = MagicMock()
    base.data_ptr.return_value = 0x7F0000000000
    region._base = base
    region.is_pinned = False
    return region


def _cudart_result(value: int):
    return SimpleNamespace(value=value)


def test_failed_registration_raises_after_retries(monkeypatch):
    calls = []

    def always_fail(ptr, size, flags):
        calls.append((ptr, size, flags))
        return _cudart_result(1)

    cudart = SimpleNamespace(cudaHostRegister=always_fail)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)
    region = _make_region()

    with pytest.raises(RuntimeError, match="cudaHostRegister failed"):
        pin_mmap_region(region)
    assert len(calls) == 3
    assert not region.is_pinned


def test_transient_failure_recovers_within_retries(monkeypatch):
    results = [_cudart_result(1), _cudart_result(0)]

    def flaky(ptr, size, flags):
        return results.pop(0)

    cudart = SimpleNamespace(cudaHostRegister=flaky)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)
    region = _make_region()

    pin_mmap_region(region)
    assert region.is_pinned


def test_success_pins_without_warning(monkeypatch):
    def ok(ptr, size, flags):
        assert size == 4096
        return _cudart_result(0)

    cudart = SimpleNamespace(cudaHostRegister=ok)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)
    region = _make_region()

    pin_mmap_region(region)
    assert region.is_pinned


def test_registration_takes_flock_on_region_fd(monkeypatch, tmp_path):
    import fcntl
    import os

    def ok(ptr, size, flags):
        return _cudart_result(0)

    cudart = SimpleNamespace(cudaHostRegister=ok)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)

    lock_path = tmp_path / "region"
    lock_path.write_bytes(b"\0" * 4096)
    fd = os.open(lock_path, os.O_RDWR)
    try:
        region = _make_region(fd=fd)
        pin_mmap_region(region)
        assert region.is_pinned
        # After pinning returns, the fd must be unlocked: an exclusive flock
        # from this process (a new open of the same inode) must succeed.
        probe = os.open(lock_path, os.O_RDWR)
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(probe, fcntl.LOCK_UN)
        os.close(probe)
    finally:
        os.close(fd)


def _pin_region_child(region_path: str, rank: int, events, barrier) -> None:
    """Child rank: mock the registration, then race pin_mmap_region.

    The mocked cudaHostRegister announces ("enter", rank) / ("exit", rank)
    around a hold interval, so the parent can reconstruct the exact
    overlap structure of the two ranks' registration sections."""
    from unittest.mock import MagicMock

    import vllm.v1.kv_offload.cpu.gpu_worker as gpu_worker

    class _FakeCudaLikePlatform:
        device_name = "test"

        def is_cuda_alike(self):
            return True

    gpu_worker.current_platform = _FakeCudaLikePlatform()

    def register_with_hold(ptr, size, flags):
        events.put(("enter", rank))
        time.sleep(REGISTER_HOLD_SECONDS)
        events.put(("exit", rank))
        return _cudart_result(0)

    torch.cuda.cudart = lambda: SimpleNamespace(cudaHostRegister=register_with_hold)

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
    their cudaHostRegister sections exclusively: each enter is immediately
    followed by its own exit before the other rank enters. Without the
    flock the barrier-synchronized ranks overlap their sections."""
    from vllm.utils.system_utils import get_mp_context

    ctx = get_mp_context()
    events = ctx.Queue()
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

        seen = [events.get(timeout=5) for _ in range(4)]
        # Serialized: enter_r, exit_r, enter_s, exit_s with no interleaving.
        assert seen[0][0] == "enter"
        assert seen[1] == ("exit", seen[0][1])
        assert seen[2][0] == "enter"
        assert seen[2][1] != seen[0][1]
        assert seen[3] == ("exit", seen[2][1])
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(timeout=10)
