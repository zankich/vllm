# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""pin_mmap_region must fail the boot on registration failure.

A failed cudaHostRegister poisons the CUDA context on affected drivers:
the next CUDA operation aborts with cudaErrorInvalidValue (observed on
otto 2026-09-08: rank 0's registration failed, the warning fired, and
the subsequent torch.arange in kernel warmup killed the worker). The
warn-and-continue fallback is therefore not survivable.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.kv_offload.cpu.gpu_worker import pin_mmap_region


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
