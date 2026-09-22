# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""pin_mmap_region takes an exclusive flock on the region fd across its
whole chunked registration, so concurrent ranks on the same shared mmap
serialize against one another and the host_register mock receives the
full sequence under the lock.
"""

import fcntl
import os
from unittest.mock import MagicMock

from vllm.v1.kv_offload.cpu import gpu_worker


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
    rec_register = MagicMock(return_value=True)
    rec_unregister = MagicMock()
    monkeypatch.setattr(gpu_worker, "host_register", rec_register)
    monkeypatch.setattr(gpu_worker, "host_unregister", rec_unregister)

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
