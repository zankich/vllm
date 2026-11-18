# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P2P-aware fully-connected probe for the custom all-reduce path.

Stock is_fully_connected only accepts NVLink between every pair, so
PCIe-only boxes with working P2P across all GPU pairs (nvidia-smi topo
-p2p OK on every pair) are rejected from the CUSTOM all-reduce path at
world_size > 2, a path the hardware supports (one-shot IPC over P2P),
leaving PYNCCL's per-hop latency floor on decode's small messages. This
tree accepts generic P2P when NVLink is absent: the custom kernel's
requirement is direct peer access, not NVLink specifically.
"""

import pytest


def _probe(monkeypatch, nvlink_ok, read_ok, write_ok):
    """Mock one P2P status answer per caps index, the way the probe fans
    out: NVLink first, then READ and WRITE separately. The mesh needs
    READ and WRITE both, so each caps index gets its own answer."""
    from vllm.platforms import cuda as cuda_mod

    nvml = cuda_mod.pynvml
    result = {
        nvml.NVML_P2P_CAPS_INDEX_NVLINK: 0 if nvlink_ok else 1,
        nvml.NVML_P2P_CAPS_INDEX_READ: 0 if read_ok else 1,
        nvml.NVML_P2P_CAPS_INDEX_WRITE: 0 if write_ok else 1,
    }

    def fake_status(handle, peer, caps_index):
        return result[caps_index]

    monkeypatch.setattr(nvml, "NVML_P2P_STATUS_OK", 0, raising=False)
    monkeypatch.setattr(
        nvml, "nvmlDeviceGetHandleByIndex", lambda i: f"handle{i}", raising=False
    )
    monkeypatch.setattr(nvml, "nvmlDeviceGetP2PStatus", fake_status, raising=False)
    # Bypass @with_nvml_context's process-global NVML init. The decorator
    # captured the wrapper at class-definition time, so patching
    # cuda_mod.with_nvml_context now would be a no-op and the real
    # pynvml.nvmlInit() would run; re-bind the method to the undecorated
    # function (@wraps left it on __wrapped__) instead.
    undecorated = cuda_mod.NvmlCudaPlatform.is_fully_connected.__func__.__wrapped__
    monkeypatch.setattr(
        cuda_mod.NvmlCudaPlatform,
        "is_fully_connected",
        classmethod(undecorated),
        raising=False,
    )
    monkeypatch.setattr(
        cuda_mod.NvmlCudaPlatform,
        "device_id_to_physical_device_id",
        classmethod(lambda cls, i: i),
        raising=False,
    )
    return cuda_mod.NvmlCudaPlatform.is_fully_connected([0, 1, 2, 3])


def test_probe_answers_without_real_nvml_init(monkeypatch):
    # The probe must not depend on the host's NVML: patching
    # cuda_mod.with_nvml_context after is_fully_connected was decorated
    # is a no-op (the decorator is bound at class-definition time), so a
    # broken patch silently runs pynvml.nvmlInit() for real and the suite
    # only passes on machines with a healthy NVML. If init raises here,
    # the probe must still answer from the mocked P2P statuses.
    from vllm.platforms import cuda as cuda_mod
    from vllm.third_party import pynvml as real_pynvml

    def broken_init():
        raise real_pynvml.NVMLError(real_pynvml.NVML_ERROR_UNINITIALIZED)

    monkeypatch.setattr(cuda_mod.pynvml, "nvmlInit", broken_init, raising=False)
    assert _probe(monkeypatch, nvlink_ok=False, read_ok=True, write_ok=True) is True


def test_p2p_caps_constants_are_ints():
    # nvmlDeviceGetP2PStatus's ctypes binding rejects non-int caps indexes;
    # the vendored READ constant is a trailing-comma tuple, and passing it
    # through crashes worker boot with a ctypes ArgumentError (argument 3).
    from vllm.third_party import pynvml

    assert isinstance(pynvml.NVML_P2P_CAPS_INDEX_READ, int)
    assert isinstance(pynvml.NVML_P2P_CAPS_INDEX_WRITE, int)
    assert isinstance(pynvml.NVML_P2P_CAPS_INDEX_NVLINK, int)


def test_p2p_mesh_accepted_without_nvlink(monkeypatch):
    # No NVLink anywhere, READ and WRITE OK on every pair: fully connected
    # for the IPC all-reduce path.
    assert _probe(monkeypatch, nvlink_ok=False, read_ok=True, write_ok=True) is True


def test_no_p2p_rejected(monkeypatch):
    # Neither NVLink nor P2P: not fully connected, same as stock.
    assert _probe(monkeypatch, nvlink_ok=False, read_ok=False, write_ok=False) is False


@pytest.mark.parametrize(
    "read_ok,write_ok",
    [(True, False), (False, True)],
)
def test_half_p2p_mesh_rejected(monkeypatch, read_ok, write_ok):
    # The probe requires direct peer access in both directions: a mesh
    # that answers OK on READ but not WRITE (or the reverse) is not
    # fully connected for the one-shot IPC all-reduce.
    assert (
        _probe(monkeypatch, nvlink_ok=False, read_ok=read_ok, write_ok=write_ok)
        is False
    )


def test_nvlink_fast_path(monkeypatch):
    # NVLink on every pair: fully connected without consulting P2P.
    assert _probe(monkeypatch, nvlink_ok=True, read_ok=False, write_ok=False) is True
