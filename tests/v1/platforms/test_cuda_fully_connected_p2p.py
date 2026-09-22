# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P2P-aware fully-connected probe for the custom all-reduce path.

Stock is_fully_connected only accepts NVLink between every pair, so
PCIe-only boxes with working P2P across all GPUs (4x 3090 Ti on this
fleet, nvidia-smi topo -p2p OK on every pair) are rejected from the
CUSTOM all-reduce path at world_size > 2 — a path the hardware supports
(one-shot IPC over P2P), leaving PYNCCL's per-hop latency floor on
decode's small messages. The fork accepts generic P2P when NVLink is
absent: the custom kernel's requirement is direct peer access, not
NVLink specifically.
"""

_P2P_INDEX = 999  # sentinel distinguishing the generic-P2P query


def _probe(monkeypatch, nvlink_ok, p2p_ok):
    from vllm.platforms import cuda as cuda_mod

    nvml = cuda_mod.pynvml
    result = {
        nvml.NVML_P2P_CAPS_INDEX_NVLINK: 0 if nvlink_ok else 1,
        _P2P_INDEX: 0 if p2p_ok else 1,
    }

    def fake_status(handle, peer, caps_index):
        if caps_index != nvml.NVML_P2P_CAPS_INDEX_NVLINK:
            caps_index = _P2P_INDEX
        return result[caps_index]

    monkeypatch.setattr(nvml, "NVML_P2P_STATUS_OK", 0, raising=False)
    monkeypatch.setattr(
        nvml, "nvmlDeviceGetHandleByIndex", lambda i: f"handle{i}", raising=False
    )
    monkeypatch.setattr(nvml, "nvmlDeviceGetP2PStatus", fake_status, raising=False)
    # bypass @with_nvml_context's process-global NVML init
    monkeypatch.setattr(
        cuda_mod,
        "with_nvml_context",
        lambda fn: classmethod(lambda cls, ids: fn(cls, ids)),
        raising=False,
    )
    monkeypatch.setattr(
        cuda_mod.NvmlCudaPlatform,
        "device_id_to_physical_device_id",
        classmethod(lambda cls, i: i),
        raising=False,
    )
    return cuda_mod.NvmlCudaPlatform.is_fully_connected([0, 1, 2, 3])


def test_p2p_caps_constants_are_ints():
    # nvmlDeviceGetP2PStatus's ctypes binding rejects non-int caps indexes;
    # the vendored READ constant was a trailing-comma tuple and crashed
    # worker boot when the fork's P2P probe first passed it through
    # (observed 2026-09-20, ArgumentError argument 3).
    from vllm.third_party import pynvml

    assert isinstance(pynvml.NVML_P2P_CAPS_INDEX_READ, int)
    assert isinstance(pynvml.NVML_P2P_CAPS_INDEX_WRITE, int)
    assert isinstance(pynvml.NVML_P2P_CAPS_INDEX_NVLINK, int)


def test_p2p_mesh_accepted_without_nvlink(monkeypatch):
    # The fork's change: no NVLink anywhere, generic P2P OK on every pair
    # → fully connected for the IPC all-reduce path.
    assert _probe(monkeypatch, nvlink_ok=False, p2p_ok=True) is True


def test_no_p2p_rejected(monkeypatch):
    # Neither NVLink nor P2P → not fully connected, same as stock.
    assert _probe(monkeypatch, nvlink_ok=False, p2p_ok=False) is False


def test_nvlink_fast_path(monkeypatch):
    # NVLink on every pair → fully connected without consulting P2P.
    assert _probe(monkeypatch, nvlink_ok=True, p2p_ok=True) is True
