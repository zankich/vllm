# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom all-reduce group policy: which process groups may build CustomAllreduce.

#54371 gated TP-only backends behind a 'tp' prefix so Engram (etp) groups
would not register IPC buffers. The exact-prefix match also excluded the
expert-parallel group, whose decode-time MoE combine all-reduces pay the
NCCL latency floor on PCIe-only boxes. This tree also admits single-node
'ep' groups, scoped to CustomAllreduce: symm mem, FlashInfer and AITER
stay TP-only, and the dispatch chain still falls through to PYNCCL on
any size or dtype gate, so large EP combines keep the ring path.
"""

import pytest
import torch

import vllm.envs as envs
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.parallel import ParallelConfig
from vllm.distributed.device_communicators import cuda_communicator as cc_mod


class _FakeGroup:
    """Not registered with torch.distributed, so the base communicator
    __init__ takes its stateless path and reads rank()/size() only."""

    def __init__(self, size: int):
        self._size = size

    def rank(self):
        return 0

    def size(self):
        return self._size


class _FakeComm:
    def __init__(self, *args, **kwargs):
        self.disabled = False
        self.world_size = 4

    def all_reduce(self, inp):
        return inp


class _FakePynccl:
    marker = object()

    def __init__(self, *args, **kwargs):
        self.disabled = False
        self.world_size = 4

    def all_reduce(self, inp):
        return _FakePynccl.marker


class _FakeCustomAllreduce:
    """Answers dispatch like the real CustomAllreduce on a four-GPU mesh
    that is fully connected through P2P rather than NVLink: should_custom_ar
    follows the stock rule, including the batch-invariant branch that
    accepts every size when VLLM_BATCH_INVARIANT is set."""

    def __init__(self, *args, **kwargs):
        self.disabled = False
        self.world_size = 4
        self.max_size = 8192 * 1024
        self.batch_invariant = envs.VLLM_BATCH_INVARIANT
        self.custom_calls = 0

    def should_custom_ar(self, inp):
        if inp.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            return False
        if inp.nbytes % 16 != 0:
            return False
        # world_size 4, fully connected via the P2P-widened probe
        return self.batch_invariant or inp.nbytes < self.max_size

    def custom_all_reduce(self, inp):
        self.custom_calls += 1
        return inp


_LAZY_COMMUNICATORS = (
    "custom_all_reduce.CustomAllreduce",
    "symm_mem.SymmMemCommunicator",
    "flashinfer_all_reduce.FlashInferAllReduce",
    "flashinfer_pcie_ipc_all_reduce.FlashInferPcieIpcAllReduce",
    "quick_all_reduce.QuickAllReduce",
    "aiter_custom_all_reduce.AiterCustomAllreduce",
)


def _build(monkeypatch, unique_name, ranks, world=None, ca_cls=_FakeComm):
    for name in _LAZY_COMMUNICATORS:
        module, cls = name.rsplit(".", 1)
        monkeypatch.setattr(
            f"vllm.distributed.device_communicators.{module}.{cls}", ca_cls
        )
    monkeypatch.setattr(
        "vllm.distributed.device_communicators.pynccl.PyNcclCommunicator",
        _FakePynccl,
    )
    return cc_mod.CudaCommunicator(
        cpu_group=_FakeGroup(len(ranks)),
        device=torch.device("cpu"),
        unique_name=unique_name,
        global_ranks=list(ranks),
        global_world_size=world if world is not None else len(ranks),
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("tp:0", True),
        ("tp:1", True),
        ("ep:0", True),  # EP combine all-reduces take the IPC path
        ("etp:0", False),  # #54371's exclusion stands
        ("etp:1", False),
        ("pp:0", False),
        ("dp:0", False),
        ("", False),
        ("tpep:0", False),  # no substring matching
    ],
)
def test_group_policy(monkeypatch, name, expected):
    comm = _build(monkeypatch, name, [0, 1, 2, 3])
    assert comm.use_custom_allreduce is expected


def test_tp_admission_unchanged(monkeypatch):
    comm = _build(monkeypatch, "tp:0", [0, 1, 2, 3])
    assert comm.use_custom_allreduce is True
    assert comm.ca_comm is not None


def test_single_node_ep_group_admitted(monkeypatch):
    with set_current_vllm_config(VllmConfig(parallel_config=ParallelConfig(nnodes=1))):
        comm = _build(monkeypatch, "ep:0", [0, 1, 2, 3])
    assert comm.use_custom_allreduce is True
    assert comm.ca_comm is not None


def test_multi_node_ep_group_rejected(monkeypatch):
    # dp=2, pp=2, tp=2 on two nodes: the EP group of pp stage 0 gathers
    # ranks {0, 1, 4, 5}, and each node holds ranks 0-3 then 4-7.
    with set_current_vllm_config(VllmConfig(parallel_config=ParallelConfig(nnodes=2))):
        comm = _build(monkeypatch, "ep:0", [0, 1, 4, 5], world=8)
    assert comm.use_custom_allreduce is False
    assert comm.ca_comm is None


def test_ep_admission_scoped_to_custom_allreduce(monkeypatch):
    # The symm-mem and FlashInfer envs default to on; the EP admission
    # must still keep those backends off.
    comm = _build(monkeypatch, "ep:0", [0, 1, 2, 3])
    assert comm.use_custom_allreduce is True
    assert comm.use_torch_symm_mem is False
    assert comm.use_flashinfer_allreduce is False
    assert comm.use_flashinfer_pcie_ipc_allreduce is False
    assert comm.use_aiter_allreduce is False
    assert comm.symm_mem_comm is None
    assert comm.fi_ar_comm is None
    assert comm.fi_pcie_ipc_ar_comm is None
    assert comm.aiter_ar_comm is None


def test_batch_invariant_ep_not_every_size(monkeypatch):
    # On a four-GPU PCIe mesh the fully-connected probe passes through
    # P2P, and under VLLM_BATCH_INVARIANT should_custom_ar alone would
    # accept every size; the communicator's EP size gate keeps the large
    # combines on the PYNCCL ring.
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    comm = _build(monkeypatch, "ep:0", [0, 1, 2, 3], ca_cls=_FakeCustomAllreduce)
    assert comm.ca_comm is not None
    small = torch.empty(16 * 1024, dtype=torch.bfloat16)
    large = torch.empty(comm.ca_comm.max_size * 2, dtype=torch.bfloat16)
    # the stock gate alone would take the large tensor through custom AR
    assert comm.ca_comm.should_custom_ar(large) is True
    assert comm.all_reduce(small) is small
    assert comm.ca_comm.custom_calls == 1
    assert comm.all_reduce(large) is _FakePynccl.marker
    assert comm.ca_comm.custom_calls == 1
