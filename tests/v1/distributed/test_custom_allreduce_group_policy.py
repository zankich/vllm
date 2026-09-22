# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom all-reduce group policy: which process groups may build CustomAllreduce.

#54371 gated TP-only backends behind a 'tp' prefix so Engram (etp) groups
would not register IPC buffers. The exact-prefix match also excluded the
expert-parallel group, whose decode-time MoE combine all-reduces pay the
NCCL latency floor on PCIe-only boxes. The fork allows 'ep' groups: the
dispatch chain still falls through to PYNCCL on any size/dtype gate, so
large EP combines keep the ring path while small ones take IPC.
"""

import pytest

from vllm.distributed.device_communicators.cuda_communicator import (
    group_allows_custom_allreduce,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("tp:0", True),
        ("tp:1", True),
        ("ep:0", True),  # fork: EP combine all-reduces take the IPC path
        ("etp:0", False),  # #54371's exclusion stands
        ("etp:1", False),
        ("pp:0", False),
        ("dp:0", False),
        ("", False),
        ("tpep:0", False),  # no substring matching
    ],
)
def test_group_policy(name, expected):
    assert group_allows_custom_allreduce(name) is expected
