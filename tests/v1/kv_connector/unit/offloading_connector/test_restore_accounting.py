# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Restore-accounting instrumentation for the OffloadingConnector.

One log line per offload restore carries the restore-assembly arithmetic,
plus loud violations when the assembly over-claims or under-covers. The
byte-verified tiers cannot catch a wrong SET of correctly-hashed blocks;
these invariants can. These tests pin the summary/violation logic that
update_state_after_alloc logs per restore.
"""

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    restore_accounting_summary,
)


def _summary(**kw):
    base = dict(
        req_id="req-1",
        num_prompt_tokens=16_620,
        num_request_tokens=16_620,
        num_locally_computed=1_950,
        num_external=14_400,
        keys_loaded=18,
        group_tokens_per_chunk=800,
    )
    base.update(kw)
    return restore_accounting_summary(**base)


def test_summary_line_carries_the_correlation_fields():
    line, violations = _summary()
    assert violations == []
    for needle in (
        "req-1",
        "prompt=16620",
        "local=1950",
        "ext=14400",
        "boundary=16350",
        "keys=18",
        "chunk=800",
    ):
        assert needle in line, line


def test_boundary_beyond_request_tokens_is_a_violation():
    # An over-claiming boundary shape.
    _, violations = _summary(num_external=15_000)
    assert any("boundary exceeds request" in v for v in violations)


def test_boundary_covering_generated_tokens_on_resume_is_clean():
    # A recompute-preempted request keeps its sampled output tokens, so a
    # restore may legitimately cover tokens past the prompt boundary: the
    # bound is the request's total length, not the prompt.
    line, violations = _summary(num_request_tokens=17_000, num_external=15_000)
    assert violations == []
    assert "boundary=16950" in line


def test_boundary_misaligned_to_chunk_is_not_flagged():
    # Boundary = GPU-prefix hits (16-token blocks) + offload chunks (800):
    # mixed granularity, alignment is NOT an invariant and the arithmetic
    # (1950 + 14400 = 16350) is legitimately misaligned.
    line, violations = _summary()
    assert violations == []
    assert "boundary=16350" in line


def test_zero_external_tokens_reports_skipped():
    line, violations = _summary(num_external=0)
    assert "skipped" in line
    assert violations == []


def test_keys_loaded_below_restored_chunks_is_a_violation():
    # Non-window groups' capacity must cover the restored extent: 14,400
    # tokens of capacity against 14,400 restored is clean; less is not.
    line, violations = _summary(full_group_capacity=14_400, has_full_group=True)
    assert violations == []
    _, violations = _summary(full_group_capacity=12_624, has_full_group=True)
    assert any("cannot cover ext" in v for v in violations)


def test_window_groups_do_not_count_toward_coverage_capacity():
    # Mixed-chunk geometry: five window groups at chunk 16 load only
    # their sliding windows (64 keys each), one full group at chunk 32
    # covers the whole extent. Total keys x group-0 chunk undercounts;
    # the invariant must hold anyway.
    line, violations = _summary(
        keys_loaded=883,
        full_group_capacity=18_016,
        has_full_group=True,
        group_detail="16:64s1062,16:64s1062,32:563s0",
    )
    assert violations == []
    assert "groups=16:64s1062,16:64s1062,32:563s0" in line


def test_no_full_group_disables_the_capacity_invariant():
    # All-window geometry: no group owes the full extent, so the
    # capacity invariant is vacuous rather than violated.
    _, violations = _summary(full_group_capacity=0, has_full_group=False)
    assert violations == []


def test_group_detail_field_carries_per_group_chunk_and_key_counts():
    line, _ = _summary(group_detail="16:938s0,16:64s874")
    assert "groups=16:938s0,16:64s874" in line
