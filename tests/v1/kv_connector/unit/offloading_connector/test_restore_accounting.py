# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Restore-accounting instrumentation for the OffloadingConnector.

Fork-local (2026-09-17): corruption incidents left every
storage tier byte-verified and the client usage frames arithmetic-only.
The remaining suspect is the restore ASSEMBLY: which token range the
connector claims as cached versus what the request's own prefix hashes
support. These tests pin the summary/violation logic that
update_state_after_alloc now logs per restore.
"""

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    restore_accounting_summary,
)


def _summary(**kw):
    base = dict(
        req_id="req-1",
        num_prompt_tokens=16_620,
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
    for needle in ("req-1", "prompt=16620", "local=1950", "ext=14400",
                   "boundary=16350", "keys=18", "chunk=800"):
        assert needle in line, line


def test_boundary_beyond_prompt_is_a_violation():
    # An over-claiming boundary shape.
    _, violations = _summary(num_external=15_000)
    assert any("boundary exceeds prompt" in v for v in violations)


def test_boundary_misaligned_to_chunk_is_not_flagged():
    # Boundary = GPU-prefix hits (16-token blocks) + offload chunks (800):
    # mixed granularity, alignment is NOT an invariant and the real r3
    # arithmetic (1950 + 14400 = 16350) is legitimately misaligned.
    line, violations = _summary()
    assert violations == []
    assert "boundary=16350" in line


def test_zero_external_tokens_reports_skipped():
    line, violations = _summary(num_external=0)
    assert "skipped" in line
    assert violations == []


def test_keys_loaded_below_restored_chunks_is_a_violation():
    # 14,400 restored tokens need >= 18 chunks of 800; 17 cannot cover it.
    _, violations = _summary(keys_loaded=17)
    assert any("keys cannot cover" in v for v in violations)


from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    store_accounting_summary,
)


def _store(**kw):
    base = dict(
        req_id="req-s",
        watermark=14_400,
        num_prompt_tokens=14_747,
        num_tokens=16_964,
        num_computed_tokens=14_747,
        hashed_blocks=900,
        block_size=16,
        is_finished=True,
    )
    base.update(kw)
    return store_accounting_summary(**base)


def test_store_line_carries_fields_and_is_clean_when_hashed_prefix_covers():
    line, violations = _store()
    assert violations == []
    for needle in ("req-s", "watermark=14400", "prompt=14747",
                   "hashed=14400", "finished=True"):
        assert needle in line, line


def test_store_beyond_hashed_prefix_is_a_violation():
    # Hashed prefix covers 8*16=128 tokens but the store reaches 14,400:
    # keys would address slots beyond committed content.
    _, violations = _store(hashed_blocks=899)
    assert any("beyond hashed prefix" in v for v in violations)


def test_store_watermark_above_computed_tokens_is_a_violation():
    _, violations = _store(num_computed_tokens=12_800)
    assert any("exceeds computed" in v for v in violations)
