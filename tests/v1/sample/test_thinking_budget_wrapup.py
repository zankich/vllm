# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior tests for the thinking-budget wrap-up patch (multi-token
wrap-up sentence forced before ``</think>`` under MTP,
``VLLM_THINKING_WRAPUP_TOKEN_IDS``).

The two failure modes the module's own comments document:
- the monotonic counter must advance only on tokens that LANDED, never
  on drafted spec tokens the rejection sampler may discard;
- the force window pins the first diverging row plus the bonus row,
  never the whole window (pinning everything was measured worse than
  pinning nothing).
"""

import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor.interface import BatchUpdate
from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder


class _MockReasoningConfig:
    reasoning_start_token_ids = [151667]
    reasoning_end_token_ids = [151668]


WRAPUP_IDS = [9001, 9002]
END_ID = 151668
START_ID = 151667
NUM_SPEC_TOKENS = 2


def _make_holder(monkeypatch):
    monkeypatch.setenv(
        "VLLM_THINKING_WRAPUP_TOKEN_IDS", ",".join(str(t) for t in WRAPUP_IDS)
    )
    holder = ThinkingBudgetStateHolder(
        _MockReasoningConfig(),
        8,
        NUM_SPEC_TOKENS,
        torch.device("cpu"),
        False,
    )
    holder.sync_batch(
        BatchUpdate(
            batch_size=1,
            removed=(),
            added=[(0, SamplingParams(thinking_token_budget=3), None, [])],
            moved=(),
        )
    )
    return holder


def _step(holder: ThinkingBudgetStateHolder, output: list[int], spec: list[int]):
    holder.update_state([output], [spec])


def _drive_into_end_mode(holder: ThinkingBudgetStateHolder):
    """Run a request past its thinking budget so the wrap-up forcing arms.

    Budget 3 minus the 2 wrap-up ids reserves to 1 (the phrase must fit
    inside the budget); one thinking token after <think> exhausts it and
    the state enters in_end with end_count 0."""
    _step(holder, [START_ID], [])
    _step(holder, [START_ID, 111], [])
    state = holder._state[0]
    assert state["in_end"] and state["end_count"] == 0
    return state


def test_wrapup_env_builds_phrase_before_end_and_reserves_budget(monkeypatch):
    """Lines 67-76: the wrap-up sentence is forced BEFORE the end token,
    and the budget is reduced by the wrap-up length so the phrase fits."""
    holder = _make_holder(monkeypatch)
    assert holder.force_token_ids == [*WRAPUP_IDS, END_ID]
    state = holder._state[0]
    assert state["thinking_token_budget"] == 1  # 3 - 2 wrap-up ids


def test_rejected_drafts_do_not_advance_wrapup_counter(monkeypatch):
    """Monotonic counter (lines ~461-479): only tokens that landed this
    step advance end_count. Drafted-but-rejected spec tokens must not:
    the counter running ahead of reality forces the wrong phrase token."""
    holder = _make_holder(monkeypatch)
    _drive_into_end_mode(holder)

    # The first forced phrase token (9001) lands; the window drafts the
    # continuation [9002, </think>] for the next verify.
    _step(holder, [START_ID, 111, 9001], [9002, END_ID])
    state = holder._state[0]
    assert state["end_count"] == 1  # advanced on the landed 9001 only

    # The verify REJECTS the drafted continuation: a different token
    # lands and the window is re-drafted. end_count must hold at 1, not
    # count the two drafted (rejected) tokens.
    _step(holder, [START_ID, 111, 9001, 888], [9002, END_ID])
    state = holder._state[0]
    assert state["end_count"] == 1


def test_force_window_pins_first_diverging_row_plus_bonus(monkeypatch):
    """Pinning (lines ~489-503): rows before the first divergence already
    hold drafts matching the continuation and stay unpinned; the first
    diverging row and every row after it are pinned, and the bonus row
    is addressed separately via bonus_force_offset."""
    holder = _make_holder(monkeypatch)
    _drive_into_end_mode(holder)
    _step(holder, [START_ID, 111, 9001], [9002, END_ID])
    state = holder._state[0]
    assert state["end_count"] == 1

    # Next window drafts [9002, 999]: row 0 matches the continuation
    # (force_token_ids[1]), row 1 diverges (999 != </think>).
    _step(holder, [START_ID, 111, 9001, 888], [9002, 999])
    state = holder._state[0]
    assert state["force_index"] == [1]  # row 0 unpinned, row 1 pinned
    assert state["bonus_force_offset"] == 2  # bonus row addressed apart

    # A fully-matching window pins nothing: the drafts already carry the
    # phrase and only the bonus row needs forcing.
    _step(holder, [START_ID, 111, 9001, 888, 777], [9002, END_ID])
    state = holder._state[0]
    assert state["force_index"] == []
    assert state["bonus_force_offset"] == 2
