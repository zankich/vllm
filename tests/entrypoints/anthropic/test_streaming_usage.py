# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for Anthropic ``/v1/messages`` streaming usage.

Covers:

- Bug 1: ``message_start`` and ``message_delta`` carried mutually
  inconsistent usage because the OpenAI chat streaming generator only
  attached ``prompt_tokens_details`` to the terminal chunk, so
  ``message_start`` reported ``input_tokens = prompt`` with no cache
  fields while ``message_delta`` reported the split usage. Under the
  Anthropic SDK's field-wise merge, ``prompt == cached + created``
  produced ``input_tokens + cache_read + cache_creation == 2 * prompt``.
  The fix plumbs ``prompt_tokens_details`` onto every continuous usage
  chunk in the OpenAI layer so both Anthropic events carry the same
  split usage.
- Bug 2: ``ChatCompletionStreamResponse.model_validate_json`` raised a
  raw pydantic ``ValidationError`` when EngineCore emitted a payload
  that was not a valid chunk (e.g. ``{"error": ...}``). The fix
  inspects the payload and, if it is a dict with an ``error`` object,
  emits an Anthropic-shaped ``event: error`` carrying the engine's
  original message and a mapped ``type``, then stops the generator.
  Any other malformed input (non-JSON, valid JSON without ``error``)
  falls through to the existing outer ``except Exception`` handler and
  surfaces as ``type=internal_error``.
"""

import pytest

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.entrypoints.serve.engine.protocol import (
    PromptTokenUsageInfo,
    UsageInfo,
)

from .test_anthropic_messages_conversion import (
    _make_stream_chunk as make_stream_chunk,
)
from .test_anthropic_messages_conversion import (
    _make_stream_converter as make_stream_converter,
)
from .test_anthropic_messages_conversion import (
    _parse_sse_events as parse_sse_events,
)


async def _collect(converter, sse_input):
    output = []
    async for event in converter.message_stream_converter(sse_input()):
        output.append(event)
    return parse_sse_events(output)


def _usage_of(event_tuple):
    """Return the ``usage`` dict for a ``message_start`` or
    ``message_delta`` event tuple."""
    _, data = event_tuple
    if data.get("type") == "message_start":
        return data["message"]["usage"]
    return data["usage"]


class TestStreamingUsageBug1RemainderZero:
    """Contract pin: with the fix the converter sees the same fully-split
    usage on both events. See
    ``TestStreamingUsageBug1ComposedPath.test_real_generator_into_real_converter_double_count_guard``
    for the composed-path regression guard."""

    @pytest.mark.asyncio
    async def test_message_start_and_delta_carry_same_split_usage(self):
        async def sse_input():
            # First chunk: role delta with prompt_tokens_details populated
            # by the fixed OpenAI layer.
            yield make_stream_chunk(
                delta=DeltaMessage(role="assistant"),
                usage=UsageInfo(
                    prompt_tokens=65488,
                    total_tokens=65488,
                    prompt_tokens_details=PromptTokenUsageInfo(
                        cached_tokens=63472,
                        created_cache_tokens=2016,
                    ),
                ),
            )
            # Intermediate content delta.
            yield make_stream_chunk(delta=DeltaMessage(content="hi"))
            # Final chunk (empty choices + cumulative usage).
            yield make_stream_chunk(
                choices=[],
                usage=UsageInfo(
                    prompt_tokens=65488,
                    completion_tokens=4,
                    total_tokens=65492,
                    prompt_tokens_details=PromptTokenUsageInfo(
                        cached_tokens=63472,
                        created_cache_tokens=2016,
                    ),
                ),
            )
            yield "data: [DONE]"

        events = await _collect(make_stream_converter(), sse_input)

        starts = [ev for ev in events if ev[0] == "message_start"]
        deltas = [ev for ev in events if ev[0] == "message_delta"]
        assert len(starts) == 1
        assert len(deltas) == 1

        start_usage = _usage_of(starts[0])
        delta_usage = _usage_of(deltas[0])

        # Both events carry the same fully-split usage.
        assert start_usage["input_tokens"] == 0
        assert start_usage["cache_read_input_tokens"] == 63472
        assert start_usage["cache_creation_input_tokens"] == 2016
        assert delta_usage["input_tokens"] == 0
        assert delta_usage["cache_read_input_tokens"] == 63472
        assert delta_usage["cache_creation_input_tokens"] == 2016

        # Field-wise merge must reconstruct the original prompt total, not
        # double it. Both events agree, so either is the right invariant.
        merged = (
            start_usage["input_tokens"]
            + start_usage["cache_read_input_tokens"]
            + start_usage["cache_creation_input_tokens"]
        )
        assert merged == 65488
        assert (
            start_usage["input_tokens"]
            + start_usage["cache_read_input_tokens"]
            + start_usage["cache_creation_input_tokens"]
            == delta_usage["input_tokens"]
            + delta_usage["cache_read_input_tokens"]
            + delta_usage["cache_creation_input_tokens"]
        )

        # Ordering preserved: message_start first, content events,
        # message_delta, message_stop.
        event_order = [ev[0] for ev in events]
        assert event_order[0] == "message_start"
        assert event_order[-1] == "message_stop"
        assert event_order.index("message_delta") < event_order.index("message_stop")


class TestStreamingUsageBug1RemainderPositive:
    """Contract pin: same split usage on both events even when the
    prompt total does not equal cached + created. See
    ``TestStreamingUsageBug1ComposedPath.test_real_generator_into_real_converter_double_count_guard``
    for the composed-path regression guard."""

    @pytest.mark.asyncio
    async def test_message_start_and_delta_carry_same_split_usage(self):
        async def sse_input():
            yield make_stream_chunk(
                delta=DeltaMessage(role="assistant"),
                usage=UsageInfo(
                    prompt_tokens=75000,
                    total_tokens=75000,
                    prompt_tokens_details=PromptTokenUsageInfo(
                        cached_tokens=44000,
                        created_cache_tokens=30191,
                    ),
                ),
            )
            yield make_stream_chunk(delta=DeltaMessage(content="hi"))
            yield make_stream_chunk(
                choices=[],
                usage=UsageInfo(
                    prompt_tokens=75000,
                    completion_tokens=3,
                    total_tokens=75003,
                    prompt_tokens_details=PromptTokenUsageInfo(
                        cached_tokens=44000,
                        created_cache_tokens=30191,
                    ),
                ),
            )
            yield "data: [DONE]"

        events = await _collect(make_stream_converter(), sse_input)

        starts = [ev for ev in events if ev[0] == "message_start"]
        deltas = [ev for ev in events if ev[0] == "message_delta"]
        assert len(starts) == 1
        assert len(deltas) == 1

        start_usage = _usage_of(starts[0])
        delta_usage = _usage_of(deltas[0])

        assert start_usage["input_tokens"] == 809  # 75000 - 44000 - 30191
        assert start_usage["cache_read_input_tokens"] == 44000
        assert start_usage["cache_creation_input_tokens"] == 30191
        assert delta_usage["input_tokens"] == 809
        assert delta_usage["cache_read_input_tokens"] == 44000
        assert delta_usage["cache_creation_input_tokens"] == 30191

        merged = (
            start_usage["input_tokens"]
            + start_usage["cache_read_input_tokens"]
            + start_usage["cache_creation_input_tokens"]
        )
        assert merged == 75000


class TestStreamingUsageNoDetails:
    """Contract pin: with ``--enable-prompt-tokens-details off``, no
    chunk carries cache info so the field-wise merge reconstructs the
    prompt exactly. See
    ``TestStreamingUsageBug1ComposedPath.test_real_generator_into_real_converter_double_count_guard``
    for the composed-path regression guard."""

    @pytest.mark.asyncio
    async def test_no_cache_fields_present(self):
        async def sse_input():
            yield make_stream_chunk(
                delta=DeltaMessage(role="assistant"),
                usage=UsageInfo(prompt_tokens=512, total_tokens=512),
            )
            yield make_stream_chunk(
                choices=[],
                usage=UsageInfo(
                    prompt_tokens=512,
                    completion_tokens=8,
                    total_tokens=520,
                ),
            )
            yield "data: [DONE]"

        events = await _collect(make_stream_converter(), sse_input)

        starts = [ev for ev in events if ev[0] == "message_start"]
        deltas = [ev for ev in events if ev[0] == "message_delta"]
        assert len(starts) == 1
        assert len(deltas) == 1

        start_usage = _usage_of(starts[0])
        delta_usage = _usage_of(deltas[0])

        # input_tokens equals the whole prompt; no cache fields at all.
        assert start_usage["input_tokens"] == 512
        assert delta_usage["input_tokens"] == 512
        for usage in (start_usage, delta_usage):
            assert "cache_read_input_tokens" not in usage
            assert "cache_creation_input_tokens" not in usage


class TestStreamingUsageErrorChunk:
    """Bug 2: an EngineCore error payload (not a valid
    ``ChatCompletionStreamResponse``) used to leak a pydantic
    ``ValidationError`` to the client. The fix inspects the payload
    and, if it is a dict with an ``error`` object, emits an Anthropic
    ``event: error`` carrying the engine's original message and a
    mapped ``type``, then stops the generator. Any other malformed
    input (non-JSON, valid JSON without ``error``) falls through to the
    outer handler's ``internal_error`` event."""

    @pytest.mark.asyncio
    async def test_engine_error_chunk_emits_anthropic_error_event(self):
        async def sse_input():
            yield (
                'data: {"error": {"message": '
                '"EngineCore encountered an issue", "code": 500}}'
            )

        raw_output = []
        converter = make_stream_converter()
        async for event in converter.message_stream_converter(sse_input()):
            raw_output.append(event)

        events = parse_sse_events(raw_output)

        # Exactly one error event carrying the engine's original message
        # verbatim and the mapped Anthropic error type.
        error_events = [ev for ev in events if ev[0] == "error"]
        assert len(error_events) == 1
        err = error_events[0][1]["error"]
        assert err["type"] == "api_error"
        assert err["message"] == "EngineCore encountered an issue"

        # No pydantic validation text anywhere in the emitted SSE bytes.
        joined = "\n".join(raw_output)
        assert "ValidationError" not in joined
        assert "validation error" not in joined.lower()

        # message_start was never emitted (the generator stops at the
        # first error chunk).
        assert not any(ev[0] == "message_start" for ev in events)

    @pytest.mark.asyncio
    async def test_engine_error_503_maps_to_overloaded(self):
        """vLLM raises 503 for retry-worthy overloads
        (``QueueOverflowError``, ``MaxQueuedTokensError``); the Anthropic
        layer must surface those as ``overloaded_error`` so clients retry."""

        async def sse_input():
            yield ('data: {"error": {"message": "queue full", "code": 503}}')

        raw_output = []
        converter = make_stream_converter()
        async for event in converter.message_stream_converter(sse_input()):
            raw_output.append(event)

        events = parse_sse_events(raw_output)
        error_events = [ev for ev in events if ev[0] == "error"]
        assert len(error_events) == 1
        err = error_events[0][1]["error"]
        assert err["type"] == "overloaded_error"
        assert err["message"] == "queue full"

    @pytest.mark.asyncio
    async def test_malformed_chunk_emits_internal_error(self):
        """A non-error malformed chunk does not leak a pydantic
        ``ValidationError`` to the caller; the outer converter handler
        converts it into an ``event: error`` with ``type=internal_error``
        and the stream ends without raising."""

        async def sse_input():
            # Valid JSON but missing required fields and no "error" key.
            yield 'data: {"foo": 1}'

        raw_output = []
        converter = make_stream_converter()
        async for event in converter.message_stream_converter(sse_input()):
            raw_output.append(event)

        events = parse_sse_events(raw_output)
        error_events = [ev for ev in events if ev[0] == "error"]
        assert len(error_events) == 1
        assert error_events[0][1]["error"]["type"] == "internal_error"

    @pytest.mark.asyncio
    async def test_non_json_chunk_emits_internal_error(self):
        """A non-JSON malformed chunk also falls through to the outer
        handler and surfaces as an ``event: error`` with
        ``type=internal_error``."""

        async def sse_input():
            yield "data: this is not json"

        raw_output = []
        converter = make_stream_converter()
        async for event in converter.message_stream_converter(sse_input()):
            raw_output.append(event)

        events = parse_sse_events(raw_output)
        error_events = [ev for ev in events if ev[0] == "error"]
        assert len(error_events) == 1
        assert error_events[0][1]["error"]["type"] == "internal_error"

    @pytest.mark.asyncio
    async def test_engine_error_chunk_after_content_blocks_flushes_block(self):
        """Bug 2, mid-stream shape: an EngineCore error payload arriving
        *after* ``message_start`` and an open text block must still
        flush the block, surface ``type=api_error`` with the engine
        message verbatim, and emit no ``message_delta`` or ``message_stop``.

        Pre-fix the error chunk fails ``model_validate_json`` and the
        unhandled ``ValidationError`` propagates to the outer handler,
        which emits ``type=internal_error`` and skips the block flush:
        no ``content_block_stop`` before the error and the wrong type.
        """

        async def sse_input():
            # message_start: role chunk with usage.
            yield make_stream_chunk(
                delta=DeltaMessage(role="assistant"),
                usage=UsageInfo(prompt_tokens=10, total_tokens=10),
            )
            # Open a text block + first text delta.
            yield make_stream_chunk(delta=DeltaMessage(content="hel"))
            # Mid-stream engine error.
            yield ('data: {"error": {"message": "engine aborted", "code": 500}}')

        raw_output = []
        converter = make_stream_converter()
        async for event in converter.message_stream_converter(sse_input()):
            raw_output.append(event)

        events = parse_sse_events(raw_output)
        event_names = [ev[0] for ev in events]

        # Flush the open text block before the error, then the error.
        assert event_names[0] == "message_start"
        assert "content_block_start" in event_names
        assert "content_block_delta" in event_names
        # The flush must happen *before* the error and the error must
        # be the last event (the generator stops).
        assert event_names[-1] == "error"
        cbs_idx = event_names.index("content_block_stop")
        err_idx = event_names.index("error")
        assert cbs_idx < err_idx, (
            "content_block_stop must be emitted before the error chunk "
            "closes the stream"
        )

        # No message_delta and no message_stop.
        assert "message_delta" not in event_names
        assert "message_stop" not in event_names

        # Error event carries api_error and the engine message verbatim.
        error_event = next(ev for ev in events if ev[0] == "error")
        assert error_event[1]["error"]["type"] == "api_error"
        assert error_event[1]["error"]["message"] == "engine aborted"


class TestStreamingUsageBug1ComposedPath:
    """Composed-path regression guard: drive the *real* OpenAI chat
    streaming generator through the *real* Anthropic
    ``message_stream_converter``. The pre-fix code only attached
    ``prompt_tokens_details`` to the terminal OpenAI chunk, so the
    Anthropic ``message_start`` reported the prompt total with no cache
    fields while ``message_delta`` carried the split; the SDK's
    field-wise merge then summed ``input_tokens + cache_read +
    cache_creation == 2 * prompt``.

    Pair: ``test_streaming_continuous_usage_chunks_carry_prompt_tokens_details``
    in ``tests/entrypoints/openai/chat_completion/test_serving_chat.py``
    covers the OpenAI half. This test covers the composition."""

    @pytest.mark.asyncio
    async def test_real_generator_into_real_converter_double_count_guard(self):
        # Imported lazily to keep the lighter streaming tests free of
        # the heavy chat-serving imports.
        from unittest.mock import MagicMock

        from tests.entrypoints.openai.chat_completion.test_serving_chat import (
            _build_minimal_metrics_serving_chat,
            _stream_request_outputs,
        )
        from vllm.entrypoints.generate.base.protocol import (
            RequestResponseMetadata,
        )
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )
        from vllm.outputs import CompletionOutput, RequestOutput

        serving = _build_minimal_metrics_serving_chat(enable_per_request_metrics=False)
        serving.enable_prompt_tokens_details = True

        # prompt == cached + created == 80 + 16 == 96. The remainder-zero
        # case is where the SDK field-wise merge sums to 2 * prompt in
        # the pre-fix code.
        first = RequestOutput(
            request_id="test-id",
            prompt="Test prompt",
            prompt_token_ids=list(range(96)),
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="hi",
                    token_ids=[100, 101],
                    cumulative_logprob=0.0,
                    logprobs=None,
                    finish_reason=None,
                )
            ],
            finished=False,
            num_cached_tokens=80,
            num_cache_creation_tokens=16,
        )
        final = RequestOutput(
            request_id="test-id",
            prompt="Test prompt",
            prompt_token_ids=list(range(96)),
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="",
                    token_ids=[],
                    cumulative_logprob=0.0,
                    logprobs=None,
                    finish_reason="stop",
                )
            ],
            finished=True,
            num_cached_tokens=80,
            num_cache_creation_tokens=16,
        )

        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Test prompt"}],
            max_tokens=10,
            stream=True,
            stream_options={
                "include_usage": True,
                "continuous_usage_stats": True,
            },
        )

        # Collect raw SSE bytes from the real OpenAI generator.
        async def sse_input():
            async for line in serving.chat_completion_stream_generator(
                request,
                _stream_request_outputs(first, final),
                "chatcmpl-test-id",
                "test-model",
                conversation=[{"role": "user", "content": "Test"}],
                tokenizer=MagicMock(),
                request_metadata=RequestResponseMetadata(request_id="chatcmpl-test-id"),
            ):
                yield line

        # Pipe them through the real Anthropic converter.
        raw_output = []
        converter = make_stream_converter()
        async for event in converter.message_stream_converter(sse_input()):
            raw_output.append(event)

        events = parse_sse_events(raw_output)
        event_names = [ev[0] for ev in events]

        # message_start must be the first event.
        assert event_names[0] == "message_start"

        starts = [ev for ev in events if ev[0] == "message_start"]
        deltas = [ev for ev in events if ev[0] == "message_delta"]
        assert len(starts) == 1
        assert len(deltas) == 1

        start_usage = _usage_of(starts[0])
        delta_usage = _usage_of(deltas[0])

        # Both events carry the same split usage on the prompt side.
        # ``output_tokens`` is the cumulative completion token count and
        # is allowed to grow across events; only the prompt-side fields
        # gate the merge invariant.
        prompt_keys = (
            "input_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        )
        for k in prompt_keys:
            assert start_usage[k] == delta_usage[k], (
                f"prompt-side field {k!r} diverged between message_start "
                f"({start_usage[k]}) and message_delta ({delta_usage[k]})"
            )

        # Field-wise merge where delta wins only with non-zero values:
        # input + read + creation must equal prompt (96), not 2 * prompt.
        def _nonzero_merge(a, b):
            return {
                k: b.get(k) if b.get(k) else a.get(k, 0)
                for k in (
                    "input_tokens",
                    "cache_read_input_tokens",
                    "cache_creation_input_tokens",
                )
            }

        merged = _nonzero_merge(start_usage, delta_usage)
        assert (
            merged["input_tokens"]
            + merged["cache_read_input_tokens"]
            + merged["cache_creation_input_tokens"]
            == 96
        ), f"non-zero merge must reconstruct prompt total; got {merged}"
