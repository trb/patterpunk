from types import SimpleNamespace

from patterpunk.llm.models.openai_compatible import OpenAiCompatibleModel
from patterpunk.llm.streaming import StreamEventType


def make_model():
    return OpenAiCompatibleModel(
        model="llama-3.3-70b", base_url="http://localhost:8000/v1"
    )


def delta_chunk(
    content=None, reasoning_content=None, tool_calls=None, finish_reason=None
):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    if reasoning_content is not None:
        delta.reasoning_content = reasoning_content
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)],
        usage=None,
    )


def usage_chunk(prompt_tokens, completion_tokens):
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ),
    )


def tool_frame(index, call_id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


async def fake_stream(chunks):
    for chunk in chunks:
        yield chunk


async def collect(model, chunks):
    return [chunk async for chunk in model._iterate_stream(fake_stream(chunks))]


async def test_text_stream_with_usage():
    model = make_model()
    events = await collect(
        model,
        [
            delta_chunk(content="Hel"),
            delta_chunk(content="lo"),
            delta_chunk(finish_reason="stop"),
            usage_chunk(10, 5),
        ],
    )
    event_types = [event.event_type for event in events]
    assert event_types == [
        StreamEventType.CONTENT_BLOCK_START,
        StreamEventType.TEXT_DELTA,
        StreamEventType.TEXT_DELTA,
        StreamEventType.CONTENT_BLOCK_STOP,
        StreamEventType.MESSAGE_END,
    ]
    assert events[0].block_type == "text"
    assert [event.text for event in events[1:3]] == ["Hel", "lo"]
    assert events[-1].usage == {"input_tokens": 10, "output_tokens": 5}


async def test_reasoning_deltas_become_thinking_events():
    model = make_model()
    events = await collect(
        model,
        [
            delta_chunk(reasoning_content="hmm "),
            delta_chunk(reasoning_content="okay"),
            delta_chunk(content="answer"),
            delta_chunk(finish_reason="stop"),
        ],
    )
    event_types = [event.event_type for event in events]
    assert event_types == [
        StreamEventType.CONTENT_BLOCK_START,
        StreamEventType.THINKING_DELTA,
        StreamEventType.THINKING_DELTA,
        StreamEventType.CONTENT_BLOCK_STOP,
        StreamEventType.CONTENT_BLOCK_START,
        StreamEventType.TEXT_DELTA,
        StreamEventType.CONTENT_BLOCK_STOP,
        StreamEventType.MESSAGE_END,
    ]
    assert events[0].block_type == "thinking"
    assert events[4].block_type == "text"


async def test_tool_call_frames_split_across_chunks():
    model = make_model()
    events = await collect(
        model,
        [
            delta_chunk(
                tool_calls=[tool_frame(0, call_id="call_1", name="get_weather")]
            ),
            delta_chunk(tool_calls=[tool_frame(0, arguments='{"city": ')]),
            delta_chunk(tool_calls=[tool_frame(0, arguments='"Berlin"}')]),
            delta_chunk(finish_reason="tool_calls"),
        ],
    )
    event_types = [event.event_type for event in events]
    assert event_types == [
        StreamEventType.TOOL_USE_START,
        StreamEventType.TOOL_USE_DELTA,
        StreamEventType.TOOL_USE_DELTA,
        StreamEventType.CONTENT_BLOCK_STOP,
        StreamEventType.MESSAGE_END,
    ]
    assert events[0].tool_call_id == "call_1"
    assert events[0].tool_name == "get_weather"
    arguments = "".join(
        event.tool_arguments_delta
        for event in events
        if event.event_type == StreamEventType.TOOL_USE_DELTA
    )
    assert arguments == '{"city": "Berlin"}'


async def test_parallel_tool_calls_close_on_index_switch():
    model = make_model()
    events = await collect(
        model,
        [
            delta_chunk(
                tool_calls=[tool_frame(0, call_id="call_1", name="a", arguments="{}")]
            ),
            delta_chunk(
                tool_calls=[tool_frame(1, call_id="call_2", name="b", arguments="{}")]
            ),
            delta_chunk(finish_reason="tool_calls"),
        ],
    )
    event_types = [event.event_type for event in events]
    assert event_types == [
        StreamEventType.TOOL_USE_START,
        StreamEventType.TOOL_USE_DELTA,
        StreamEventType.CONTENT_BLOCK_STOP,
        StreamEventType.TOOL_USE_START,
        StreamEventType.TOOL_USE_DELTA,
        StreamEventType.CONTENT_BLOCK_STOP,
        StreamEventType.MESSAGE_END,
    ]
    starts = [
        event for event in events if event.event_type == StreamEventType.TOOL_USE_START
    ]
    assert [start.tool_call_id for start in starts] == ["call_1", "call_2"]


async def test_missing_tool_call_id_is_generated():
    model = make_model()
    events = await collect(
        model,
        [
            delta_chunk(tool_calls=[tool_frame(0, name="get_time", arguments="{}")]),
            delta_chunk(finish_reason="tool_calls"),
        ],
    )
    start = events[0]
    assert start.event_type == StreamEventType.TOOL_USE_START
    assert start.tool_call_id.startswith("call_get_time_")


async def test_usage_less_stream_still_ends():
    model = make_model()
    events = await collect(
        model,
        [
            delta_chunk(content="hi"),
            delta_chunk(finish_reason="stop"),
        ],
    )
    assert events[-1].event_type == StreamEventType.MESSAGE_END
    assert events[-1].usage is None
