import asyncio
import gc
import json
import threading
import weakref
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from openai import AsyncOpenAI

from patterpunk.lib.async_client_per_event_loop import AsyncClientPerEventLoop


class FakeClient:
    pass


def look_up_in_new_event_loop(clients):
    async def look_up():
        return clients.for_running_loop()

    return asyncio.run(look_up())


def test_one_event_loop_reuses_one_client():
    clients = AsyncClientPerEventLoop(FakeClient)

    async def look_up_twice():
        return clients.for_running_loop(), clients.for_running_loop()

    first, second = asyncio.run(look_up_twice())

    assert first is second


def test_each_event_loop_gets_its_own_client():
    clients = AsyncClientPerEventLoop(FakeClient)

    first_loop_client = look_up_in_new_event_loop(clients)
    second_loop_client = look_up_in_new_event_loop(clients)

    assert first_loop_client is not second_loop_client


def test_clients_of_closed_event_loops_are_released():
    clients = AsyncClientPerEventLoop(FakeClient)
    first_loop_client_reference = weakref.ref(look_up_in_new_event_loop(clients))

    look_up_in_new_event_loop(clients)
    gc.collect()

    assert first_loop_client_reference() is None


CHAT_COMPLETION = {
    "id": "completion",
    "object": "chat.completion",
    "created": 0,
    "model": "stub",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
}


class KeepAliveChatCompletionHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        body = json.dumps(CHAT_COMPLETION).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def keep_alive_server_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), KeepAliveChatCompletionHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()
    server.server_close()


def test_successive_event_loops_never_reuse_a_closed_loops_connection(
    keep_alive_server_url,
):
    # max_retries=0 turns a reused connection from a closed event loop into a failure.
    # SDK retries would hide that failure, and a single shared client would pass too.
    clients = AsyncClientPerEventLoop(
        lambda: AsyncOpenAI(
            base_url=keep_alive_server_url, api_key="unused", max_retries=0
        )
    )

    async def send_request():
        response = await clients.for_running_loop().chat.completions.create(
            model="stub", messages=[{"role": "user", "content": "hi"}]
        )
        return response.choices[0].message.content

    answers = [asyncio.run(send_request()) for _ in range(3)]

    assert answers == ["ok", "ok", "ok"]
