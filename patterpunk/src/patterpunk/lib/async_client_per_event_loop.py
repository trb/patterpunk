import asyncio
import threading
from typing import Callable, Dict, Generic, TypeVar

ClientType = TypeVar("ClientType")


class AsyncClientPerEventLoop(Generic[ClientType]):
    # An async SDK client keeps idle connections in a pool for reuse.
    # Each pooled connection only works inside the event loop that opened it.
    # asyncio.run() closes its loop, but the client keeps that loop's connections pooled.
    # A request from the next loop then fails with "Event loop is closed" or waits for a retry.
    # One client per event loop keeps every pooled connection usable.
    def __init__(self, build_client: Callable[[], ClientType]):
        self._build_client = build_client
        self._clients_by_event_loop: Dict[asyncio.AbstractEventLoop, ClientType] = {}
        self._lock = threading.Lock()

    def for_running_loop(self) -> ClientType:
        running_loop = asyncio.get_running_loop()
        with self._lock:
            self._release_clients_of_closed_loops()
            if running_loop not in self._clients_by_event_loop:
                self._clients_by_event_loop[running_loop] = self._build_client()
            return self._clients_by_event_loop[running_loop]

    def _release_clients_of_closed_loops(self) -> None:
        closed_loops = [
            event_loop
            for event_loop in self._clients_by_event_loop
            if event_loop.is_closed()
        ]
        for closed_loop in closed_loops:
            del self._clients_by_event_loop[closed_loop]
