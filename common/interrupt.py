import asyncio
import logging
import signal

from typing import Callable, List, Any, Coroutine

Closure = Callable[[], None]


class InterruptionHandler(object):

    def __init__(self,
                 closures: List[Closure] = [lambda: logging.info(f"Stopping instance")],
                 sigs=[signal.SIGINT, signal.SIGTERM]):
        self._closures = closures
        self._sigs = sigs
        self._released = False
        self._running = True

    def __enter__(self):
        self._original_handlers = [signal.getsignal(s) for s in self._sigs]

        def handler(signum, frame):
            self.release()
            self._running = False

        for s in self._sigs:
            signal.signal(s, handler)

        return self

    def __exit__(self, type, value, tb):
        self.release()

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.__exit__(exc_type, exc_val, exc_tb)

    def release(self):
        if self._released:
            return False

        for h, s in zip(self._original_handlers, self._sigs):
            signal.signal(s, h)

        for closure in self._closures:
            closure()

        self._released = True
        self._running = False

        return True

    def running(self):
        return self._running


def run_main(coro: Coroutine):
    try:
        asyncio.get_event_loop().run_until_complete(coro)
    except KeyboardInterrupt:
        logging.info(f"Terminating gracefully")
