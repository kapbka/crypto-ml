import asyncio
import hashlib
import logging
import os
import pickle
import socket
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from functools import partial
from typing import Callable, Any, Dict, Optional

import aio_pika
from aio_pika.patterns import RPC

from common.constants import RMQ_CONNECTION, RMQ_EXCHANGE_NAME
from common.interrupt import InterruptionHandler

_connection: Optional[aio_pika.Connection] = None


async def _get_connection() -> aio_pika.Connection:
    global _connection
    if not _connection:
        _connection = await rmq_connect()
    return _connection


async def shutdown():
    global _connection
    if _connection:
        await _connection.close()
        _connection = None


class Subscriber:
    def __init__(self, guard: InterruptionHandler, topic: str, exchange_name: str = RMQ_EXCHANGE_NAME):
        self._guard = guard
        self._exchange_name = exchange_name
        self._topic = topic
        self._connection: Optional[aio_pika.Connection] = None
        self._queue: Optional[aio_pika.Queue] = None

    async def __aenter__(self):
        self._connection = await _get_connection()
        channel = await self._connection.channel()

        exchange = await channel.declare_exchange(self._exchange_name, type=aio_pika.ExchangeType.TOPIC)
        self._queue = await channel.declare_queue(auto_delete=True, exclusive=True)
        await self._queue.bind(exchange, self._topic)

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._connection.close()

    async def get(self):
        while self._guard.running():
            msg: aio_pika.IncomingMessage = await self._queue.get(fail=False)
            if msg:
                logging.debug(f"Incoming: {msg}")
                await msg.ack()
                return pickle.loads(msg.body)
            else:
                await asyncio.sleep(0.1)


class Publisher:
    def __init__(self, topic: str = None, exchange_name: str = RMQ_EXCHANGE_NAME):
        self._exchange_name = exchange_name
        self._topic = topic
        self._connection = None
        self._exchange = None
        self._channel = None

    async def __aenter__(self):
        self._connection = await _get_connection()
        self._channel = await self._connection.channel()
        self._exchange = await self._channel.declare_exchange(self._exchange_name, type=aio_pika.ExchangeType.TOPIC)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._connection.close()

    async def send(self, msg: Any, expiry: Optional[timedelta] = None, topic: str = None):
        await self._exchange.publish(
            aio_pika.Message(body=pickle.dumps(msg), expiration=expiry),
            routing_key=topic or self._topic, mandatory=False
        )


async def rmq_connect(logger=logging.getLogger()):
    for i in range(60):
        try:
            logger.debug(f"Connecting to RMQ")
            res = await aio_pika.connect_robust(RMQ_CONNECTION,
                                                loop=asyncio.get_event_loop(),
                                                client_properties={"connection_name": socket.gethostname()})
            logging.info(f"Connected to {RMQ_CONNECTION.replace(os.getenv('RABBITMQ_PASSWORD'), '****')}")
            return res
        except Exception as e:
            logger.error(f"RMQ is not available: {e}")
            await asyncio.sleep(1)


async def _wrapper(thread_pool: ThreadPoolExecutor, function: Callable[[Any], Any], **kwargs):
    try:
        return await asyncio.get_event_loop().run_in_executor(thread_pool, partial(function, **kwargs))
    except:
        logging.exception(f"Worker failed, kwargs: {kwargs}")
        raise


async def run_consumer(method_name: str, function: Callable[[Any], Any], guard: InterruptionHandler) -> None:
    connection = await _get_connection()

    with ThreadPoolExecutor(max_workers=1) as thread_pool:
        async with connection:
            channel = await connection.channel()
            await channel.set_qos(prefetch_count=1)

            rpc = await RPC.create(channel)
            await rpc.register(method_name, partial(_wrapper, thread_pool, function), auto_delete=True)
            while guard.running():
                await asyncio.sleep(0.1)


async def _get_or_call(rpc: aio_pika.patterns.RPC, method_name: str, args: dict, cache: Dict[str, Any]) -> Any:
    hash_md5 = hashlib.md5()
    hash_md5.update(pickle.dumps(args))
    hash_md5.update(method_name.encode())

    md5 = hash_md5.hexdigest()

    cached = cache.get(md5)
    if cached:
        return cached

    cache[md5] = await rpc.call(method_name, kwargs=args)
    return cache[md5]


Client = aio_pika.patterns.RPC


async def rpc_client(method_name: str, purge=True) -> Client:
    connection = await _get_connection()
    channel = await connection.channel()
    rpc = await RPC.create(channel)

    queue = await channel.declare_queue(method_name,
                                        auto_delete=True,
                                        arguments={"x-dead-letter-exchange": rpc.DLX_NAME})
    if purge:
        await queue.purge()

    return rpc

