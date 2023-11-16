import asyncio
import logging
import pickle
import socket
import sys
from typing import Any, List, Callable, Optional

from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy.sql.expression import asc

from common.interrupt import InterruptionHandler
from db.api.db import DB
from db.model.messaging import Payload, Consumer


class FatalError(Exception):
    pass


class Notifier:
    def __init__(self, db: DB, guard: InterruptionHandler, name: str):
        self._db = db
        self._guard = guard
        self._connection = db.session.connection().connection.connection
        self._connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        self._name = name
        self._subscribers: List[Callable[[], None]] = []
        self._task = asyncio.ensure_future(self._read_notifications())

        self._start_listening()

    def subscribe(self, func: Callable[[], None]):
        self._subscribers.append(func)

    def unsubscribe(self, func: Callable[[], None]):
        self._subscribers.remove(func)

        if not self._subscribers:
            self._task.cancel()

    async def _read_notifications(self):
        while self._guard.running():
            self._read_notification(None)
            await asyncio.sleep(0.5)

    def _read_notification(self, *args):
        """
        Reads all available notifications and puts them to the queue
        :param args: ignored
        :return: None
        """
        self._connection.poll()
        handled = []
        for notify in self._connection.notifies:
            logging.debug(f"Got NOTIFY: {notify}")

            for subscriber in self._subscribers:
                subscriber()

        list(map(self._connection.notifies.remove, handled))

    def _start_listening(self):
        """
        This functions starts listening for postgres notifications on specified channels
        :return: tuple of channel name and payload
        """
        cursor = self._connection.cursor()
        cursor.execute(f"LISTEN {self._name};")

        asyncio.get_running_loop().add_reader(self._connection.fileno(), self._read_notification, None)


class Subscriber:
    def __init__(self, db: DB, guard: InterruptionHandler, queue: str, uid: str = None, exclusive=False):
        self._db = db
        self._guard = guard
        self._queue = asyncio.Queue()
        self._uuid = uid or f"{queue}@{socket.gethostname()}"
        self._exclusive = exclusive

        self._consumer = db.session.query(Consumer).filter(Consumer.id == self._uuid).first()
        if not self._consumer:
            self._consumer = Consumer(id=self._uuid, active=0)
            db.session.add(self._consumer)

        self._consumer.active += 1
        db.commit()

        self._queue_name = queue.replace('-', '_').replace('.', '_')

        self._notifier = Notifier(db=db, guard=guard, name=self._queue_name)
        self._notifier.subscribe(self._receive)

        logging.debug(f"Subscribing to {self._queue_name} id {self._consumer}")

    def __enter__(self):
        return self

    async def __aenter__(self):
        return self.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._queue.put_nowait('')
        self._consumer.active -= 1
        self._db.commit()
        self._notifier.unsubscribe(self._receive)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.__exit__(exc_type, exc_val, exc_tb)

    def _receive(self):
        logging.debug(f"{self._consumer} notified")
        self._queue.put_nowait('')

    async def get(self):
        while self._guard.running():
            query = self._db.session.query(Payload).filter(Payload.consumer == self._queue_name, Payload.available)
            if self._consumer.last_processed_payload:
                query = query.filter(Payload.id > self._consumer.last_processed_payload)
            if self._exclusive:
                query = query.with_for_update(of=Payload, nowait=True)

            payload = query.order_by(asc(Payload.id)).first()
            if not payload:
                await self._queue.get()  # wait for notification
                continue

            logging.debug(f"{self._consumer} received {payload.id}")

            if self._exclusive:
                payload.available = False
            self._consumer.last_processed_payload = payload.id
            self._db.commit()
            return pickle.loads(payload.data)


class Publisher:
    def __init__(self, db: DB, name: Optional[str] = None):
        self._db = db
        self._name = name
        self._connection = self._db.session.connection().connection.connection
        self._connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        self._uuid = f"{name}@{socket.gethostname()}"

    def __enter__(self):
        return self

    async def __aenter__(self):
        return self.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.__exit__(exc_type, exc_val, exc_tb)

    def send(self, msg: Any, name: str = None):
        target = (name or self._name).replace('-', '_').replace('.', '_')
        payload = Payload(data=pickle.dumps(msg), producer=self._uuid, consumer=target, available=True)
        self._db.session.add(payload)
        self._db.commit()

        logging.debug(f"Notifying {target} with {msg}")
        self._db.session.execute("SELECT pg_notify(:p1, '')", params=dict(p1=target))


class RPCHandler:
    def __init__(self, db: DB, guard: InterruptionHandler, name: str, func: Callable[[Any], Any], uid: str = None):
        self._func = func
        self._guard = guard
        self._subscriber = Subscriber(db=db, guard=guard, queue=name, exclusive=True, uid=uid)
        self._publisher = Publisher(db=db, name=f"{name}-response")

    def __enter__(self):
        return self

    async def __aenter__(self):
        return self.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._publisher.__exit__(exc_type, exc_val, exc_tb)
        self._subscriber.__exit__(exc_type, exc_val, exc_tb)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.__exit__(exc_type, exc_val, exc_tb)

    async def run_once(self):
        payload, reply_to = await self._subscriber.get()
        try:
            result = self._func(**payload)
            self._publisher.send(result, name=reply_to)
        except FatalError as e:
            logging.exception(f"Fatal error, payload: {payload}, reply_to: {reply_to}")
            self._publisher.send(e, name=reply_to)
            sys.exit(1)
        except Exception as e:
            logging.exception(f"Handler failed, payload: {payload}, reply_to: {reply_to}")
            self._publisher.send(e, name=reply_to)

    async def run(self):
        while self._guard.running():
            await self.run_once()


class RPCClient:
    def __init__(self, db: DB, guard: InterruptionHandler, name: str):
        self._guard = guard
        self._uuid = f"{name}_{socket.gethostname().lower().replace('-', '_')}"
        self._publisher = Publisher(db=db, name=name)
        self._subscriber = Subscriber(db=db, guard=guard, queue=self._uuid, uid=self._uuid, exclusive=True)

    def __enter__(self):
        return self

    async def __aenter__(self):
        return self.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._publisher.__exit__(exc_type, exc_val, exc_tb)
        self._subscriber.__exit__(exc_type, exc_val, exc_tb)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.__exit__(exc_type, exc_val, exc_tb)

    async def __call__(self, *args, **kwargs) -> Any:
        self._publisher.send(msg=(kwargs, self._uuid))
        result = await self._subscriber.get()
        if issubclass(type(result), Exception):
            raise result
        return result
