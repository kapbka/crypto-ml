import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Type, Callable

from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, InstanceState

from common.constants import DB_CONNECTION
from common.k8s.probes import mark_alive
from db.api.action import DBAction
from db.api.ann import DBAnn
from db.api.currency import DBCurrency
from db.api.deal import DBDeal
from db.api.history_stat import DBHistoryStat
from db.api.individual import DBIndividual
from db.api.interval import DBRunSumInterval
from db.api.portfolio import DBPortfolio
from db.api.prediction import DBPrediction
from db.api.price import DBPrice
from db.api.run_sum import DBRunSum
from db.api.scaler import DBScaler
from db.api.version import DBVersion
from db.model import Base


def serialize(obj):
    if isinstance(obj, InstanceState):
        return None

    if isinstance(obj, datetime):
        return {'_ts': obj.isoformat()}
    raise obj


def deserialize(obj):
    ts = obj.get('_ts')
    if ts is not None:
        return datetime.fromisoformat(ts)
    return obj


class DB:
    def __init__(self):
        self.engine = None
        self.session = self.connect()
        self._connection = None
        self._callbacks = defaultdict(list)
        self._pingers = list()

    async def __aenter__(self):
        return self

    def __enter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.__exit__(exc_type, exc_val, exc_tb)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.rollback()
        else:
            self.commit()

    @staticmethod
    async def _ping(subscription: str):
        engine = create_engine(DB_CONNECTION)
        session = Session(engine, expire_on_commit=False)
        while True:
            session.execute("SELECT pg_notify(:p1, :p2)", params=dict(p1=subscription, p2=''))
            session.commit()
            await asyncio.sleep(10)

    def _read_notification(self, *args):
        """
        Reads all available notifications and notifies subscribers
        :param args: ignored
        :return: None
        """
        self._connection.poll()
        while self._connection.notifies:
            notify = self._connection.notifies.pop()
            logging.debug(f"NOTIFY: {notify.pid=}, {notify.channel=}, {notify.payload=}")
            mark_alive()

            if not notify.payload:
                continue

            for entity, callback in self._callbacks[notify.channel]:
                try:
                    instance = entity()
                    for k, v in json.loads(notify.payload, object_hook=deserialize).items():
                        if k.startswith("_"):
                            continue
                        setattr(instance, k, v)

                    logging.debug(f"Notifying {callback} with {instance}")
                    callback(instance)
                except:
                    logging.exception(f"Exception in notify callback, {notify.channel=}, {notify.payload=}")

    def connect(self):
        self.engine = create_engine(DB_CONNECTION)
        session = Session(self.engine, expire_on_commit=False)

        for attempt in range(10):
            try:
                Base.metadata.create_all(self.engine)
                session.commit()
                logging.info(f"Connected to {DB_CONNECTION.replace(os.getenv('POSTGRES_PASSWORD'), '****')}")
                break
            except OperationalError:
                logging.exception(f"Unable to connect to Postgres, attempt #: {attempt}")
                time.sleep(5)

        return session

    def add(self, db_object: Base):
        body = json.dumps(db_object.__dict__, default=serialize)
        self.session.add(db_object)
        self.session.execute("SELECT pg_notify(:p1, :p2)", params=dict(p1=db_object.__tablename__, p2=body))

    def subscribe(self, entity: Type[Base], callback: Callable[[Base], None]):
        if self._connection is None:
            engine = create_engine(DB_CONNECTION)
            session = Session(engine, expire_on_commit=False)
            self._connection = session.connection().connection.connection
            self._connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            asyncio.get_event_loop().add_reader(self._connection.fileno(), self._read_notification, engine, session)

        self._callbacks[entity.__tablename__].append((entity, callback))
        if len(self._callbacks[entity.__tablename__]) == 1:
            self._pingers.append(asyncio.get_event_loop().create_task(self._ping(entity.__tablename__)))

        cursor = self._connection.cursor()
        cursor.execute(f"LISTEN {entity.__tablename__};")

    def commit(self):
        self.session.commit()

    def rollback(self):
        self.session.rollback()

    def flush(self):
        self.session.flush()

    @property
    def version(self):
        return DBVersion(self)

    @property
    def currency(self):
        return DBCurrency(self)

    @property
    def price(self):
        return DBPrice(self)

    @property
    def individual(self):
        return DBIndividual(self)

    @property
    def portfolio(self):
        return DBPortfolio(self)

    @property
    def deal(self):
        return DBDeal(self)

    @property
    def history_stat(self):
        return DBHistoryStat(self)

    @property
    def run_sum_interval(self):
        return DBRunSumInterval(self)

    @property
    def run_sum(self):
        return DBRunSum(self)

    @property
    def ann(self):
        return DBAnn(self)

    @property
    def scaler(self):
        return DBScaler(self)

    @property
    def prediction(self):
        return DBPrediction(self)

    @property
    def action(self):
        return DBAction(self)
