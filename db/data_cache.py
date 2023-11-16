import asyncio
import logging
from datetime import datetime, timedelta
from typing import Type, List, AsyncGenerator, Optional

from pandas import read_sql, DataFrame
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy.sql.expression import asc

from common.interrupt import InterruptionHandler
from db.api import DB


class DataCache:
    """
    This class is supposed to provide a data window access to cached DB data
    """

    def __init__(self, db: DB, class_type: Type, currency: str, index_column: str = 'ts',
                 currency_column: str = 'currency'):
        self._class_type = class_type
        self._currency = currency
        self._index_column = index_column
        self._currency_column = currency_column
        self._cache: List[class_type] = list()

        self._db: DB = db
        self._running = True

        self._connection = self._db.session.connection().connection.connection
        self._connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

        self._queue: Optional[asyncio.Queue] = None

    def load(self, from_ts: datetime, to_ts: Optional[datetime] = None) -> DataFrame:
        """
        Loads last objects from the database starting from 'from_ts' timestamp.
        Sorting is performed by 'ts' column ascending.
        :param from_ts: first timestamp to load data from
        :param to_ts: last timestamp to load data
        :return: pandas dataframe
        """
        query = self._db.session \
            .query(self._class_type) \
            .filter(getattr(self._class_type, self._index_column) > from_ts,
                    getattr(self._class_type, self._currency_column) == self._currency)

        if to_ts:
            query = query.filter(getattr(self._class_type, self._index_column) < to_ts)

        query = query.order_by(asc(getattr(self._class_type, self._index_column)))
        return read_sql(query.statement, query.session.bind).set_index(self._index_column)

    def _read_notification(self, *args):
        """
        Reads all available notifications and puts them to the queue
        :param args: ignored
        :return: None
        """
        self._connection.poll()
        while self._connection.notifies:
            notify = self._connection.notifies.pop()
            logging.debug(f"Got NOTIFY: {notify.pid}, {notify.channel}, {notify.payload}")
            self._queue.put_nowait(notify.channel)

    def _start_listening(self, channels: List[str]):
        """
        This functions starts listening for postgres notifications on specified channels
        :param channels: channel names to listen on
        :return: tuple of channel name and payload
        """
        cursor = self._connection.cursor()
        cursor.execute(";".join([f"LISTEN {channel}" for channel in channels]))

        asyncio.get_event_loop().add_reader(self._connection.fileno(), self._read_notification, None)

    def stop(self):
        """
        Stops the cache
        :return: None
        """
        self._running = False
        self._queue.put_nowait(None)

    def notify(self):
        """
        Raises a notification with 'class_type' table name.
        :return: None
        """
        self._connection.cursor().execute(f"NOTIFY {self._class_type.__tablename__};")

    async def get(self, window_size: int, from_ts: datetime) -> AsyncGenerator[DataFrame, None]:
        """
        Gets rolling window data from the database returning batches of 'window_size' size.
        :param window_size: size of the window
        :param from_ts: timestamp of the last element of the first batch
        :return: async generator of DataFrame
        """

        self._queue = asyncio.Queue()
        self._start_listening([self._class_type.__tablename__])

        # drop seconds to make sure we don't miss last updates
        from_ts = datetime(
            from_ts.year,
            from_ts.month,
            from_ts.day,
            from_ts.hour,
            from_ts.minute,
            tzinfo=from_ts.tzinfo
        )
        df = self.load(from_ts - timedelta(minutes=window_size))
        for i in range(len(df) - window_size + 1):
            yield df.iloc[i:i + window_size, :]

        if len(df):
            from_ts = df.index[-1]

        with InterruptionHandler() as guard:
            while guard.running() and self._running:
                self._read_notification()
                channel = await self._queue.get()
                if channel is None:
                    break

                # load new rows
                df = self.load(from_ts - timedelta(minutes=window_size - 1))
                for i in range(len(df) - window_size + 1):
                    yield df.iloc[i:i + window_size, :]

                if len(df):
                    from_ts = df.index[-1]
