from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator, List

import pytest
from pandas import DataFrame

from db.api import DB
from db.data_cache import DataCache
from db.model import Price

START_TS = datetime(2021, 10, 10, tzinfo=timezone.utc)


def generate_prices(db: DB, number: int, start_ts: datetime = START_TS):
    for num in range(number):
        db.price.set(currency='btc', ts=start_ts + timedelta(minutes=num),
                     close=num + 1, low=num + 1, high=num + 1, volume=num + 1)

    db.commit()


async def get_updates(generator: AsyncGenerator[DataFrame, None], num: int) -> List[DataFrame]:
    updates = []
    async for data in generator:
        updates.append(data)
        if len(updates) == num:
            return updates


def validate(updates: List[DataFrame], window_size: int, last_ts: datetime) -> datetime:
    # check initial data loaded on start
    assert sum(map(len, updates)) == window_size * len(updates)

    for update in updates:
        assert update.index[0] == last_ts
        assert update.index[1] == last_ts + timedelta(minutes=1)
        last_ts = update.index[1]

    return last_ts


@pytest.mark.asyncio
async def test_retrieving_data():
    # clean data first
    with DB() as db:
        db.session.execute(f'TRUNCATE TABLE {Price.__tablename__} CASCADE')

    # now emulate data upload
    with DB() as db:
        db.currency.set('btc', 'Bitcoin')
        generate_prices(db, 5)

        # check that we will iterate over them
        window = DataCache(db, Price, 'btc')

        # we have 5 records with window size 2
        # means we should get 4 updates on start
        # then we emulate 3 row insert which should give
        # 3 one more updates
        generator = window.get(2, START_TS)
        updates = await get_updates(generator, 4)
        last_ts = validate(updates, 2, START_TS)

        # insert a few rows
        generate_prices(db, 3, last_ts + timedelta(minutes=1))

        # notify multiple times, it shouldn't matter
        for _ in range(5):
            window.notify()

        # get them back and check
        updates = await get_updates(generator, 3)
        last_ts = validate(updates, 2, last_ts)

        # emulate price update, insert price with odd minutes
        ts = datetime(
            last_ts.year,
            last_ts.month,
            last_ts.day,
            last_ts.hour,
            last_ts.minute + 1, 30,
            tzinfo=timezone.utc
        )
        generate_prices(db, 1, start_ts=ts)

        updates = await get_updates(generator, 1)
        update = updates[0]
        assert update.index[0] == last_ts
        assert update.index[1] == ts

        window.stop()

