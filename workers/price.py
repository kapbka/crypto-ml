import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

from dateutil.parser import parse
from sqlalchemy import func, or_, asc

from common.constants import MIN_DATE, CURRENCIES
from common.exchanges.binance_api import Binance
from common.interrupt import run_main
from common.k8s.probes import mark_alive, mark_ready
from common.log import setup_logging
from common.zk import ZooKeeper
from db.api import DB
from db.model import Price


def clean_odd_timestamps(db: DB, currency: str):
    # clean odd timestamps
    db.session.query(Price).\
        filter(or_(func.extract('milliseconds', Price.ts) != 0,
                   func.extract('seconds', Price.ts) != 0,
                   Price.update_ts.isnot(None))).delete(synchronize_session=False)

    logging.info(f"Checking gaps in prices")

    last: Optional[Price] = None
    for price in db.session.query(Price).filter(Price.currency == currency).order_by(asc(Price.ts)):
        while last and last.ts + timedelta(minutes=1) != price.ts:
            missing = Price(
                currency=last.currency,
                ts=last.ts + timedelta(minutes=1),
                close=last.close,
                low=last.low,
                high=last.high,
                volume=last.volume,
            )
            logging.info(f"Found missing price between {last} and {price}, filling with {missing}")
            db.session.add(missing)
            last = missing
        last = price

    db.commit()


async def fill_missing_prices(db: DB, exchange: Binance, start_date: datetime, currency: str, ticker: str) -> datetime:
    last_price = db.price.get_last(currency)
    if last_price:
        start_date = last_price.ts

    db.currency.set(code=currency, name=ticker)
    db.commit()

    time_started = datetime.now(timezone.utc)
    ds_all = await exchange.history(from_dt=start_date, to_dt=None)
    ds_all = ds_all[:time_started] if len(ds_all) and time_started < ds_all.index[-1] else ds_all

    ds = ds_all[start_date + timedelta(minutes=1):] if last_price else ds_all
    if not len(ds):
        return last_price.ts

    logging.info(f"Saving {len(ds)} historical prices from {ds.index[0]} to {ds.index[-1]}")

    copy = ds.copy()
    copy['currency'] = currency
    copy.to_sql('price', con=db.session.connection(), if_exists='append', index_label='ts')

    db.commit()
    return ds_all.index[-1] if len(ds_all) else start_date


async def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="This script runs a worker that fetches prices in real-time and updates the database")
    parser.add_argument("-v", "--verbosity", type=str, default='INFO', help="verbosity level: INFO/DEBUG")
    parser.add_argument("-c", "--currency", type=str, help="currency to use explicitly")
    parser.add_argument("--start-date", type=str, default=MIN_DATE.isoformat(),
                        help=f"date to start from, default: {MIN_DATE}")

    args = parser.parse_args(args)
    setup_logging(args.verbosity, filename='/tmp/price-worker.log')

    async with ZooKeeper('price') as zk:
        currency = args.currency or await zk.get_shared_value(CURRENCIES)

    with DB() as db:
        ticker = f"{currency.upper()}USDT"

        mark_alive()
        async with Binance(ticker=ticker) as exchange:
            last_ts = await fill_missing_prices(
                db=db,
                exchange=exchange,
                start_date=parse(args.start_date) - timedelta(minutes=1),
                currency=currency,
                ticker=ticker
            )

            clean_odd_timestamps(db, currency)

            async for is_final, data in exchange.subscribe():
                mark_alive()

                if not is_final:
                    continue

                obj = Price(currency=currency, **data)
                if (obj.ts - last_ts).seconds > 60:
                    logging.warning(f"Unable to get last kline from subscription: {obj}, last ts: {last_ts}")
                    last_ts = await fill_missing_prices(
                        db=db,
                        exchange=exchange,
                        start_date=last_ts,
                        currency=currency,
                        ticker=ticker
                    )

                # sanity check
                last = db.price.get_last(currency)
                if obj.ts <= last.ts:
                    logging.info(f"Skipping update {obj}, last saved: {last}")
                    continue

                assert last.ts == obj.ts - timedelta(minutes=1), (obj, last)

                # save new price, DB will handle notification
                db.add(obj)
                db.commit()

                logging.debug(f"Last price: {obj}")
                last_ts = obj.ts

                mark_ready()


if __name__ == '__main__':
    run_main(main())
