import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone, timedelta
from functools import partial
from typing import List, Tuple, Dict

import numpy as np

from common.constants import DEFAULT_CURRENCY, IsPortfolio
from common.interrupt import run_main
from common.k8s.probes import mark_ready, mark_alive
from common.log import setup_logging
from common.metrics.evaluator import Limits
from common.metrics.evaluator import portfolio
from common.portfolio import deals_to_trades, calculate_shares
from db.api import DB
from db.model import Price, IndividualAttribute, Deal
from workers.ann import get_deals


async def get_local_deals(**kwargs):
    return get_deals(**kwargs)


def append_deals(db: DB, currency: str, results: List[Tuple[List[Deal], Limits]], attrs: List[IndividualAttribute]):
    for idx, (deals, limits) in enumerate(results):
        for deal in deals:
            deal.is_realtime = 1

        if deals:
            logging.debug(f"Saving {len(deals)} deals for {attrs[idx].md5}, last: {deals[-1]}")

            db.deal.set_batch(
                version=attrs[idx].version_id,
                individual_id=attrs[idx].individual_id,
                currency=currency,
                deals=deals,
                replace=False,
            )
    db.commit()


async def process_realtime(db: DB,
                           currency: str,
                           attrs: List[IndividualAttribute],
                           ts: datetime,
                           rpc: callable) -> Dict[Limits, float]:
    results = await asyncio.gather(*[rpc(attributes=attr, to_ts=ts) for attr in attrs])
    append_deals(db=db, currency=currency, results=results, attrs=attrs)

    # take all deals and run portfolio on them
    trades = [deals_to_trades(deals) for deals, limits in results if deals]
    result = {}
    if trades:
        data = np.concatenate(trades)
        data = data[data[:, 3].argsort(kind='stable')]
        _, _, shares = portfolio(events=data, total_shares=len(results), generate_deals=False)
        result = calculate_shares(limits=list(map(lambda x: x[1], results)), shares=shares)
    return result


async def process_historical(db: DB,
                             currency: str,
                             attrs: List[IndividualAttribute],
                             ts: datetime,
                             rpc: callable,
                             batch_size: int) -> int:
    assert batch_size

    deadline = ts + timedelta(days=365)
    count = 0
    while datetime.now(timezone.utc) < deadline and attrs:
        current_attrs = attrs[:batch_size]
        attrs = attrs[batch_size:]

        logging.debug(f"Processing batch {current_attrs}")
        results = await asyncio.gather(*[rpc(attributes=attr, to_ts=ts) for attr in current_attrs])
        count += len(results)

        for attr, (deals, limits) in zip(current_attrs, results):
            if deals:
                logging.debug(f"Individual {attr.md5} has {len(deals)} deals, last: {deals[-1] if deals else None}, "
                              f"v: {attr.version_id}, s: {attr.scaler_id}, c: {currency}")
                db.deal.set_batch(
                    version=attr.version_id,
                    individual_id=attr.individual_id,
                    currency=currency,
                    deals=deals,
                    replace=False,
                )

                db.commit()
    return count


def load_individuals(db: DB, currency: str) -> Tuple[List[IndividualAttribute], List[IndividualAttribute]]:
    portfolio_individuals = db.portfolio.get_members(currency_code=currency)
    portfolio_individual_attributes = db.portfolio.get_members_attrs(
        currency_code=currency,
        portfolio_members=portfolio_individuals
    )
    history = db.individual.get_history(portfolio=IsPortfolio.No)

    logging.debug(f"Loaded portfolio {portfolio_individuals} and {len(history)} history enabled bots")

    return portfolio_individual_attributes, list(map(lambda x: x[1], history))


async def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="This script runs a worker that fetches inputs in real-time and generates deals")
    parser.add_argument("-v", "--verbosity", type=str, default='INFO', help="verbosity level: INFO/DEBUG")
    parser.add_argument("--currency", type=str, default=DEFAULT_CURRENCY,
                        help=f"currency to use, default: {DEFAULT_CURRENCY}")

    args = parser.parse_args(args)
    setup_logging(args.verbosity, filename='/tmp/deals-worker.log')

    queue = asyncio.Queue()

    def on_new_price(price: Price):
        nonlocal queue
        if price.currency == args.currency:
            queue.put_nowait(price)

    async with DB() as db:
        db.subscribe(Price, on_new_price)

        rpc = partial(get_local_deals, db=db)

        # 1. get list of all bots
        # 2. on each price update call 'get_deals' for real-time first
        # 3. calculate portfolios for real-time bots
        # 4. then call 'get_deals' for all historical, check if processing of historical takes more than a minute,
        # than start with real-time again

        realtime, history = load_individuals(db=db, currency=args.currency)

        total_count = len(realtime) + len(history)

        while True:
            update: Price = await queue.get()
            if not update:
                break

            # on each update we want to process all real-time bots and as many historical bots as possible
            if realtime:
                actions = await process_realtime(
                    db=db,
                    currency=args.currency,
                    attrs=realtime,
                    ts=update.ts,
                    rpc=rpc,
                )

                db.action.set(limits=actions, currency_code=args.currency, ts=update.ts, close=update.close)

                logging.debug(f"Last update: {update}, actions: {actions}, close: {update.close}")

            count = await process_historical(
                db=db,
                currency=args.currency,
                attrs=history,
                ts=update.ts,
                rpc=rpc,
                batch_size=1,
            )

            cycle_count = count + len(realtime)
            if cycle_count:
                mark_ready()
            mark_alive()

            if total_count > cycle_count:
                logging.info(f"{total_count - cycle_count} individuals left unprocessed")

            new_realtime, history = load_individuals(db=db, currency=args.currency)
            if new_realtime != realtime:
                logging.info(f"Reloaded realtime bots: {[bot.md5 for bot in new_realtime]}")
                realtime = new_realtime


if __name__ == '__main__':
    run_main(main())
