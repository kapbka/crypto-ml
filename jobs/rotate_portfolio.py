import argparse
import logging
import sys
from datetime import datetime, timezone, timedelta

import numpy as np

from common.constants import DEFAULT_CURRENCY, RealtimeStatus, HistoryStatus
from common.log import setup_logging
from common.portfolio import deals_to_trades, find_best, save_portfolio, portfolio
from common.reporting import send_message
from db.api import DB


def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="This script finds best portfolio on a given date range and enables it for deal processing")
    parser.add_argument("-v", "--verbosity", type=str, default='INFO', help="verbosity level: INFO/DEBUG")
    parser.add_argument("--currency", type=str, default=DEFAULT_CURRENCY,
                        help=f"currency to use, default: {DEFAULT_CURRENCY}")
    parser.add_argument("--version", type=int, default=5, help=f"version to use")
    parser.add_argument("--size", type=int, default=3, help=f"number of individuals to combine together")
    parser.add_argument("--days", type=str, default=5, help=f"days offset to start from")

    args = parser.parse_args(args)
    setup_logging(args.verbosity)

    logging.info(args)

    deals_cache = dict()
    trades_cache = dict()
    with DB() as db:
        for individual in db.individual.get_all(include_portfolio=False):
            deals_cache[individual.md5] = db.deal.get_all(
                version_id=args.version,
                currency=args.currency,
                individual_id=individual.id,
                ts_from=datetime.now(tz=timezone.utc) - timedelta(days=args.days)
            )
            if not deals_cache[individual.md5]:
                logging.info(f"Skipping {individual}, count: {len(deals_cache[individual.md5])}")
                continue

            trades = deals_to_trades(deals_cache[individual.md5])
            if trades:
                trades_cache[individual.md5] = trades

    sorted_results = find_best(trades_cache, args.size)
    if sorted_results:
        with DB() as db:
            portfolio_individuals = db.portfolio.get_members(currency_code=args.currency)
            bots = set([t.md5 for t in portfolio_individuals])

            # now evaluate current one
            percent = 0
            if bots:
                data = [trades_cache[bot] for bot in bots if trades_cache.get(bot)]
                if data:
                    data = np.concatenate(data)
                    data = data[data[:, 3].argsort(kind='stable')]

                    percent, _, _ = portfolio(data, len(bots), generate_deals=False)
                    diff = sorted_results[0][0] - percent
                    if diff < 0.1:
                        logging.info(f"Current portfolio {bots} with profit {percent:.2f}% will not be replaced with "
                                     f"new best {sorted_results[0][0]:.2f}%")
                        return

            result = save_portfolio(db, args.currency, sorted_results[0])

            send_message(f"New portfolio {result.md5}, profit on last {args.days} days is {sorted_results[0][0]:.2f}%, "
                         f"previous profit: {percent:.2f}%, new members: {sorted_results[0][1]}")

            for attr in db.portfolio.get_realtime(currency_code=args.currency):
                attr.history_enabled = HistoryStatus.Enabled.value,
                attr.realtime_enabled = RealtimeStatus.Disabled.value

            db.individual.attribute.set_defaults(
                individual=result,
                history_enabled=HistoryStatus.Disabled.value,
                realtime_enabled=RealtimeStatus.Enabled.value
            )


if __name__ == "__main__":
    main()
