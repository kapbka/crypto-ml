import argparse
import logging
import sys
from datetime import datetime, timezone
from functools import partial

import numpy as np
from dateutil import parser as date_parser

from common.constants import DEFAULT_CURRENCY
from common.log import setup_logging
from common.metrics.evaluator import profit
from common.plot_tools import plot_columns
from common.portfolio import deals_to_trades, run_portfolio_detailed, find_best, save_portfolio
from db.api import DB
from db.data_cache import DataCache
from db.model import Price


def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="This script runs portfolio evaluation")
    parser.add_argument("-v", "--verbosity", type=str, default='INFO', help="verbosity level: INFO/DEBUG")
    parser.add_argument("--graphs", action='store_true', default=False, help=f"dump graphs")
    parser.add_argument("--deals", action='store_true', default=False, help=f"dump deals to log")
    parser.add_argument("--currency", type=str, default=DEFAULT_CURRENCY,
                        help=f"currency to use, default: {DEFAULT_CURRENCY}")
    parser.add_argument("--version", type=int, default=5, help=f"version to use")
    parser.add_argument("--min-run-sum-percent", type=float, default=0, help=f"minimum running sum percent to use")
    parser.add_argument("--size", type=int, default=2, help=f"number of individuals to combine together")
    parser.add_argument("--save-count", type=int, default=10, help=f"number of top portfolios to save")
    parser.add_argument("--begin", type=str, default=datetime(2021, 11, 1).isoformat(),
                        help=f"datetime to start from, default: {datetime(2021, 11, 1).isoformat()}")
    parser.add_argument("md5", type=str, nargs='*', help=f"list of md5 hashes of individuals")

    args = parser.parse_args(args)
    setup_logging(args.verbosity, filename='/tmp/portfolio.log')

    if args.md5:
        args.size = len(args.md5)

    logging.info(args)

    deals_cache = dict()
    trades_cache = dict()
    with DB() as db:
        for individual in [db.individual.get_db(i) for i in args.md5] or db.individual.get_all(include_portfolio=False):
            deals_cache[individual.md5] = db.deal.get_all(
                version_id=args.version,
                currency=args.currency,
                individual_id=individual.id,
                ts_from=date_parser.parse(args.begin).replace(tzinfo=timezone.utc)
            )
            if not deals_cache[individual.md5] or profit(deals_cache[individual.md5]) < args.min_run_sum_percent:
                logging.info(f"Skipping {individual}, count: {len(deals_cache[individual.md5])}")
                continue

            trades = deals_to_trades(deals_cache[individual.md5])
            if trades:
                trades_cache[individual.md5] = trades

        df_all = DataCache(db, Price, args.currency).load(
            from_ts=date_parser.parse(args.begin).replace(tzinfo=timezone.utc)
        )

    sorted_results = find_best(trades_cache, args.size)
    list(map(logging.info, sorted_results[:30]))
    with DB() as db:
        list(map(partial(save_portfolio, db, args.currency), sorted_results[:args.save_count]))

    if args.graphs:
        bots = sorted_results[0][1]
        data = np.concatenate([trades_cache[bot] for bot in bots if len(trades_cache[bot])])
        data = data[data[:, 3].argsort(kind='stable')]

        ts = np.array([np.datetime64(t).astype(int) for t in df_all.index.values])
        money = run_portfolio_detailed(data, df_all['close'].values, ts, args.size)
        df_all['money'] = [m[0] for m in money]
        df_all['shares'] = [m[1] for m in money]

        fig = plot_columns(df_all, ['shares', ['close', 'money']], size_x=100, size_y=20)
        fig.savefig('portfolio.png')


if __name__ == "__main__":
    main()
