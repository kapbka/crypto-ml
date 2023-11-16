import argparse
import json
import logging
import sys
from datetime import timedelta, datetime, timezone
from functools import partial

from dateutil.parser import parse as parse_dt

from common.constants import DEFAULT_CURRENCY, EPOCH, VERSION_TO_METHOD, LABEL_COLUMNS
from common.deal_generator import DealGenerator
from common.log import setup_logging
from common.plot_tools import plot_to_file
from db.api import DB
from db.data_cache import DataCache
from db.model import Price
from models.ann import max_offset


def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="This script runs evaluation of one or multiple individuals")
    parser.add_argument("-v", "--verbosity", type=str, default='INFO', help="verbosity level: INFO/DEBUG")
    parser.add_argument("--currency", type=str, default=DEFAULT_CURRENCY,
                        help=f"currency to use, default: {DEFAULT_CURRENCY}")
    parser.add_argument("md5", type=str, nargs='*', help=f"list of individual md5 hashes")
    parser.add_argument("--history-ts", type=str, default=EPOCH.isoformat(),
                        help=f"datetime to start history from, default: {EPOCH.isoformat()}")
    parser.add_argument("--end-ts", type=str, default=datetime.now().isoformat(),
                        help=f"datetime to stop evaluation at, default: {datetime.now().isoformat()}")
    parser.add_argument("--realtime-offset", type=int, default=0,
                        help=f"minutes offset from the end to start real-time from, default: 0")
    parser.add_argument("--version", type=int, default=5, help=f"version to use")
    parser.add_argument("--graphs", action='store_true', default=False, help=f"dump graphs")
    parser.add_argument("--deals", action='store_true', default=False, help=f"dump deals to log")
    parser.add_argument("--all", action='store_true', default=False, help=f"process all bots")

    args = parser.parse_args(args)
    setup_logging(args.verbosity, filename='/tmp/local-eval.log')

    method = VERSION_TO_METHOD[args.version]

    with DB() as db:
        if args.all:
            args.md5 = list(map(lambda x: x.md5, db.individual.get_all(include_portfolio=False)))

        individuals = list(map(db.individual.get, args.md5))
        attributes = list(map(partial(db.individual.attribute.get, args.currency, args.version), args.md5))
        ann_params = [db.ann.get(db.individual.get_db(i.md5()).ann_id) for i in individuals]

        assert len(individuals) == len(args.md5)
        assert len(attributes) == len(args.md5)
        assert len(ann_params) == len(args.md5)

        offset = 0
        for ann in ann_params:
            offsets = json.loads(ann.offsets)
            indicators = json.loads(ann.indicators)
            offset = max(offset, max_offset(indicators, offsets) - 1)

        from_ts = parse_dt(args.history_ts).replace(tzinfo=timezone.utc) - timedelta(minutes=offset)
        to_ts = parse_dt(args.end_ts).replace(tzinfo=timezone.utc)
        df_all = DataCache(db, Price, args.currency).load(from_ts=from_ts, to_ts=to_ts)[LABEL_COLUMNS].copy()
        df_preload = df_all.head(len(df_all) - args.realtime_offset).copy()

        moneys = []
        for individual, attr in zip(individuals, attributes):
            generator = DealGenerator(db=db, attributes=attr, read_only=True)
            # deals, _ = generator.process_pending()
            deals, _, _ = generator.process(current_usd=0, df=df_preload, ts_from=from_ts)

            deals = deals.copy()
            money = generator.money
            if args.realtime_offset:
                new_deals, _ = generator.process_pending(ts_to=parse_dt(args.end_ts).replace(tzinfo=timezone.utc))
                deals.extend(new_deals)

            if args.deals:
                list(map(logging.info, deals))

            moneys.append(money)

        min_length = min(map(len, moneys))
        df_graph = df_all.tail(min_length).copy()
        for individual, money in zip(individuals, moneys):
            df_graph[individual.md5()] = money[-min_length:]

        df_graph.to_csv('data/local-eval.csv')
        if args.graphs:
            unique_name = f"data/logs/pic/{'-'.join(args.md5)}_{method.name}.png"
            plot_to_file(df=df_graph, bots=args.md5, graph_path=unique_name, size_x=200, size_y=16)


if __name__ == "__main__":
    main()
