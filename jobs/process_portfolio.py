import argparse
from datetime import datetime, timezone
import logging
import sys

from common.constants import EPOCH, IsPortfolio
from common.log import setup_logging
from common.portfolio import process_historical_portfolios
from db.api import DB


def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="This script runs portfolio deals calculation")
    parser.add_argument("-v", "--verbosity", type=str, default='INFO', help="verbosity level: INFO/DEBUG")

    args = parser.parse_args(args)
    setup_logging(args.verbosity, filename='/tmp/portfolio_deals.log')

    with DB() as db:
        portfolios = db.individual.get_history(portfolio=IsPortfolio.Yes)
        logging.info(f"Processing {len(portfolios)} portfolios")

        completed = 0
        total = len(portfolios)
        time_started = datetime.now(tz=timezone.utc)
        for _, attributes in portfolios:
            logging.info(f"Processing portfolio {attributes.md5}")
            completed += process_historical_portfolios(
                db=db,
                version=attributes.version_id,
                currency=attributes.currency_code,
                ts_from=EPOCH,
                portfolio_md5=attributes.md5
            )

            speed = (datetime.now(tz=timezone.utc) - time_started) / completed
            remaining_time = (total - completed) * speed
            eta = datetime.now(tz=timezone.utc) + remaining_time
            db.history_stat.set(
                speed=speed,
                eta=eta,
                processed_count=completed,
                remaining_count=len(portfolios) - completed
            )

            db.commit()

        db.commit()
        logging.info(f"Done processing of {completed} portfolios")


if __name__ == '__main__':
    main()
