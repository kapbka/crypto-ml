import argparse
import asyncio
import logging
import os
import sys

import joblib

from common.constants import DEFAULT_CURRENCY, TRAIN_FILE
from common.log import setup_logging
from common.serialization import load_individuals, OLD_LSTM
from common.storage.api import Storage
from db.api import DB

SCALER_PATTERN = 'data/cloud/scaled/{currency}_last_{days}days_1min_01september.csv/MinMaxScaler/scaler.dat'


def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="This script imports individuals from a folder to a DB")
    parser.add_argument("-v", "--verbosity", type=str, default='INFO', help="verbosity level: INFO/DEBUG")
    parser.add_argument("md5", type=str, nargs='*', help=f"list of individual md5 hashes")
    parser.add_argument("--folder", type=str, required=True, help=f"folder to import from")

    args = parser.parse_args(args)
    setup_logging(args.verbosity)

    with DB() as db:
        if not os.path.exists(args.folder):
            asyncio.get_event_loop().run_until_complete(Storage().sync_folder(args.folder))

        individuals, params, scalers = load_individuals(args.folder)
        logging.info(f"Loaded {len(individuals)} individuals from {args.folder}")

        # also save default scaler
        scaler_path = SCALER_PATTERN.format(days=TRAIN_FILE.days, currency=DEFAULT_CURRENCY)
        if not os.path.exists(scaler_path):
            asyncio.new_event_loop().run_until_complete(Storage().download(scaler_path, scaler_path))

        db.currency.set(DEFAULT_CURRENCY, f"{DEFAULT_CURRENCY.upper()}USDT")
        old_scaler = db.scaler.set(DEFAULT_CURRENCY, joblib.load(scaler_path))
        db.flush()

        logging.info(f"Old scaler: {old_scaler}")

        counter = 0
        for individual, (layers, offsets, indicators, is_scaled) in zip(individuals, params):
            if args.md5 and individual.md5() not in args.md5:
                continue

            scaler_id = None
            if layers == OLD_LSTM:
                # this is an old individual that should be scaled using minmax scaler from the cloud
                is_scaled = True
                scaler_id = old_scaler.id
            elif is_scaled:
                # this is a new individual that is scaled with STD scaler, we need to create it here
                scaler = db.scaler.set(DEFAULT_CURRENCY, scalers[0])
                db.flush()
                scaler_id = scaler.id
                logging.info(f"New scaler: {scaler}")

            ann = db.ann.set(layers=layers, offsets=offsets, indicators=indicators, scaled=is_scaled)
            db.flush()
            db.individual.set(individual, ann_id=ann.id, scaler_id=scaler_id, train_currency=DEFAULT_CURRENCY)

            logging.info(f"Saving {individual.md5()} with {individual.fitness}")

            counter += 1

        logging.info(f"Saved {counter} individuals")


if __name__ == "__main__":
    main()
