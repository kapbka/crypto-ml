import argparse
import logging
import sys

import numpy as np

from db.api import DB
from common.constants import DEFAULT_CURRENCY


def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="This script patches an individual with passed weights")
    parser.add_argument("--parent-md5", type=str, help=f"parent individual md5", required=True)
    parser.add_argument("weights", type=str, nargs='+', help=f"list of floats to add")

    args = parser.parse_args(args)

    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(name)s:%(message)s', level=logging.INFO)

    with DB() as db:
        individual = db.individual.get(args.parent_md5)
        logging.info(f"Existing weights: {individual.weights[-1]}")

        new_weights = np.array(list(map(float, args.weights)))
        individual.weights[-1] = new_weights
        logging.info(f"Received weights: {new_weights}")

        logging.info(f"New md5: {individual.md5()}")
        db_data = db.individual.get_db(args.parent_md5)
        attrs = db.individual.attribute.get(DEFAULT_CURRENCY, 5, args.parent_md5)

        db.individual.set(
            individual,
            ann_id=db_data.ann_id,
            parent_md5=args.parent_md5,
            scaler_id=attrs.scaler_id,
            train_currency=DEFAULT_CURRENCY
        )


if __name__ == '__main__':
    main()
