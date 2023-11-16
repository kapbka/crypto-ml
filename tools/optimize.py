import argparse
import datetime
import json
import logging
import pickle
import sys

import numpy as np
from dateutil.parser import parse as parse_dt
from scipy.optimize import minimize

from common.constants import EPOCH, DEFAULT_CURRENCY, VERSION_TO_METHOD, LABEL_COLUMNS
from common.metrics.evaluator import evaluate
from db.api import DB
from db.data_cache import DataCache
from db.model import Price
from models.ann import make_inputs, create_ann, columns_from_indicators
from preprocessing.batch_rsi import Preprocessor


def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="This script runs evaluation of one or multiple individuals")
    parser.add_argument("-v", "--verbosity", type=str, default='INFO', help="verbosity level: INFO/DEBUG")
    parser.add_argument("--parent-md5", type=str, help=f"parent bot md5", required=True)
    parser.add_argument("--fitness-index", type=int, help=f"index of the fitness tuple to use", default=0)
    parser.add_argument("--min-fitness", type=float, help=f"minimum fitness value to start from", default=0.3)
    parser.add_argument("--initial", action='store_true', default=False, help=f"optimize initial weights")
    parser.add_argument("--version", type=int, required=True, help=f"version id to use")
    parser.add_argument("--start-ts", type=str, default=EPOCH.isoformat(),
                        help=f"datetime to start evaluation from, default: {EPOCH.isoformat()}")
    parser.add_argument("--end-ts", type=str, default=datetime.datetime.now().isoformat(),
                        help=f"datetime to stop evaluation at, default: {datetime.datetime.now().isoformat()}")
    parser.add_argument("--algorithm", type=str, default='Nelder-Mead', help=f"algorithm to use, default: Nelder-Mead")
    parser.add_argument("--weights", type=str, help=f"list of floats to use instead of last layer weights")
    parser.add_argument("--distribution", type=str,
                        help=f"list of random distribution parameters to apply on top of initial weights")
    parser.add_argument("--currency", type=str, default=DEFAULT_CURRENCY,
                        help=f"currency to use, default: {DEFAULT_CURRENCY}")

    args = parser.parse_args(args)

    method = VERSION_TO_METHOD[args.version]

    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(name)s:%(message)s', level=logging.INFO)

    with DB() as db:
        individual = db.individual.get(args.parent_md5)
        logging.info(f"Last layer weights: {individual.weights[-1]}")

        attr = db.individual.attribute.get(args.currency, args.version, args.parent_md5)
        db_individual = db.individual.get_db(individual.md5())
        ann_params = db.ann.get(db_individual.ann_id)
        offsets = json.loads(ann_params.offsets)
        layers = json.loads(ann_params.layers)
        indicators = json.loads(ann_params.indicators)

        df_all = DataCache(db, Price, args.currency).load(from_ts=parse_dt(args.start_ts),
                                                          to_ts=parse_dt(args.end_ts))[LABEL_COLUMNS].copy()

    columns = columns_from_indicators(indicators)
    ann = create_ann(layers, offsets, columns)

    Preprocessor().apply_indicators(df_all, indicators)
    df_all.dropna(inplace=True)

    scaler_obj = db.scaler.get(attr.scaler_id) if attr.scaler_id and ann_params.is_scaled else None
    scaler = pickle.loads(scaler_obj.data) if scaler_obj else None
    if scaler:
        df_all[columns] = scaler.transform(df_all[columns].values)

    data_inputs, prices, ts = make_inputs(df_all, offsets, columns)

    def run_with_weights(weights: np.ndarray):
        individual.weights[-1] = weights

        all_weights, eval_args = individual.weights_and_limits(layers)

        ann.set_weights(all_weights)

        predict_array = ann.predict(data_inputs)

        money = np.zeros(len(prices))

        result, _, _ = evaluate(prices, predict_array, money, method, 0, 0, 0, 0, False, eval_args[0], eval_args[1], 0)
        logging.info(f"Res: {result}, weights: {weights}")
        if not result[args.fitness_index]:
            return 100
        return -result[args.fitness_index]

    if args.weights:
        new_weights = [w for w in args.weights.split(" ") if w]
        new_weights = np.array(list(map(float, new_weights)))
        logging.info(f"Using provided weights instead of existing: {new_weights}")
        individual.weights[-1] = new_weights

    distribution_params = []
    if args.distribution:
        for distr in args.distribution.split(" "):
            distribution_params.append(list(map(float, distr.split(","))))

    if args.initial:
        x0 = np.concatenate((individual.weights[-1], np.array([0.02, 0.02])))

        res = minimize(fun=run_with_weights, method=args.algorithm, x0=x0)
        logging.info(f"Initial res: {res}")

    logging.info(f"Using distribution params: {distribution_params}")

    while True:
        x0 = np.array([])

        for param in distribution_params:
            x0 = np.concatenate((x0, np.random.uniform(param[0], param[1], 1)))

        initial = run_with_weights(x0)
        if initial < -args.min_fitness:
            res = minimize(fun=run_with_weights, method=args.algorithm, x0=x0)
            logging.info(f"Min res: {res}")


if __name__ == '__main__':
    main()
