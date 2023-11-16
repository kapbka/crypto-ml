import argparse
import gc
import logging
import os.path
import pickle
import sys
import time
from datetime import datetime, timezone, date
from typing import List, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from dateutil import parser as date_parser
from sklearn.preprocessing import StandardScaler

from common.constants import DEFAULT_CURRENCY, EvaluationMethod, LABEL_COLUMNS
from common.individual import Individual
from common.k8s.probes import mark_ready, mark_alive
from common.log import setup_logging
from common.metrics.evaluator import evaluate as eval_ds
from common.storage.file import File
from db.api.db import DB
from db.data_cache import DataCache
from db.model.processing import Price
from models.ann import create_ann, make_inputs, stringify, ModelParams
from preprocessing.batch_rsi import Preprocessor
from preprocessing.oco_marker import create_labels_with_predictions

BUY_LIMIT = 0.02
SELL_LIMIT = -BUY_LIMIT
DEFAULT_OFFSETS = list(reversed(range(0, 240 + 1, 10)))


class Evaluator:
    def __init__(self, ann, data_inputs: np.ndarray, prices: np.ndarray, ts: np.ndarray, base_path: str,
                 offsets: List[int], indicators: Dict[str, List[int]], is_scaled: bool, ann_params: ModelParams,
                 save_profit: float, scaler: StandardScaler, currency: str):
        self._ann = ann
        self._data_inputs = data_inputs
        self._prices = prices
        self._ts = ts
        self._base_path = base_path
        self._offsets = offsets
        self._indicators = indicators
        self._is_scaled = is_scaled
        self._ann_params = ann_params
        self._save_profit = save_profit
        self._scaler = scaler
        self._currency = currency

        self._best_profit = -100
        self.best_path = ''
        self.best_weights = []

    def evaluate_and_save(self, epoch, logs):
        mark_alive()

        res, _ = evaluate(self._ann, self._data_inputs, self._prices, self._ts, BUY_LIMIT, SELL_LIMIT)

        if res[0] > self._best_profit:
            self._best_profit = res[0]
            self.best_weights = self._ann.get_weights()

            individual = Individual(weights=self._ann.get_weights(), fitness=res)

            logging.info(f"New best profit: {res}, md5: {individual.md5()}")

            self.best_path = os.path.join(self._base_path, f"{epoch}_{'-'.join(map(str, res))}_{individual.md5()}")

            with open(self.best_path, "wb") as out:
                pickle.dump((individual, self._ann_params, self._offsets, self._indicators, self._is_scaled), out)

            if res[1] > self._save_profit and res[2]:
                with DB() as db:
                    scaler_id = None
                    if self._is_scaled:
                        scaler = db.scaler.set(self._currency, self._scaler)
                        db.flush()
                        scaler_id = scaler.id

                    ann = db.ann.set(layers=self._ann_params, offsets=self._offsets,
                                     indicators=self._indicators, scaled=self._is_scaled)
                    db.flush()

                    db.individual.set(bot=individual, ann_id=ann.id, scaler_id=scaler_id, train_currency=self._currency)

                    logging.info(f"Saved {individual.md5()} with {res}, scaler id: {scaler_id}, ann: {ann}")


def compile_and_fit(model, data_inputs: np.ndarray, labels: np.ndarray, epochs: int, evaluator: Evaluator):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, mode='min')
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    mark_ready()
    return model.fit(data_inputs, labels, verbose=2, epochs=epochs, callbacks=[
        tf.keras.callbacks.LambdaCallback(on_epoch_end=evaluator.evaluate_and_save)])


def evaluate(ann, data_inputs: np.ndarray, prices: np.ndarray, ts: np.ndarray, buy_percent: float, sell_percent: float):
    predictions = ann.predict(data_inputs, verbose=0)
    tf.keras.backend.clear_session()
    gc.collect()

    money = np.zeros(len(predictions))
    res, _, deals, = eval_ds(prices=prices, predictions=predictions, money=money,
                             method=EvaluationMethod.OCO, current_usd=0, start_usd=0, buy_limit_cap=0,
                             sell_limit_cap=0, use_max_sell_cap=False, oco_buy_percent=buy_percent,
                             oco_sell_percent=sell_percent, oco_rise_percent=0.0001)

    logging.info(res)

    df_graph = pd.DataFrame({'close': prices[:, 0],
                             'up_predicted': predictions[:, 0],
                             'down_predicted': predictions[:, 1]},
                            index=ts)
    df_graph['money'] = money

    return res, df_graph


def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Creates labels based on OCO orders and trains ANN")
    parser.add_argument("-v", "--verbosity", type=str, default='INFO', help="verbosity level: INFO/DEBUG")
    parser.add_argument("--currency", type=str, default=DEFAULT_CURRENCY,
                        help=f"currency to use, default: {DEFAULT_CURRENCY}")
    parser.add_argument("--begin", type=str, default=datetime(2021, 4, 1).isoformat(),
                        help=f"datetime to start from, default: {datetime(2021, 4, 1).isoformat()}")
    parser.add_argument("--end", type=str, default=datetime(2021, 11, 1).isoformat(),
                        help=f"datetime to stop training at, default: {datetime(2021, 11, 1).isoformat()}")
    parser.add_argument("--cells", type=int, help=f"number of LSTM cells", default=256)
    parser.add_argument("--epoch", type=int, help=f"number of train epoch", default=100)
    parser.add_argument("--seed", type=int, help=f"seed number", default=int((time.time() * 1000) % 2**32))
    parser.add_argument("--std-scaler", action='store_true', default=False, help=f"use StandardScaler")
    parser.add_argument("--offsets", type=str, default=','.join(map(str, DEFAULT_OFFSETS)),
                        help=f"data offsets to use, default: {','.join(map(str, DEFAULT_OFFSETS))}")
    parser.add_argument("--save-profit", type=float, help=f"minimum profit value to save to a DB", default=0.2)
    parser.add_argument("--parent-md5", type=str, help=f"parent bot md5")
    parser.add_argument("--labels", type=str, help=f"label configuration, default: 0.02,20 0.03,30",
                        default="0.02,20 0.03,30")
    parser.add_argument("--only-crashes", action='store_true', default=False, help=f"train on crashes only")
    parser.add_argument("--infinite", action='store_true', default=False, help=f"run infinitely")
    parser.add_argument("--crash-params", type=str, help=f"crash detection config, default: {60*24*1},{60*24*1},{5}",
                        default=f"{60*24*1},{60*24*1},{5}")

    args = parser.parse_args(args)

    setup_logging(filename='/tmp/trainer.log')

    logging.info(args)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        logging.info(f"GPU: {gpus[0]}")
        tf.config.experimental.set_memory_growth(gpus[0], True)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    label_config = np.array([np.array(list(map(float, label.split(",")))) for label in args.labels.split(" ")])

    start_weights = []
    with DB() as db:
        from_ts = date_parser.parse(args.begin).replace(tzinfo=timezone.utc)
        df_all = DataCache(db, Price, args.currency).load(from_ts=from_ts)
        logging.info(f"Clean data: {df_all.index[0]} -> {df_all.index[-1]}")

        if args.parent_md5:
            start_weights = db.individual.get(args.parent_md5).weights

    df_all_inputs = df_all[LABEL_COLUMNS + ['volume']].copy()

    train_columns, indicators = Preprocessor.create_indicators(df_all_inputs)
    intervals = list(map(int, args.crash_params.split(",")))
    Preprocessor.mark_crashes(
        df=df_all_inputs,
        detection_length=intervals[0],
        recovery_length=intervals[1],
        amount=intervals[2] / 100,
    )
    if not args.only_crashes:
        df_all_inputs['crash'] = np.where(df_all_inputs['crash'] == 1, 0, 1)

    df_all_inputs.dropna(inplace=True)
    logging.info(f"Preprocessed data: {df_all.index[0]} -> {df_all.index[-1]}")
    logging.info(df_all_inputs.value_counts(["crash"]))

    model = [('LSTM', dict(units=args.cells)), ('Dense', dict(units=3, activation='sigmoid'))]

    ts_end = date_parser.parse(args.end).replace(tzinfo=timezone.utc)
    interval = ts_end.date() - date(df_all.index[0].year, df_all.index[0].month, df_all.index[0].day)

    scaler = StandardScaler()
    train_file = File(ts=ts_end.date(), days=interval.days, ticker=args.currency)
    unique_name = stringify(model, scaler, train_file)
    label_cfg = "-".join(map(lambda x: f"{x[0]}_{x[1]}", label_config))
    offsets = list(map(int, args.offsets.split(",")))
    saved_path = f"data/cloud/checkpoints/models/{args.currency}/{unique_name}_{args.seed}_{label_cfg}_{'-'.join(map(str, offsets))}"
    os.makedirs(saved_path, exist_ok=True)

    df = df_all_inputs[:ts_end].copy()
    df_test = df_all_inputs[ts_end:].copy()

    logging.info(f"Train data: {df.index[0]} -> {df.index[-1]}")
    logging.info(f"Test data: {df_test.index[0]} -> {df_test.index[-1]}")

    if args.std_scaler:
        df[train_columns] = scaler.fit_transform(df[train_columns].values)
        df_test[train_columns] = scaler.transform(df_test[train_columns].values)
        with open(os.path.join(saved_path, "scaler.pickle"), "wb") as out:
            pickle.dump(scaler, out)

    labels = np.zeros((len(df), 3))
    create_labels_with_predictions(
        close=df['close'].values,
        high=df['high'].values,
        low=df['low'].values,
        labels=labels,
        limits=label_config,
        filter=df['crash'].values,  # use positive values as filter(mark only in a crash)
    )

    df['buy'] = labels[:, 0]
    df['sell'] = labels[:, 1]
    df['do_nothing'] = labels[:, 2]

    logging.info(df.value_counts(["buy", "sell", "do_nothing"]))

    data_inputs, labels, ts = make_inputs(df,
                                          offsets=offsets,
                                          label_col=['buy', 'sell', 'do_nothing', 'close', 'high', 'low'],
                                          columns=train_columns)
    data_inputs_test, prices_test, ts_test = make_inputs(df_test, offsets=offsets, columns=train_columns)

    logging.info(f"Data len: {data_inputs.shape}, path: {saved_path}")

    ann = create_ann(model, offsets, train_columns)
    ann.summary(print_fn=logging.info)

    while True:
        weights = [tf.keras.initializers.glorot_uniform(seed=np.random.randint(0, 1000))(w.shape) if w.ndim > 1 else w
                   for w in ann.get_weights()]
        ann.set_weights(weights)

        if start_weights:
            ann.set_weights(start_weights)
            if not args.save_profit:
                parent_res, _ = evaluate(ann, data_inputs_test, prices_test, ts_test, BUY_LIMIT, SELL_LIMIT)
                logging.info(f"Treating parent result as minimum profit for save: {parent_res}")
                args.save_profit = parent_res[1]

        evaluator = Evaluator(ann=ann, data_inputs=data_inputs_test, prices=prices_test, ts=ts_test,
                              base_path=saved_path, offsets=offsets, indicators=indicators, is_scaled=args.std_scaler,
                              ann_params=model, save_profit=args.save_profit, scaler=scaler, currency=args.currency)
        compile_and_fit(ann, data_inputs=data_inputs, labels=labels[:, :3], epochs=args.epoch, evaluator=evaluator)

        ann.set_weights(evaluator.best_weights)

        evaluate(ann, data_inputs, labels[:, 3:], ts, BUY_LIMIT, SELL_LIMIT)
        evaluate(ann, data_inputs_test, prices_test, ts_test, BUY_LIMIT, SELL_LIMIT)

        if not args.infinite:
            break


if __name__ == '__main__':
    main()
