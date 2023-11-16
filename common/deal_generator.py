import json
import logging
import pickle
from datetime import timedelta, datetime
from functools import partial
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from common.constants import DealStatus, LABEL_COLUMNS, VERSION_TO_METHOD, EPOCH
from common.individual import Individual
from common.metrics.evaluator import Fitness, Limits, evaluate, convert_deal
from db.api import DB
from db.data_cache import DataCache
from db.model import Price, Deal, IndividualAttribute
from models.ann import make_inputs, Ann, columns_from_indicators, create_ann
from preprocessing.batch_rsi import Preprocessor


PREDICTION_COLUMNS = ['buy', 'sell', 'idle']


class DealGenerator:
    models_cache = {}
    data_cache = {}

    def __init__(self, db: DB, attributes: IndividualAttribute, read_only=False):
        self._db = db
        self._read_only = read_only

        db_individual = db.individual.get_db(attributes.individual_id)
        ann_params = db.ann.get(db_individual.ann_id)
        layers = json.loads(ann_params.layers)

        self._offsets = json.loads(ann_params.offsets)
        self._indicators = json.loads(ann_params.indicators)
        self._currency = attributes.currency_code
        self._bot_id = attributes.individual_id
        self._bot_md5 = attributes.md5

        self._columns = columns_from_indicators(self._indicators)
        self._ann = self._get_model(model=layers, offsets=self._offsets)
        self._version = attributes.version_id
        self._method = VERSION_TO_METHOD[attributes.version_id]
        self._eval_args: List[float] = list()
        self._eval_kwargs: Dict[str, float] = dict()
        self._scaler_id: int = attributes.scaler_id

        for attr in ['oco_buy_percent', 'oco_sell_percent', 'oco_rise_percent']:
            self._eval_kwargs[attr] = getattr(attributes, attr)

        self._scaler = pickle.loads(db.scaler.get(attributes.scaler_id).data) if attributes.scaler_id else None

        weights, self._eval_args = Individual(weights=pickle.loads(db_individual.weights)).weights_and_limits(layers)
        self._ann.set_weights(weights)

        self._start_usd = db.price.get_first(self._currency).close

        self.money: List[float] = list()

    def _load_predictions(self, ts_from: datetime, ts_to: datetime) -> pd.DataFrame:
        """
        Loads predictions from the database using currency, version and bot id
        @param ts_from: ts to start from
        @param ts_to: ts to end with
        @return: pandas dataframe containing data
        """
        ts = datetime.now()
        data = self._db.prediction.load(
            ts_from=ts_from,
            currency=self._currency,
            version=self._version,
            bot_id=self._bot_id
        )
        spent = datetime.now() - ts
        logging.debug(f"Loaded {len(data)} cached predictions for {self._bot_md5} from {ts_from}, spent: {spent}")
        return data[:ts_to]

    def _predict_with_cache(self, df: pd.DataFrame, ts_from: datetime, ann: Ann) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Checks cache before running ANN prediction, updates DB cache after ANN prediction is done
        @param df: dataset containing prices
        @param ts_from: timestamp to start from
        @param ann: ANN reference
        @return: returns predictions, prices, timestamps
        """
        # try to load from the cache first
        cached = self._load_predictions(ts_from=ts_from, ts_to=df.index[-1])
        empty = not len(cached)

        # make sure latest price is newer or equal to the latest prediction
        assert empty or cached.index[-1] <= df.index[-1]

        result_predictions = cached[PREDICTION_COLUMNS].values if not empty else np.array([])
        result_ts = cached.index.values.astype(np.datetime64) if not empty else np.array([])

        # this much data is missing and needs to be predicted
        predict_from = (cached.index[-1] if not empty else ts_from) + timedelta(minutes=1)

        # check that we have enough data in the dataset to generate new predictions
        if predict_from < df.index[-1]:
            # make sure we have enough data to get first prediction
            predict_from -= timedelta(minutes=max(self._offsets))

            # generate data for this interval
            data_inputs, _, ts = make_inputs(
                df=df[predict_from:].dropna(),
                offsets=self._offsets,
                columns=columns_from_indicators(self._indicators)
            )

            # make sure there is no data gap or overlap
            assert empty or cached.index[-1] + timedelta(minutes=1) == pd.to_datetime(ts[0], utc=True)
            assert int((ts[-1] - ts[0]) / np.timedelta64(1, 'm')) == len(ts) - 1

            # run actual ANN
            predicted_array = ann.predict(data_inputs, verbose=0)

            if not self._read_only:
                # store results to the cache
                logging.debug(f"Saving {len(predicted_array)} predictions for {self._bot_md5}")

                df_save = pd.DataFrame(predicted_array, index=ts, columns=PREDICTION_COLUMNS)
                df_save['currency_code'] = self._currency
                df_save['version_id'] = self._version
                df_save['individual_id'] = self._bot_id

                assert int((df_save.index[-1] - df_save.index[0]).total_seconds() // 60) == len(df_save) - 1

                self._db.prediction.save(df_save)
                self._db.commit()

            # merge arrays
            result_predictions = np.concatenate((result_predictions, predicted_array)) if not empty else predicted_array
            result_ts = np.concatenate((result_ts, ts)) if not empty else ts

            assert len(result_ts) == len(result_predictions)

        prices = df[pd.to_datetime(result_ts[0], utc=True):
                    pd.to_datetime(result_ts[-1], utc=True)] if len(result_ts) else df.head(0)
        assert len(prices) == len(result_predictions)

        return result_predictions, prices[LABEL_COLUMNS].values, result_ts

    def _get_model(self, model: List[Tuple[str, dict]], offsets: List[int]):
        """
        Loads ANN model using cache
        @param model: model config
        @param offsets: data label offsets
        @return: ANN
        """
        key = pickle.dumps((model, offsets, self._columns))
        if key not in self.models_cache:
            self.models_cache[key] = create_ann(params=model, offsets=offsets, train_columns=self._columns)
        return self.models_cache[key]

    def _append_data(self, df: pd.DataFrame):
        """
        Appends data to the price cache
        @param df: new data
        @return: None
        """
        key = json.dumps([self._currency, self._indicators, self._scaler_id])
        logging.debug(f"Appending data for {self._currency} from {self.data_cache[key].index[-1]}, size: {len(df)}")

        # make sure there is no gap here
        assert self.data_cache[key].index[-1] + timedelta(minutes=1) == df.index[0]

        self.data_cache[key] = pd.concat([self.data_cache[key], df])
        self._preprocess(self.data_cache[key])

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocessed dataframe with optional scaling
        @param df: dataframe containing prices
        @return: dataframe containing indicators
        """
        Preprocessor().apply_indicators(df, self._indicators)

        if self._scaler:
            df[self._columns] = self._scaler.transform(df[self._columns].values)
        return df

    def _fetch_prices(self, from_ts: datetime, df: Optional[pd.DataFrame]):
        """
        Wrapper for loading prices from the database or taking them from passed dataframe
        @param from_ts: timestamp to start from
        @param df: dataframe containing prices, optional
        @return: dataframe containing prices
        """
        if df is not None:
            logging.info(f"Using provided prices from {df.index[0]} to {df.index[-1]}")
            return df

        logging.debug(f"Downloading data for {self._currency} from {from_ts}, scaler: {self._scaler_id}")
        res = DataCache(self._db, Price, self._currency).load(from_ts=from_ts)[LABEL_COLUMNS]
        logging.debug(f"Downloaded {len(res)} rows for {self._indicators.keys()}, "
                      f"last ts: {res.index[-1]}")
        return res

    def _load_data(self, ts_to: Optional[datetime], df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Loads preprocessed prices data, uses cache if already preprocessed
        @param ts_to: ts to start from
        @param df: dataframe containing prices, optional
        @return: preprocessed dataframe
        """
        key = json.dumps([self._currency, self._indicators, self._scaler_id])
        if key not in self.data_cache:
            self.data_cache[key] = self._fetch_prices(from_ts=EPOCH, df=df)
            self._preprocess(self.data_cache[key])

        elif ts_to and self.data_cache[key].index[-1] < ts_to:
            # check if we can download more
            more = self._fetch_prices(from_ts=self.data_cache[key].index[-1], df=None)
            if len(more):
                self._append_data(more)

        return self.data_cache[key][:ts_to]

    def _run_ann(self, df: pd.DataFrame, ts_from: datetime, start_usd: float, current_usd: float) -> \
            Tuple[Fitness, np.ndarray, Limits, np.ndarray, List[Deal]]:
        """
        Runs evaluation of a bot with current weights on a dataset
        @return: fitness value, last prices, last limits, money, deals
        """

        predict_array, prices, ts = self._predict_with_cache(df=df, ts_from=ts_from, ann=self._ann)
        if not len(prices):
            return (0, 0, 0), np.array([]), (0, 0, 0, 0), np.array([]), []

        money = np.zeros(len(prices))
        eval_res, actions, deals = evaluate(prices, predict_array, money, self._method, current_usd, start_usd,
                                            *self._eval_args, **self._eval_kwargs)

        return eval_res, prices[-1], actions, money, list(map(partial(convert_deal, ts, self._bot_id), deals))

    def process(self, current_usd: float, df: Optional[pd.DataFrame] = None,
                ts_from: datetime = EPOCH, ts_to: Optional[datetime] = None) -> Tuple[List[Deal], Fitness, Limits]:
        """
        Processes prices up to the latest and generates deals.
        @param current_usd: current running sum USD to start with
        @param df: dataframe to load data from
        @param ts_from: timestamp to start from, EPOCH by default
        @param ts_to: timestamp to end, the latest by default
        @return: returns fitness and last action(limit order prices)
        """
        df = self._load_data(ts_to=ts_to, df=df)
        eval_res, prices, limit_prices, self.money, deals = self._run_ann(
            df=df,
            ts_from=ts_from,
            start_usd=self._start_usd if df is None else 0,
            current_usd=current_usd
        )
        return deals, eval_res, limit_prices

    def process_pending(self, ts_to: Optional[datetime] = None) -> Tuple[List[Deal], Limits]:
        """
        Processes prices starting from last deal.
        @param ts_to: last known price timestamp
        @return:
        """
        last_deal = self._db.deal.get_last(
            currency=self._currency,
            individual=self._bot_id,
            version=self._version,
            status=DealStatus.Close
        )

        start_ts = EPOCH if not last_deal else last_deal.sell_ts
        current_usd = 0 if not last_deal else last_deal.run_usd
        deals, _, limit_prices = self.process(current_usd=current_usd, ts_from=start_ts, ts_to=ts_to)
        return deals, limit_prices
