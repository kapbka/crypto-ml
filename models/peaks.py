from functools import partial

import numpy as np
import pandas as pd
import pandas_ta as ta
from joblib import dump
from joblib import load
from preprocessing.indicators.ta.fi import FI
from preprocessing.indicators.ta.macd import MACD
from preprocessing.indicators.ta.rsi import RSI
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class PeakBasedModel:
    GROUPING_MINUTES = 60

    def __init__(self, df):
        self._peak_model = load('models/trained/peak.joblib')
        self._fast_model = load('models/trained/fast.joblib')

        self._close = []
        self._volume = []
        self._ts = []

        self._df = df
        self._peaks = []
        self._last_action = 0
        self._buy_price = 0
        self._time_buy = None
        self._price_raise_count = 0

    @staticmethod
    def _peak(prev_name, current_name, x):
        if x[prev_name] < 0 < x[current_name]:
            return -1
        if x[prev_name] > 0 > x[current_name]:
            return 1
        return 0

    def _action(self, x):
        if self._buy_price:
            if x['close'] > self._buy_price:
                self._price_raise_count += 1
            else:
                self._price_raise_count -= 1

        if x['predicted_change_fast'] > 0.02:  # 0.01 gives a lot on the crash
            # react on huge raise
            self._last_action = 1
        elif x['predicted_change_fast'] < -0.02:  # -0.01 gives a lot on the crash
            # react on huge drop
            self._last_action = -1
        elif not x['predicted_peak_prev'] and x['predicted_peak']:
            # buy action if prediction changed from 0 to 1
            self._last_action = 1
        elif x['rolling_max_diff'] > 0.014:
            # if price is dropping rapidly
            self._last_action = -1
        elif x['slope_fast'] < 0 and not x['predicted_peak'] and x['mean_slope'] <= 0:
            # sell prediction with two additional conditions:
            # 1. make sure price is actually dropping(slope_fast)
            # 2. make sure mean peak slope is not raising(mean_slope)
            self._last_action = -1
        elif self._price_raise_count < -6:
            # check how many drops in prices we have since we bought, -6 works the best ...
            self._last_action = -1
        else:
            self._last_action = 0

        # track price we bought and time
        if self._last_action == 1:
            self._buy_price = x['close']
            self._time_buy = x.name
            self._price_raise_count = 0
        elif self._last_action == -1:
            self._buy_price = 0
            self._time_buy = None
            self._price_raise_count = 0

        return self._last_action

    @staticmethod
    def preprocess(df, train=True, indicator_window=16, close_window=16, mean_window=3, peak_history_size=6,
                   rolling_max_window=2):
        x_columns = []
        for i in [FI(indicator_window * 2),
                  MACD(indicator_window * 2, indicator_window * 4, indicator_window * 2 - 1, indicator_window * 2),
                  RSI(indicator_window * 2)]:
            x_columns.extend(i.apply(df))
        x_columns *= 2

        df.dropna(inplace=True)

        df['close_rolling'] = df['close'].rolling(window=close_window).mean()
        df['rolling_slope'] = ta.slope(df['close_rolling'], length=1)

        df['slope_on_slope'] = ta.slope(df['rolling_slope'])
        df['slope_on_slope_prev'] = df['slope_on_slope'].shift(1)

        df['peak'] = df.apply(partial(PeakBasedModel._peak, 'slope_on_slope_prev', 'slope_on_slope'), axis=1)

        peaks = df[df['peak'] != 0].copy()
        peaks['mean'] = peaks['rolling_slope'].rolling(window=mean_window).mean()
        if train:
            next_peak = peaks['mean'].shift(-1)
            change_peak = (next_peak - peaks['mean']) / peaks['mean']
            peaks['peak_change'] = np.where(change_peak > 0, 1, 0)

        for i in range(0, peak_history_size):
            peaks[f'slope_shifted_{i}'] = peaks['rolling_slope'].shift(i)
            peaks[f'mean_{i}'] = peaks[f'mean'].shift(i)

            x_columns.append(f'slope_shifted_{i}')
            x_columns.append(f'mean_{i}')

        peaks.dropna(inplace=True)
        joined = peaks.reindex(df.index, method='pad')

        if train:
            df['peak_change'] = joined['peak_change']

        df['mean_peak'] = joined['mean']

        for i in range(0, peak_history_size):
            df[f'slope_shifted_{i}'] = joined[f'slope_shifted_{i}']
            df[f'mean_{i}'] = joined[f'mean_{i}']

        df.dropna(inplace=True)

        # fast prediction
        if train:
            df['change'] = (df['close'].shift(-1) - df['close']) / df['close']
        else:
            # stuff for additional algorithm on top
            df['rolling_max'] = df['close'].rolling(window=rolling_max_window).max()
            df['rolling_max_diff'] = (df['rolling_max'] - df['close']) / df['rolling_max']
            df['slope_fast'] = ta.slope(df['close'], length=1)
            df['mean_slope'] = ta.slope(df['mean_peak'], length=1)

        return x_columns

    def _predict(self, x_columns):
        self._df['predicted_peak'] = self._peak_model.predict(self._df[x_columns])
        self._df['predicted_peak_prev'] = self._df['predicted_peak'].shift(1)
        self._df['predicted_change_fast'] = self._fast_model.predict(self._df[x_columns])

    def evaluate_preprocessing(self):
        x_columns = self.preprocess(self._df, train=False)
        assert x_columns
        self._df['predicted_peak'] = self._df['peak_change']
        self._df['predicted_peak_prev'] = self._df['predicted_peak'].shift(1)
        self._df['predicted_change_fast'] = np.array([0 for i in range(len(self._df))])
        self._df['action'] = self._df.apply(self._action, axis=1)
        return self._df

    def process(self):
        x_columns = self.preprocess(self._df, train=False)
        assert x_columns
        self._predict(x_columns)
        self._df['action'] = self._df.apply(self._action, axis=1)
        return self._df

    def incremental_process(self, close: float, volume: float, ts: pd.Timestamp):
        div = ts.minute % self.GROUPING_MINUTES

        self._close.append(close)
        self._volume.append(volume)
        self._ts.append(ts)

        if len(self._close) < 10000:
            return None
        else:
            price = []
            volume = []
            index = []

            current_volume = 0
            for c, v, t in zip(self._close, self._volume, self._ts):
                current_volume += v
                if t.minute % self.GROUPING_MINUTES == div:
                    price.append(c)
                    volume.append(current_volume)
                    index.append(t)

                    current_volume = 0

            self._df = pd.DataFrame({'close': price, 'volume': volume}, index=index)
            x_columns = self.preprocess(self._df)
            if x_columns:
                self._predict(x_columns)
                self._df['action'] = self._df.apply(self._action, axis=1)
            else:
                self._df['action'] = np.array([0 for i in range(len(self._df))])
        return self._df

    @staticmethod
    def train(df, split_point, x_columns, skip_fast=False):
        x_all_train = df[:split_point][x_columns]
        x_all_test = df[split_point:][x_columns]

        y_all_train = df[:split_point]['peak_change']
        y_all_test = df[split_point:]['peak_change']

        model = RandomForestClassifier(random_state=0, n_jobs=12)
        model.fit(x_all_train, y_all_train)

        from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

        y_pred = model.predict(x_all_test)
        accuracy = accuracy_score(y_all_test, y_pred)
        print('Model Training Score is:', model.score(x_all_train, y_all_train))
        print('Model Confusion is:\n', confusion_matrix(y_all_test, y_pred))
        print(f'Model Accuracy is: {accuracy}')
        print('Model Report is:\n', classification_report(y_all_test, y_pred))

        df['predicted_peak'] = model.predict(df[x_columns])

        y_all_train = df[:split_point]['change']
        y_all_test = df[split_point:]['change']

        tf = df[split_point:].copy()

        tf['action'] = np.where(tf['predicted_peak'] > 0, 1, -1)

        with Evaluator(tf, 'action', 0.001) as evaluator:
            tf['money'] = tf.apply(evaluator.buy_sell, axis=1)

        if not skip_fast:
            fast_model = RandomForestRegressor(random_state=0, n_jobs=6)
            fast_model.fit(x_all_train, y_all_train)

            print('Fast model Training Score is:', fast_model.score(x_all_train, y_all_train))
            df['predicted_change_fast'] = fast_model.predict(df[x_columns])
            dump(fast_model, 'models/trained/fast.joblib')

        dump(model, 'models/trained/peak.joblib')

        return tf['money'][-1]


if __name__ == '__main__':
    df = pd.read_csv('data/btc_last_30days.csv', index_col='ts', parse_dates=["ts"])
    # df = pd.read_csv('data/binance.csv', index_col='ts', parse_dates=["ts"])
    model = PeakBasedModel(df)
    df = model.process()

    split_point = '2021-03-03 00:00:00'  # slow raise after the crash
    split_point = '2021-04-05 17:00:00'  # when bot bought

    tf = df[split_point:].copy()
    with Evaluator(tf, 'action', 0.001) as evaluator:
        tf['money'] = tf.apply(evaluator.buy_sell, axis=1)

    from common.plot_tools import plot_columns
    import matplotlib.pyplot as plt

    plot_columns(tf, ['close', 'action', 'money'])
    plt.savefig('peaks.png')
