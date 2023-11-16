from collections import defaultdict
from functools import partial
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from common.constants import INDICATOR_LENGTHS

SLOPE_COUNT = 30
COLUMNS = ['close_rolling_slope', 'rsi_smooth_10h', 'rsi_smooth_30m', 'slope_count', 'close_slope',
           'close_diff_30m', 'close_diff_60m_max', 'jump_count', 'slope_speed', 'volume']


class Preprocessor:
    def __init__(self, close_rolling_window=100):
        self._close_rolling_window = close_rolling_window

        self._x_buffer = None
        self._x_buffer_len = 0

        self._annotations: List[Tuple[pd.Timestamp, str]] = []

    def predict(self, degree: int, series: pd.Series):
        if not self._x_buffer_len or self._x_buffer_len != len(series):
            self._x_buffer = np.array(range(len(series))).reshape((len(series), 1))
            self._x_buffer_len = len(series)

        poly_reg = PolynomialFeatures(degree=degree)
        X_poly = poly_reg.fit_transform(self._x_buffer)
        lin_reg_2 = LinearRegression()
        lin_reg_2.fit(X_poly, series)

        return lin_reg_2.predict(poly_reg.fit_transform([[self._x_buffer_len]]))[0]

    def preprocess(self, df, close_window=90):
        close_smooth = df['close'].rolling(window=close_window).apply(partial(self.predict, 2))
        df['close_slope'] = ta.slope(close_smooth, length=5)

        rsi = ta.rsi(df['close'], length=30)
        df['rsi_smooth_30m'] = rsi.rolling(window=30).apply(partial(self.predict, 2))
        df['rsi_smooth_10h'] = rsi.rolling(window=60 * 10).apply(partial(self.predict, 2))

        return self

    def process(self, df):
        close_rolling = df['close'].rolling(window=self._close_rolling_window).mean()
        df['close_rolling_slope'] = ta.slope(close_rolling, length=5)

        for i in range(1, SLOPE_COUNT + 1):
            df[f'slope_{i}'] = ta.slope(df['rsi_smooth_10h'], length=i)

        def count_slopes(x: pd.Series):
            slopes = map(lambda i: x[f'slope_{i}'], range(1, SLOPE_COUNT + 1))
            return len(list(filter(lambda s: s > 0, slopes)))

        df['slope_count'] = df.apply(count_slopes, axis=1)
        df['slope_count_prev'] = df['slope_count'].shift(1)

        for i in range(1, SLOPE_COUNT + 1):
            df.pop(f'slope_{i}')

        current_jump_count = -1

        def jump_count(x: pd.Series):
            nonlocal current_jump_count

            if x['slope_count'] and not x['slope_count_prev']:
                current_jump_count = 0
            elif not x['slope_count']:
                current_jump_count = -1
            elif x['slope_count_prev'] > x['slope_count'] and current_jump_count != -1:
                current_jump_count += 1

            if x['slope_count'] == SLOPE_COUNT and x['slope_count_prev'] != SLOPE_COUNT:
                res = current_jump_count
                current_jump_count = -1
                return res

            return -1

        df['jump_count'] = df.apply(jump_count, axis=1)

        zero_slopes_last_ts = None

        def slope_speed(x: pd.Series):
            nonlocal zero_slopes_last_ts
            if not x['slope_count']:
                zero_slopes_last_ts = x.name

            if zero_slopes_last_ts and x['slope_count'] == SLOPE_COUNT and x['slope_count_prev'] != SLOPE_COUNT:
                res = (x.name - zero_slopes_last_ts).seconds // 60
                zero_slopes_last_ts = None
                return res

            return 0

        df['slope_speed'] = df.apply(slope_speed, axis=1)

        max_30 = df['close'].rolling(window=30).max()
        df['close_diff_30m'] = (max_30 - df['close']) / max_30

        rolling_60 = df['close'].rolling(window=60)
        df['close_diff_60m_max'] = (rolling_60.max() - rolling_60.min()) / rolling_60.min()

        return self

    def rsi(self, length: int, df: pd.DataFrame):
        return ta.rsi(df['close'], length=length, scalar=3) - 1.5

    def slope(self, length: int, df: pd.DataFrame):
        return ta.slope(ta.ema(df['close'], length=length), length=length, as_angle=True)

    def apply_indicators(self, df: pd.DataFrame, indicators: Dict[str, List[int]]):
        for name, lengths in indicators.items():
            for length in lengths:
                df[f"{name}_{length}"] = getattr(self, name)(length, df)

    @staticmethod
    def create_indicators(df: pd.DataFrame,
                          lengths: List[int] = INDICATOR_LENGTHS) -> Tuple[List[str], Dict[str, List[int]]]:
        train_columns = []
        indicators = defaultdict(list)
        for length in lengths:
            indicators['rsi'].append(length)
            indicators['slope'].append(length)

            df[f'rsi_{length}'] = ta.rsi(df['close'], length=length, scalar=3) - 1.5
            df[f'slope_{length}'] = ta.slope(ta.ema(df['close'], length=length), length=length, as_angle=True)

            train_columns.append(f'rsi_{length}')
            train_columns.append(f'slope_{length}')
        return train_columns, indicators

    @staticmethod
    def mark_crashes(df: pd.DataFrame, detection_length: int, recovery_length: int, amount: float):
        max_price = df['close'].rolling(window=detection_length).max()
        df['crash'] = ((max_price - df['close']) / max_price > amount).rolling(window=recovery_length).max()

