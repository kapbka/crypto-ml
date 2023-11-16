import numpy as np

from preprocessing.batch_rsi import SLOPE_COUNT
from preprocessing.indicators.avg import Average
from preprocessing.indicators.min_max import Maximum, Minimum
from preprocessing.indicators.regression import Regression
from preprocessing.indicators.rsi import RSI
from preprocessing.indicators.slope import Slope


class Preprocessor:
    def __init__(self):
        # indicators
        self._close_smooth = Regression(window=90)
        self._close_slope = Slope(window=5)
        self._close_rolling = Average(window=100)
        self._close_rolling_slope = Slope(window=5)
        self._rsi = RSI(window=30, buffer_size=30 * 40)
        self._rsi_smooth_30m = Regression(window=30)
        self._rsi_smooth_10h = Regression(window=60 * 10)
        self._max_30m = Maximum(window=30)
        self._max_60m = Maximum(window=60)
        self._min_60m = Minimum(window=60)

        self._rsi_smooth_10h_slopes = [Slope(window=i) for i in range(1, SLOPE_COUNT + 1)]

        # state
        self._index = 0
        self._slope_count = 0
        self._current_jump_count = -1
        self._zero_slopes_last_ts = 0

    def update(self, price: float) -> tuple:
        saved_state = self._index, self._slope_count, self._current_jump_count, self._zero_slopes_last_ts
        result = self._process(price, 'update')
        self._index, self._slope_count, self._current_jump_count, self._zero_slopes_last_ts = saved_state
        return result

    def insert(self, price: float) -> tuple:
        return self._process(price, 'insert')

    def _process(self, price: float, action: str):
        close_slope = getattr(self._close_slope, action)(getattr(self._close_smooth, action)(price))

        rsi = getattr(self._rsi, action)(price)
        rsi_smooth_30m = getattr(self._rsi_smooth_30m, action)(rsi)
        rsi_smooth_10h = getattr(self._rsi_smooth_10h, action)(rsi)
        slopes = list(map(lambda slope: getattr(slope, action)(rsi_smooth_10h), self._rsi_smooth_10h_slopes))

        close_rolling_slope = getattr(self._close_rolling_slope, action)(getattr(self._close_rolling, action)(price))
        slope_count = sum(1 for s in slopes if s > 0)
        slope_count_prev = self._slope_count
        self._slope_count = slope_count

        jump_count = self._jump_count(slope_count, slope_count_prev)
        slope_speed = self._slope_speed(slope_count, slope_count_prev)

        max_30m = getattr(self._max_30m, action)(price)
        close_diff_30m = (max_30m - price) / max_30m

        min_60m = getattr(self._min_60m, action)(price)
        max_60m = getattr(self._max_60m, action)(price)
        close_diff_60m_max = (max_60m - min_60m) / min_60m if not np.isnan(min_60m) else np.nan

        self._index += 1

        return close_rolling_slope, rsi_smooth_10h, rsi_smooth_30m, slope_count, close_slope, close_diff_30m, \
            close_diff_60m_max, jump_count, slope_speed

    def _jump_count(self, slope_count: int, slope_count_prev: int):
        if slope_count and not slope_count_prev:
            self._current_jump_count = 0
        elif not slope_count:
            self._current_jump_count = -1
        elif slope_count_prev > slope_count and self._current_jump_count != -1:
            self._current_jump_count += 1

        if slope_count == SLOPE_COUNT and slope_count_prev != SLOPE_COUNT:
            res = self._current_jump_count
            self._current_jump_count = -1
            return res

        return -1

    def _slope_speed(self, slope_count: int, slope_count_prev: int):
        if not slope_count:
            self._zero_slopes_last_ts = self._index

        if self._zero_slopes_last_ts and slope_count == SLOPE_COUNT and slope_count_prev != SLOPE_COUNT:
            res = self._index - self._zero_slopes_last_ts
            self._zero_slopes_last_ts = 0
            return res

        return 0
