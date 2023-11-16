from pandas import Series
from pandas_ta import rsi

from preprocessing.indicators.base import Indicator


class RSI(Indicator):
    def __init__(self, window: int, buffer_size: int):
        self._length = window
        super().__init__(window=buffer_size)

    def is_pending(self):
        return self._index < self._length + 1

    def process(self, previous: float) -> float:
        return rsi(Series(self._buffer[:self._index]), length=self._length).values[-1]
