from abc import ABC, abstractmethod

import numpy as np
from numba import njit


@njit
def _shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    result[num:] = fill_value
    result[:num] = arr[-num:]
    return result


class Indicator(ABC):
    """
    This base class provides an interface for manipulating
    stateful indicators, for example average price.
    """
    def __init__(self, window: int):
        self._window = window
        self._buffer = np.zeros(window)
        self._index = 0

    @abstractmethod
    def process(self, previous: float) -> float:
        raise NotImplementedError("This method must be overridden")

    def is_pending(self):
        return self._index < self._window

    def update(self, price: float) -> float:
        """
        This method accepts latest price and changes internal state of the indicator
        returning last value of the indicator.
        :param price: last price
        :return: new indicator value
        """
        if self.is_pending():
            return np.nan

        previous = self._buffer[-1]
        self._buffer[-1] = price
        return self.process(previous)

    def insert(self, price: float) -> float:
        """
        This method accepts latest price and inserts it as new series to internal state
        of the indicator, returning last value of the indicator.
        :param price: last price
        :return: new indicator value
        """
        previous = None
        if self._index == self._window:
            previous = self._buffer[0]
            self._buffer = _shift(self._buffer, -1)
            self._buffer[-1] = price
        else:
            self._buffer[self._index] = price
            self._index += 1

        if self.is_pending():
            return np.nan

        return self.process(previous)
