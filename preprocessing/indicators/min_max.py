from sortedcontainers import SortedDict

from preprocessing.indicators.base import Indicator


class MinMax(Indicator):
    _reversed = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data = SortedDict()

    def process(self, previous: float) -> float:
        if previous is None:
            for price in self._buffer:
                self._data[price] = self._data.get(price, 0) + 1
        else:
            self._data[self._buffer[-1]] = self._data.get(self._buffer[-1], 0) + 1
            self._data[previous] -= 1
            if not self._data[previous]:
                self._data.pop(previous)

        it = iter(self._data.keys()) if self._reversed else iter(reversed(self._data.keys()))
        return next(it)


class Minimum(MinMax):
    _reversed = True


class Maximum(MinMax):
    _reversed = False
