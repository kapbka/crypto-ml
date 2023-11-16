from preprocessing.indicators.base import Indicator


class Average(Indicator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sum: float = 0

    def process(self, previous: float) -> float:
        if previous is None:
            self._sum = self._buffer.sum()
        else:
            self._sum -= previous
            self._sum += self._buffer[-1]

        return self._sum / self._window
