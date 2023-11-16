from preprocessing.indicators.base import Indicator


class Slope(Indicator):
    def __init__(self, window: int):
        super().__init__(window=window + 1)

    def process(self, previous: float) -> float:
        return (self._buffer[-1] - self._buffer[0]) / (self._window - 1)

