import numpy as np
from pandas import DataFrame
from pandas_ta import rsi

from preprocessing.indicators.rsi import RSI


def test_rsi_same_as_pandas():
    df = DataFrame(np.random.randint(0, 100, size=1000), columns=['close'])
    size = 30
    expected_values = rsi(df['close'], length=size).values
    s = RSI(size, size * 10)
    values = np.array(list(map(s.insert, df['close'])))

    assert len(expected_values) == len(values)
    assert np.alltrue(np.isnan(expected_values[:size]))
    assert np.alltrue(np.isnan(values[:size]))
    assert np.allclose(expected_values[size:], values[size:])
