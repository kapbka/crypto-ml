from functools import partial

import numpy as np
from pandas import DataFrame

from preprocessing.batch_rsi import Preprocessor
from preprocessing.indicators.regression import Regression


def test_regression_same_as_in_preprocessor():
    df = DataFrame(np.random.randint(0, 100, size=1000), columns=['close'])

    size = 30

    df['close_smooth'] = df['close'].rolling(window=30).apply(partial(Preprocessor().predict, 2))
    expected_values = df['close_smooth'].values
    s = Regression(size)
    values = np.array(list(map(s.insert, df['close'])))

    assert len(expected_values) == len(values)
    assert np.alltrue(np.isnan(expected_values[:size - 1]))
    assert np.alltrue(np.isnan(values[:size - 1]))
    assert np.allclose(expected_values[size - 1:], values[size - 1:])
