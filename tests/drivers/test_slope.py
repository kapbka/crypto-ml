import numpy as np
from pandas import DataFrame
from pandas_ta import slope

from preprocessing.indicators.slope import Slope


def test_slope():
    s = Slope(2)

    assert np.isnan(s.update(1))
    assert np.isnan(s.insert(1))

    assert np.isnan(s.update(2))
    assert np.isnan(s.insert(2))

    assert np.isnan(s.update(3))
    assert np.isclose(s.insert(3), (3 - 1) / 2)
    assert np.isclose(s.update(2), (2 - 1) / 2)

    assert np.isclose(s.update(4), (4 - 1) / 2)

    assert np.isclose(s.insert(4), (4 - 2) / 2)
    assert np.isclose(s.insert(5), (5 - 4) / 2)
    assert np.isclose(s.insert(6), (6 - 4) / 2)


def test_slope_same_as_pandas():
    df = DataFrame(np.random.randint(0, 100, size=100), columns=['close'])
    expected_values = slope(df['close'], length=5).values
    s = Slope(5)
    values = np.array(list(map(s.insert, df['close'])))

    assert len(expected_values) == len(values)
    assert np.alltrue(np.isnan(expected_values[:5]))
    assert np.alltrue(np.isnan(values[:5]))
    assert np.allclose(expected_values[5:], values[5:])
