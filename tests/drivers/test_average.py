import numpy as np
from pandas import DataFrame

from preprocessing.indicators.avg import Average


def test_average():
    a = Average(3)

    assert np.isnan(a.update(1))
    assert np.isnan(a.insert(1))

    assert np.isnan(a.update(2))
    assert np.isnan(a.insert(2))

    assert np.isnan(a.update(3))
    assert np.isclose(a.insert(3), (1 + 2 + 3) / 3)
    assert np.isclose(a.update(2), (1 + 2 + 2) / 3)

    assert np.isclose(a.update(4), (1 + 2 + 4) / 3)
    assert np.isclose(a.update(5), (1 + 2 + 5) / 3)

    assert np.isclose(a.insert(4), (2 + 5 + 4) / 3)
    assert np.isclose(a.insert(5), (5 + 4 + 5) / 3)
    assert np.isclose(a.insert(6), (4 + 5 + 6) / 3)


def test_average_same_as_pandas():
    df = DataFrame(np.random.randint(0, 100, size=100), columns=['close'])
    expected_values = df['close'].rolling(window=10).mean().values
    average = Average(10)
    values = np.array(list(map(average.insert, df['close'])))

    assert len(expected_values) == len(values)
    assert np.alltrue(np.isnan(expected_values[:9]))
    assert np.alltrue(np.isnan(values[:9]))
    assert np.allclose(expected_values[9:], values[9:])
