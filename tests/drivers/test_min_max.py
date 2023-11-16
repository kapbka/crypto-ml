import numpy as np
from pandas import DataFrame

from preprocessing.indicators.min_max import Minimum, Maximum


def test_minimum():
    minimum = Minimum(3)

    assert np.isnan(minimum.update(1))
    assert np.isnan(minimum.insert(1))

    assert np.isnan(minimum.update(2))
    assert np.isnan(minimum.insert(2))

    assert np.isnan(minimum.update(3))
    assert minimum.insert(0) == 0
    assert minimum.update(1) == 1
    assert minimum.insert(3) == 1
    assert minimum.insert(3) == 1
    assert minimum.insert(3) == 3


def test_min_max_same_as_pandas():
    df = DataFrame(np.random.randint(0, 100, size=100), columns=['close'])

    minimum = Minimum(20)
    maximum = Maximum(20)
    values = [np.array(list(map(agg.insert, df['close']))) for agg in (minimum, maximum)]
    for expected, calculated in zip([df['close'].rolling(window=20).min().values,
                                     df['close'].rolling(window=20).max().values], values):
        assert len(expected) == len(calculated)
        assert np.alltrue(np.isnan(expected[:19]))
        assert np.alltrue(np.isnan(calculated[:19]))
        assert np.allclose(expected[19:], calculated[19:])
