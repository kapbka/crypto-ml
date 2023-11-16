from datetime import datetime

import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

from common.constants import LABEL_COLUMNS, TRAIN_FILE
from models.ann import make_batches, create_ann, make_inputs, stringify, columns_from_indicators, max_offset

TEST_INDICATORS = {'rsi': [1, 5, 100], 'slope': [1, 20, 50]}
TEST_COLUMNS = columns_from_indicators(TEST_INDICATORS)
TEST_OFFSETS = list(reversed(range(0, 20 + 1, 10)))
TEST_MODEL = [('LSTM', dict(units=1)), ('Dense', dict(units=3))]


def test_create_ann():
    ann = create_ann(params=TEST_MODEL, offsets=TEST_OFFSETS, train_columns=TEST_COLUMNS)

    length = max(TEST_OFFSETS) + 1
    index_data = np.array([datetime.fromtimestamp(i * 60) for i in range(length)])

    inputs = np.random.randint(0, 1, (length, len(TEST_COLUMNS)))
    columns = np.random.randint(0, 1, (length, 3))  # close, high, low

    data_inputs, labels, ts = make_batches(inputs, columns, index_data, window_offsets=TEST_OFFSETS, label_offset=0)
    assert len(ann.predict(data_inputs)) == 1


def test_using_dataframe():
    ann = create_ann(params=TEST_MODEL, offsets=TEST_OFFSETS, train_columns=TEST_COLUMNS)
    length = max(TEST_OFFSETS) + 1
    np.random.seed(42)
    data = {c: np.random.randint(0, 1, length) for c in TEST_COLUMNS + LABEL_COLUMNS}
    df = DataFrame(data, index=np.array([datetime.fromtimestamp(i * 60) for i in range(length)]))

    data_inputs, labels, ts = make_inputs(df, offsets=TEST_OFFSETS, columns=TEST_COLUMNS, label_col=LABEL_COLUMNS)

    assert len(ann.predict(data_inputs)) == 1


def test_stringify():
    assert 'btc_90_2021-09-01/minmaxscaler/lstm_1_dense_3' == \
           stringify(TEST_MODEL, scaler=MinMaxScaler(), dataset=TRAIN_FILE)


def test_columns():
    assert columns_from_indicators({'rsi': [1, 5], 'slope': [1, 5]}) == ['rsi_1', 'slope_1', 'rsi_5', 'slope_5']


def test_offsets():
    assert max_offset({'rsi': [10, 20], 'slope': [5]}, [1, 30]) == 70
