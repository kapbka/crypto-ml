from datetime import datetime

import numpy as np
from pandas import DataFrame

from preprocessing.batch_rsi import Preprocessor as Batch, COLUMNS
from preprocessing.stream_rsi import Preprocessor as Stream


def test_stream_preprocessor_same_as_batch():
    length = 60 * 24 * 1  # 1 day

    columns = COLUMNS.copy()
    columns.remove('volume')

    index_data = [datetime.fromtimestamp(i * 60) for i in range(length)]

    np.random.seed(42)
    batch = DataFrame({'close': np.random.randint(1, 100, size=length)}, index=index_data)
    Batch().preprocess(batch).process(batch)

    s = Stream()
    values = list(map(s.insert, batch['close']))
    stream = DataFrame({column: [v[idx] for v in values] for idx, column in enumerate(columns)},
                       index=index_data)

    result = batch[columns].compare(stream, align_axis=0)
    assert not len(result), result

    # update last price and make sure the result values are different
    updated_values = s.update(123)
    assert updated_values != values[-1]

    # restore back the value and check result values are the same
    restored_values = s.update(batch['close'].values[-1])
    assert restored_values == values[-1]
