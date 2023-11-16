from datetime import datetime

import numpy as np
from pandas import DataFrame

from preprocessing.batch_rsi import Preprocessor, SLOPE_COUNT, COLUMNS


def test_batch_preprocessor():
    length = 60 * 24 * 3
    df = DataFrame({'close': np.random.randint(0, 100, size=length),
                    'volume': np.random.randint(0, 100, size=length), },
                   index=[datetime.fromtimestamp(i) for i in range(length)])

    started = datetime.now()
    Preprocessor().preprocess(df).process(df)
    print(f"Batch: {datetime.now() - started}")

    df.dropna(inplace=True)

    data = df[COLUMNS]
    c = data['slope_count'].value_counts().index.values
    c.sort()
    assert np.alltrue(np.equal(c, np.array(range(SLOPE_COUNT + 1))))

    started = datetime.now()
    Preprocessor().apply_indicators(df, {'slope': [1], 'rsi': [1]})
    print(f"Indicators: {datetime.now() - started}")

    assert 'slope_1' in df
    assert 'rsi_1' in df

    cols, indicators = Preprocessor.create_indicators(df)
    assert cols
    assert indicators

    Preprocessor.mark_crashes(df, 10, 10, 0.01)
    assert 'crash' in df

