import json
import pickle
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf
from asynctest.mock import patch, Mock, MagicMock
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

from common.constants import LABEL_COLUMNS
from common.deal_generator import DealGenerator
from common.individual import Individual
from db.model.processing import IndividualAttribute, Individual as DBIndividual
from models.ann import create_ann, columns_from_indicators

ACTIONS = [np.array([0, 0, 20]),
           np.array([20, 0, 0]),
           np.array([0, 20, 0])]
TEST_INDICATORS = {'rsi': [1, 5, 100], 'slope': [1, 20, 50]}
TEST_COLUMNS = columns_from_indicators(TEST_INDICATORS)
TEST_MODEL = [('LSTM', dict(units=1)), ('Dense', dict(units=3))]
TEST_OFFSETS = list(reversed(range(0, 20 + 1, 10)))


def test_generate_deals():
    np.random.seed(42)

    initial_length = 60 * 24 * 2
    data = {c: np.random.uniform(1, 100, initial_length) for c in TEST_COLUMNS + LABEL_COLUMNS}
    df_initial = DataFrame(data, index=np.array([datetime.fromtimestamp(i * 60, timezone.utc)
                                                 for i in range(initial_length)]))
    scaler = MinMaxScaler()
    scaler.fit(df_initial[TEST_COLUMNS].values)

    ann = create_ann(TEST_MODEL, TEST_OFFSETS, TEST_COLUMNS)
    individual = Individual(weights=ann.get_weights())

    db = MagicMock()
    db.ann.get.return_value = MagicMock(layers=json.dumps(TEST_MODEL),
                                        offsets=json.dumps(TEST_OFFSETS),
                                        indicators=json.dumps(TEST_INDICATORS))
    db.individual.get_db.return_value = DBIndividual(weights=pickle.dumps(individual.weights))
    db.scaler.get.return_value = MagicMock(data=pickle.dumps(scaler))
    db.price.get_first.return_value = MagicMock(close=100)
    db.session.add_all = MagicMock()
    db.commit = MagicMock()
    db.prediction.save = MagicMock()
    db.prediction.load = MagicMock(return_value=DataFrame())

    generator = DealGenerator(db=db, attributes=IndividualAttribute(version_id=5, scaler_id=1))

    initial_predictions = np.array([ACTIONS[idx % 3] for idx in range(initial_length - max(TEST_OFFSETS) - 100)])

    # call generator first time, it will generate deals for all available data points
    with patch.object(tf.keras.models.Sequential, 'predict', new=Mock(return_value=initial_predictions)):
        deals, _, limits = generator.process(current_usd=0, df=df_initial)

        assert deals
        assert limits

    initial_deals = deals.copy()

    # call it second time to emulate generating predictions only for 10 appended rows
    append_length = 10
    data = {c: np.random.uniform(1, 100, append_length) for c in TEST_COLUMNS + LABEL_COLUMNS}
    df = DataFrame(data, index=np.array([datetime.fromtimestamp(initial_length * 60 + i * 60, timezone.utc)
                                         for i in range(append_length)]))
    append_predictions = np.array([ACTIONS[idx % 3] for idx in range(append_length)])
    with patch.object(tf.keras.models.Sequential, 'predict', new=Mock(return_value=append_predictions)), \
            patch('db.data_cache.DataCache.load', new=MagicMock(return_value=df)):
        db.prediction.load = MagicMock(return_value=DataFrame(initial_predictions,
                                                              columns=['buy', 'sell', 'idle'],
                                                              index=df_initial.tail(len(initial_predictions)).index))

        db.deal.get_last.return_value = initial_deals[-1]
        deals, limits = generator.process_pending(ts_to=datetime.now(tz=timezone.utc))
        assert deals
        assert limits
        assert len(deals) > len(initial_deals)
