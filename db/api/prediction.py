from datetime import datetime

from pandas import DataFrame
from pandas.io.sql import read_sql
from sqlalchemy.sql.expression import asc
import logging

from db.model import Prediction


class DBPrediction:
    def __init__(self, db):
        self.db = db

    def load(self, ts_from: datetime, currency: str, version: int, bot_id: int) -> DataFrame:
        query = self.db.session.query(Prediction) \
            .filter(Prediction.currency_code == currency,
                    Prediction.version_id == version,
                    Prediction.individual_id == bot_id,
                    Prediction.ts > ts_from) \
            .order_by(asc(Prediction.ts))

        res = read_sql(query.statement, query.session.bind).set_index('ts')
        expected_minutes = int((res.index[-1] - res.index[0]).total_seconds() // 60) if len(res) else 0
        valid = len(res) == 1 or len(res) and expected_minutes == len(res) - 1

        # detect gaps in the cache and invalidate it
        if not valid and len(res):
            logging.warning(f"Invalidating cache for bot {bot_id}, currency {currency}, version {version}, "
                            f"expected {expected_minutes}, got {len(res)}, res: {res}")
            self.db.session.query(Prediction) \
                .filter(Prediction.currency_code == currency,
                        Prediction.version_id == version,
                        Prediction.individual_id == bot_id).delete()
            self.db.commit()
            res = res.head(0)
        return res

    def save(self, predictions: DataFrame):
        predictions.to_sql(
            Prediction.__tablename__,
            con=self.db.session.connection(),
            if_exists='append',
            index_label='ts'
        )
        self.db.commit()
