import logging
from datetime import datetime, timedelta

from sqlalchemy import text
from sqlalchemy.sql.expression import asc, desc

from common.constants import RunSumIntervalType
from db.model.analytics import RunSumInterval


class DBRunSumInterval:
    def __init__(self, db):
        self.db = db

    def delete(self, run_sum_interval_id: int):
        self.db.session.query(RunSumInterval).filter(RunSumInterval.id == run_sum_interval_id).delete()

    def get(self, interval_code: str):
        return self.db.session.query(RunSumInterval).filter(RunSumInterval.code == interval_code).first()

    def get_batch(self):
        return self.db.session.query(RunSumInterval).order_by(asc(RunSumInterval.id)).all()

    def set(self, interval_code: str, interval_name: str, interval_type: int,
            from_now: timedelta = None, start_ts: datetime = None, end_ts: datetime = None):

        # validation checks
        if interval_type == RunSumIntervalType.FromNow.value and not from_now:
            raise ValueError('Empty from_now parameter!')

        if interval_type == RunSumIntervalType.FromTo.value and not start_ts:
            raise ValueError('Empty start_ts or end_ts parameter(-s)!')

        run_sum_interval = self.db.session.query(RunSumInterval). \
            filter(RunSumInterval.code == interval_code,
                   RunSumInterval.type == interval_type).first()

        # insert
        if not run_sum_interval:
            run_sum_interval = RunSumInterval(
                code=interval_code,
                name=interval_name,
                type=interval_type,
                from_now=from_now,
                start_ts=start_ts,
                end_ts=end_ts)
            self.db.session.add(run_sum_interval)
            self.db.flush()
        # update
        else:
            run_sum_interval.name = interval_name
            run_sum_interval.from_now = from_now
            run_sum_interval.start_ts = start_ts
            run_sum_interval.end_ts = end_ts

        return run_sum_interval
