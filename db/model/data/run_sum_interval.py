import logging
from datetime import datetime, timedelta

from db.api import DB
from common.constants import RunSumIntervalType
from db.model.analytics import RunSumInterval


def insert_run_sum_interval():
    logging.info('insert_run_sum_interval start')

    intervals = [
        # From Now
        {'code': 'quarter', 'name': 'Quarter', 'from_now': timedelta(days=90), 'start_ts': None, 'end_ts': None},
        {'code': 'month', 'name': 'Month', 'from_now': timedelta(days=30), 'start_ts': None, 'end_ts': None},
        {'code': 'week', 'name': 'Week', 'from_now': timedelta(days=7), 'start_ts': None, 'end_ts': None},
        # From To
        {'code': 'test', 'name': 'Test', 'from_now': None,
         'start_ts': datetime(day=1, month=11, year=2021), 'end_ts': None},
        {'code': 'fall_3-5_dec_2021', 'name': 'Fall 3-5 Dec 2021', 'from_now': None,
         'start_ts': datetime(day=3, month=12, year=2021), 'end_ts': datetime(day=5, month=12, year=2021)},
        {'code': 'fall_5_jan_2022', 'name': 'Fall 5 Jan 2022', 'from_now': None,
         'start_ts': datetime(day=5, month=1, year=2022, hour=20), 'end_ts': datetime(day=5, month=1, year=2022, hour=23)},
        {'code': 'fall_20-23_jan_2022', 'name': 'Fall 20-23 Jan 2022', 'from_now': None,
         'start_ts': datetime(day=20, month=1, year=2022, hour=15), 'end_ts': datetime(day=23, month=1, year=2022, hour=00)},
        {'code': 'rise_24-25_jan_2022', 'name': 'Rise 24-25 Jan 2022', 'from_now': None,
         'start_ts': datetime(day=24, month=1, year=2022, hour=8), 'end_ts': datetime(day=25, month=1, year=2022, hour=00)}
    ]

    cnt = 0
    with DB() as db:
        for interval in intervals:
            run_sum_interval = RunSumInterval(
                code=interval['code'],
                name=interval['name'],
                type=RunSumIntervalType.FromNow.value if interval['from_now'] else RunSumIntervalType.FromTo.value,
                from_now=interval['from_now'],
                start_ts=interval['start_ts'],
                end_ts=interval['end_ts'])
            db.session.add(run_sum_interval)
            cnt += 1
        db.commit()

    logging.info(f'insert_run_sum_interval end, inserted {cnt} rows')
