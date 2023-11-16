import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from functools import partial

from common.constants import RunSumStatus, RunSumIntervalType

from db.api import DB

NUMBER_OF_BATCHES = 3
PERCENT_COMMISSION = 0.1


def _upload_interval_run_sum(currency_code: str, version_id: int, interval_id: int,
                             current_batch_num: int, number_of_batches: int,
                             percent_commission: float):
    with DB() as db:
        # for the current interval type update run_sum_header as done
        db.run_sum.upload(currency_code=currency_code,
                          version_id=version_id,
                          interval_id=interval_id,
                          current_batch_num=current_batch_num,
                          number_of_batches=number_of_batches,
                          percent_commission=percent_commission)


def upload_run_sum(number_of_batches: int, percent_commission: float = PERCENT_COMMISSION):
    dt = datetime.now()
    # truncate current date to minutes
    dt = datetime(day=dt.day, month=dt.month, year=dt.year, hour=dt.hour, minute=dt.minute)

    with DB() as db:
        currencies = db.currency.get_batch()
        versions = db.version.get_batch()
        intervals = db.run_sum_interval.get_batch()

    # for each currency
    for currency in currencies:
        logging.info(f'CURRENCY {currency.code}')
        # for each version
        for version in versions:
            logging.info(f'VERSION {version.id} - {version.name}')
            # for each interval type
            for interval in intervals:
                logging.info(f'interval_type={interval.code}')

                if interval.type == RunSumIntervalType.FromNow.value:
                    start_ts = dt - interval.from_now
                    end_ts = dt
                elif interval.type == RunSumIntervalType.FromTo.value:
                    start_ts = interval.start_ts
                    end_ts = interval.end_ts if interval.end_ts else datetime.now()
                else:
                    # just additional validation check if model was changed but this script wasn't
                    raise ValueError(f'Unknown interval type {interval.type} for interval code {interval.code}!')

                # insert a new header and set header_id
                with DB() as db:
                    db.run_sum.set_header(currency_code=currency.code,
                                          version_id=version.id,
                                          interval_id=interval.id,
                                          interval_code=interval.code,
                                          start_ts=start_ts,
                                          end_ts=end_ts,
                                          status=RunSumStatus.InProgress.value)

                # for each interval run necessary number of batches and waiting for the completion
                with ThreadPoolExecutor(max_workers=number_of_batches) as e:
                    workers = []
                    for current_batch_num in range(number_of_batches):
                        worker = partial(_upload_interval_run_sum,
                                         currency_code=currency.code,
                                         version_id=version.id,
                                         interval_id=interval.id,
                                         current_batch_num=current_batch_num,
                                         number_of_batches=number_of_batches,
                                         percent_commission=percent_commission)
                        workers.append(e.submit(worker))
                    for i in workers:
                        i.result()

                with DB() as db:
                    db.run_sum.set_header(currency_code=currency.code,
                                          version_id=version.id,
                                          interval_id=interval.id,
                                          interval_code=interval.code,
                                          start_ts=start_ts,
                                          end_ts=end_ts,
                                          status=RunSumStatus.Done.value)
            logging.info(f'-- END VERSION {version.id} -- ')
        logging.info(f'------------ END CURRENCY {currency.code} ------------ ')
