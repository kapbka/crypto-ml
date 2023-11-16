import logging
from datetime import datetime

from sqlalchemy import text

from common.constants import RunSumStatus
from db.model.analytics import RunSumHeader, RunSum


class DBRunSum:
    def __init__(self, db):
        self.db = db

    def _delete(self, run_sum_header_id: int):
        self.db.session.query(RunSum).filter(RunSum.header_id == run_sum_header_id).delete()
        self.db.session.query(RunSumHeader).filter(RunSumHeader.id == run_sum_header_id).delete()

    def set_header(self, currency_code: str, version_id: int, interval_id: int, interval_code: str,
                   start_ts: datetime, end_ts: datetime, status: int):
        run_sum_header = self.db.session.query(RunSumHeader). \
            filter(RunSumHeader.currency_code == currency_code,
                   RunSumHeader.version_id == version_id,
                   RunSumHeader.interval_id == interval_id).first()

        # insert
        if status == RunSumStatus.InProgress.value:
            # clean if exists
            if run_sum_header:
                self._delete(run_sum_header_id=run_sum_header.id)

            init_usd = self.db.session.execute(
                text(
                    """select get_init_usd
                       (
                           p_currency_code := :currency_code,
                           p_start_ts      := :start_ts
                       );
                    """
                ).bindparams(
                    currency_code=currency_code,
                    start_ts=start_ts
                )
            )

            run_sum_header = RunSumHeader(
                currency_code=currency_code,
                version_id=version_id,
                interval_id=interval_id,
                interval_code=interval_code,
                start_ts=start_ts,
                end_ts=end_ts,
                init_usd=init_usd.fetchone()[0],
                status=status)
            self.db.session.add(run_sum_header)
            self.db.flush()
        # update
        elif status == RunSumStatus.Done.value:
            if run_sum_header and run_sum_header.status == RunSumStatus.InProgress.value:
                run_sum_header.status = RunSumStatus.Done.value
                # once all batches uploaded and we mark loading as done
                # we generate pivot table to use in Grafana's dashboards
                self.db.session.execute(
                    text(
                        """select generate_pivot_table
                           (
                             'run_sum_pivoted',
                             'select rsh.currency_code,
                                     rsh.version_id,
                                     rs.individual_id,
                                     rs.md5,
                                     rsi.name,
                                     rsh.init_usd,
                                     rs.percent,
                                     rs.crypto,
                                     rs.usd
                                from run_sum_interval rsi
                                left join run_sum_header rsh
                                  on rsi.id = rsh.interval_id
                                left join run_sum rs
                                  on rsh.id = rs.header_id',
                             array['currency_code', 'version_id', 'individual_id', 'md5'],
                             array['name'], '#.percent', null
                           );
                        """
                    )
                )
            else:
                raise ValueError(f'No in progress record found with interval type {interval_code}')
        return run_sum_header

    def upload(self, currency_code: str, version_id: int, interval_id: int,
               current_batch_num: int, number_of_batches: int,
               percent_commission: float):
        logging.info(f'run_sum.upload start, current_batch_num {current_batch_num}, number_of_batches {number_of_batches}')
        self.db.session.execute(
            text(
                """call calc_run_sum
                   (
                       p_currency_code      := :currency_code,
                       p_version_id         := :version_id,
                       p_interval_id        := :interval_id,
                       p_current_batch      := :current_batch_num,
                       p_number_of_batches  := :number_of_batches,
                       p_percent_commission := :percent_commission,
                       p_md5_arr            := :md5_arr,
                       p_process_mode       := :process_mode
                   );
                """
            ).bindparams(
                currency_code=currency_code,
                version_id=version_id,
                interval_id=interval_id,
                current_batch_num=current_batch_num,
                number_of_batches=number_of_batches,
                percent_commission=percent_commission,
                md5_arr=None,  # '{"8743da89d23c3e9735d56b2b93c4433c", "846608702e15a57c76af0455c4f616ff"}',
                process_mode=0
            )
        )
        logging.info(f'run_sum.upload end, current_batch_num {current_batch_num}, number_of_batches {number_of_batches}')
