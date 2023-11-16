from datetime import datetime, timedelta

from sqlalchemy.sql.expression import desc

from common.constants import HistoryStatStatus
from db.model.processing import HistoryStat


class DBHistoryStat:
    def __init__(self, db):
        self.db = db

    def get(self, hist_id: int):
        return self.db.session.query(HistoryStat).filter(HistoryStat.id == hist_id).first()

    def get_last(self):
        return self.db.session.query(HistoryStat).order_by(desc(HistoryStat.id)).first()

    def get_in_progress(self):
        return self.db.session.query(HistoryStat).filter(HistoryStat.status == HistoryStatStatus.InProgress.value)\
            .order_by(desc(HistoryStat.id)).first()

    def get_batch(self):
        return self.db.session.query(HistoryStat).all()

    def set(self, speed: timedelta, eta: datetime, processed_count: int, remaining_count: int):
        status = HistoryStatStatus.Done.value if remaining_count == 0 else HistoryStatStatus.InProgress.value
        history_stat = self.get_in_progress()

        if not history_stat:
            history_stat = HistoryStat(
                speed=speed,
                eta=eta,
                processed_count=processed_count,
                remaining_count=remaining_count,
                status=status
            )
            self.db.session.add(history_stat)
        else:
            history_stat.speed = speed
            history_stat.eta = eta
            history_stat.processed_count = processed_count
            history_stat.remaining_count = remaining_count
            history_stat.status = status

        return history_stat
