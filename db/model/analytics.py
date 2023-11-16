from enum import IntEnum

from sqlalchemy import Column, Sequence, ForeignKey, CheckConstraint, UniqueConstraint, ForeignKeyConstraint
from sqlalchemy import Integer, String, DateTime, Float, Interval, ARRAY

from sqlalchemy import func

from db.model.base import Base


class RunSumInterval(Base):
    __tablename__ = 'run_sum_interval'

    id = Column(Integer, Sequence('run_sum_interval_id_seq'), primary_key=True)

    code = Column(String, nullable=False, unique=True)
    name = Column(String, nullable=False, unique=True)

    # interval properties
    # 0 - from now (last n days, for example), 1 - certain dates from / to (for falls and rises)
    type = Column(Integer, CheckConstraint('type in (0,1)'), nullable=False)
    from_now = Column(Interval)
    start_ts = Column(DateTime)
    end_ts = Column(DateTime)

    # system info dates
    create_ts = Column(DateTime, server_default=func.now(), nullable=False)
    update_ts = Column(DateTime, onupdate=func.now())

    __table_args__ = (CheckConstraint('(type = 0 and from_now is not null) or '
                                      '(type = 1 and start_ts is not null)'),
                      {'extend_existing': True})

    def __repr__(self):
        return f"<RunSumInterval(id='{self.id}', code='{self.code}', type='{self.type}," \
               f"from_now='{self.from_now}', start_ts='{self.start_ts}, end_ts='{self.end_ts}')>"


class RunSumHeader(Base):
    __tablename__ = 'run_sum_header'

    id = Column(Integer, Sequence('run_sum_header_id_seq'), primary_key=True)

    currency_code = Column(String(6), ForeignKey('currency.code'), nullable=False)
    version_id = Column(Integer, ForeignKey('version.id'), nullable=False)
    interval_id = Column(Integer, ForeignKey('run_sum_interval.id'), nullable=False)
    interval_code = Column(String, nullable=False)

    status = Column(Integer, CheckConstraint('status in (0,1)'), nullable=False)
    start_ts = Column(DateTime, nullable=False)
    end_ts = Column(DateTime, nullable=False)
    init_usd = Column(Float, nullable=False)

    # system info dates
    create_ts = Column(DateTime, server_default=func.now(), nullable=False)
    update_ts = Column(DateTime, onupdate=func.now())

    __table_args__ = (UniqueConstraint('currency_code', 'version_id', 'interval_id', name='uc_run_sum_header_1'),
                      {'extend_existing': True})

    def __repr__(self):
        return f"<RunSumHeader(id='{self.id}', interval_code='{self.interval_code}', status='{self.status}')>"


class RunSum(Base):
    __tablename__ = 'run_sum'

    header_id = Column(Integer, ForeignKey('run_sum_header.id'), primary_key=True, nullable=False)
    individual_id = Column(Integer, ForeignKey('individual.id'), primary_key=True, nullable=False)
    md5 = Column(String(32), nullable=False)
    usd = Column(Float, nullable=False)
    percent = Column(Float, nullable=False)
    crypto = Column(Float, nullable=False)
    percent_by_week = Column(ARRAY(Float))
    # system info dates
    create_ts = Column(DateTime, server_default=func.now(), nullable=False)
    update_ts = Column(DateTime, onupdate=func.now())

    __table_args__ = (UniqueConstraint('header_id', 'individual_id', name='uc_run_sum_1'),
                      {'extend_existing': True})
