import json
from typing import Tuple, Dict

from sqlalchemy import Column, Sequence, ForeignKey, CheckConstraint, UniqueConstraint, ForeignKeyConstraint
from sqlalchemy import Integer, String, LargeBinary, DateTime, Float, Interval
from sqlalchemy.orm import deferred
from sqlalchemy.sql import func

from common.constants import DealStatus
from db.model.base import Base
from db.model.types import UTCDateTime


class Version(Base):
    __tablename__ = 'version'

    id = Column(Integer, Sequence('version_id_seq'), primary_key=True)
    name = Column(String, unique=True, nullable=False)
    comment = Column(String, nullable=False)

    # system info dates
    create_ts = Column(DateTime, server_default=func.now(), nullable=False)
    update_ts = Column(DateTime, onupdate=func.now())

    __table_args__ = ({'extend_existing': True})

    def __repr__(self):
        return f"<Version(id='{self.id}', comment='{self.comment}')>"


class Currency(Base):
    __tablename__ = 'currency'

    code = Column(String(6), primary_key=True, nullable=False)
    name = Column(String(30), nullable=False)

    # system info dates
    create_ts = Column(DateTime, server_default=func.now(), nullable=False)
    update_ts = Column(DateTime, onupdate=func.now())

    __table_args__ = ({'extend_existing': True})

    def __repr__(self):
        return f"<Currency(code='{self.code}', name='{self.name}')>"


class Ann(Base):
    __tablename__ = 'ann'

    id = Column(Integer, Sequence('ann_id_seq'), primary_key=True, nullable=False)

    layers = Column(String, nullable=False)
    offsets = Column(String, nullable=False)
    indicators = Column(String, nullable=False)
    is_scaled = Column(Integer, CheckConstraint('is_scaled in (0,1)'), nullable=False, default=0)

    # system info dates
    create_ts = Column(DateTime, server_default=func.now(), nullable=False)
    update_ts = Column(DateTime, onupdate=func.now())

    __table_args__ = ({'extend_existing': True})

    def __repr__(self):
        return f"<Ann(layers='{self.layers}', offsets='{self.offsets}', indicators='{self.indicators}')>"


class Scaler(Base):
    __tablename__ = 'scaler'

    id = Column(Integer, Sequence('scaler_id_seq'), primary_key=True, nullable=False)

    currency_code = Column(String(6), ForeignKey('currency.code'), nullable=False)
    name = Column(String, nullable=False)

    # contains pickled scaler data
    data = Column(LargeBinary, nullable=False)

    # system info dates
    create_ts = Column(DateTime, server_default=func.now(), nullable=False)
    update_ts = Column(DateTime, onupdate=func.now())

    __table_args__ = ({'extend_existing': True})

    def __repr__(self):
        return f"<Scaler(id='{self.id}', name='{self.name}', currency='{self.currency_code}')>"


class Individual(Base):
    __tablename__ = 'individual'

    id = Column(Integer, Sequence('individual_id_seq'), primary_key=True, nullable=False)

    # md5 is the md5 hash of the weights
    md5 = Column(String(32), nullable=False, unique=True)
    parent_md5 = Column(String(32))
    is_portfolio = Column(Integer, CheckConstraint('is_portfolio in (0, 1)'), nullable=False, default=0)

    # the currency used to train an individual
    train_currency = Column(String(6))

    # neural network weights that define behaviour of the individual
    weights = deferred(Column(LargeBinary, nullable=False))

    # link to an ANN config
    ann_id = Column(Integer, ForeignKey('ann.id'), nullable=False)

    # system info dates
    create_ts = Column(DateTime, server_default=func.now(), nullable=False)
    update_ts = Column(DateTime, onupdate=func.now())

    __table_args__ = ({'extend_existing': True})

    def __repr__(self):
        return f"<Individual(id='{self.id}', md5='{self.md5}')>"


class IndividualAttribute(Base):
    __tablename__ = 'individual_attribute'

    currency_code = Column(String(6), ForeignKey('currency.code'), primary_key=True, nullable=False)
    version_id = Column(Integer, ForeignKey('version.id'), primary_key=True, nullable=False)
    individual_id = Column(Integer, ForeignKey('individual.id'), primary_key=True, nullable=False)
    md5 = Column(String(32), nullable=False)

    # various attributes
    realtime_enabled = Column(Integer, CheckConstraint('realtime_enabled in (0, 1)'), nullable=False, default=0)
    history_enabled = Column(Integer, CheckConstraint('history_enabled in (0, 1)'), nullable=False, default=1)
    # whether minimize() was applied to last neuron layer weights
    is_optimized = Column(Integer, CheckConstraint('is_optimized in (0, 1)'), nullable=False, default=0)
    # portfolio attributes
    priority = Column(Integer, default=0)
    share = Column(Float, CheckConstraint('share between 0 and 1'), default=0)

    # reference to a scaler it should be scaled with, NULL means not scaled
    scaler_id = Column(Integer, ForeignKey('scaler.id'), nullable=True)

    # parameters of evaluate()
    oco_buy_percent = Column(Float, CheckConstraint('oco_buy_percent between 0 and 1'), default=0.02)
    oco_sell_percent = Column(Float, CheckConstraint('oco_sell_percent between -1 and 0'), default=-0.02)
    oco_rise_percent = Column(Float, CheckConstraint('oco_rise_percent between 0 and 1'), default=0.0)

    # system info dates
    create_ts = Column(DateTime, server_default=func.now(), nullable=False)
    update_ts = Column(DateTime, onupdate=func.now())

    __table_args__ = (UniqueConstraint('currency_code', 'version_id', 'individual_id', name='uc_individual_attribute_1'),
                      CheckConstraint('realtime_enabled != history_enabled or '
                                      '(realtime_enabled = 0 and history_enabled = 0)'),
                      # if realtime is enabled share value has to be populated
                      CheckConstraint('realtime_enabled = 0 or share > 0'),
                      {'extend_existing': True})

    def __repr__(self):
        return f"<IndividualAttribute(currency_code='{self.currency_code}', version_id='{self.version_id}', md5='{self.md5}', " \
               f"realtime_enabled='{self.realtime_enabled}', history_enabled='{self.history_enabled}')>"


class IndividualAttributeHistory(Base):
    __tablename__ = 'individual_attribute_history'

    id = Column(Integer, Sequence('individual_attribute_history_id_seq'), primary_key=True)
    #
    currency_code = Column(String(6), ForeignKey('currency.code'), nullable=False)
    version_id = Column(Integer, ForeignKey('version.id'), nullable=False)
    individual_id = Column(Integer, ForeignKey('individual.id'), nullable=False)
    md5 = Column(String(32), nullable=False)

    # various attributes
    realtime_enabled_old = Column(Integer, nullable=False)
    realtime_enabled_new = Column(Integer, nullable=False)
    history_enabled_old = Column(Integer, nullable=False)
    history_enabled_new = Column(Integer, nullable=False)
    # portfolio attributes
    priority_old = Column(Integer)
    priority_new = Column(Integer)
    share_old = Column(Float)
    share_new = Column(Float)

    # reference to a scaler it should be scaled with, NULL means not scaled
    scaler_id_old = Column(Integer, ForeignKey('scaler.id'), nullable=True)
    scaler_id_new = Column(Integer, ForeignKey('scaler.id'), nullable=True)

    # parameters of evaluate()
    oco_buy_percent_old = Column(Float)
    oco_buy_percent_new = Column(Float)
    #
    oco_sell_percent_old = Column(Float)
    oco_sell_percent_new = Column(Float)
    #
    oco_rise_percent_old = Column(Float)
    oco_rise_percent_new = Column(Float)

    # system info dates
    create_ts = Column(DateTime, server_default=func.now(), nullable=False)
    update_ts = Column(DateTime, onupdate=func.now())

    __table_args__ = ({'extend_existing': True})

    def __repr__(self):
        return f"<IndividualAttributeHistory(id='{self.id}', " \
               f"currency_code='{self.currency_code}', version_id='{self.version_id}', md5='{self.md5}'," \
               f"realtime_enabled_new='{self.realtime_enabled_new}', share_new='{self.share_new}')>"


class Portfolio(Base):
    __tablename__ = 'portfolio'

    portfolio_md5 = Column(String(32), nullable=False, primary_key=True)
    md5 = Column(String(32), nullable=False, primary_key=True)
    share = Column(Float, CheckConstraint('share between 0 and 1'))

    # system info dates
    create_ts = Column(DateTime, server_default=func.now(), nullable=False)
    update_ts = Column(DateTime, onupdate=func.now())

    __table_args__ = ({'extend_existing': True})

    def __repr__(self):
        return f"<Portfolio(portfolio_md5='{self.portfolio_md5}', md5='{self.md5}', share='{self.share}')>"


class Price(Base):
    __tablename__ = 'price'

    currency = Column(String(6), ForeignKey('currency.code'), primary_key=True, nullable=False)
    ts = Column(UTCDateTime, primary_key=True, nullable=False)
    close = Column(Float, CheckConstraint('close > 0'), nullable=False)
    low = Column(Float, CheckConstraint('low > 0'))
    high = Column(Float, CheckConstraint('high > 0'))
    volume = Column(Float, CheckConstraint('volume >= 0'))
    # system info dates
    create_ts = Column(DateTime, server_default=func.now(), nullable=False)
    update_ts = Column(DateTime, onupdate=func.now())

    __table_args__ = (UniqueConstraint('currency', 'ts', name='uc_price_1'),
                      {'extend_existing': True})

    def __repr__(self):
        return f"<Price(ts='{self.ts}', close='{self.close}', volume='{self.volume}')>"


class Deal(Base):
    __tablename__ = 'deal'

    id = Column(Integer, Sequence('deal_id_seq'), primary_key=True)
    version = Column(Integer, ForeignKey('version.id'), nullable=False)
    individual = Column(Integer, ForeignKey('individual.id'), nullable=False)
    currency = Column(String(6), nullable=False)
    is_realtime = Column(Integer, CheckConstraint('is_realtime in (0,1)'), nullable=False, default=0)
    # buy
    buy_ts = Column(UTCDateTime, nullable=False)
    buy_price = Column(Float, CheckConstraint('buy_price > 0'), nullable=False)  # buy limit order price
    # sell
    sell_ts = Column(UTCDateTime)
    sell_price = Column(Float)  # sell limit order price
    # 0 - opened (an individual bought crypto but hasn't sold yet)
    # 1 - closed (an individual sold crypto)
    status = Column(Integer, CheckConstraint('status in (0,1)'), nullable=False)
    # running sums starting with the first individual deal
    run_usd = Column(Float)
    run_crypto = Column(Float)
    run_percent = Column(Float)
    # system info dates
    create_ts = Column(DateTime, server_default=func.now(), nullable=False)
    update_ts = Column(DateTime, onupdate=func.now())

    __table_args__ = (UniqueConstraint('version', 'individual', 'currency', 'buy_ts', 'sell_ts', name='uc_deal_1'),
                      ForeignKeyConstraint((currency, buy_ts), [Price.currency, Price.ts]),
                      CheckConstraint('(status = 0 and run_usd is null and run_percent is not null and run_crypto is not null)'
                                      'or'
                                      '(status = 1 and run_usd is not null and run_percent is not null and run_crypto = 0)'),
                      {'extend_existing': True}
                      )

    def percent(self):
        return (self.sell_price - self.buy_price) * 100 / self.buy_price if self.sell_price else 0

    def __repr__(self):
        status = DealStatus(self.status).name
        sell_part = f" -> {self.sell_price:.1f}$ @ {self.sell_ts}, profit: {self.percent():.2f}%, run: " \
                    f"{self.run_percent or 0:.2f}%" if self.sell_price else ""

        return f"<Deal([{status}] {self.buy_price:.1f}$ @ {self.buy_ts}{sell_part})>"


class IndividualBalance(Base):
    __tablename__ = 'individual_balance'

    currency_code = Column(String(6), ForeignKey('currency.code'), primary_key=True, nullable=False)
    version_id = Column(Integer, ForeignKey('version.id'), primary_key=True, nullable=False)
    individual_id = Column(Integer, ForeignKey('individual.id'), primary_key=True, nullable=False)
    md5 = Column(String(32), nullable=False)

    # balances
    crypto = Column(Float, nullable=False)
    usd = Column(Float, nullable=False)
    percent = Column(Float, nullable=False)
    # reference to deal id
    deal_id = Column(Integer, ForeignKey('deal.id'), nullable=False)

    # system info dates
    create_ts = Column(DateTime, server_default=func.now(), nullable=False)
    update_ts = Column(DateTime, onupdate=func.now())

    __table_args__ = (UniqueConstraint('currency_code', 'version_id', 'individual_id', name='uc_individual_balance_1'),
                      {'extend_existing': True})

    def __repr__(self):
        return f"<IndividualBalance(currency_code='{self.currency_code}', version_id='{self.version_id}', md5='{self.md5}', " \
               f"usd='{self.usd}', percent='{self.percent}', crypto='{self.crypto}'>"


class HistoryStat(Base):
    __tablename__ = 'history_stat'

    id = Column(Integer, Sequence('history_stat_id_seq'), primary_key=True, nullable=False)

    # processing stat
    speed = Column(Interval, nullable=False)
    eta = Column(UTCDateTime, nullable=False)
    processed_count = Column(Integer, CheckConstraint('processed_count >= 0'), nullable=False)
    remaining_count = Column(Integer, CheckConstraint('remaining_count >= 0'), nullable=False)
    status = Column(Integer, CheckConstraint('status in (0,1)'), nullable=False)

    # system info dates
    create_ts = Column(DateTime, server_default=func.now(), nullable=False)
    update_ts = Column(DateTime, onupdate=func.now())

    __table_args__ = (CheckConstraint('status = 0 or remaining_count = 0'),
                      {'extend_existing': True})

    def __repr__(self):
        return f"<HistoryStat(id='{self.id}', status='{self.status}', " \
               f"speed='{self.speed}', processed_count='{self.processed_count}', remaining_count='{self.remaining_count}'>"


class Prediction(Base):
    __tablename__ = 'prediction'

    currency_code = Column(String(6), ForeignKey('currency.code'), primary_key=True, nullable=False)
    version_id = Column(Integer, ForeignKey('version.id'), primary_key=True, nullable=False)
    individual_id = Column(Integer, ForeignKey('individual.id'), primary_key=True, nullable=False)
    ts = Column(UTCDateTime, primary_key=True, nullable=False)

    # ANN prediction results
    buy = Column(Float, nullable=False)
    sell = Column(Float, nullable=False)
    idle = Column(Float, nullable=False)

    # system info dates
    create_ts = Column(DateTime, server_default=func.now(), nullable=False)

    __table_args__ = (UniqueConstraint('currency_code', 'version_id', 'individual_id', 'ts',
                                       name='uc_prediction_1'),
                      {'extend_existing': True})

    def __repr__(self):
        return f"<Prediction(currency_code='{self.currency_code}', ts='{self.ts}', " \
               f"buy='{self.buy}', sell='{self.sell}', idle='{self.idle}'>"


class Action(Base):
    __tablename__ = 'action'

    currency_code = Column(String(6), ForeignKey('currency.code'), primary_key=True, nullable=False)
    ts = Column(UTCDateTime, primary_key=True, nullable=False)

    # limit prices serialized to json dict
    limits = Column(String, nullable=False)

    # price at the moment of action
    close = Column(Float, CheckConstraint('close > 0'), nullable=False)

    # system info dates
    create_ts = Column(DateTime, server_default=func.now(), nullable=False)
    update_ts = Column(DateTime, onupdate=func.now())

    def get_limits(self) -> Tuple[Dict[tuple, float], float]:
        return {tuple(o["limit"]): o["share"] for o in json.loads(self.limits)}, self.close

    __table_args__ = (UniqueConstraint('currency_code', 'ts', name='uc_action_1'),
                      {'extend_existing': True})

    def __repr__(self):
        return f"<Action(limits='{self.limits}', ts='{self.ts}'>"
