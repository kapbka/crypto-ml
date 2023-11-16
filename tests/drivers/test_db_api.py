import asyncio
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from asynctest.mock import MagicMock
from sklearn.preprocessing import StandardScaler

from common.constants import DEFAULT_CURRENCY, RealtimeStatus, HistoryStatus, DealStatus, OptimizedStatus, \
    HistoryStatStatus, RunSumStatus, RunSumIntervalType, EPOCH, IsPortfolio
from common.individual import Individual as Bot
from db.api import DB
from db.api.currency import DBCurrency
from db.api.interval import DBRunSumInterval
from db.api.run_sum import DBRunSum
from db.api.version import DBVersion
from db.model import Version, Currency, Deal, RunSumInterval, RunSumHeader, Prediction, Price
from db.tools.run_sum import upload_run_sum

VERSION_NUM = 1
NEW_VERSION_NUM = 2
TEST_INDICATORS = {'rsi': [1, 5, 100], 'slope': [1, 20, 50]}
TEST_OFFSETS = list(reversed(range(0, 20 + 1, 10)))
TEST_MODEL = [('LSTM', dict(units=1)), ('Dense', dict(units=3))]


class Data:
    def __init__(self):
        self.db = DB()
        # currency
        self.currency = self.db.currency.set(code=DEFAULT_CURRENCY, name=DEFAULT_CURRENCY)
        self.db.flush()
        print(self.currency)

        # ann config
        self.ann = self.db.ann.set(layers=TEST_MODEL, offsets=TEST_OFFSETS, indicators=TEST_INDICATORS, scaled=False)
        self.db.flush()
        self.ann = self.db.ann.get(ann_id=self.ann.id)
        print(self.ann)

        # scaler
        scaler = self.db.scaler.set(currency=DEFAULT_CURRENCY, scaler=StandardScaler())
        self.db.flush()
        self.scaler = self.db.scaler.get(scaler_id=scaler.id)
        print(self.scaler)

        # version
        self.db.version.set(version_num=VERSION_NUM, name=str(VERSION_NUM), comment='test')
        self.bot = Bot(weights=[np.array([1234])])
        self.parent_bot = Bot(weights=[np.array([4321])])

        # individual
        self.db.individual.set(bot=self.bot,
                               parent_md5=self.parent_bot.md5(),
                               ann_id=self.ann.id,
                               scaler_id=self.scaler.id,
                               train_currency='btc')
        self.individual = self.db.individual.get_db(self.bot.md5())
        assert self.individual
        assert self.individual.parent_md5 == self.parent_bot.md5()
        assert self.individual.train_currency == 'btc'
        assert self.individual.is_portfolio == IsPortfolio.No.value
        print(self.individual)

        # individual attributes
        self.individual_attribute = self.db.individual.attribute.get(
            currency_code=self.currency.code,
            version_id=VERSION_NUM,
            md5_or_id=self.bot.md5())
        assert self.individual_attribute
        assert self.individual_attribute.realtime_enabled == RealtimeStatus.Disabled.value
        assert self.individual_attribute.history_enabled == HistoryStatus.Enabled.value
        assert self.individual_attribute.is_optimized == OptimizedStatus.Disabled.value
        assert self.individual_attribute.priority == 0
        assert self.individual_attribute.share == 0
        print(self.individual_attribute)


class DataPortfolio:
    def __init__(self):
        self.db = DB()
        # currency
        self.currency = self.db.currency.set(code=DEFAULT_CURRENCY, name=DEFAULT_CURRENCY)
        print(self.currency)

        # ann config
        self.ann = self.db.ann.set(layers=TEST_MODEL, offsets=TEST_OFFSETS, indicators=TEST_INDICATORS, scaled=False)
        self.db.flush()
        self.ann = self.db.ann.get(ann_id=self.ann.id)
        print(self.ann)

        # scaler
        scaler = self.db.scaler.set(currency=DEFAULT_CURRENCY, scaler=StandardScaler())
        self.db.flush()
        self.scaler = self.db.scaler.get(scaler_id=scaler.id)
        print(self.scaler)

        # version
        self.db.version.set(version_num=VERSION_NUM, name=str(VERSION_NUM), comment='test')
        self.bot_1 = Bot(weights=[np.array([1234])])
        self.bot_2 = Bot(weights=[np.array([12345])])
        self.bot_portfolio = Bot(weights=[np.array([123456])])

        # member individual 1
        self.db.individual.set(bot=self.bot_1,
                               ann_id=self.ann.id,
                               scaler_id=self.scaler.id,
                               train_currency='btc')
        self.individual_1 = self.db.individual.get_db(self.bot_1.md5())
        assert self.individual_1
        assert self.individual_1.train_currency == 'btc'
        assert self.individual_1.is_portfolio == IsPortfolio.No.value

        # member individual 2
        self.db.individual.set(bot=self.bot_2,
                               ann_id=self.ann.id,
                               scaler_id=self.scaler.id,
                               train_currency='btc')
        self.individual_2 = self.db.individual.get_db(self.bot_2.md5())
        assert self.individual_2
        assert self.individual_2.train_currency == 'btc'
        assert self.individual_2.is_portfolio == IsPortfolio.No.value

        # portfolio
        self.db.individual.set(bot=self.bot_portfolio,
                               ann_id=self.ann.id,
                               scaler_id=self.scaler.id,
                               train_currency='btc',
                               is_portfolio=IsPortfolio.Yes.value)
        self.individual_portfolio = self.db.individual.get_db(self.bot_portfolio.md5())
        assert self.individual_portfolio
        assert self.individual_portfolio.train_currency == 'btc'
        assert self.individual_portfolio.is_portfolio == IsPortfolio.Yes.value


def make_deal(data: Data,
              currency_code: str, ts: datetime,
              close: float, low: float, high: float, volume: float,
              version_id: int, status: int,
              buy_ts: Optional[datetime], buy_price: Optional[float],
              run_usd: Optional[float], run_crypto: Optional[float], run_percent: Optional[float]):
    for dt in [buy_ts, ts]:
        if dt:
            price = data.db.price.get(currency_code=currency_code, ts=dt)
            if not price:
                price = data.db.price.set(currency=currency_code,
                                          ts=dt,
                                          close=close,
                                          low=low,
                                          high=high,
                                          volume=volume)
            else:
                price.close = close
                price.low = low
                price.high = high
                price.volume = volume
            assert price
            assert price.ts == dt
            assert price.close == close
            print(price)

    _deal = Deal(version=VERSION_NUM,
                 individual=data.individual.id,
                 currency=DEFAULT_CURRENCY,
                 is_realtime=RealtimeStatus.Disabled.value,
                 buy_ts=price.ts if status == DealStatus.Open.value else buy_ts,
                 buy_price=price.close if status == DealStatus.Open.value else buy_price,
                 sell_ts=price.ts if status == DealStatus.Close.value else None,
                 sell_price=price.close if status == DealStatus.Close.value else None,
                 status=status,
                 run_usd=run_usd,
                 run_crypto=run_crypto,
                 run_percent=run_percent)
    data.db.deal.set_batch(version=version_id,
                           currency=currency_code,
                           individual_id=data.individual.id,
                           deals=[_deal],
                           replace=False)
    deal = data.db.deal.get(version_id=VERSION_NUM,
                            currency=DEFAULT_CURRENCY,
                            individual_id=data.individual.id,
                            buy_ts=buy_ts if buy_ts else price.ts)

    assert deal
    assert deal.status == status
    assert deal.buy_ts == buy_ts if buy_ts else price.ts
    assert deal.buy_price == buy_price if buy_price else price.close
    print(deal)

    if status == DealStatus.Close.value:
        assert deal.sell_ts == price.ts
        assert deal.sell_price == price.close
    assert deal.run_usd == run_usd
    assert deal.run_crypto == run_crypto
    assert deal.run_percent == run_percent

    if status == DealStatus.Close.value:
        # balance
        individual_balance = data.db.individual.balance.set(md5=data.bot.md5(), deal=deal)
        assert individual_balance
        assert individual_balance.usd == run_usd
        print(individual_balance)

    return deal


def test_save_version():
    data = Data()

    version = data.db.version.set(version_num=NEW_VERSION_NUM, name=str(NEW_VERSION_NUM), comment='New version')
    assert version
    assert version.id == NEW_VERSION_NUM
    print(version)

    individual_attribute = data.db.individual.attribute.get(
        currency_code=data.currency.code,
        version_id=NEW_VERSION_NUM,
        md5_or_id=data.individual.md5)
    assert individual_attribute
    assert individual_attribute.realtime_enabled == RealtimeStatus.Disabled.value
    assert individual_attribute.history_enabled == HistoryStatus.Enabled.value
    assert individual_attribute.is_optimized == OptimizedStatus.Disabled.value
    assert individual_attribute.priority == 0
    assert individual_attribute.share == 0

    # check that default attributes are not logged
    individual_attribute_history = data.db.individual.attribute.history.get_all(
        currency_code=data.currency.code,
        version_id=NEW_VERSION_NUM,
        md5_or_id=data.individual.md5
    )
    assert len(individual_attribute_history) == 0

    data.db.rollback()


def test_save_price_batch():
    data = Data()
    frame = pd.DataFrame({'close': [1, 2],
                          'low': [1, 2],
                          'high': [1, 2],
                          'volume': [1, 2],
                          }, index=[datetime.now(tz=timezone.utc) - timedelta(minutes=1),
                                    datetime.now(tz=timezone.utc)])
    data.db.price.set_batch(data.currency.code, frame)
    assert data.db.price.get_last(data.currency.code)
    assert data.db.price.get_first(data.currency.code)

    data.db.rollback()


def test_save_individual():
    data = Data()

    ind1 = data.db.individual.get(data.individual.id)
    ind2 = data.db.individual.get(data.individual.md5)
    assert ind1.md5() == ind2.md5()

    ind3 = data.db.individual.search(ind1.md5()[:5], 1)[0]
    assert ind1.md5() == ind3.md5

    data.db.rollback()


def test_invalid_individual_attributes():
    data = Data()

    # realtime_enabled and history_enabled are both enabled
    with pytest.raises(ValueError):
        data.db.individual.attribute.set(currency_code=data.currency.code,
                                         version_id=VERSION_NUM,
                                         individual=data.individual,
                                         realtime_enabled=RealtimeStatus.Enabled.value,
                                         history_enabled=HistoryStatus.Enabled.value,
                                         is_optimized=OptimizedStatus.Disabled.value,
                                         priority=0,
                                         share=0,
                                         scaler_id=None,
                                         oco_buy_percent=0.02,
                                         oco_sell_percent=-0.02,
                                         oco_rise_percent=0.0)

    # realtime is enabled but share is not populated correctly within (0,1] interval
    with pytest.raises(ValueError):
        data.db.individual.attribute.set(currency_code=data.currency.code,
                                         version_id=VERSION_NUM,
                                         individual=data.individual,
                                         realtime_enabled=RealtimeStatus.Enabled.value,
                                         history_enabled=HistoryStatus.Disabled.value,
                                         is_optimized=OptimizedStatus.Disabled.value,
                                         priority=0,
                                         share=0,
                                         scaler_id=None,
                                         oco_buy_percent=0.02,
                                         oco_sell_percent=-0.02,
                                         oco_rise_percent=0.0)
    data.db.rollback()


def test_change_individual_attributes():
    data = Data()

    # change realtime enabled, history disabled, optimized = on
    data.db.individual.attribute.set(currency_code=data.currency.code,
                                     version_id=VERSION_NUM,
                                     individual=data.individual,
                                     realtime_enabled=RealtimeStatus.Enabled.value,
                                     history_enabled=HistoryStatus.Disabled.value,
                                     is_optimized=OptimizedStatus.Enabled.value,
                                     priority=1,
                                     share=0.5,
                                     scaler_id=data.scaler.id,
                                     oco_buy_percent=0.02,
                                     oco_sell_percent=-0.02,
                                     oco_rise_percent=0.0)
    assert data.individual_attribute.realtime_enabled == RealtimeStatus.Enabled.value
    assert data.individual_attribute.history_enabled == HistoryStatus.Disabled.value
    assert data.individual_attribute.is_optimized == OptimizedStatus.Enabled.value
    assert data.individual_attribute.priority == 1
    assert data.individual_attribute.share == 0.5

    enabled = data.db.individual.get_realtime(currency_code=data.currency.code)
    assert enabled

    individual_attribute_history_last = data.db.individual.attribute.history.get_last(
        currency_code=data.currency.code,
        version_id=VERSION_NUM,
        md5_or_id=data.individual.md5
    )
    assert individual_attribute_history_last
    assert individual_attribute_history_last.realtime_enabled_old == RealtimeStatus.Disabled.value
    assert individual_attribute_history_last.realtime_enabled_new == RealtimeStatus.Enabled.value
    assert individual_attribute_history_last.history_enabled_old == HistoryStatus.Enabled.value
    assert individual_attribute_history_last.history_enabled_new == HistoryStatus.Disabled.value
    assert individual_attribute_history_last.priority_old == 0
    assert individual_attribute_history_last.priority_new == 1
    assert individual_attribute_history_last.share_old == 0.0
    assert individual_attribute_history_last.share_new == 0.5

    print(individual_attribute_history_last)

    individual_attribute_history_all = data.db.individual.attribute.history.get_all(
        currency_code=data.currency.code,
        version_id=VERSION_NUM,
        md5_or_id=data.individual.md5
    )
    assert len(individual_attribute_history_all) == 1

    # change realtime disabled, history enabled
    data.db.individual.attribute.set(currency_code=data.currency.code,
                                     version_id=VERSION_NUM,
                                     individual=data.individual,
                                     realtime_enabled=RealtimeStatus.Disabled.value,
                                     history_enabled=HistoryStatus.Enabled.value,
                                     is_optimized=OptimizedStatus.Disabled.value,
                                     priority=1,
                                     share=0.5,
                                     scaler_id=data.scaler.id,
                                     oco_buy_percent=0.02,
                                     oco_sell_percent=-0.02,
                                     oco_rise_percent=0.0)
    assert data.individual_attribute.realtime_enabled == RealtimeStatus.Disabled.value
    assert data.individual_attribute.history_enabled == HistoryStatus.Enabled.value
    assert data.individual_attribute.is_optimized == OptimizedStatus.Disabled.value
    assert data.individual_attribute.priority == 1
    assert data.individual_attribute.share == 0.5

    individual_attribute_history_last = data.db.individual.attribute.history.get_last(
        currency_code=data.currency.code,
        version_id=VERSION_NUM,
        md5_or_id=data.individual.md5
    )
    assert individual_attribute_history_last
    assert individual_attribute_history_last.realtime_enabled_old == RealtimeStatus.Enabled.value
    assert individual_attribute_history_last.realtime_enabled_new == RealtimeStatus.Disabled.value
    assert individual_attribute_history_last.history_enabled_old == HistoryStatus.Disabled.value
    assert individual_attribute_history_last.history_enabled_new == HistoryStatus.Enabled.value
    assert individual_attribute_history_last.priority_old == 1
    assert individual_attribute_history_last.priority_new == 1
    assert individual_attribute_history_last.share_old == 0.5
    assert individual_attribute_history_last.share_new == 0.5

    individual_attribute_history_all = data.db.individual.attribute.history.get_all(
        currency_code=data.currency.code,
        version_id=VERSION_NUM,
        md5_or_id=data.individual.md5
    )
    assert len(individual_attribute_history_all) == 2

    # disable both
    data.db.individual.attribute.set(currency_code=data.currency.code,
                                     version_id=VERSION_NUM,
                                     individual=data.individual,
                                     realtime_enabled=RealtimeStatus.Disabled.value,
                                     history_enabled=HistoryStatus.Disabled.value,
                                     is_optimized=OptimizedStatus.Enabled.value,
                                     priority=1,
                                     share=1,
                                     scaler_id=data.scaler.id,
                                     oco_buy_percent=0.02,
                                     oco_sell_percent=-0.02,
                                     oco_rise_percent=0.0)
    assert data.individual_attribute.realtime_enabled == RealtimeStatus.Disabled.value
    assert data.individual_attribute.history_enabled == HistoryStatus.Disabled.value
    assert data.individual_attribute.is_optimized == OptimizedStatus.Enabled.value
    assert data.individual_attribute.priority == 1
    assert data.individual_attribute.share == 1

    individual_attribute_history_last = data.db.individual.attribute.history.get_last(
        currency_code=data.currency.code,
        version_id=VERSION_NUM,
        md5_or_id=data.individual.md5
    )
    assert individual_attribute_history_last
    assert individual_attribute_history_last.realtime_enabled_old == RealtimeStatus.Disabled.value
    assert individual_attribute_history_last.realtime_enabled_new == RealtimeStatus.Disabled.value
    assert individual_attribute_history_last.history_enabled_old == HistoryStatus.Enabled.value
    assert individual_attribute_history_last.history_enabled_new == HistoryStatus.Disabled.value
    assert individual_attribute_history_last.priority_old == 1
    assert individual_attribute_history_last.priority_new == 1
    assert individual_attribute_history_last.share_old == 0.5
    assert individual_attribute_history_last.share_new == 1

    individual_attribute_history_all = data.db.individual.attribute.history.get_all(
        currency_code=data.currency.code,
        version_id=VERSION_NUM,
        md5_or_id=data.individual.md5
    )
    assert len(individual_attribute_history_all) == 3

    data.db.rollback()


def test_get_history():
    data = Data()

    individuals = data.db.individual.get_history(portfolio=IsPortfolio.No)
    assert individuals

    data.db.rollback()


def test_set_invalid_individual_attribute_history():
    data = Data()

    # change realtime enabled, history disabled, optimized = on
    data.db.individual.attribute.set(currency_code=data.currency.code,
                                     version_id=VERSION_NUM,
                                     individual=data.individual,
                                     realtime_enabled=RealtimeStatus.Enabled.value,
                                     history_enabled=HistoryStatus.Disabled.value,
                                     is_optimized=OptimizedStatus.Enabled.value,
                                     priority=1,
                                     share=0.5,
                                     scaler_id=data.scaler.id,
                                     oco_buy_percent=0.02,
                                     oco_sell_percent=-0.02,
                                     oco_rise_percent=0.0)
    assert data.individual_attribute.realtime_enabled == RealtimeStatus.Enabled.value
    assert data.individual_attribute.history_enabled == HistoryStatus.Disabled.value
    assert data.individual_attribute.is_optimized == OptimizedStatus.Enabled.value
    assert data.individual_attribute.priority == 1
    assert data.individual_attribute.share == 0.5

    individual_attribute_history_last = data.db.individual.attribute.history.get_last(
        currency_code=data.currency.code,
        version_id=VERSION_NUM,
        md5_or_id=data.individual.md5
    )
    assert individual_attribute_history_last
    assert individual_attribute_history_last.realtime_enabled_old == RealtimeStatus.Disabled.value
    assert individual_attribute_history_last.realtime_enabled_new == RealtimeStatus.Enabled.value
    assert individual_attribute_history_last.history_enabled_old == HistoryStatus.Enabled.value
    assert individual_attribute_history_last.history_enabled_new == HistoryStatus.Disabled.value
    assert individual_attribute_history_last.priority_old == 0
    assert individual_attribute_history_last.priority_new == 1
    assert individual_attribute_history_last.share_old == 0.0
    assert individual_attribute_history_last.share_new == 0.5

    individual_attribute_new = deepcopy(data.individual_attribute)
    individual_attribute_new.currency_code = 'tst'

    with pytest.raises(ValueError):
        data.db.individual.attribute.history.set(
            individual_attribute_old=data.individual_attribute,
            individual_attribute_new=individual_attribute_new)

    data.db.rollback()


def test_portfolio():
    data = DataPortfolio()

    link_1 = data.db.portfolio.set(
        portfolio_md5=data.individual_portfolio.md5,
        md5=data.individual_1.md5,
        share=0.2
    )
    assert link_1
    print(link_1)
    link_1 = data.db.portfolio.get(
        portfolio_md5=data.individual_portfolio.md5,
        md5=data.individual_1.md5
    )
    assert link_1
    assert link_1.share == 0.2

    # update
    link_1 = data.db.portfolio.set(
        portfolio_md5=data.individual_portfolio.md5,
        md5=data.individual_1.md5,
        share=0.3
    )
    assert link_1
    link_1 = data.db.portfolio.get(
        portfolio_md5=data.individual_portfolio.md5,
        md5=data.individual_1.md5
    )
    assert link_1
    assert link_1.share == 0.3

    link_2 = data.db.portfolio.set(
        portfolio_md5=data.individual_portfolio.md5,
        md5=data.individual_2.md5,
        share=0.7)
    assert link_2
    link_2 = data.db.portfolio.get(
        portfolio_md5=data.individual_portfolio.md5,
        md5=data.individual_2.md5
    )
    assert link_2
    assert link_2.share == 0.7

    all_links = data.db.portfolio.get_batch(portfolio_md5=data.individual_portfolio.md5)
    assert all_links
    assert len(all_links) == 2

    # set realtime enabled
    member_attr_1 = data.db.individual.attribute.set(currency_code=data.currency.code,
                                                     version_id=VERSION_NUM,
                                                     individual=data.individual_1,
                                                     realtime_enabled=RealtimeStatus.Enabled.value,
                                                     history_enabled=HistoryStatus.Disabled.value,
                                                     is_optimized=OptimizedStatus.Enabled.value,
                                                     priority=1,
                                                     share=0.5,
                                                     scaler_id=data.scaler.id,
                                                     oco_buy_percent=0.02,
                                                     oco_sell_percent=-0.02,
                                                     oco_rise_percent=0.0)
    assert member_attr_1.realtime_enabled == RealtimeStatus.Enabled.value
    assert member_attr_1.history_enabled == HistoryStatus.Disabled.value
    assert member_attr_1.is_optimized == OptimizedStatus.Enabled.value
    assert member_attr_1.priority == 1
    assert member_attr_1.share == 0.5

    member_attr_2 = data.db.individual.attribute.set(currency_code=data.currency.code,
                                                     version_id=VERSION_NUM,
                                                     individual=data.individual_2,
                                                     realtime_enabled=RealtimeStatus.Enabled.value,
                                                     history_enabled=HistoryStatus.Disabled.value,
                                                     is_optimized=OptimizedStatus.Enabled.value,
                                                     priority=1,
                                                     share=0.5,
                                                     scaler_id=data.scaler.id,
                                                     oco_buy_percent=0.02,
                                                     oco_sell_percent=-0.02,
                                                     oco_rise_percent=0.0)
    assert member_attr_2.realtime_enabled == RealtimeStatus.Enabled.value
    assert member_attr_2.history_enabled == HistoryStatus.Disabled.value
    assert member_attr_2.is_optimized == OptimizedStatus.Enabled.value
    assert member_attr_2.priority == 1
    assert member_attr_2.share == 0.5

    portfolio_attr = data.db.individual.attribute.set(currency_code=data.currency.code,
                                                      version_id=VERSION_NUM,
                                                      individual=data.individual_portfolio,
                                                      realtime_enabled=RealtimeStatus.Enabled.value,
                                                      history_enabled=HistoryStatus.Disabled.value,
                                                      is_optimized=OptimizedStatus.Enabled.value,
                                                      priority=1,
                                                      share=1,
                                                      scaler_id=data.scaler.id,
                                                      oco_buy_percent=0.02,
                                                      oco_sell_percent=-0.02,
                                                      oco_rise_percent=0.0)
    assert portfolio_attr.realtime_enabled == RealtimeStatus.Enabled.value
    assert portfolio_attr.history_enabled == HistoryStatus.Disabled.value
    assert portfolio_attr.is_optimized == OptimizedStatus.Enabled.value
    assert portfolio_attr.priority == 1
    assert portfolio_attr.share == 1

    # get realtime portfolio
    realtime_portfolio = data.db.portfolio.get_realtime(currency_code=data.currency.code)
    assert realtime_portfolio
    assert len(realtime_portfolio) == 1

    # portfolio individuals with attributes without portfolio_md5 populated
    portfolio_individuals_with_attrs = data.db.portfolio.get_realtime_members_with_attrs(
        currency_code=data.currency.code,
        version_id=VERSION_NUM
    )
    assert portfolio_individuals_with_attrs
    assert len(portfolio_individuals_with_attrs) == 2

    # portfolio individuals with attributes with portfolio_md5 populated
    portfolio_individuals_with_attrs = data.db.portfolio.get_realtime_members_with_attrs(
        currency_code=data.currency.code,
        version_id=VERSION_NUM,
        portfolio_md5=realtime_portfolio[0].md5
    )
    assert portfolio_individuals_with_attrs
    assert len(portfolio_individuals_with_attrs) == 2

    # deletion
    data.db.portfolio.delete(
        portfolio_md5=data.individual_portfolio.md5,
        md5=data.individual_2.md5
    )
    all_links = data.db.portfolio.get_batch(portfolio_md5=data.individual_portfolio.md5)
    assert all_links
    assert len(all_links) == 1

    # one more time
    with pytest.raises(ValueError):
        data.db.portfolio.delete(
            portfolio_md5=data.individual_portfolio.md5,
            md5=data.individual_2.md5
        )

    data.db.rollback()


# https://photos.app.goo.gl/sVyc55CMe2i9B16o6
def test_save_deal():
    data = Data()

    # 0. the very first deal
    deal_buy = make_deal(data=data,
                         currency_code=data.currency.code,
                         ts=datetime(day=1, month=10, year=2021, hour=00, minute=00, tzinfo=timezone.utc),
                         close=58500, low=58000, high=59000, volume=100,
                         version_id=VERSION_NUM, status=DealStatus.Open.value,
                         buy_ts=None, buy_price=None,
                         run_usd=None, run_crypto=0.999, run_percent=4)

    # close
    deal_sell = make_deal(data=data,
                          currency_code=data.currency.code,
                          ts=datetime(day=5, month=10, year=2021, hour=10, minute=00, tzinfo=timezone.utc),
                          close=60000, low=59000, high=63000, volume=500,
                          version_id=VERSION_NUM, status=DealStatus.Close.value,
                          buy_ts=deal_buy.buy_ts, buy_price=deal_buy.buy_price,
                          run_usd=1000, run_crypto=0, run_percent=5)

    ###################################################################################################################

    # 1. deal_new earlier than deal_last, no intersection
    with pytest.raises(ValueError):
        deal = make_deal(data=data,
                         currency_code=data.currency.code,
                         ts=datetime(day=29, month=9, year=2021, hour=10, minute=00, tzinfo=timezone.utc),
                         close=50000, low=45000, high=52000, volume=50,
                         version_id=VERSION_NUM, status=DealStatus.Close.value,
                         buy_ts=datetime(day=28, month=9, year=2021, hour=5, minute=00, tzinfo=timezone.utc),
                         buy_price=51800,
                         run_usd=55000, run_crypto=0, run_percent=4)

    ###################################################################################################################

    # 2. deal_new later than deal_last, no intersection
    deal = make_deal(data=data,
                     currency_code=data.currency.code,
                     ts=datetime(day=12, month=10, year=2021, hour=12, minute=00, tzinfo=timezone.utc),
                     close=54600, low=45000, high=52000, volume=50,
                     version_id=VERSION_NUM, status=DealStatus.Close.value,
                     buy_ts=datetime(day=7, month=10, year=2021, hour=9, minute=00, tzinfo=timezone.utc),
                     buy_price=51800, run_usd=53200, run_crypto=0, run_percent=4)
    deals = data.db.deal.get_all(
        currency=data.currency.code,
        version_id=VERSION_NUM,
        individual_id=data.individual.id,
        ts_from=datetime(day=1, month=9, year=2021, hour=00, minute=00, tzinfo=timezone.utc)
    )
    assert len(deals) == 2
    last = data.db.deal.get_last(currency=data.currency.code, version=VERSION_NUM, individual=data.individual.id)
    assert last.buy_ts == deal.buy_ts
    assert last.buy_price == deal.buy_price
    assert last.sell_ts == deal.sell_ts
    assert last.sell_price == deal.sell_price

    ###################################################################################################################

    # 3-4. deal_new.buy_ts between deal_last.buy_ts and deal_last.sell_ts (opened)
    with pytest.raises(ValueError):
        deal = make_deal(data=data,
                         currency_code=data.currency.code,
                         ts=datetime(day=9, month=10, year=2021, hour=12, minute=00, tzinfo=timezone.utc),
                         close=54600, low=45000, high=52000, volume=50,
                         version_id=VERSION_NUM, status=DealStatus.Open.value,
                         buy_ts=None, buy_price=None,
                         run_usd=None, run_crypto=0.987, run_percent=3)

    data.db.rollback()


def test_save_deal_invalid():
    data = Data()

    # 1. buy_ts == sell_ts
    with pytest.raises(ValueError):
        deal = make_deal(data=data,
                         currency_code=data.currency.code,
                         ts=datetime(day=6, month=10, year=2021, hour=10, minute=00, tzinfo=timezone.utc),
                         close=54600, low=45000, high=52000, volume=50,
                         version_id=VERSION_NUM, status=DealStatus.Close.value,
                         buy_ts=datetime(day=6, month=10, year=2021, hour=10, minute=00, tzinfo=timezone.utc),
                         buy_price=51800, run_usd=53200, run_crypto=0, run_percent=4)

    ###################################################################################################################

    # 2. buy_ts > sell_ts
    with pytest.raises(ValueError):
        deal = make_deal(data=data,
                         currency_code=data.currency.code,
                         ts=datetime(day=6, month=10, year=2021, hour=10, minute=00, tzinfo=timezone.utc),
                         close=54600, low=45000, high=52000, volume=50,
                         version_id=VERSION_NUM, status=DealStatus.Close.value,
                         buy_ts=datetime(day=8, month=10, year=2021, hour=10, minute=00, tzinfo=timezone.utc),
                         buy_price=51800, run_usd=53200, run_crypto=0, run_percent=4)

    data.db.rollback()


def test_predictions():
    data = Data()

    predictions = pd.DataFrame(
        dict(currency_code=[data.currency.code],
             version_id=[VERSION_NUM],
             individual_id=[data.individual.id],
             buy=[1],
             sell=[0],
             idle=[0]),
        index=[datetime.now()]
    )

    prediction = Prediction(
        currency_code=data.currency.code,
        version_id=VERSION_NUM,
        individual_id=data.individual.id,
        ts=datetime.now(),
        buy=1,
        sell=0,
        idle=0
    )
    print(prediction)

    data.db.prediction.save(predictions)

    res = data.db.prediction.load(
        ts_from=EPOCH,
        currency=data.currency.code,
        version=VERSION_NUM,
        bot_id=data.individual.id,
    )
    assert len(res), res

    # create a gap
    data.db.prediction.save(pd.DataFrame(
        dict(currency_code=[data.currency.code],
             version_id=[VERSION_NUM],
             individual_id=[data.individual.id],
             buy=[1],
             sell=[0],
             idle=[0]),
        index=[datetime.now() + timedelta(minutes=30)]
    ))

    # make sure the cache is invalidated
    result = data.db.prediction.load(
        ts_from=EPOCH,
        currency=data.currency.code,
        version=VERSION_NUM,
        bot_id=data.individual.id,
    )

    assert not len(result), result

    data.db.rollback()


def test_history_stat():
    data = Data()

    # create
    history_stat = data.db.history_stat.set(
        speed=timedelta(hours=2),
        eta=datetime(day=15, month=12, year=2021, hour=12, minute=00, tzinfo=timezone.utc),
        processed_count=0,
        remaining_count=20000
    )
    assert history_stat
    assert history_stat.speed == timedelta(hours=2)
    assert history_stat.eta == datetime(day=15, month=12, year=2021, hour=12, minute=00, tzinfo=timezone.utc)
    assert history_stat.processed_count == 0
    assert history_stat.remaining_count == 20000
    assert history_stat.status == HistoryStatStatus.InProgress.value
    print(history_stat)

    # in progress update
    history_stat = data.db.history_stat.set(
        speed=timedelta(hours=1),
        eta=datetime(day=15, month=12, year=2021, hour=9, minute=00, tzinfo=timezone.utc),
        processed_count=3000,
        remaining_count=17000
    )
    assert history_stat
    assert history_stat.speed == timedelta(hours=1)
    assert history_stat.eta == datetime(day=15, month=12, year=2021, hour=9, minute=00, tzinfo=timezone.utc)
    assert history_stat.processed_count == 3000
    assert history_stat.remaining_count == 17000
    assert history_stat.status == HistoryStatStatus.InProgress.value
    print(history_stat)

    # done
    history_stat = data.db.history_stat.set(
        speed=timedelta(hours=1),
        eta=datetime(day=15, month=12, year=2021, hour=9, minute=00, tzinfo=timezone.utc),
        processed_count=20000,
        remaining_count=0
    )
    data.db.flush()
    assert history_stat
    assert history_stat.speed == timedelta(hours=1)
    assert history_stat.eta == datetime(day=15, month=12, year=2021, hour=9, minute=00, tzinfo=timezone.utc)
    assert history_stat.processed_count == 20000
    assert history_stat.remaining_count == 0
    assert history_stat.status == HistoryStatStatus.Done.value
    print(history_stat)

    stat = data.db.history_stat.get(history_stat.id)
    assert stat

    last = data.db.history_stat.get_last()
    assert last

    batch = data.db.history_stat.get_batch()
    assert batch

    data.db.rollback()


def test_run_sum_interval():
    data = Data()

    run_sum_interval_month = data.db.run_sum_interval.set(
        interval_code='month_',
        interval_name='Month_',
        interval_type=RunSumIntervalType.FromNow,
        from_now=timedelta(days=30)
    )
    assert run_sum_interval_month
    print(run_sum_interval_month)

    run_sum_interval_month_get = data.db.run_sum_interval.get(interval_code='month_')
    assert run_sum_interval_month_get

    run_sum_interval_fall = data.db.run_sum_interval.set(
        interval_code='fall_dec',
        interval_name='Fall Dec',
        interval_type=RunSumIntervalType.FromTo,
        start_ts=datetime(day=3, month=12, year=2021),
        end_ts=datetime(day=5, month=12, year=2021)
    )
    assert run_sum_interval_fall
    print(run_sum_interval_fall)

    run_sum_intervals = data.db.run_sum_interval.get_batch()
    run_sum_intervals_len = len(run_sum_intervals)
    assert len(run_sum_intervals) > 1

    data.db.run_sum_interval.delete(run_sum_interval_id=run_sum_interval_month_get.id)

    run_sum_intervals = data.db.run_sum_interval.get_batch()
    assert len(run_sum_intervals) == run_sum_intervals_len - 1

    # incorrectly passed parameters
    with pytest.raises(ValueError):
        run_sum_interval_month = data.db.run_sum_interval.set(
            interval_code='month_',
            interval_name='Month_',
            interval_type=RunSumIntervalType.FromTo,
            from_now=timedelta(days=30)
        )

    with pytest.raises(ValueError):
        run_sum_interval_fall = data.db.run_sum_interval.set(
            interval_code='fall_dec',
            interval_name='Fall Dec',
            interval_type=RunSumIntervalType.FromNow,
            start_ts=datetime(day=3, month=12, year=2021),
            end_ts=datetime(day=5, month=12, year=2021)
        )

    data.db.rollback()


def test_run_sum_from_now():
    data = Data()

    interval_code = 'week'
    interval_len = 7
    dt = datetime.now()
    # truncate current date to minutes
    dt = datetime(day=dt.day, month=dt.month, year=dt.year, hour=dt.hour, minute=dt.minute, tzinfo=timezone.utc)

    start_ts = dt - timedelta(days=7)
    end_ts = dt

    price = data.db.price.get(currency_code=data.currency.code, ts=start_ts)
    if not price:
        price = data.db.price.set(currency=data.currency.code,
                                  ts=start_ts,
                                  close=54000,
                                  low=51200,
                                  high=55000,
                                  volume=500)

    run_sum_interval_week = data.db.run_sum_interval.set(
        interval_code=interval_code,
        interval_name=interval_code,
        interval_type=RunSumIntervalType.FromNow,
        from_now=timedelta(days=interval_len)
    )
    assert run_sum_interval_week

    # in progress
    run_sum_header = data.db.run_sum.set_header(
        currency_code=data.currency.code,
        version_id=VERSION_NUM,
        interval_id=run_sum_interval_week.id,
        interval_code=interval_code,
        start_ts=start_ts,
        end_ts=end_ts,
        status=RunSumStatus.InProgress.value)
    assert run_sum_header
    assert run_sum_header.status == RunSumStatus.InProgress.value
    print(run_sum_header)

    # upload run sum
    data.db.run_sum.upload(currency_code=data.currency.code,
                           version_id=VERSION_NUM,
                           interval_id=run_sum_interval_week.id,
                           current_batch_num=0,
                           number_of_batches=1,
                           percent_commission=0.1)

    # done
    data.db.run_sum.set_header(currency_code=data.currency.code,
                               version_id=VERSION_NUM,
                               interval_id=run_sum_interval_week.id,
                               interval_code=interval_code,
                               start_ts=start_ts,
                               end_ts=end_ts,
                               status=RunSumStatus.Done.value)
    assert run_sum_header
    assert run_sum_header.status == RunSumStatus.Done.value

    # done second time to mark as done
    with pytest.raises(ValueError):
        data.db.run_sum.set_header(currency_code=data.currency.code,
                                   version_id=VERSION_NUM,
                                   interval_id=run_sum_interval_week.id,
                                   interval_code=interval_code,
                                   start_ts=start_ts,
                                   end_ts=end_ts,
                                   status=RunSumStatus.Done.value)

    # reload header as new in progress (to cover delete method)
    run_sum_header = data.db.run_sum.set_header(
        currency_code=data.currency.code,
        version_id=VERSION_NUM,
        interval_id=run_sum_interval_week.id,
        interval_code=interval_code,
        start_ts=start_ts,
        end_ts=end_ts,
        status=RunSumStatus.InProgress.value)
    assert run_sum_header
    assert run_sum_header.status == RunSumStatus.InProgress.value

    data.db.rollback()


def test_run_sum_from_to():
    data = Data()

    interval_code = 'fall'
    dt = datetime.now()
    # truncate current date to minutes
    dt = datetime(day=dt.day, month=dt.month, year=dt.year, hour=dt.hour, minute=dt.minute, tzinfo=timezone.utc)

    start_ts = dt - timedelta(days=5)
    end_ts = dt

    price = data.db.price.get(currency_code=data.currency.code, ts=start_ts)
    if not price:
        price = data.db.price.set(currency=data.currency.code,
                                  ts=start_ts,
                                  close=54000,
                                  low=51200,
                                  high=55000,
                                  volume=500)

    run_sum_interval_custom = data.db.run_sum_interval.set(
        interval_code=interval_code,
        interval_name=interval_code,
        interval_type=RunSumIntervalType.FromTo,
        start_ts=start_ts,
        end_ts=end_ts
    )
    assert run_sum_interval_custom

    # in progress
    run_sum_header = data.db.run_sum.set_header(
        currency_code=data.currency.code,
        version_id=VERSION_NUM,
        interval_id=run_sum_interval_custom.id,
        interval_code=interval_code,
        start_ts=start_ts,
        end_ts=end_ts,
        status=RunSumStatus.InProgress.value)
    assert run_sum_header
    assert run_sum_header.status == RunSumStatus.InProgress.value
    print(run_sum_header)

    # upload run sum
    data.db.run_sum.upload(currency_code=data.currency.code,
                           version_id=VERSION_NUM,
                           interval_id=run_sum_interval_custom.id,
                           current_batch_num=0,
                           number_of_batches=1,
                           percent_commission=0.1)

    # done
    data.db.run_sum.set_header(currency_code=data.currency.code,
                               version_id=VERSION_NUM,
                               interval_id=run_sum_interval_custom.id,
                               interval_code=interval_code,
                               start_ts=start_ts,
                               end_ts=end_ts,
                               status=RunSumStatus.Done.value)
    assert run_sum_header
    assert run_sum_header.status == RunSumStatus.Done.value

    data.db.rollback()


@patch.object(DBVersion, 'get_batch', new=MagicMock(return_value=[Version(id=VERSION_NUM, name=str(VERSION_NUM))]))
@patch.object(DBCurrency, 'get_batch', new=MagicMock(return_value=[Currency(code=DEFAULT_CURRENCY)]))
@patch.object(DBRunSumInterval, 'get_batch', new=MagicMock(return_value=[RunSumInterval(type=5)]))
def test_run_sum_upload_script_unknown_interval():
    # upload run sum
    with pytest.raises(ValueError):
        upload_run_sum(number_of_batches=3)


@patch.object(DBVersion, 'get_batch', new=MagicMock(return_value=[Version(id=VERSION_NUM, name=str(VERSION_NUM))]))
@patch.object(DBCurrency, 'get_batch', new=MagicMock(return_value=[Currency(code=DEFAULT_CURRENCY)]))
@patch.object(DBRunSumInterval, 'get_batch', new=MagicMock(return_value=[RunSumInterval(id=100,
                                                                                        code='week',
                                                                                        type=RunSumIntervalType.FromTo.value,
                                                                                        start_ts=(datetime.now()-timedelta(days=5)),
                                                                                        end_ts=datetime.now)]))
@patch.object(DBRunSum, 'set_header', new=MagicMock(return_value=RunSumHeader(id=1)))
@patch.object(DBRunSum, 'upload', new=MagicMock())
def test_run_sum_upload_script():
    upload_run_sum(number_of_batches=3)


def test_set_ann():
    data = Data()

    ann = data.db.ann.set(layers=TEST_MODEL, offsets=TEST_OFFSETS, indicators=TEST_INDICATORS, scaled=True)
    data.db.flush()
    got = data.db.ann.get(ann_id=ann.id)

    assert got.is_scaled
    assert got.layers == json.dumps(TEST_MODEL)
    assert got.offsets == json.dumps(TEST_OFFSETS)
    assert got.indicators == json.dumps(TEST_INDICATORS)

    data.db.rollback()


@pytest.mark.asyncio
async def test_context_manager():
    async with DB() as db:
        db.rollback()

    with pytest.raises(Exception):
        with DB() as db:
            raise Exception()


def test_additional_currency():
    data = Data()
    currency = data.db.currency.set(code='TST', name='Test')
    attr = data.db.individual.attribute.get(currency_code=currency.code,
                                            version_id=VERSION_NUM,
                                            md5_or_id=data.bot.md5())
    assert attr


def test_action_set_get():
    data = Data()
    assert not data.db.action.get(DEFAULT_CURRENCY)

    limits = {(1, 2, 3, 4): 0.7, (0, 0, 0, 0): 0.3}
    for _ in range(2):
        data.db.action.set(
            limits=limits,
            currency_code=DEFAULT_CURRENCY,
            ts=datetime(day=1, month=10, year=2022, hour=00, minute=00, tzinfo=timezone.utc),
            close=123.0,
        )

    found, close = data.db.action.get(DEFAULT_CURRENCY).get_limits()
    assert found == limits
    assert close == 123.0


def test_add_notifications():
    with DB() as db:
        dt = datetime(day=1, month=10, year=2022, hour=00, minute=00, tzinfo=timezone.utc)
        obj = Price(currency='btc', close=1, low=1, high=1, volume=1, ts=dt)

        received = None

        def callback(x: Price):
            nonlocal received
            received = x
            raise Exception("test")

        db.subscribe(Price, callback)
        db.add(obj)
        db.commit()

        spent = 0
        while received is None:
            assert spent < 10
            asyncio.get_event_loop().run_until_complete(asyncio.sleep(1))
            spent += 1

        db.session.delete(obj)
        db.commit()
