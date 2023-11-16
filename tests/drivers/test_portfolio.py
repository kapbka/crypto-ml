from datetime import datetime, timedelta, timezone

import numpy as np
from asynctest.mock import MagicMock

from common.metrics.evaluator import profit, portfolio
from common.portfolio import process_historical_portfolios, deals_to_trades, calculate_shares, run_portfolio_detailed
from db.model.processing import Portfolio, Deal


def ts(v: int):
    return datetime(2022, 1, 10, tzinfo=timezone.utc) + timedelta(minutes=v)


def test_deals_to_trades():
    assert np.allclose(deals_to_trades(
        [
            Deal(buy_price=100, buy_ts=ts(1), sell_price=105, sell_ts=ts(3), status=1, individual=1),
            Deal(buy_price=100, buy_ts=ts(5), sell_price=110, sell_ts=ts(9), status=1, individual=1),
        ]), [np.array([1.00000000e+00, 1.00000000e+00, 1.00000000e+02, 1.64177286e+15]),
             np.array([1.00000000e+00, -1.00000000e+00, 1.05000000e+02, 1.64177298e+15]),
             np.array([1.0000000e+00, 1.0000000e+00, 1.0000000e+02, 1.6417731e+15]),
             np.array([1.00000000e+00, -1.00000000e+00, 1.10000000e+02, 1.64177334e+15])])
    assert not deals_to_trades(
        [
            Deal(buy_price=100, buy_ts=ts(1), status=0, individual=1),
            Deal(buy_price=100, buy_ts=ts(5), sell_price=110, sell_ts=ts(9), status=1, individual=1),
        ])
    assert not deals_to_trades(
        [
            Deal(buy_price=100, buy_ts=ts(1), sell_price=105, sell_ts=ts(3), status=1, individual=1),
            Deal(buy_price=100, buy_ts=ts(2), sell_price=110, sell_ts=ts(9), status=1, individual=1),
        ])


def test_portfolio():
    db = MagicMock()
    db.portfolio.get_batch.return_value = [Portfolio(portfolio_md5='parent', md5='bot1'),
                                           Portfolio(portfolio_md5='parent', md5='bot2')]

    # https://photos.app.goo.gl/KsggbE9xDdJTMgQ47
    db.deal.get_all.side_effect = [
        [
            Deal(buy_price=100, buy_ts=ts(1), sell_price=105, sell_ts=ts(3), status=1, individual=1),
            Deal(buy_price=100, buy_ts=ts(5), sell_price=110, sell_ts=ts(9), status=1, individual=1),
        ],
        [
            Deal(buy_price=105, buy_ts=ts(2), sell_price=110, sell_ts=ts(4), status=1, individual=2),
            Deal(buy_price=115, buy_ts=ts(6), sell_price=120, sell_ts=ts(8), status=1, individual=2),
        ]
    ]

    assert process_historical_portfolios(db, version=1, currency='btc', ts_from=ts(0)) == 1

    deals = list(db.deal.set_batch.call_args.kwargs['deals'])
    running_sum = profit(deals)
    assert np.isclose(running_sum, 23.1065)

    usd = deals[0].buy_price + deals[0].buy_price * deals[-1].run_percent / 100
    assert np.isclose(usd, 123.13124126)
    assert len(deals) == 2

    assert deals[0].buy_ts == ts(1)
    assert deals[0].sell_ts == ts(4)

    assert deals[1].buy_ts == ts(5)
    assert deals[1].sell_ts == ts(9)


def test_portfolio_zero_shares():
    data = np.array(
        deals_to_trades([
            Deal(buy_price=100, buy_ts=ts(1), sell_price=105, sell_ts=ts(3), status=1, individual=1),
            Deal(buy_price=105, buy_ts=ts(4), sell_price=110, sell_ts=ts(5), status=1, individual=2),
        ])
    )

    data = data[data[:, 3].argsort(kind='stable')]
    _, _, shares = portfolio(events=data, total_shares=2, generate_deals=False)
    assert shares == []


def test_portfolio_1_share():
    data = np.concatenate([
        deals_to_trades([Deal(buy_price=100, buy_ts=ts(1), sell_price=105, sell_ts=ts(3), status=1, individual=1)]),
        deals_to_trades([Deal(buy_price=105, buy_ts=ts(2), status=0, individual=2)])
    ])

    data = data[data[:, 3].argsort(kind='stable')]
    _, _, shares = portfolio(events=data, total_shares=2, generate_deals=False)
    assert shares == [1 / 2]


def test_portfolio_2_shares():
    data = np.concatenate([
        deals_to_trades([Deal(buy_price=100, buy_ts=ts(1), status=0, individual=1)]),
        deals_to_trades([Deal(buy_price=105, buy_ts=ts(2), status=0, individual=2)])
    ])

    data = data[data[:, 3].argsort(kind='stable')]
    _, _, shares = portfolio(events=data, total_shares=2, generate_deals=False)
    assert shares == [1 /
                      2, 1 / 2]


def test_portfolio_3_shares():
    data = np.concatenate([
        deals_to_trades([Deal(buy_price=100, buy_ts=ts(1), status=0, individual=1)]),
        deals_to_trades([Deal(buy_price=105, buy_ts=ts(2), status=0, individual=2)]),
        deals_to_trades([Deal(buy_price=105, buy_ts=ts(3), status=0, individual=3)]),
    ])

    data = data[data[:, 3].argsort(kind='stable')]
    _, _, shares = portfolio(events=data, total_shares=3, generate_deals=False)
    assert shares == [0.33333333333333337, 0.4444444444444444, 0.2222222222222222]


def test_portfolio_4_shares():
    data = np.concatenate([
        deals_to_trades([Deal(buy_price=100, buy_ts=ts(1), status=0, individual=1)]),
        deals_to_trades([Deal(buy_price=105, buy_ts=ts(2), status=0, individual=2)]),
        deals_to_trades([Deal(buy_price=105, buy_ts=ts(3), status=0, individual=3)]),
        deals_to_trades([Deal(buy_price=105, buy_ts=ts(3), status=0, individual=4)]),
    ])

    data = data[data[:, 3].argsort(kind='stable')]
    _, _, shares = portfolio(events=data, total_shares=4, generate_deals=False)
    assert shares == [0.25, 0.375, 0.28125, 0.09375]


def test_calculate_shares():
    # no money in market
    assert calculate_shares(limits=[(0, 0, 0, 0),
                                    (0, 0, 0, 0),
                                    (0, 0, 0, 0),
                                    (0, 0, 0, 0)],
                            shares=[]) == {}

    # one bot buys and gets all money
    assert calculate_shares(limits=[(0, 0, 200, 100),
                                    (0, 0, 0, 0),
                                    (0, 0, 0, 0),
                                    (0, 0, 0, 0)],
                            shares=[1]) == {(0, 0, 200, 100): 1.}

    # second bot buys with different limits
    assert calculate_shares(limits=[(0, 0, 200, 100),
                                    (0, 0, 210, 110),
                                    (0, 0, 0, 0),
                                    (0, 0, 0, 0)],
                            shares=[3 / 4, 1 / 4]) == {(0, 0, 200, 110): 3 / 4,
                                                       (0, 0, 210, 100): 1 / 4}

    # third bot buys with different limits
    assert calculate_shares(limits=[(0, 0, 200, 100),
                                    (0, 0, 210, 110),
                                    (0, 0, 220, 120),
                                    (0, 0, 0, 0)],
                            shares=[0.4, 0.35, 0.25]) == {(0, 0, 200, 120): 0.4,
                                                          (0, 0, 210, 110): 0.35,
                                                          (0, 0, 220, 100): 0.25}


def test_run_detailed_portfolio():
    data = np.concatenate([
        deals_to_trades([Deal(buy_price=100, buy_ts=ts(1), sell_price=110, sell_ts=ts(4), status=1, individual=1)]),
        deals_to_trades([Deal(buy_price=105, buy_ts=ts(2), status=0, individual=2)]),
        deals_to_trades([Deal(buy_price=105, buy_ts=ts(3), status=0, individual=3)]),
    ])

    data = data[data[:, 3].argsort(kind='stable')]
    _, _, shares = portfolio(events=data, total_shares=3, generate_deals=False)

    money = run_portfolio_detailed(
        trades=data,
        prices=np.array([100, 105, 105, 110]),
        all_ts=np.array([int(np.datetime64(ts(i)).astype(int) * 1000) for i in range(1, 5)]),
        total_shares=3,
    )
    assert money == [(99.9, 1.0),
                     (104.89500000000001, 2.0),
                     (104.89500000000001, 3.0),
                     (109.85337000000001, 2.0)]
