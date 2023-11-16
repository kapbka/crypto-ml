import json
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest
from asynctest.mock import CoroutineMock, MagicMock
from binance.exceptions import BinanceAPIException

from common.exchanges.binance_api import Binance, BinanceSocketManager


class FakeSocket:
    def __init__(self, values=MagicMock()):
        self._values = values

    async def __aenter__(self):
        return self

    async def recv(self):
        return self._values()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


EXCHANGE_INFO = {"timezone": "UTC", "serverTime": 1638955490070, "rateLimits": [
    {"rateLimitType": "REQUEST_WEIGHT", "interval": "MINUTE", "intervalNum": 1, "limit": 1200},
    {"rateLimitType": "ORDERS", "interval": "SECOND", "intervalNum": 10, "limit": 50},
    {"rateLimitType": "ORDERS", "interval": "DAY", "intervalNum": 1, "limit": 160000}], "exchangeFilters": [],
                 "symbols": [{"symbol": "BTCUSDT", "status": "TRADING", "baseAsset": "BTC", "baseAssetPrecision": 8,
                              "quoteAsset": "USDT", "quotePrecision": 8, "quoteAssetPrecision": 8,
                              "baseCommissionPrecision": 8, "quoteCommissionPrecision": 8,
                              "orderTypes": ["LIMIT", "LIMIT_MAKER", "MARKET", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"],
                              "icebergAllowed": True, "ocoAllowed": True, "quoteOrderQtyMarketAllowed": True,
                              "isSpotTradingAllowed": True, "isMarginTradingAllowed": False, "filters": [
                         {"filterType": "PRICE_FILTER", "minPrice": "0.01000000",
                          "maxPrice": "1000000.00000000", "tickSize": "0.01000000"},
                         {"filterType": "PERCENT_PRICE", "multiplierUp": "5", "multiplierDown": "0.2",
                          "avgPriceMins": 5},
                         {"filterType": "LOT_SIZE", "minQty": "0.00000100", "maxQty": "900.00000000",
                          "stepSize": "0.00000100"},
                         {"filterType": "MIN_NOTIONAL", "minNotional": "10.00000000", "applyToMarket": True,
                          "avgPriceMins": 5}, {"filterType": "ICEBERG_PARTS", "limit": 10},
                         {"filterType": "MARKET_LOT_SIZE", "minQty": "0.00000000", "maxQty": "100.00000000",
                          "stepSize": "0.00000000"}, {"filterType": "MAX_NUM_ORDERS", "maxNumOrders": 200},
                         {"filterType": "MAX_NUM_ALGO_ORDERS", "maxNumAlgoOrders": 5}],
                              "permissions": ["SPOT"]}]}

AVG_PRICE = {'price': 100}
ORDER1 = {'orderId': 1}
ORDER2 = {'orderId': 2}
ORDER3 = {'orderId': 3}
EXCEPT = BinanceAPIException(None, 500, json.dumps({'code': '', 'msg': ''}))


@pytest.mark.asyncio
async def test_oco_orders():
    client = CoroutineMock(get_exchange_info=CoroutineMock(return_value=EXCHANGE_INFO),
                           get_avg_price=CoroutineMock(return_value=AVG_PRICE),
                           order_oco_sell=CoroutineMock(return_value={'orders': [ORDER1, ORDER2]}),
                           cancel_order=CoroutineMock(side_effect=[EXCEPT, True]),
                           close_connection=CoroutineMock(),
                           get_order=CoroutineMock())
    with patch('binance.AsyncClient.create', new=CoroutineMock(return_value=client)):
        async with Binance(ticker="BTCUSDT") as api:
            order_id_sell = await api.sell_oco(amount=0.1, lower=api.average_price * 0.9, upper=api.average_price * 1.1)
            await api.cancel_order(order_id_sell)

            client.cancel_order.assert_called_with(symbol='BTCUSDT', orderId=ORDER1['orderId'])
            client.order_oco_sell.assert_called_with(symbol='BTCUSDT',
                                                     quantity=0.1,
                                                     price=110.0,
                                                     stopPrice=90.0,
                                                     stopLimitPrice=90.0,
                                                     stopLimitTimeInForce=client.TIME_IN_FORCE_GTC)


@pytest.mark.asyncio
async def test_orders():
    client = CoroutineMock(get_exchange_info=CoroutineMock(return_value=EXCHANGE_INFO),
                           get_avg_price=CoroutineMock(return_value=AVG_PRICE),
                           create_order=CoroutineMock(side_effect=[ORDER1, ORDER2, ORDER3]),
                           get_open_orders=CoroutineMock(return_value=[ORDER2, ORDER3]),
                           cancel_order=CoroutineMock(),
                           close_connection=CoroutineMock(),
                           get_order=CoroutineMock())
    with patch('binance.AsyncClient.create', new=CoroutineMock(return_value=client)):
        async with Binance(ticker="BTCUSDT") as api:
            order_id_buy = await api.buy(0.1)
            client.create_order.assert_called_with(symbol='BTCUSDT',
                                                   side=client.SIDE_BUY,
                                                   quoteOrderQty=0.1,
                                                   type=client.ORDER_TYPE_MARKET)

            limit_order_id_buy = await api.buy(0.1, price=api.average_price * 1.1)

            client.create_order.assert_called_with(symbol='BTCUSDT',
                                                   quantity=0.000909,
                                                   side=client.SIDE_BUY,
                                                   type=client.ORDER_TYPE_STOP_LOSS_LIMIT,
                                                   stopPrice=110.0,
                                                   price=110.0,
                                                   timeInForce=client.TIME_IN_FORCE_GTC)

            limit_order_id_sell = await api.sell(0.1, price=api.average_price * 0.9)

            client.create_order.assert_called_with(symbol='BTCUSDT',
                                                   quantity=0.1,
                                                   side=client.SIDE_SELL,
                                                   type=client.ORDER_TYPE_STOP_LOSS_LIMIT,
                                                   stopPrice=90.0,
                                                   price=90.0,
                                                   timeInForce=client.TIME_IN_FORCE_GTC)

            orders = await api.get_orders()
            assert limit_order_id_buy in orders
            assert limit_order_id_sell in orders

            for order in orders:
                await api.get_order(order)
                await api.cancel_order(order)

            client.get_avg_price.assert_called_with(symbol="BTCUSDT")
            assert client.cancel_order.call_count == 2
            client.cancel_order.assert_called_with(orderId=ORDER3['orderId'], symbol="BTCUSDT")


@pytest.mark.asyncio
async def test_trades():
    client = CoroutineMock(get_exchange_info=CoroutineMock(return_value=EXCHANGE_INFO),
                           get_avg_price=CoroutineMock(return_value=AVG_PRICE),
                           create_order=CoroutineMock(side_effect=[ORDER1, ORDER2, ORDER3]),
                           get_open_orders=CoroutineMock(return_value=[ORDER2, ORDER3]),
                           cancel_order=CoroutineMock(),
                           close_connection=CoroutineMock(),
                           get_order=CoroutineMock(),
                           get_my_trades=CoroutineMock(return_value=[MagicMock]))
    with patch('binance.AsyncClient.create', new=CoroutineMock(return_value=client)):
        async with Binance(ticker="BTCUSDT") as api:
            trades = await api.get_trades()
            assert trades
            client.get_my_trades.assert_called_once()


@pytest.mark.asyncio
async def test_balance():
    client = CoroutineMock(get_exchange_info=CoroutineMock(return_value=EXCHANGE_INFO),
                           get_avg_price=CoroutineMock(return_value=AVG_PRICE),
                           create_order=CoroutineMock(side_effect=[ORDER1, ORDER2, ORDER3]),
                           get_open_orders=CoroutineMock(return_value=[ORDER2, ORDER3]),
                           cancel_order=CoroutineMock(),
                           close_connection=CoroutineMock(),
                           get_order=CoroutineMock(),
                           get_asset_balance=CoroutineMock(return_value=dict(free=1, locked=2)))
    with patch('binance.AsyncClient.create', new=CoroutineMock(return_value=client)):
        async with Binance(ticker="BTCUSDT") as api:
            btc, usdt = await api.get_balance()
            assert btc == 3
            assert usdt == 3


@pytest.mark.asyncio
async def test_history():
    client = CoroutineMock(get_exchange_info=CoroutineMock(return_value=EXCHANGE_INFO),
                           get_avg_price=CoroutineMock(return_value=AVG_PRICE),
                           create_order=CoroutineMock(side_effect=[ORDER1, ORDER2, ORDER3]),
                           get_open_orders=CoroutineMock(return_value=[ORDER2, ORDER3]),
                           cancel_order=CoroutineMock(),
                           close_connection=CoroutineMock(),
                           get_order=CoroutineMock(),
                           get_historical_klines=CoroutineMock(return_value=[
                               [
                                   1499040000000,  # Open time
                                   "0.01634790",  # Open
                                   "0.80000000",  # High
                                   "0.01575800",  # Low
                                   "0.01577100",  # Close
                                   "148976.11427815",  # Volume
                                   1499644799999,  # Close time
                                   "2434.19055334",  # Quote asset volume
                                   308,  # Number of trades
                                   "1756.87402397",  # Taker buy base asset volume
                                   "28.46694368",  # Taker buy quote asset volume
                                   "17928899.62484339"  # Ignore.
                               ]
                           ]))
    with patch('binance.AsyncClient.create', new=CoroutineMock(return_value=client)):
        async with Binance(ticker="BTCUSDT") as api:
            start_ts = datetime.now()
            df = await api.history(start_ts - timedelta(minutes=5), to_dt=None)
            assert len(df) > 0
            assert (df['low'] <= df['close']).all()
            assert (df['high'] >= df['close']).all()
            assert (df['volume'] > 0).all()


@pytest.mark.asyncio
async def test_klines():
    client = CoroutineMock(get_exchange_info=CoroutineMock(return_value=EXCHANGE_INFO),
                           get_avg_price=CoroutineMock(return_value=AVG_PRICE),
                           create_order=CoroutineMock(side_effect=[ORDER1, ORDER2, ORDER3]),
                           get_open_orders=CoroutineMock(return_value=[ORDER2, ORDER3]),
                           cancel_order=CoroutineMock(),
                           close_connection=CoroutineMock(),
                           get_order=CoroutineMock(),
                           get_klines=CoroutineMock(return_value=[
                               [
                                   1499040000000,  # Open time
                                   "0.01634790",  # Open
                                   "0.80000000",  # High
                                   "0.01575800",  # Low
                                   "0.01577100",  # Close
                                   "148976.11427815",  # Volume
                                   1499644799999,  # Close time
                                   "2434.19055334",  # Quote asset volume
                                   308,  # Number of trades
                                   "1756.87402397",  # Taker buy base asset volume
                                   "28.46694368",  # Taker buy quote asset volume
                                   "17928899.62484339"  # Ignore.
                               ]
                           ]))
    with patch('binance.AsyncClient.create', new=CoroutineMock(return_value=client)):
        async with Binance(ticker="BTCUSDT") as api:
            df = await api.klines(limit=1)
            assert len(df) > 0
            assert (df['low'] <= df['close']).all()
            assert (df['high'] >= df['close']).all()
            assert (df['volume'] > 0).all()


@pytest.mark.asyncio
@patch.object(BinanceSocketManager, 'kline_socket', new=MagicMock(return_value=FakeSocket()))
async def test_subscription():
    client = CoroutineMock(get_exchange_info=CoroutineMock(return_value=EXCHANGE_INFO),
                           get_avg_price=CoroutineMock(return_value=AVG_PRICE),
                           create_order=CoroutineMock(side_effect=[ORDER1, ORDER2, ORDER3]),
                           get_open_orders=CoroutineMock(return_value=[ORDER2, ORDER3]),
                           cancel_order=CoroutineMock(),
                           close_connection=CoroutineMock())
    with patch('binance.AsyncClient.create', new=CoroutineMock(return_value=client)):
        async with Binance(ticker="BTCUSDT") as api:
            async for final, data in api.subscribe():
                for col in ['low', 'high', 'close', 'ts', 'volume']:
                    assert col in data
                break


@pytest.mark.asyncio
@patch.object(BinanceSocketManager, 'trade_socket', new=MagicMock(return_value=FakeSocket()))
async def test_trade_subscription():
    client = CoroutineMock(get_exchange_info=CoroutineMock(return_value=EXCHANGE_INFO),
                           get_avg_price=CoroutineMock(return_value=AVG_PRICE),
                           create_order=CoroutineMock(side_effect=[ORDER1, ORDER2, ORDER3]),
                           get_open_orders=CoroutineMock(return_value=[ORDER2, ORDER3]),
                           cancel_order=CoroutineMock(),
                           close_connection=CoroutineMock())
    with patch('binance.AsyncClient.create', new=CoroutineMock(return_value=client)):
        async with Binance(ticker="BTCUSDT") as api:
            mock = CoroutineMock(return_value=False)
            await api.monitor_trades(callback=mock)
            mock.assert_called()


@pytest.mark.asyncio
async def test_order_subscription():
    client = CoroutineMock(get_exchange_info=CoroutineMock(return_value=EXCHANGE_INFO),
                           get_avg_price=CoroutineMock(return_value=AVG_PRICE),
                           create_order=CoroutineMock(side_effect=[ORDER1, ORDER2, ORDER3]),
                           get_open_orders=CoroutineMock(return_value=[ORDER2, ORDER3]),
                           cancel_order=CoroutineMock(),
                           close_connection=CoroutineMock())
    values = MagicMock(side_effect=[
        {"e": "executionReport", "E": 1659957840751, "s": "BTCUSDT", "c": "LsT4spgtok6YLLMEqSbeVc", "S": "BUY",
         "o": "MARKET", "f": "GTC", "q": "0.41532200", "p": "0.00000000", "P": "0.00000000", "F": "0.00000000", "g": -1,
         "C": "", "x": "NEW", "X": "NEW", "r": "NONE", "i": 1681273, "l": "0.00000000", "z": "0.00000000",
         "L": "0.00000000", "n": "0", "N": None, "T": 1659957840751, "t": -1, "I": 3821353, "w": True, "m": False,
         "M": False, "O": 1659957840751, "Z": "0.00000000", "Y": "0.00000000", "Q": "10000.00000000"},
        {"e": "executionReport", "E": 1659957840751, "s": "BTCUSDT", "c": "LsT4spgtok6YLLMEqSbeVc", "S": "BUY",
         "o": "MARKET", "f": "GTC", "q": "0.41532200", "p": "0.00000000", "P": "0.00000000", "F": "0.00000000", "g": -1,
         "C": "", "x": "TRADE", "X": "FILLED", "r": "NONE", "i": 1681273, "l": "0.01452800", "z": "0.41532200",
         "L": "24080.34000000", "n": "0.00000000", "N": "BTC", "T": 1659957840751, "t": 468530, "I": 3821368,
         "w": False, "m": False, "M": True, "O": 1659957840751, "Z": "9999.98132985", "Y": "349.83917952",
         "Q": "10000.00000000"},
        {"e": "outboundAccountPosition", "E": 1659957840751, "u": 1659957840751,
         "B": [{"a": "BTC", "f": "0.73487500", "l": "0.00000000"},
               {"a": "USDT", "f": "13940.97680782", "l": "0.00000000"}]},
        {"e": "outboundAccountPosition", "E": 1660135983374, "u": 1660135983373,
         "B": [{"a": "BTC", "f": "0.00000850", "l": "0.02369000"}, {"a": "BNB", "f": "0.00000000", "l": "0.00000000"},
               {"a": "USDT", "f": "0.02038840", "l": "0.00000000"}]},
        {"e": "error", "msg": "test"}
    ])

    with patch('binance.AsyncClient.create', new=CoroutineMock(return_value=client)), \
        patch.object(BinanceSocketManager, 'user_socket', new=MagicMock(return_value=FakeSocket(values))):
        async with Binance(ticker="BTCUSDT") as api:
            results = list()
            async for data in api.subscribe_to_user_updates():
                results.append(data)

            assert [{'id': 1681273, 'price': 24080.34000000, 'qty': 10000.00000000, 'side': 'BUY', 'status': 'FILLED'},
                    (0.734875, 13940.97680782), (0.0236985, 0.0203884)] == results
