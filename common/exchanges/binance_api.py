import asyncio
import logging
import math
import os
from datetime import datetime, timezone
from typing import Optional, Dict, Tuple, AsyncGenerator, Callable, Awaitable, Union

import pandas as pd
from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException
from binance.helpers import round_step_size

from common.constants import PRICE_INTERVAL_MINUTES
from common.retry import retry

PRECISION = 5

# Currency and USDT
Balance = Tuple[float, float]


class Binance:
    OPEN_TIME_IDX = 0
    HIGH_IDX = 2
    LOW_IDX = 3
    CLOSE_IDX = 4
    VOLUME_IDX = 5
    CLOSE_TIME_IDX = 6

    def __init__(self, ticker: str):
        """
        :param ticker: ticker name, example: BTCUSDT
        :return:
        """
        self._client = None
        self._ticker = ticker
        self.average_price = 0.0

    async def __aenter__(self):
        self._client = await AsyncClient.create(api_key=os.environ['BINANCE_API_KEY'],
                                                api_secret=os.environ['BINANCE_SECRET_KEY'],
                                                testnet=int(os.environ['BINANCE_TESTNET_ENABLED']) > 0)

        info = await self._client.get_exchange_info()
        info = next(filter(lambda x: x['symbol'] == self._ticker, info['symbols']))

        self.price_step = float(next(filter(lambda x: x['filterType'] == 'PRICE_FILTER', info['filters']))['tickSize'])
        self.qty_step = float(next(filter(lambda x: x['filterType'] == 'LOT_SIZE', info['filters']))['stepSize'])

        self.average_price = float((await self._client.get_avg_price(symbol=self._ticker))['price'])
        logging.info(f"Connected to Binance, test mode: {self._client.testnet}, average price: {self.average_price}, "
                     f"price step: {self.price_step}, quantity step: {self.qty_step}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._client.close_connection()

    def _klines_to_pandas(self, klines: list) -> pd.DataFrame:
        df = pd.DataFrame({
            'low': [float(line[self.LOW_IDX]) for line in klines],
            'high': [float(line[self.HIGH_IDX]) for line in klines],
            'close': [float(line[self.CLOSE_IDX]) for line in klines],
            'volume': [float(line[self.VOLUME_IDX]) for line in klines],
            'ts': [datetime.fromtimestamp((line[self.CLOSE_TIME_IDX] + 1) / 1000.0, timezone.utc)
                   for line in klines]
        })
        return df.drop_duplicates('ts').set_index('ts')

    async def history(self, from_dt: datetime, to_dt: Optional[datetime]):
        """
        :param from_dt: datetime from to get prices
        :param to_dt: datetime to end or None
        :return:
        """
        logging.info(f"Downloading {self._ticker}, from_dt {from_dt}, to_dt: {to_dt}")

        interval = f'{PRICE_INTERVAL_MINUTES}m'
        klines = await self._client.get_historical_klines(symbol=self._ticker,
                                                          interval=interval,
                                                          start_str=from_dt.isoformat(),
                                                          end_str=to_dt.isoformat() if to_dt else None)

        return self._klines_to_pandas(klines)

    async def subscribe(self) -> AsyncGenerator[Tuple[bool, Dict[str, float]], None]:
        """
        Subscribes to kline updates
        :return: None
        """
        bsm = BinanceSocketManager(self._client)
        async with bsm.kline_socket(self._ticker) as ts:
            while True:
                data = await ts.recv()
                if data['e'] == 'error':
                    break

                kl = data['k']
                yield kl['x'], {
                    'close': float(kl['c']),
                    'high': float(kl['h']),
                    'low': float(kl['l']),
                    'volume': float(kl['v']),
                    'ts': datetime.fromtimestamp((kl['T'] + 1) / 1000.0, timezone.utc)
                }

    async def monitor_trades(self, callback: Callable[[int], Awaitable[bool]]) -> None:
        """
        :param callback: callback that will handle executed order ids, if False is returned this method exits
        :return: None
        """
        while True:
            try:
                bsm = BinanceSocketManager(self._client)
                async with bsm.trade_socket(self._ticker) as ts:
                    while True:
                        data = await ts.recv()
                        if data['e'] == 'error':
                            raise Exception(data['e'])

                        if not await callback(data['b']) or not await callback(data['a']):
                            return
            except Exception:
                logging.exception(f"Error in trade monitor")

    async def klines(self, limit: int = 1):
        """
        :param limit: number of last lines
        :return:
        """
        start = datetime.now(tz=timezone.utc)
        klines = await self._client.get_klines(symbol=self._ticker,
                                               interval=f'{PRICE_INTERVAL_MINUTES}m',
                                               limit=limit + 1)
        return self._klines_to_pandas(klines)[:start]

    @retry((asyncio.TimeoutError,))
    async def get_orders(self):
        return [o['orderId'] for o in await self._client.get_open_orders()]

    @retry((asyncio.TimeoutError,))
    async def get_order(self, order_id: int):
        return await self._client.get_order(symbol=self._ticker, orderId=order_id)

    @retry((asyncio.TimeoutError,))
    async def cancel_order(self, order_id: int):
        try:
            res = await self._client.cancel_order(symbol=self._ticker, orderId=order_id)
            logging.info(f"Cancelled {order_id}, res: {res}")
        except BinanceAPIException:
            logging.error(f"Unable to cancel order: {order_id}")

    def _adjust_amount_and_price(self, amount: float, price: float) -> Tuple[float, float]:
        amount_orig = amount

        amount_precision: int = int(round(-math.log(self.qty_step, 10), 0))
        factor = 10 ** amount_precision
        amount = math.floor(amount * factor) / factor

        assert amount <= amount_orig
        return amount, round_step_size(price, self.price_step)

    async def sell_oco(self, amount: float, lower: float, upper: float) -> int:
        amount, lower = self._adjust_amount_and_price(amount=amount, price=lower)
        _, upper = self._adjust_amount_and_price(amount=amount, price=upper)
        params = {'symbol': self._ticker,
                  'quantity': amount,
                  'price': upper,
                  'stopPrice': lower,
                  'stopLimitPrice': lower,
                  'stopLimitTimeInForce': self._client.TIME_IN_FORCE_GTC}

        logging.info(f"Placing order: {params}")
        result = await self._client.order_oco_sell(**params)
        return result['orders'][0]['orderId']

    async def _make_order(self, amount: float, price: float, side: str):
        if price:
            if side == self._client.SIDE_BUY:
                amount = amount / price

            amount, price = self._adjust_amount_and_price(amount=amount, price=price)
            params = {'symbol': self._ticker,
                      'quantity': amount,
                      'side': side,
                      'type': self._client.ORDER_TYPE_STOP_LOSS_LIMIT,
                      'stopPrice': price,
                      'price': price,
                      'timeInForce': self._client.TIME_IN_FORCE_GTC}
        else:
            amount, _ = self._adjust_amount_and_price(amount=amount, price=price)
            qty_name = 'quantity' if side == self._client.SIDE_SELL else 'quoteOrderQty'
            params = {'symbol': self._ticker,
                      'side': side,
                      qty_name: amount,
                      'type': self._client.ORDER_TYPE_MARKET}

        logging.info(f"Placing order: {params}")
        order = await self._client.create_order(**params)
        return order['orderId']

    async def buy(self, amount: float, price: float = 0) -> int:
        return await self._make_order(amount=amount, price=price, side=self._client.SIDE_BUY)

    async def sell(self, amount: float, price: float = 0) -> int:
        return await self._make_order(amount=amount, price=price, side=self._client.SIDE_SELL)

    @retry((asyncio.TimeoutError,))
    async def get_balance(self) -> Balance:
        currency = self._ticker.replace('USDT', '')
        currency = await self._client.get_asset_balance(asset=currency)
        usd = await self._client.get_asset_balance(asset='USDT')
        return float(currency['free']) + float(currency['locked']), float(usd['free']) + float(usd['locked'])

    @retry((asyncio.TimeoutError, ))
    async def get_trades(self):
        return await self._client.get_my_trades(symbol=self._ticker)

    async def subscribe_to_user_updates(self) -> AsyncGenerator[Union[Balance, Dict[str, float]], None]:
        """
        Subscribes to order execution updates
        :return: yields either balance update or order executions/cancellations
        """
        currency_name = self._ticker.replace('USDT', '')
        bsm = BinanceSocketManager(self._client)
        async with bsm.user_socket() as ts:
            while True:
                data = await ts.recv()
                if data['e'] == 'error':
                    logging.error(data)
                    break

                logging.info(f"User update: {data}")
                if data['e'] == 'outboundAccountPosition':
                    # balance update
                    currency = next(filter(lambda x: x['a'] == currency_name, data['B']))
                    usd = next(filter(lambda x: x['a'] == "USDT", data['B']))
                    yield float(currency['f']) + float(currency['l']), float(usd['f']) + float(usd['l'])
                elif data['e'] == 'executionReport' and data.get('X') in ['FILLED', 'CANCELED']:
                    # trade execution
                    yield {'id': data['i'],
                           'side': data['S'],
                           'qty': float(data['Q']),
                           'price': float(data['L']),
                           'status': data['X']}
