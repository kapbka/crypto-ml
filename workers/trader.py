import argparse
import asyncio
import datetime
import logging
import sys
from typing import Tuple, List, Dict

from common.constants import DEFAULT_CURRENCY
from common.exchanges.binance_api import Binance
from common.interrupt import run_main
from common.k8s.probes import mark_ready
from common.log import setup_logging
from common.metrics.evaluator import Limits
from common.reporting import send_message
from db.api.action import Action
from db.api.db import DB

MINIMUM_SHARE_DIFFERENCE = 0.01


class Trader:
    def __init__(self, api: Binance):
        self._api = api
        self._currency = 0.0
        self._usd = 0.0
        self._reported_currency = 0.0
        self._reported_usd = 0.0
        self._active_limit_orders: List[int] = list()
        self._active_limit_order_prices: List[Tuple[Limits, float]] = list()
        self._updater = None
        self._running = True

    async def __aenter__(self):
        # first we need to sync current state
        # to do that cleanly cancel all active orders
        await self._cancel_all()
        await self._update_status(self._api.average_price)
        self._updater = asyncio.create_task(self.process_executed_orders())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._running = False
        await self._updater
        await self._cancel_all()

    async def get_share(self, price: float) -> float:
        """
        Gets balance from exchange and determines current share in currency compared to total money
        :param price: current price
        :return: currency shares, value from 0 to 1
        """
        potentially_in_currency = self._currency + self._usd / price
        return self._currency / potentially_in_currency

    async def process_executed_orders(self):
        while self._running:
            try:
                async for update in self._api.subscribe_to_user_updates():
                    logging.info(f"User update: {update}")

                    if isinstance(update, dict):
                        # order execution/cancellation
                        order_id = update['id']
                        if order_id in self._active_limit_orders:
                            idx = self._active_limit_orders.index(order_id)
                            self._active_limit_orders[idx] = 0
                            self._active_limit_order_prices[idx] = (0, 0, 0, 0), 0

                        if update['status'] in ['FILLED', 'EXPIRED']:
                            send_message(f"Order {update['status']}: {update}")
                    else:
                        # balance update
                        currency, usd = update
                        await self._update_balance(price=self._api.average_price, currency=currency, usd=usd)

                    await asyncio.sleep(0.01)
            except:
                logging.exception(f"Error in order subscription API")

    async def _cancel_all(self):
        orders = await self._api.get_orders()
        await asyncio.gather(*list(map(self._api.cancel_order, orders)))
        self._active_limit_orders.clear()

    async def _update_status(self, price: float):
        currency, usd = await self._api.get_balance()
        await self._update_balance(price=price, currency=currency, usd=usd)

    async def _update_balance(self, price: float, currency: float, usd: float):
        if (self._currency, self._usd) != (currency, usd):
            if price and (currency, usd) != (self._reported_currency, self._reported_usd):
                total = usd + currency * price
                send_message(f"Balance has changed, currency: {self._reported_currency:.5f} -> {currency:.5f}, "
                             f"usd: {self._reported_usd:.1f} -> {usd:.1f}$, total: {total:.1f}$")
                self._reported_currency, self._reported_usd = currency, usd
            self._currency, self._usd = currency, usd

    async def sync_state(self, share: float, price: float):
        """
        This function synchronizes current money in market with expected share of currency
        @param share: total proportion of currency to be bought, values from 0 to 1
        @param price: current market price of the currency
        @return:
        """
        actual_currency_shares = await self.get_share(price=price)
        total_in_currency = self._currency + self._usd / price

        order_id = 0
        if share - actual_currency_shares > MINIMUM_SHARE_DIFFERENCE:
            # buy currency
            shares_to_buy = share - actual_currency_shares
            to_buy = shares_to_buy * total_in_currency * price

            # adjust amount if this is the last share
            to_buy = self._usd if share == 1 else to_buy

            logging.info(f"Buying with market price {shares_to_buy:.2f} shares, to spend: {to_buy:.2f}$")
            await self._cancel_all()
            order_id = await self._api.buy(amount=to_buy)
        elif actual_currency_shares - share > MINIMUM_SHARE_DIFFERENCE:
            # sell currency
            shares_to_sell = actual_currency_shares - share
            to_sell = shares_to_sell * total_in_currency

            # adjust amount if this is the last share
            to_sell = self._currency if share == 0 else to_sell

            logging.info(f"Selling with market price {shares_to_sell:.2f} shares, to spend: {to_sell:.4f} currency")
            await self._cancel_all()
            order_id = await self._api.sell(amount=to_sell)

        started = datetime.datetime.now()
        while order_id:
            order = await self._api.get_order(order_id)
            if order['status'] in ['FILLED', 'CANCELED', 'EXPIRED']:
                send_message(f"Market order {order['status']}: {order}")
                break

            await asyncio.sleep(1)
            assert datetime.datetime.now() - started < datetime.timedelta(minutes=1)

    async def sync_limit_orders(self, limit_prices: Dict[Limits, float], price: float):
        """
        This function raises OCO limit orders if they're not matching intended
        @param limit_prices: limit prices mapped to share amount
        @param price: current market price
        @return:
        """
        if self._active_limit_order_prices == list(limit_prices.items()):
            return

        logging.info(f"Re-syncing orders {self._active_limit_order_prices} != {limit_prices}")

        # cancel all
        await self._cancel_all()
        await self._update_status(price)
        self._active_limit_orders.clear()
        self._active_limit_order_prices.clear()

        total_in_currency = self._currency + self._usd / price

        for (buy_high, buy_low, sell_high, sell_low), share in limit_prices.items():
            # sell this much currency with these limits
            to_sell = share * total_in_currency

            to_sell = min(to_sell, self._currency)
            self._currency -= to_sell
            logging.info(f"Raising limit order, to sell: {to_sell}, total: {total_in_currency}, left: {self._currency}")

            order_id = await self._api.sell_oco(amount=to_sell, lower=sell_low, upper=sell_high)

            self._active_limit_orders.append(order_id)

        self._active_limit_order_prices = list(limit_prices.items())

    async def handle(self, limit_prices: Dict[Limits, float], price: float):
        logging.info(f"Handling: {limit_prices}")

        await self.sync_state(share=sum(limit_prices.values()), price=price)
        await self.sync_limit_orders(limit_prices=limit_prices, price=price)

        logging.info(f"Handled: {limit_prices}")


async def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="This script runs a worker that places orders on Binance exchange")
    parser.add_argument("-v", "--verbosity", type=str, default='INFO', help="verbosity level: INFO/DEBUG")
    parser.add_argument("--currency", type=str, default=DEFAULT_CURRENCY,
                        help=f"currency to use, default: {DEFAULT_CURRENCY}")

    args = parser.parse_args(args)
    setup_logging(args.verbosity, filename='/tmp/trader.log')

    async with DB() as db, \
            Binance(ticker=f"{args.currency.upper()}USDT") as exchange, \
            Trader(api=exchange) as trader:

        queue = asyncio.Queue()
        queue.put_nowait((db.action.get(currency_code=args.currency), False))

        db.subscribe(Action, lambda x: queue.put_nowait((x, True)))

        while True:
            item, is_subscription = await queue.get()
            if item:
                await trader.handle(*item.get_limits())

            if item and is_subscription:
                mark_ready()


if __name__ == '__main__':
    run_main(main())
