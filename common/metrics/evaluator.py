from typing import Tuple, List, Optional

import numpy as np
from numba import njit
from pandas import to_datetime

from common.constants import DEAL_COEFFICIENT, COMMISSION, EvaluationMethod, DealStatus
from db.model.processing import Deal as DBDeal

# fitness, clean profit, deal count
Fitness = Tuple[float, float, float]

# buy high, buy low, sell high, sell low
Limits = Tuple[float, float, float, float]

# buy_ts, buy_price, sell_ts, sell_price, run_usd, run_crypto, run_percent
Deal = Tuple[int, float, int, float, float, float, float]
Result = Tuple[Fitness, Limits, List[Deal]]


# set env NUMBA_DISABLE_JIT=1 if you want to debug
@njit
def is_crash(prices: np.ndarray, limit: float):
    idx = len(prices) - 1
    while idx > 0:
        profit = (prices[idx] - prices[-1]) / prices[-1]
        if abs(profit) > limit:
            return profit > 0
        idx -= 1

    return False


def convert_deal(ts: Optional[np.ndarray], bot_id: int, evaluator_deal: Deal):
    def none_if_0(x): return None if x == 0 else x
    def to_dt(arg, **kwargs): return to_datetime(arg, utc=True, **kwargs)

    buy_ts, buy_price, sell_ts, sell_price, run_usd, run_crypto, run_percent = evaluator_deal
    sell_ts_res = None
    usd = None
    if sell_price:
        usd = run_usd
        sell_ts_res = to_dt(ts[sell_ts]) if ts is not None else to_dt(sell_ts, unit='us')

    return DBDeal(buy_ts=to_dt(ts[buy_ts]) if ts is not None else to_dt(buy_ts, unit='us'),
                  buy_price=buy_price,
                  sell_ts=sell_ts_res,
                  status=DealStatus.Close.value if sell_price else DealStatus.Open.value,
                  sell_price=none_if_0(sell_price),
                  run_usd=usd,
                  run_crypto=run_crypto,
                  run_percent=none_if_0(run_percent),
                  is_realtime=0,
                  individual=bot_id)


def profit(deals: List[DBDeal]) -> float:
    usd = deals[0].buy_price

    for deal in deals:
        crypto = (usd * 0.999) / deal.buy_price
        usd = 0
        if deal.status == DealStatus.Close:
            usd = crypto * 0.999 * deal.sell_price
            crypto = 0

    return ((usd + crypto * (deal.sell_price or deal.buy_price)) - deals[0].buy_price) * 100 / deals[0].buy_price


@njit
def evaluate(prices: np.ndarray, predictions: np.ndarray, money: np.ndarray, method: EvaluationMethod,
             current_usd: float, start_usd: float, buy_limit_cap: float = None, sell_limit_cap: float = None,
             use_max_sell_cap: bool = None, oco_buy_percent: float = None, oco_sell_percent: float = None,
             oco_rise_percent: float = None) -> Result:
    assert len(prices) == len(predictions)

    buy_limit_cap = buy_limit_cap or 0.
    sell_limit_cap = sell_limit_cap or 0.
    use_max_sell_cap = use_max_sell_cap or 0
    oco_buy_percent = oco_buy_percent or 0.02
    oco_sell_percent = oco_sell_percent or -0.02
    if oco_rise_percent is None:
        oco_rise_percent = 0.015

    # state variables
    crypto = 0.0
    usd = current_usd or prices[0][0]
    start_usd = start_usd or prices[0][0]
    max_price = 0.0
    deals: List[Deal] = []

    # this is to track crashes and profit
    last_profit_and_time = 0, 0
    buy_price = 0.0

    # buy high, buy low, sell high, sell low
    next_limits = np.zeros(4)

    def handle_buy(price_buy: float, close: float, idx: int):
        """
        Handles buy action and updates state variables, creates an open deal
        @param price_buy: price the order was executed with
        @param close: current closing price, used for OCO sell
        @param idx: current timestamp
        @return: nothing
        """
        nonlocal usd, crypto, max_price, buy_price

        crypto = usd / price_buy - COMMISSION * (usd / price_buy)
        usd = 0.
        max_price = close
        buy_price = price_buy

        # buy_ts, buy_price, sell_ts, sell_price, run_usd, run_crypto, run_percent
        deals.append((idx, price_buy, 0, 0., usd, crypto, (crypto * price_buy - start_usd) * 100 / start_usd))

        next_limits[0] = 0
        next_limits[1] = 0

        if method == EvaluationMethod.OCO:
            next_limits[2] = close + close * abs(oco_buy_percent)
            next_limits[3] = max(next_limits[3], close - close * abs(oco_sell_percent))

    def handle_sell(price_sell: float, idx: int):
        """
        Handles sell order execution, updates internal state, closes active deal
        @param price_sell: price the order was executed with
        @param idx: current timestamp
        @return: nothing
        """
        nonlocal usd, crypto, max_price, buy_price, deals, next_limits

        usd = price_sell * crypto - COMMISSION * (price_sell * crypto)
        crypto = 0.0

        # buy_ts, buy_price, sell_ts, sell_price, run_usd, run_crypto, run_percent
        deals[-1] = deals[-1][0], deals[-1][1], idx, price_sell, usd, crypto, (usd - start_usd) * 100 / start_usd
        next_limits = np.zeros(4)

    # iterate over current prices and predictions
    for idx, current_prices in enumerate(prices):
        close, high, low = current_prices

        # this is required only for first two versions for backward compatibility due to a bug
        # and can be removed later
        if method == EvaluationMethod.ClosePrice or method == EvaluationMethod.CloseLO:
            # we're checking Limit Orders using only close price
            high = close
            low = close

        # sometimes limit prices raised on previous step are preserved for backward compatibility
        max_price = max(max_price, high)
        prev_limits = next_limits
        next_limits = np.zeros(4) if method == EvaluationMethod.CleanLOPrice else prev_limits

        # actual predictions
        buy_limit, sell_limit, do_nothing = predictions[idx]

        # variables indicating if an order has been executed on the previous step
        buy_trigger_high = 0 < prev_limits[0] < high
        buy_trigger_low = 0 < low < prev_limits[1]
        sell_trigger_high = 0 < prev_limits[2] < high
        sell_trigger_low = 0 < low < prev_limits[3]

        if usd > 0 and (buy_trigger_low or buy_trigger_high):
            # means that buy order was executed with either high or low limit prices
            handle_buy(close if method == EvaluationMethod.ClosePrice else prev_limits[int(buy_trigger_low)], close, idx)
        elif crypto > 0 and (sell_trigger_high or sell_trigger_low):
            # means that sell order was executed with either high or low limit prices
            price = close if method == EvaluationMethod.ClosePrice else prev_limits[2 + int(sell_trigger_low)]
            last_profit_and_time = (price - buy_price) / buy_price, idx
            handle_sell(price, idx)
        elif method == EvaluationMethod.OCO and buy_limit > sell_limit and buy_limit > do_nothing:
            # in this branch we handle buy prediction for OCO orders,
            # OCO orders are meant to have fixed sell prices and be updated evey time bot predicts buy
            if usd > 0:
                # means we still have not bough
                last_profit, time = last_profit_and_time
                high_profit_limit = oco_buy_percent * 0.8
                crash = is_crash(prices[:idx, 0], high_profit_limit)
                if oco_rise_percent > 0 and last_profit > high_profit_limit and idx - time < 24 * 60 and crash:
                    # this means that current price is in the crash that is most likely
                    # cause by previous rise, so we have to be careful here and not buy with market order
                    next_limits = np.array([close + close * oco_rise_percent, 0., 0., 0.])
                else:
                    # buy using market price
                    handle_buy(close, close, idx)
            else:
                # we've already bought so just update OCO sell prices
                next_limits[2] = close + close * abs(oco_buy_percent)
                next_limits[3] = max(next_limits[3], close - close * abs(oco_sell_percent))
        elif method != EvaluationMethod.OCO and usd > 0 and buy_limit > do_nothing:
            # it's a limit order price update
            next_limits = np.array([close + close * (10 - buy_limit) / 1000, 0., 0., 0.])
            if next_limits[0] < close:
                # limit price is already matching so just buy using market price
                handle_buy(close, close, idx)
        elif crypto > 0 and sell_limit > do_nothing:
            # it's a limit order price update
            next_limits = np.array([0., 0., 0., max_price - max_price * (10 - sell_limit) / 1000])
            if method == EvaluationMethod.OCO or close < next_limits[3]:
                # current price is already less than the limit price we wanted to sell
                # so just sell with market price, laos sell if this is OCO order mode
                handle_sell(close, idx)

        # now adjust limit orders if they're higher than specified caps
        if method == EvaluationMethod.CappedLOPrice:
            if next_limits[0] > 0 and usd > 0 and buy_limit_cap != 0.:
                # adjust next limit to be not higher than buy_limit_cap
                percent = (next_limits[0] - close) / close
                if percent > abs(buy_limit_cap):
                    next_limits[0] = close + close * abs(buy_limit_cap)
            elif crypto > 0 and next_limits[3] > 0 and sell_limit_cap != 0.:
                # adjust next limit to be not higher than sell_limit_cap
                price = max_price if use_max_sell_cap > 0 else close
                percent = (price - next_limits[3]) / price
                if percent > abs(sell_limit_cap):
                    next_limits[3] = min(price - price * abs(sell_limit_cap), close)

        # calculate current money on each step
        money[idx] = close * crypto + usd

    clean_profit = (usd + prices[-1][0] * crypto - prices[0][0]) / prices[0][0]
    expected_deals = len(prices) / DEAL_COEFFICIENT
    missing = expected_deals - len(deals)
    diff = (missing / expected_deals) ** 2 if missing > 0 and clean_profit > 0 else 0
    fitness = clean_profit - clean_profit * diff, clean_profit, len(deals)
    limits = next_limits[0], next_limits[1], next_limits[2], next_limits[3],
    return fitness, limits, deals


@njit
def portfolio(events: np.ndarray, total_shares: int, generate_deals: bool = False) \
        -> Tuple[float, List[Deal], List[float]]:
    start_usd = events[0][2]
    usd = start_usd
    crypto = 0
    shares = 0
    deals: List[Deal] = list()
    shares_in_market = 0

    for idx, event in enumerate(events):
        bot_id, share_change, price, ts = event

        prev = shares
        shares += share_change
        assert 0 <= shares <= total_shares

        if prev < shares and usd:
            # buy with all money
            crypto += usd * (1 - COMMISSION) / price
            usd = 0
            shares_in_market = 1

            if generate_deals and not prev:
                # TODO: potentially need an optimisation here, there might be a case when multiple trades happen
                # at exact same time so that we might buy and then sell immediately, that would cause unnecessary
                # deal to happen, we should check here first if last deal we closed has exactly the same timestamp
                # as the current one, and if it's the case we should cancel sell and buy only if shares changed
                # between cancelled sell and current buy

                # buy_ts, buy_price, sell_ts, sell_price, run_usd, run_crypto, run_percent
                deals.append((ts, price, 0, 0., usd, crypto, (crypto * price - start_usd) * 100 / start_usd))

        elif crypto and shares < prev:
            # someone sold
            shares_to_sell = shares_in_market - shares_in_market * float(shares) / total_shares
            shares_in_market -= shares_to_sell
            assert 0 <= shares_in_market < total_shares

            to_sell = crypto - crypto * float(shares) / total_shares
            crypto -= to_sell
            usd += to_sell * price * (1 - COMMISSION)

            if generate_deals and not shares:
                # adjust the price to make sure the profit percent is correct
                buy_ts, buy_price, sell_ts, sell_price, run_usd, run_crypto, run_percent = deals[-1]

                usd_when_bought = buy_price * run_crypto
                percent = (usd - usd_when_bought) / usd_when_bought
                adjusted_sell = (1 + percent + COMMISSION) * buy_price

                # buy_ts, buy_price, sell_ts, sell_price, run_usd, run_crypto, run_percent
                deals[-1] = buy_ts, buy_price, ts, adjusted_sell, usd, crypto, (usd - start_usd) * 100 / start_usd

    # instead of returning just one share, it's better to return an array of shares because we know all of them
    # for example, if total 3 bots and all 3 in market return: (1-2/3), (1-(1-2/3)*1/3), (1-(1-(1-2/3)*1/3))
    # then these shares will be combined with sorted limit prices in trade and corresponding orders will be placed
    result_shares = list()
    shares = shares - 1 if shares else 0
    while shares:
        shares_to_sell = shares_in_market - shares_in_market * float(shares) / total_shares
        result_shares.append(shares_to_sell)

        shares_in_market -= shares_to_sell
        assert 0 <= shares_in_market < total_shares
        shares -= 1

    if shares_in_market:
        result_shares.append(shares_in_market)

    return (usd + crypto * events[-1][2] - start_usd) * 100 / start_usd, deals, result_shares
