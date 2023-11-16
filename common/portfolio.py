import logging
from collections import defaultdict
from datetime import datetime
from functools import partial
from itertools import permutations
from typing import List, Tuple, Dict, Optional

import numpy as np
from numba import njit

from common.constants import IsPortfolio
from common.individual import Individual
from common.metrics.evaluator import portfolio, convert_deal, Limits
from db.api import DB
from db.model.processing import Deal, Individual as DBIndividual


def deals_to_trades(deals: List[Deal]) -> List[np.ndarray]:
    trades: List[np.ndarray] = []

    for idx, d in enumerate(deals):
        if idx and d.buy_ts < deals[idx - 1].sell_ts:
            logging.error(f"Deal intersection for bot {d.individual}: {deals[idx - 1]} and {d}")
            return []  # deals have intersection

        buy_ts = np.datetime64(d.buy_ts).astype(int)
        if not d.sell_price and (idx != len(deals) - 1 or trades and trades[-1][-1] > buy_ts):
            logging.error(f"Open deal in the middle: {d}, {d.id}, {d.individual}")
            return []  # incorrect unclosed deal in the middle

        trades.append(np.array([float(d.individual), 1., d.buy_price, float(buy_ts)]))
        if d.sell_price:
            sell_ts = np.datetime64(d.sell_ts).astype(int)
            trades.append(np.array([float(d.individual), -1., d.sell_price, float(sell_ts)]))
    return trades


def process_historical_portfolios(db: DB, version: int, currency: str, ts_from: datetime,
                                  portfolio_md5: Optional[str] = None) -> int:
    portfolios = defaultdict(list)
    all_bots = set()
    for db_portfolio in db.portfolio.get_batch(portfolio_md5=portfolio_md5):
        portfolios[db_portfolio.portfolio_md5].append(db_portfolio.md5)
        all_bots.add(db_portfolio.md5)

    # load deals
    deals = {md5: db.deal.get_all(
        version_id=version,
        currency=currency,
        individual_id=db.individual.get_db(md5).id,
        ts_from=ts_from
    ) for md5 in all_bots}

    # load trades
    trades = {md5: deals_to_trades(deals[md5]) for md5 in all_bots}

    # now for each one generate deals
    for portfolio_md5, bots in portfolios.items():
        all_trades = [trades[bot] for bot in bots if trades[bot]]
        if len(all_trades) == len(all_bots):
            data = np.concatenate(all_trades)
            data = data[data[:, 3].argsort(kind='stable')]
            _, deals, _ = portfolio(events=data, total_shares=len(all_trades), generate_deals=True)

            portfolio_id = db.individual.get_db(portfolio_md5).id
            db.deal.set_batch(
                version=version,
                individual_id=portfolio_id,
                currency=currency,
                deals=map(partial(convert_deal, None, portfolio_id), deals)
            )

    return len(portfolios)


def calculate_shares(limits: List[Limits], shares: List[float]) -> Dict[Limits, float]:
    high = sorted(filter(lambda x: x, map(lambda x: x[2], limits)), reverse=False)
    low = sorted(filter(lambda x: x, map(lambda x: x[3], limits)), reverse=True)

    res: Dict[Limits, float] = defaultdict(float)
    for share, up, down in zip(shares, high, low):
        res[(0, 0, up, down)] += share

    return {**res}


def find_best(trades_cache: Dict[str, List[np.ndarray]], size: int) -> List[Tuple[float, Tuple[str]]]:
    logging.info(f"{len(trades_cache)} bots to consider")

    all_combinations = set(map(tuple, map(lambda x: sorted(x), permutations(trades_cache.keys(), size))))
    logging.info(f"{len(all_combinations)} combinations to consider")

    results = list()
    prev_percent = 0
    for idx, bots in enumerate(all_combinations):
        data = np.concatenate([trades_cache[bot] for bot in bots if len(trades_cache[bot])])
        data = data[data[:, 3].argsort(kind='stable')]

        percent, deals, _ = portfolio(data, len(bots), generate_deals=True)
        logging.debug(f"Portfolio: {percent:.1f}%, bots: {bots}")

        results.append((percent, bots))

        done_percent = int(idx * 100 / len(all_combinations))
        if done_percent != prev_percent:
            logging.info(f"Done {done_percent}%, best: {sorted(results, key=lambda x: x[0], reverse=True)[0]}")
            prev_percent = done_percent

    return sorted(results, key=lambda x: x[0], reverse=True)


@njit
def run_portfolio_detailed(trades: List[np.ndarray], prices: np.ndarray, all_ts: np.ndarray, total_shares: int) -> \
        List[Tuple[float, int]]:
    money = prices[0]
    crypto = 0
    shares = 0

    moneys = list()
    trade_idx = 0
    next_ts = int(trades[trade_idx][3] * 1000)
    for idx, (ts, price) in enumerate(zip(all_ts, prices)):
        while trade_idx < len(trades) and ts == next_ts:
            individual_id, share_change, price, _ = trades[trade_idx]

            prev = shares
            shares += share_change
            assert 0 <= shares <= total_shares

            if prev < shares and money:
                # buy with all money
                crypto += money * 0.999 / price
                money = 0
            elif crypto and shares < prev:
                # someone sold
                to_sell = crypto - crypto * float(shares) / total_shares
                crypto -= to_sell
                money += to_sell * price * 0.999

            trade_idx += 1
            if trade_idx < len(trades):
                next_ts = int(trades[trade_idx][3] * 1000)

        moneys.append((crypto * price + money, shares))

    assert trade_idx == len(trades)
    if crypto:
        money += crypto * price

    return moneys


def save_portfolio(db: DB, currency: str, portfolio_data: Tuple[float, Tuple[str]]) -> DBIndividual:
    percent, bots = portfolio_data

    # load bots from the DB
    individuals = list(map(db.individual.get, bots))
    result_individual = Individual(weights=list(map(lambda i: np.array(i.md5()), individuals)))

    saved = db.individual.get_db(result_individual.md5())
    new = not saved
    if new:
        saved = db.individual.set(
            bot=result_individual,
            ann_id=db.individual.get_db(bots[0]).ann_id,
            train_currency=currency,
            is_portfolio=IsPortfolio.Yes
        )

        list(map(partial(db.portfolio.set, result_individual.md5(), 1), bots))

    action = "has been saved" if new else "already exists"
    logging.info(f"Portfolio {result_individual.md5()} with profit {percent:.1f}% {action}")
    return saved
