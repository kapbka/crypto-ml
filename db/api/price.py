import logging
from datetime import datetime
from typing import List, Optional

from pandas import DataFrame
from sqlalchemy.sql.expression import desc, asc

from db.model import Price


class DBPrice:
    def __init__(self, db):
        self.db = db

    def get(self, currency_code: str, ts: datetime):
        return self.db.session.query(Price).filter(Price.currency == currency_code, Price.ts == ts).first()

    def get_first(self, currency: str) -> Optional[Price]:
        return self.db.session.query(Price).filter(Price.currency == currency).order_by(asc(Price.ts)).first()

    def get_last(self, currency: str) -> Optional[Price]:
        return self.db.session.query(Price).filter(Price.currency == currency).order_by(desc(Price.ts)).first()

    def set(self, currency: str, ts: datetime, close: float, low: float, high: float, volume: float):
        # returns a price instance without id property populated
        # if at some point you need id to be populated
        # then add session.flush() before return
        price = Price(currency=currency,
                      ts=ts,
                      close=close,
                      low=low,
                      high=high,
                      volume=volume
                      )
        self.db.session.add(price)
        return price

    def set_batch(self, currency: str, df: DataFrame) -> List[Price]:
        logging.info(f"Saving {len(df)} prices for {currency}, last: {df.index[-1] if len(df) else None}")

        prices = list()

        for ts, close, low, high, volume in zip(df.index, df['close'], df['low'], df['high'], df['volume']):
            prices.append(self.set(currency=currency, ts=ts, close=close, low=low, high=high, volume=volume))

        return prices
