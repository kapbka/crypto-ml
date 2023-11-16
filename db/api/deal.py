from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy.sql.expression import desc

from common.constants import DealStatus
from db.model import Deal


class DBDeal:
    def __init__(self, db):
        self.db = db

    def get(self, version_id: int, currency: str, individual_id: int, buy_ts: datetime):
        return self.db.session.query(Deal).filter(Deal.version == version_id,
                                                  Deal.individual == individual_id,
                                                  Deal.currency == currency,
                                                  Deal.buy_ts == buy_ts
                                                  ).first()

    def get_all(self, version_id: int, currency: str, individual_id: int, ts_from: datetime):
        return self.db.session.query(Deal).filter(Deal.version == version_id,
                                                  Deal.individual == individual_id,
                                                  Deal.currency == currency,
                                                  Deal.buy_ts >= ts_from,
                                                  ).all()

    def get_last(self, currency: str, individual: int, version: int, status: Optional[DealStatus] = None) -> Deal:
        query = self.db.session.query(Deal). \
            filter(Deal.currency == currency,
                   Deal.version == version,
                   Deal.individual == individual
                   )
        query = query.filter(Deal.status == status) if status else query
        return query.order_by(desc(Deal.buy_ts)).first()

    def delete(self, currency_code: str, version_id: int, individual_id: int):
        self.db.individual.balance.delete(currency_code=currency_code, version_id=version_id, md5_or_id=individual_id)
        self.db.session.query(Deal).filter(Deal.currency == currency_code,
                                           Deal.version == version_id,
                                           Deal.individual == individual_id).delete()

    def set_batch(self, version: int, individual_id: int, currency: str, deals: List[Deal], replace: bool = True):
        # individual
        individual = self.db.individual.get_db(md5_or_id=individual_id)

        # clean before saving if replace is true
        if replace:
            self.delete(currency_code=currency, version_id=version, individual_id=individual_id)

        nvl = lambda a, b: a or b
        dt = datetime.now(tz=timezone.utc)

        # set deals
        deal = None
        sell_ts = None
        deal_count = 0
        for ev_deal in deals:
            deal_count += 1
            is_new = True

            if ev_deal.buy_ts >= nvl(ev_deal.sell_ts, dt):
                raise ValueError(f'Invalid deal dates: <{ev_deal}>')

            if deal_count == 1:
                deal = self.get(
                    currency=currency,
                    version_id=version,
                    individual_id=individual_id,
                    buy_ts=ev_deal.buy_ts
                )
                if deal:
                    if deal.status == DealStatus.Close:
                        raise ValueError(f'Deal {deal.id} has already been closed earlier!')

                    is_new = False

                    deal.is_realtime = ev_deal.is_realtime
                    deal.buy_ts = ev_deal.buy_ts
                    deal.buy_price = ev_deal.buy_price
                    deal.sell_ts = ev_deal.sell_ts
                    deal.sell_price = ev_deal.sell_price
                    deal.status = ev_deal.status
                    deal.run_usd = ev_deal.run_usd
                    deal.run_percent = ev_deal.run_percent
                    deal.run_crypto = ev_deal.run_crypto
                else:
                    deal_last = self.get_last(
                        currency=currency,
                        individual=individual_id,
                        version=version
                    )
                    if deal_last and ev_deal.buy_ts < nvl(deal_last.sell_ts, dt):
                        raise ValueError(f'New deal buy_ts {ev_deal.buy_ts} is earlier than the last deal in DB '
                                         f'{nvl(deal_last.sell_ts, dt)}!')
            else:
                if ev_deal.buy_ts < sell_ts:
                    raise ValueError(f'Individual {individual_id}: deal with buy_ts {ev_deal.buy_ts} is intersected '
                                     f'with the previous sell_ts {sell_ts}')

            if is_new:
                deal = Deal(version=version,
                            individual=individual_id,
                            currency=currency,
                            is_realtime=ev_deal.is_realtime,
                            buy_ts=ev_deal.buy_ts,
                            buy_price=ev_deal.buy_price,
                            sell_ts=ev_deal.sell_ts,
                            sell_price=ev_deal.sell_price,
                            status=ev_deal.status,
                            run_usd=ev_deal.run_usd,
                            run_percent=ev_deal.run_percent,
                            run_crypto=ev_deal.run_crypto,
                            )
                self.db.session.add(deal)

            sell_ts = nvl(deal.sell_ts, dt)

        # if last deal closed, update balance
        if deal and deal.status == DealStatus.Close:
            self.db.session.flush()
            self.db.individual.balance.set(md5=individual.md5, deal=deal)
