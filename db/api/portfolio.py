from functools import partial
from typing import Tuple, List, Optional

from common.constants import RealtimeStatus
from db.model import Portfolio, Individual, IndividualAttribute


class DBPortfolio:
    def __init__(self, db):
        self.db = db

    def get(self, portfolio_md5: str, md5: str) -> Portfolio:
        return self.db.session.query(Portfolio).filter(Portfolio.portfolio_md5 == portfolio_md5,
                                                       Portfolio.md5 == md5).first()

    def get_batch(self, portfolio_md5: Optional[str] = None) -> List[Portfolio]:
        query = self.db.session.query(Portfolio)
        query = query.filter(Portfolio.portfolio_md5 == portfolio_md5) if portfolio_md5 else query
        return query.all()

    def get_realtime(self, currency_code: str) -> List[IndividualAttribute]:
        query = self.db.session.query(IndividualAttribute) \
            .join(Portfolio, Portfolio.portfolio_md5 == IndividualAttribute.md5) \
            .filter(IndividualAttribute.currency_code == currency_code,
                    IndividualAttribute.realtime_enabled == RealtimeStatus.Enabled.value)
        return query.distinct().all()

    def get_members(self, currency_code: str, portfolio_md5: str = None) -> List[Individual]:
        if not portfolio_md5:
            portfolio_attrs = self.get_realtime(currency_code=currency_code)
            portfolio_md5 = portfolio_attrs[0].md5 if portfolio_attrs else None

        members = []
        if portfolio_md5:
            members = self.get_batch(portfolio_md5=portfolio_md5)
            members = list(map(self.db.individual.get_db, map(lambda x: x.md5, members)))

        return members

    def get_members_attrs(self, currency_code: str, version_id: int = None, portfolio_members: List[Individual] = None) \
            -> List[IndividualAttribute]:
        if not version_id:
            portfolio_attrs = self.get_realtime(currency_code=currency_code)
            version_id = portfolio_attrs[0].version_id if portfolio_attrs else None

        if not portfolio_members:
            portfolio_members = self.get_members(currency_code=currency_code)

        member_attrs = []
        if portfolio_members:
            get_attr = partial(self.db.individual.attribute.get, currency_code, version_id)
            member_attrs = list(map(get_attr, map(lambda x: x.md5, portfolio_members)))

        return member_attrs

    def get_realtime_members_with_attrs(self, currency_code: str, version_id: int = None, portfolio_md5: str = None) \
            -> List[Tuple[Individual, IndividualAttribute]]:
        portfolio_members: List[Individual] = self.get_members(currency_code=currency_code, portfolio_md5=portfolio_md5)

        member_attrs: List[IndividualAttribute] = self.get_members_attrs(
            currency_code=currency_code,
            version_id=version_id,
            portfolio_members=portfolio_members
        )

        return list(zip(portfolio_members, map(lambda x: x, member_attrs)))

    def set(self, portfolio_md5: str, share: float, md5: str):
        portfolio = self.get(portfolio_md5=portfolio_md5, md5=md5)
        if not portfolio:
            portfolio = Portfolio(portfolio_md5=portfolio_md5, md5=md5, share=share)
            self.db.session.add(portfolio)
            self.db.session.flush()
        else:
            portfolio.share = share

        return portfolio

    def delete(self, portfolio_md5: str, md5: str):
        portfolio = self.get(portfolio_md5=portfolio_md5, md5=md5)

        if not portfolio:
            raise ValueError(f'No such member {md5} in portfolio {portfolio_md5}')

        self.db.session.query(Portfolio).filter(Portfolio.portfolio_md5 == portfolio_md5,
                                                Portfolio.md5 == md5).delete()
