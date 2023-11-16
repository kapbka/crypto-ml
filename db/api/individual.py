import pickle
from typing import Union, List, Tuple, Optional

from sqlalchemy import func, or_
from sqlalchemy.orm import undefer
from sqlalchemy.sql.expression import asc

from common.constants import RealtimeStatus, HistoryStatus, IsPortfolio, EPOCH, DEFAULT_CURRENCY
from common.individual import Individual as Bot
from db.api.attribute import DBIndividualAttribute
from db.api.balance import DBIndividualBalance
from db.model import Individual, Deal, IndividualAttribute


class DBIndividual:
    def __init__(self, db):
        self.db = db

    def get(self, md5_or_id: Union[str, int]) -> Bot:
        query = self.db.session.query(Individual)
        query = query.filter(Individual.md5 == md5_or_id) if isinstance(md5_or_id, str) else query.filter(
            Individual.id == md5_or_id)
        return Bot(weights=pickle.loads(query.options(undefer(Individual.weights)).first().weights))

    def get_db(self, md5_or_id: Union[str, int]) -> Individual:
        query = self.db.session.query(Individual)
        individual = query.filter(Individual.md5 == md5_or_id).first() if isinstance(md5_or_id, str) \
            else query.filter(Individual.id == md5_or_id).first()
        return individual

    def get_all(self, include_portfolio: bool = True) -> List[Individual]:
        query = self.db.session.query(Individual)\
            .filter(or_(include_portfolio, func.coalesce(Individual.is_portfolio, IsPortfolio.No) == IsPortfolio.No))\
            .order_by(asc(Individual.id))
        return query.all()

    def get_realtime(self, currency_code: str) -> List[Tuple[Individual, IndividualAttribute]]:
        return self.db.session.query(Individual, IndividualAttribute) \
            .join(IndividualAttribute, Individual.id == IndividualAttribute.individual_id) \
            .filter(IndividualAttribute.currency_code == currency_code,
                    IndividualAttribute.realtime_enabled == RealtimeStatus.Enabled.value,
                    func.coalesce(Individual.is_portfolio, IsPortfolio.No) == IsPortfolio.No).all()

    def get_history(self, portfolio: IsPortfolio) -> List[Tuple[Individual, IndividualAttribute]]:
        query = self.db.session.query(Individual, IndividualAttribute) \
            .join(IndividualAttribute, Individual.id == IndividualAttribute.individual_id) \
            .join(Deal, Individual.id == Deal.individual, isouter=True) \
            .filter(IndividualAttribute.history_enabled == HistoryStatus.Enabled.value,
                    func.coalesce(Individual.is_portfolio, IsPortfolio.No) == portfolio) \
            .group_by(IndividualAttribute.version_id,
                      IndividualAttribute.individual_id,
                      IndividualAttribute.currency_code,
                      Individual.id) \
            .order_by(asc(func.coalesce(func.max(Deal.buy_ts), EPOCH)))
        return query.all()

    def search(self, query: str, limit: int):
        return self.db.session.query(Individual) \
            .filter(Individual.md5.like(query + "%")) \
            .limit(limit).all()

    def set(self, bot: Bot, ann_id: int, scaler_id: Optional[int] = None, parent_md5: Optional[str] = None,
            train_currency: str = None, is_portfolio: int = IsPortfolio.No.value):
        individual = self.db.session.query(Individual).filter(Individual.md5 == bot.md5()).first()
        if not individual:
            individual = Individual(md5=bot.md5(),
                                    weights=pickle.dumps(bot.weights),
                                    parent_md5=parent_md5,
                                    ann_id=ann_id,
                                    train_currency=DEFAULT_CURRENCY if not train_currency else train_currency,
                                    is_portfolio=is_portfolio
                                    )
            self.db.session.add(individual)
            self.db.session.flush()
            self.attribute.set_defaults(individual=individual, scaler_id=scaler_id, currency_code=train_currency)

        return individual

    @property
    def attribute(self):
        return DBIndividualAttribute(self.db)

    @property
    def balance(self):
        return DBIndividualBalance(self.db)
