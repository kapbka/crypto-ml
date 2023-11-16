from sqlalchemy.sql.expression import asc

from db.model import Currency


class DBCurrency:
    def __init__(self, db):
        self.db = db

    def get(self, code: str):
        return self.db.session.query(Currency).filter(Currency.code == code).first()

    def get_batch(self):
        return self.db.session.query(Currency).order_by(asc(Currency.code)).all()

    def set(self, code: str, name: str):
        currency = self.get(code=code)

        if not currency:
            currency = Currency(
                code=code,
                name=name
            )
            self.db.session.add(currency)

            # generate default attributes for all individuals
            for individual in self.db.individual.get_all():
                self.db.individual.attribute.set_defaults(individual=individual, currency_code=code)

        return currency
