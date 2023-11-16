from typing import Union

from db.model import Deal, IndividualBalance


class DBIndividualBalance:
    def __init__(self, db):
        self.db = db

    def get(self, currency_code: str, version_id: int, md5_or_id: Union[str, int]):
        query = self.db.session.query(IndividualBalance)
        individual_balance = query.filter(IndividualBalance.currency_code == currency_code,
                                          IndividualBalance.version_id == version_id,
                                          IndividualBalance.md5 == md5_or_id).first() if isinstance(md5_or_id, str) \
            else query.filter(IndividualBalance.currency_code == currency_code,
                              IndividualBalance.version_id == version_id,
                              IndividualBalance.individual_id == md5_or_id).first()
        return individual_balance

    def set(self, md5: str, deal: Deal):
        individual_balance = self.get(currency_code=deal.currency, version_id=deal.version, md5_or_id=deal.individual)

        if not individual_balance:
            individual_balance = IndividualBalance(currency_code=deal.currency,
                                                   version_id=deal.version,
                                                   individual_id=deal.individual,
                                                   md5=md5,
                                                   crypto=deal.run_crypto,
                                                   usd=deal.run_usd,
                                                   percent=deal.run_percent,
                                                   deal_id=deal.id
                                                   )
            self.db.session.add(individual_balance)
            self.db.session.flush()
        else:
            individual_balance.crypto = deal.run_crypto
            individual_balance.usd = deal.run_usd
            individual_balance.percent = deal.run_percent
            individual_balance.deal_id = deal.id

        return individual_balance

    def delete(self, currency_code: str, version_id: int, md5_or_id: Union[str, int]):
        if isinstance(md5_or_id, str):
            self.db.session.query(IndividualBalance).filter(IndividualBalance.currency_code == currency_code,
                                                            IndividualBalance.version_id == version_id,
                                                            IndividualBalance.md5 == md5_or_id).delete()
        else:
            self.db.session.query(IndividualBalance).filter(IndividualBalance.currency_code == currency_code,
                                                            IndividualBalance.version_id == version_id,
                                                            IndividualBalance.individual_id == md5_or_id).delete()
