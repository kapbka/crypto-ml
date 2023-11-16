import pickle

from sklearn.preprocessing import StandardScaler

from db.model import Scaler


class DBScaler:
    def __init__(self, db):
        self.db = db

    def get(self, scaler_id: int) -> Scaler:
        return self.db.session.query(Scaler).filter(Scaler.id == scaler_id).first()

    def set(self, currency: str, scaler: StandardScaler):
        name = type(scaler).__name__.lower()
        result = self.db.session.query(Scaler).filter(Scaler.currency_code == currency,
                                                      Scaler.name == name).first()

        if not result:
            result = Scaler(
                currency_code=currency,
                name=name,
                data=pickle.dumps(scaler),
            )
            self.db.session.add(result)

        return result
