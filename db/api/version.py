from sqlalchemy.sql.expression import asc

from db.model import Version


class DBVersion:
    def __init__(self, db):
        self.db = db

    def get(self, version_id: int):
        return self.db.session.query(Version).filter(Version.id == version_id).first()

    def get_batch(self):
        return self.db.session.query(Version).order_by(asc(Version.id)).all()

    def set(self, version_num: int, name: str, comment: str):
        version = self.get(version_id=version_num)

        if not version:
            version = Version(
                id=version_num,
                name=name,
                comment=comment,
            )
            self.db.session.add(version)

            # generate default attributes for all individuals
            for individual in self.db.individual.get_all():
                self.db.individual.attribute.set_defaults(individual=individual, currency_code=individual.train_currency)

        return version
