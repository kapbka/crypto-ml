from typing import Union

from sqlalchemy.sql.expression import desc

from db.model import IndividualAttribute, IndividualAttributeHistory


class DBIndividualAttributeHistory:
    def __init__(self, db):
        self.db = db

    def get_last(self, currency_code: str, version_id: int, md5_or_id: Union[str, int]):
        query = self.db.session.query(IndividualAttributeHistory)
        individual_attribute_history = query.filter(IndividualAttributeHistory.currency_code == currency_code,
                                                    IndividualAttributeHistory.version_id == version_id,
                                                    IndividualAttributeHistory.md5 == md5_or_id). \
            order_by(desc(IndividualAttributeHistory.id)).first() if isinstance(md5_or_id, str) \
            else query.filter(IndividualAttributeHistory.currency_code == currency_code,
                              IndividualAttributeHistory.version_id == version_id,
                              IndividualAttributeHistory.individual_id == md5_or_id). \
            order_by(desc(IndividualAttributeHistory.id)).first()
        return individual_attribute_history

    def get_all(self, currency_code: str, version_id: int, md5_or_id: Union[str, int]):
        query = self.db.session.query(IndividualAttributeHistory)
        individual_attribute_history = query.filter(IndividualAttributeHistory.currency_code == currency_code,
                                                    IndividualAttributeHistory.version_id == version_id,
                                                    IndividualAttributeHistory.md5 == md5_or_id). \
            order_by(desc(IndividualAttributeHistory.id)).all() if isinstance(md5_or_id, str) \
            else query.filter(IndividualAttributeHistory.currency_code == currency_code,
                              IndividualAttributeHistory.version_id == version_id,
                              IndividualAttributeHistory.individual_id == md5_or_id). \
            rder_by(desc(IndividualAttributeHistory.id)).all()
        return individual_attribute_history

    def set(self, individual_attribute_old: IndividualAttribute, individual_attribute_new: IndividualAttribute):
        # validation check
        if (individual_attribute_old.currency_code != individual_attribute_new.currency_code or
                individual_attribute_old.version_id != individual_attribute_new.version_id or
                individual_attribute_old.individual_id != individual_attribute_new.individual_id or
                individual_attribute_old.md5 != individual_attribute_new.md5):
            raise ValueError('individual_attribute_old and individual_attribute_new have not matching natural key!')

        individual_attribute_history = IndividualAttributeHistory(
            currency_code=individual_attribute_new.currency_code,
            version_id=individual_attribute_new.version_id,
            individual_id=individual_attribute_new.individual_id,
            md5=individual_attribute_new.md5,
            #
            realtime_enabled_old=individual_attribute_old.realtime_enabled,
            realtime_enabled_new=individual_attribute_new.realtime_enabled,
            #
            history_enabled_old=individual_attribute_old.history_enabled,
            history_enabled_new=individual_attribute_new.history_enabled,
            #
            priority_old=individual_attribute_old.priority,
            priority_new=individual_attribute_new.priority,
            share_old=individual_attribute_old.share,
            share_new=individual_attribute_new.share,
            #
            scaler_id_old=individual_attribute_old.scaler_id,
            scaler_id_new=individual_attribute_new.scaler_id,
            #
            oco_buy_percent_old=individual_attribute_old.oco_buy_percent,
            oco_buy_percent_new=individual_attribute_new.oco_buy_percent,
            oco_sell_percent_old=individual_attribute_old.oco_sell_percent,
            oco_sell_percent_new=individual_attribute_new.oco_sell_percent,
            oco_rise_percent_old=individual_attribute_old.oco_rise_percent,
            oco_rise_percent_new=individual_attribute_new.oco_rise_percent
        )
        self.db.session.add(individual_attribute_history)
        self.db.session.flush()

        return individual_attribute_history
