from copy import deepcopy
from typing import Union, Optional

from common.constants import RealtimeStatus, HistoryStatus, OptimizedStatus, WriteAttributeHistoryStatus
from db.api.attribute_history import DBIndividualAttributeHistory
from db.model import Individual, IndividualAttribute, Currency


class DBIndividualAttribute:
    def __init__(self, db):
        self.db = db

    def get(self, currency_code: str, version_id: int, md5_or_id: Union[str, int]) -> IndividualAttribute:
        query = self.db.session.query(IndividualAttribute)
        individual_attribute = query.filter(IndividualAttribute.currency_code == currency_code,
                                            IndividualAttribute.version_id == version_id,
                                            IndividualAttribute.md5 == md5_or_id).first() if isinstance(md5_or_id, str)\
            else query.filter(IndividualAttribute.currency_code == currency_code,
                              IndividualAttribute.version_id == version_id,
                              IndividualAttribute.individual_id == md5_or_id).first()
        return individual_attribute

    def set(self, currency_code: str, version_id: int, individual: Individual,
            realtime_enabled: int, history_enabled: int, is_optimized: int, scaler_id: Optional[int],
            # portfolio attributes
            priority: int, share: float,
            # oco orders specifics
            oco_buy_percent: float, oco_sell_percent: float, oco_rise_percent: float,
            # history
            write_history: int = WriteAttributeHistoryStatus.On.value):

        # attribute validation
        if realtime_enabled == RealtimeStatus.Enabled.value:
            if history_enabled == HistoryStatus.Enabled.value:
                raise ValueError('Realtime and history attributes can not be activated simultaneously!')
            if share <= 0 or share > 1:
                raise ValueError('If realtime is enabled then share value has to be within (0, 1] interval!')

        individual_attribute = self.get(currency_code=currency_code, version_id=version_id, md5_or_id=individual.id)
        if not individual_attribute:
            individual_attribute = IndividualAttribute(currency_code=currency_code,
                                                       version_id=version_id,
                                                       individual_id=individual.id,
                                                       md5=individual.md5,
                                                       realtime_enabled=realtime_enabled,
                                                       history_enabled=history_enabled,
                                                       is_optimized=is_optimized,
                                                       priority=priority,
                                                       share=share,
                                                       scaler_id=scaler_id,
                                                       oco_buy_percent=oco_buy_percent,
                                                       oco_sell_percent=oco_sell_percent,
                                                       oco_rise_percent=oco_rise_percent,
                                                       )
            self.db.session.add(individual_attribute)
            self.db.session.flush()
        else:
            individual_attribute_old = None
            if write_history == WriteAttributeHistoryStatus.On.value:
                individual_attribute_old = deepcopy(individual_attribute)

            individual_attribute.realtime_enabled = realtime_enabled
            individual_attribute.history_enabled = history_enabled
            individual_attribute.is_optimized = is_optimized
            individual_attribute.priority = priority
            individual_attribute.share = share
            individual_attribute.scaler_id = scaler_id
            individual_attribute.oco_buy_percent = oco_buy_percent
            individual_attribute.oco_sell_percent = oco_sell_percent
            individual_attribute.oco_rise_percent = oco_rise_percent

            if write_history == WriteAttributeHistoryStatus.On.value:
                self.history.set(
                    individual_attribute_old=individual_attribute_old,
                    individual_attribute_new=individual_attribute
                )

        return individual_attribute

    def set_defaults(self, individual: Individual,
                     realtime_enabled: int = RealtimeStatus.Disabled.value,
                     history_enabled: int = HistoryStatus.Enabled.value,
                     is_optimized: int = OptimizedStatus.Disabled.value,
                     scaler_id: int = None,
                     currency_code: str = None):
        """will be used in individual.set to populated default values for all currencies and versions"""
        for currency in [Currency(code=currency_code)] if currency_code else self.db.currency.get_batch():
            for version in self.db.version.get_batch():
                self.set(currency_code=currency.code,
                         version_id=version.id,
                         individual=individual,
                         realtime_enabled=realtime_enabled,
                         history_enabled=history_enabled,
                         is_optimized=is_optimized,
                         priority=0,
                         share=1 if realtime_enabled else 0,
                         scaler_id=scaler_id,
                         oco_buy_percent=0.02,
                         oco_sell_percent=-0.02,
                         oco_rise_percent=0.0,
                         write_history=WriteAttributeHistoryStatus.Off.value
                         )

    @property
    def history(self):
        return DBIndividualAttributeHistory(self.db)
