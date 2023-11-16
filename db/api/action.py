import json
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

from sqlalchemy.sql.expression import desc

from common.metrics.evaluator import Limits
from db.model import Action


class DBAction:
    def __init__(self, db):
        self.db = db

    def set(self, limits: Dict[Limits, float], currency_code: str, ts: datetime, close: float):
        # get last action and check if we need to create new one
        body = json.dumps([{"limit": k, "share": v} for k, v in limits.items()])
        found: Action = self.db.session.query(Action)\
            .filter(Action.currency_code == currency_code) \
            .order_by(desc(Action.ts)).first()
        if found and found.limits == body:
            return

        action = Action(currency_code=currency_code, ts=ts, close=close, limits=body)
        logging.info(f"New action: {action}")
        self.db.add(action)

    def get(self, currency_code: str) -> Optional[Action]:
        return self.db.session.query(Action).filter(Action.currency_code == currency_code) \
            .order_by(desc(Action.ts)).first()
