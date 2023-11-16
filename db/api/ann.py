import json
from typing import Dict, List, Tuple

from db.model import Ann

Layers = List[Tuple[str, Dict[str, int]]]


class DBAnn:
    def __init__(self, db):
        self.db = db

    def get(self, ann_id: int) -> Ann:
        return self.db.session.query(Ann).filter(Ann.id == ann_id).first()

    def set(self, layers: Layers, offsets: List[int], indicators: Dict[str, List[int]], scaled: bool):
        layers = json.dumps(layers)
        offsets = json.dumps(offsets)
        indicators = json.dumps(indicators)
        is_scaled = int(scaled)

        ann = self.db.session.query(Ann).filter(Ann.layers == layers,
                                                Ann.offsets == offsets,
                                                Ann.indicators == indicators,
                                                Ann.is_scaled == is_scaled).first()

        if not ann:
            ann = Ann(
                layers=layers,
                offsets=offsets,
                indicators=indicators,
                is_scaled=is_scaled,
            )
            self.db.session.add(ann)

        return ann
