import hashlib
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Individual:
    weights: List[np.ndarray]
    fitness: Tuple[float, float, float] = (0, 0, 0)

    def weights_and_limits(self, params: List[Tuple[str, dict]]) -> Tuple[List[np.ndarray], Tuple[float, float, bool]]:
        limits = [0, 0, 0]
        weights = self.weights.copy()
        if len(weights[-1]) != params[-1][1]['units']:
            limits = weights[-1][params[-1][1]['units']:]
            weights[-1] = weights[-1][:params[-1][1]['units']]
        return weights, limits

    def md5(self) -> str:
        hash_md5 = hashlib.md5()
        for w in self.weights:
            hash_md5.update(w.tobytes())

        return hash_md5.hexdigest()
