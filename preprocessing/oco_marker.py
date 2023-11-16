import numpy as np
from numba import njit

BUY = 0
SELL = 1
DO_NOTHING = 2


@njit
def create_labels_with_predictions(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                                   labels: np.ndarray, limits: np.ndarray, filter: np.ndarray):
    labels[:, DO_NOTHING] = 1
    time_limits = limits[:, 1]

    for start in range(len(high)):
        if filter is not None and not filter[start]:
            continue  # skipping this point because it's filtered out

        for current in range(start + 1, min(len(high), start + int(max(time_limits)) + 1)):
            profit_buy = (high[current] - close[start]) / close[start]
            profit_sell = (low[current] - close[start]) / close[start]

            label_idx = None
            for profit_limit, time_limit in limits:
                if profit_buy > profit_limit or profit_sell < -profit_limit or current == len(high) - 1:
                    label_idx = SELL if profit_sell < -profit_limit else BUY
                    labels[start][label_idx] = 1
                    labels[start][DO_NOTHING] = 0
                    break

            if label_idx is not None:
                break
