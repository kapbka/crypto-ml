import os
from typing import List, Tuple, Any, Union, Dict

import numpy as np
import pandas as pd
import tensorflow as tf

from common.constants import LABEL_COLUMNS
from common.storage.file import File

ModelParams = List[Tuple[str, dict]]
Ann = tf.keras.models.Sequential


def make_batches(frame: np.ndarray, label_data: np.ndarray, ts: np.ndarray, window_offsets: List[int],
                 label_offset: int = 0):
    total_rows = frame.shape[0]
    max_offset = max(*window_offsets)
    total_result_rows = total_rows - max_offset - label_offset

    res = np.zeros([total_result_rows, len(window_offsets), frame.shape[1]])
    label_shape = label_data.shape
    timestamps = np.zeros(total_result_rows, dtype=ts.dtype)
    labels = np.zeros(total_result_rows) if len(label_shape) == 1 else np.zeros([total_result_rows, label_shape[1]])
    for idx in range(total_result_rows):
        max_idx = idx + max_offset
        res[idx] = np.array([frame[max_idx - offset] for offset in window_offsets])
        labels[idx] = label_data[max_idx + label_offset]
        timestamps[idx] = ts[max_idx + label_offset]
    return res, labels, timestamps


def make_inputs(df: pd.DataFrame, offsets: List[int], columns: List[str],
                label_col: Union[str, List[str]] = LABEL_COLUMNS,
                label_offset: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return make_batches(df[columns].values, df[label_col].values, df.index.values, window_offsets=offsets,
                        label_offset=label_offset)


def create_ann(params: ModelParams, offsets: List[int], train_columns: List[str]) -> Ann:
    ann = tf.keras.models.Sequential()
    for layer_name, kwargs in params:
        ann.add(getattr(tf.keras.layers, layer_name)(**kwargs))

    data_inputs = np.zeros([1, len(offsets), len(train_columns)])
    ann(data_inputs)
    return ann


def stringify(params: ModelParams, scaler: Any, dataset: File) -> str:
    res = [f"{dataset.ticker}_{dataset.days}_{dataset.ts.isoformat()}", type(scaler).__name__.lower()]

    ann_parts = list()
    for name, kwargs in params:
        params = "_".join([str(t) for t in kwargs.values()])
        ann_parts.append(f"{name.lower()}{f'_{params}' if kwargs else ''}")
    return os.path.join(*res, "_".join(ann_parts))


def columns_from_indicators(indicators: Dict[str, List[int]]) -> List[str]:
    train_columns = []
    for indicator, lengths in indicators.items():
        train_columns.extend([f"{indicator}_{length}" for length in lengths])

    train_columns.sort(key=lambda x: int(x.split('_')[-1]))
    return train_columns


def max_offset(indicators: Dict[str, List[int]], offsets: List[int]) -> int:
    return max(offsets) + max(map(max, indicators.values())) * 2
