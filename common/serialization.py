import logging
import os
import pickle
from typing import List, Tuple, Union

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from common.individual import Individual

OLD_LSTM = [('LSTM', dict(units=32)), ('Dense', dict(units=3))]
OLD_OFFSETS = list(reversed([0, 10, 30, 60, 120]))
OLD_INDICATORS = {'rsi_prediction': []}


def load_file(full_path: str, individuals: List[Individual], result_params: List[tuple]) -> \
        List[Union[StandardScaler, MinMaxScaler]]:
    logging.info(f"Loading {full_path}")
    with open(full_path, "rb") as cp_file:
        obj = pickle.load(cp_file)

    scalers = []
    if isinstance(obj, MinMaxScaler) or isinstance(obj, StandardScaler):
        logging.info(f"Ignoring scaler: {full_path}")
        scalers.append(obj)
    elif isinstance(obj, tuple):
        obj, *params = obj
        individuals.append(obj)
        result_params.append(params)
    elif isinstance(obj, Individual):
        individuals.append(obj)
        result_params.append((OLD_LSTM, OLD_OFFSETS, OLD_INDICATORS, False))
    elif isinstance(obj, list):
        individuals.extend(obj)
        result_params.extend([(OLD_LSTM, OLD_OFFSETS, OLD_INDICATORS, False) for _ in obj])
    else:
        individuals.extend(obj["halloffame"].items)
        result_params.extend([(OLD_LSTM, OLD_OFFSETS, OLD_INDICATORS, False) for _ in obj["halloffame"].items])

    return scalers


def load_individuals(file_path: str) -> Tuple[List[Individual], List[tuple], List[Union[StandardScaler, MinMaxScaler]]]:
    individuals = list()
    params = list()
    scalers = list()

    if os.path.isfile(file_path):
        scalers.extend(load_file(file_path, individuals, params))
    elif os.path.isdir(file_path):
        logging.info(f"Loading directory {file_path}")
        for path, directories, files in os.walk(file_path):
            for file in files:
                full_path = os.path.join(path, file)
                scalers.extend(load_file(full_path, individuals, params))

    return individuals, params, scalers
