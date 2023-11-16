import json
from collections import defaultdict
from typing import Dict

from common.constants import DICTIONARY_SIZE


def from_stats(data: Dict[str, int]):
    counter = 0
    result = defaultdict(int)
    for k, v in sorted([(k, v) for k, v in data.items()], key=lambda x: x[1], reverse=True):
        result[k] = counter
        counter += 1
        if counter >= DICTIONARY_SIZE:
            break

    return result


def get_dict() -> Dict[str, int]:
    with open('data/words.json', 'r') as in_file:
        data = json.load(in_file)
    return from_stats(data)
