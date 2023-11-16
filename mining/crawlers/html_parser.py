import re
from typing import List

import requests


def parse_page(url: str, regexps: List[str]) -> List[str]:
    results = set()
    for pattern in regexps:
        for r in re.findall(pattern, url, re.MULTILINE):
            results.add(r)

        for r in re.findall(pattern, requests.get(url).text, re.MULTILINE):
            results.add(r)

    return list(results)