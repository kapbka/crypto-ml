import json
import logging
import os
import sys
from pathlib import Path
from http import client

import requests

from common.constants import GRAFANA_SERVER
from common.log import setup_logging
from grafana.tools.backup import backup


def get_dashboards(folder_path: str):
    res = dict()

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_name = file.split('.')[0]
            with open(os.path.join(root, file), 'r') as f:
                res[file_name] = json.load(f)

    return res


def main(folder_path: str):
    logging.info(f"folder_path {folder_path}")

    gd = os.path.join(folder_path, 'grafana/dashboards')
    dashboards = get_dashboards(gd)

    headers = {"Authorization": f"Bearer {os.environ['GRAFANA_TOKEN']}"}
    url_upd = f"{GRAFANA_SERVER}/api/dashboards/db"

    for dk in dashboards:
        # get current version from Grafana
        url_get = f"{GRAFANA_SERVER}/api/dashboards/uid/{dk}"
        r_get = requests.get(url=url_get, headers=headers)
        r_get.raise_for_status()

        dashboards[dk]['meta']['version'] = r_get.json()['meta']['version']
        dashboards[dk]['dashboard']['version'] = r_get.json()['dashboard']['version']
        r_upd = requests.post(url=url_upd, headers=headers, json=dashboards[dk])
        logging.info(f"{client.responses[r_upd.status_code]} {dk}: {r_upd.json()}")
        r_upd.raise_for_status()


if __name__ == '__main__':
    setup_logging(filename='/tmp/grafana_deploy.log')
    main(sys.argv[1])
