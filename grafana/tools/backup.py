import json
import logging
import os
import sys
import time
import zipfile
from tempfile import TemporaryDirectory

import requests

from common.constants import GRAFANA_SERVER
from common.log import setup_logging


def zip_directory(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, mode='w') as zipf:
        len_dir_path = len(folder_path)
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, file_path[len_dir_path:])


def backup(folder_path: str, dashboard_path: str = None):
    logging.info(f"folder_path {folder_path}, dashboard_path: {dashboard_path}")

    headers = {"Authorization": f"Bearer {os.environ['GRAFANA_TOKEN']}"}
    url_list = f"{GRAFANA_SERVER}/api/search?query=%"
    r_list = requests.get(url=url_list, headers=headers)
    r_list.raise_for_status()
    uids = [item['uid'] for item in r_list.json()]

    with TemporaryDirectory() as temp_dir:
        for uid in uids:
            url_uid = f"{GRAFANA_SERVER}/api/dashboards/uid/{uid}"
            r_uid = requests.get(url=url_uid, headers=headers)
            r_uid.raise_for_status()
            with open(f"{dashboard_path or temp_dir}/{uid}.json", 'w+') as f:
                json.dump(r_uid.json(), f, indent=4)

        if not dashboard_path:
            os.makedirs(name=folder_path, exist_ok=True)
            zip_directory(temp_dir, os.path.join(folder_path, f"grafana_dump_{time.strftime('%d%m%Y')}.zip"))


if __name__ == '__main__':
    setup_logging()
    backup(sys.argv[1], sys.argv[2])
