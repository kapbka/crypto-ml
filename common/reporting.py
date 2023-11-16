import logging
import os
from typing import List

import requests

from common.constants import TELEGRAM_NOTIFY_CHATS

DEFAULT_CHATS = list(
    map(int, (os.getenv('TELEGRAM_NOTIFY_CHATS') or ",".join(map(str, TELEGRAM_NOTIFY_CHATS))).split(",")))


def send_message(text: str,
                 chats: List[int] = DEFAULT_CHATS,
                 bot_id: str = os.getenv('TELEGRAM_NOTIFIER_BOT_ID')):
    try:
        url = f"https://api.telegram.org/bot{bot_id}/sendMessage"
        for chat in chats:
            requests.get(url, params={'chat_id': chat, "text": text})
        logging.info(f"Sent '{text}' to {chats}, bot id: {bot_id}")
    except:
        logging.exception(f"Unable to send '{text}' to {chats}, bot id: {bot_id}")
