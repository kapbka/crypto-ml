import logging
import os

from mongoengine import connect as _mongo_connect


def connect():
    connection = _mongo_connect('twitter', host='ml.kapbka.dev', username='bot', password=os.getenv('PASSWORD'))
    logging.info(f"Connecting to {connection}")
