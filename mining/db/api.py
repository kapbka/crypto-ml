import logging

from mongoengine import connect as _mongo_connect


def connect():
    connection = _mongo_connect('twitter', host='ml.clrn.dev', username='bot', password='sE]W.c<J~Me74dgE')
    logging.info(f"Connecting to {connection}")
