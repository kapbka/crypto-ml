import logging
import sys


def setup_logging(level: str = 'info', out=sys.stderr, filename: str = None):
    format = '[%(asctime)s][%(levelname)s] %(message)s'

    handler = logging.StreamHandler(out)
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.addHandler(handler)

    if filename:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.setLevel(getattr(logging, level.upper()))
