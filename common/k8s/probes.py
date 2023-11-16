import logging
import os

READINESS_FILE = os.getenv('READINESS_FILE', '/tmp/ready')
LIVENESS_FILE = os.getenv('LIVENESS_FILE', '/tmp/alive')

_marked_ready = False


def mark_ready(force: bool = False):
    global _marked_ready
    if _marked_ready and not force:
        return

    with open(READINESS_FILE, 'w') as r:
        r.write("ready")

    if not _marked_ready:
        logging.info(f"Instance is ready")
    _marked_ready = True


def mark_alive():
    with open(LIVENESS_FILE, 'w') as r:
        r.write("alive")
