import argparse
import asyncio
import logging
import sys
from datetime import datetime
from functools import partial
from typing import List, Tuple

import tensorflow as tf
from funcy import log_durations

from common.deal_generator import DealGenerator
from common.interrupt import InterruptionHandler
from common.k8s.probes import mark_ready, mark_alive
from common.log import setup_logging
from common.metrics.evaluator import Limits
from common.queues.postgres import RPCHandler, FatalError
from db.api import DB
from db.model import Deal, IndividualAttribute

generators = dict()


@log_durations(logging.debug)
def get_deals(db: DB, attributes: IndividualAttribute, to_ts: datetime) -> Tuple[List[Deal], Limits]:
    key = attributes.individual_id, attributes.currency_code, attributes.version_id, attributes.scaler_id
    if key not in generators:
        generators[key] = DealGenerator(db=db, attributes=attributes)

    try:
        deals, limits = generators[key].process_pending(ts_to=to_ts)
        if deals:
            logging.debug(f"New {len(deals)} deals for {attributes}, last: {deals[-1]}")
        mark_alive()
        return deals, limits
    except tf.errors.InternalError:
        raise FatalError("Critical error, terminating the worker")


async def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="This script runs a worker that generates deals by running prices through an ANN")
    parser.add_argument("-v", "--verbosity", type=str, default='INFO', help="verbosity level: INFO/DEBUG")

    args = parser.parse_args(args)
    setup_logging(args.verbosity, filename='/tmp/ann-worker.log')

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        logging.info(f"GPU: {gpus[0]}")
        tf.config.experimental.set_memory_growth(gpus[0], True)

    with DB() as db, InterruptionHandler() as guard, RPCHandler(db, guard, 'get_deals', partial(get_deals, db)) as h:
        mark_ready()
        await h.run()


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())
