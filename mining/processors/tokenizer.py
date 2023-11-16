import logging
import multiprocessing
import time
from itertools import product
from typing import Tuple

from common.tokenizer import tokenize
from mining.db.api import connect
from mining.db.models import Tweet


def tokenize_tweet(tweet: Tuple[int, str]):
    return tweet[0], tokenize(tweet[1])


def main():
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(name)s:%(message)s', level=logging.INFO)

    connect()

    counter = 0
    while True:
        tweets = [(t.id, t.text) for t in Tweet.objects(tokens=None).limit(1000)]
        logging.info(f"Received {len(tweets)} tweets")
        if not tweets:
            time.sleep(60)
            continue

        with multiprocessing.Pool(processes=4) as pool:
            results = pool.starmap(tokenize_tweet, product(tweets))

        counter += len(results)
        logging.info(f"Processed {counter} tweets")

        for tweet_id, tokens in results:
            Tweet.objects(id=tweet_id).update_one(set__tokens=tokens)


if __name__ == '__main__':
    main()