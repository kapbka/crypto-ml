import logging
import multiprocessing
import time
from itertools import product
from typing import Tuple, List

from textblob import TextBlob

from mining.db.api import connect
from mining.db.models import Tweet


def analyze_tweet(tweet: Tuple[int, List[str]]):
    blob = TextBlob(" ".join(tweet[1]))
    return tweet[0], blob.sentiment.polarity, blob.sentiment.subjectivity


def main():
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(name)s:%(message)s', level=logging.INFO)

    connect()

    counter = 0
    while True:
        tweets = [(t.id, t.tokens) for t in Tweet.objects(blob_polarity=None).limit(10000)]
        if not tweets:
            time.sleep(60)
            continue

        with multiprocessing.Pool(processes=8) as pool:
            results = pool.starmap(analyze_tweet, product(tweets))

        counter += len(results)
        logging.info(f"Processed {counter} tweets")

        for tweet_id, polarity, subjectivity in results:
            Tweet.objects(id=tweet_id).update_one(set__blob_polarity=polarity,
                                                  set__blob_objectivity=subjectivity)


if __name__ == '__main__':
    main()