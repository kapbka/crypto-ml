import datetime
import logging
import time
from typing import List

from common.constants import MIN_DATE, PRICE_INTERVAL_MINUTES
from mining.db.api import connect
from mining.db.models import Tweet, Price, DataPoint


def get_tweets(date: datetime.datetime):
    end = date + datetime.timedelta(days=1)
    logging.info(f"Loading tweets {date} -> {end}")

    pipeline = [
        {
            "$match": {
                "ts": {
                    "$gte": date,
                    "$lt": end
                }
            },
        },
        {
            "$lookup": {
                "from": "twitter_account",
                "localField": "user_name",
                "foreignField": "name",
                "as": "account_info"
            }
        },
        {
            "$sort": {
                "ts": 1
            }
        }
    ]
    res = list(Tweet.objects.aggregate(*pipeline, allowDiskUse=True))
    logging.info(f"Loaded {len(res)} tweets")
    return res


def update_stats(tweets: List[dict], res: DataPoint):
    for t in tweets:
        followers = (t.get('account_info') or [{'followers': 1}])[-1]['followers']

        res.polarity_avg += t.get('blob_polarity', 0)
        res.polarity_sum += t.get('blob_polarity', 0)
        res.polarity_scaled_sum += t.get('blob_polarity', 0) * followers
        res.objectivity_avg += t.get('blob_objectivity', 0)
        res.objectivity_sum += t.get('blob_objectivity', 0)
        res.objectivity_scaled_sum += t.get('blob_objectivity') * followers

        res.followers_sum += followers

    if tweets:
        res.polarity_avg = res.polarity_avg / len(tweets)
        res.objectivity_avg = res.objectivity_avg / len(tweets)

    res.tweet_count = len(tweets)


def dataset_generator(last_dp: datetime.datetime):
    counter = 0

    tweets_cache = []
    last_date = None
    last_price_dt = None

    logging.debug(f"Loading prices from {last_dp}")
    prices = list(Price.objects(ts__gt=last_dp).order_by('ts'))
    if prices:
        logging.info(f"Loaded {len(prices)} prices for {last_dp}")

    for price in prices:
        if price.ts.date() != last_date:
            last_date = price.ts.date()
            tweets_cache.extend(get_tweets(datetime.datetime(last_date.year, last_date.month, last_date.day)))

        if last_price_dt == price.ts:
            logging.debug(f"Skipping duplicated price {last_price_dt}")
            continue

        last_price_dt = price.ts

        counter += 1
        if counter % 1000 == 0:
            logging.info(f"Processed {counter} prices, last: {price.ts}, cache size: {len(tweets_cache)}, "
                         f"progress: {int((counter * 100) / len(prices))}%")

        from_dt = price.ts - datetime.timedelta(minutes=1)
        tweets = []
        while tweets_cache and tweets_cache[0]['ts'] < price.ts:
            if tweets_cache[0]['ts'] >= from_dt:
                tweets.append(tweets_cache[0])
            del tweets_cache[0]

        logging.debug(f"Processing {len(tweets)} tweets for {price.ts}")

        point = DataPoint()
        update_stats(tweets, point)

        point.ts = price.ts
        point.open = price.open
        point.close = price.close
        point.low = price.low
        point.high = price.high
        point.volume = price.volume
        yield point


def main():
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(name)s:%(message)s', level=logging.INFO)

    connect()

    while True:
        # find last available datapoint
        objs = DataPoint.objects().order_by('-ts').limit(1)
        if objs:
            last_dp = objs[0].ts
        else:
            last_dp = MIN_DATE - datetime.timedelta(minutes=PRICE_INTERVAL_MINUTES)

        if datetime.datetime.now() - last_dp > datetime.timedelta(minutes=PRICE_INTERVAL_MINUTES):
            for obj in dataset_generator(last_dp):
                obj.save()

        time.sleep(10)


if __name__ == '__main__':
    main()
