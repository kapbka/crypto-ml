import logging
import time
from datetime import datetime
from typing import List, Optional

import stweet as st
from arrow import Arrow
from textblob import TextBlob

from common.constants import MIN_DATE
from common.tokenizer import tokenize
from mining.db.api import connect
from mining.db.models import TwitterAccount, Tweet


class RunCallback(st.TweetOutput):
    def __init__(self, callback):
        self._cb = callback
        self.counter = 0

    def export_tweets(self, tweets: List[st.Tweet]):
        for t in tweets:
            self._cb(t)
            self.counter += 1


def fetch_tweets(query: str, callback: callable, since: Optional[datetime], until: Optional[datetime]):
    logging.info(f"Fetching tweets since {since} until {until}, query: '{query}'")

    search_tweets_task = st.SearchTweetsTask(
        all_words=query,
        language=st.Language.ENGLISH,
        since=Arrow.fromdatetime(since) if since else None,
        until=Arrow.fromdatetime(until) if until else None,
    )

    cb = RunCallback(callback)
    st.TweetSearchRunner(
        search_tweets_task=search_tweets_task,
        tweet_outputs=[cb]
    ).run()

    logging.info(f"Fetch completed since {since} until {until}, query: '{query}', collected: {cb.counter}")


class TweetProcessor:
    def __init__(self):
        self._counter = 0

    def handle_tweet(self, tweet: st.Tweet):
        TwitterAccount.objects(name=tweet.user_name).update_one(set__name=tweet.user_name, upsert=True)

        tokens = tokenize(tweet.full_text)
        blob = TextBlob(" ".join(tokens))
        Tweet.objects(tweet_id=int(tweet.id_str)).update_one(set__tweet_id=int(tweet.id_str),
                                                             set__text=tweet.full_text,
                                                             set__ts=tweet.created_at.datetime,
                                                             set__retweet=tweet.retweeted,
                                                             set__user_name=tweet.user_name,
                                                             set__tokens=tokens,
                                                             set__blob_polarity=blob.sentiment.polarity,
                                                             set__blob_objectivity=blob.sentiment.subjectivity,
                                                             upsert=True)
        self._counter += 1
        if self._counter % 1000 == 0:
            logging.info(f"Processed {self._counter} tweets, last dt: {tweet.created_at}")


def fetch_all_tweets(processor: TweetProcessor):
    min_dt = None
    objs = Tweet.objects().order_by('ts').limit(1)
    if objs:
        min_dt = objs[0].ts

    max_dt = None
    objs = Tweet.objects().order_by('-ts').limit(1)
    if objs:
        max_dt = objs[0].ts

    query = 'bitcoin'
    if min_dt and max_dt:
        fetch_tweets(query, processor.handle_tweet, since=max_dt, until=None)
        if min_dt != MIN_DATE:
            fetch_tweets(query, processor.handle_tweet, since=MIN_DATE, until=min_dt)
    else:
        fetch_tweets(query, processor.handle_tweet, since=MIN_DATE, until=None)


def main():
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(name)s:%(message)s', level=logging.INFO)

    connect()

    processor = TweetProcessor()
    while True:
        try:
            fetch_all_tweets(processor)
        except:
            logging.exception(f"Failed to fetch tweets")

        time.sleep(60)


if __name__ == '__main__':
    main()
