from enum import IntEnum

from mongoengine import Document, StringField, DateTimeField, BooleanField, IntField, DecimalField, \
    ListField, EnumField, FloatField


class TweetClass(IntEnum):
    Skip = -1
    Opinion = 0
    News = 1


class TweetSentiment(IntEnum):
    Negative = -1
    Neutral = 0
    Positive = 1


class TwitterAccount(Document):
    name = StringField(required=True)
    followers = IntField()
    meta = {'indexes': ['name', 'followers']}


class Tweet(Document):
    tweet_id = IntField(required=True)
    text = StringField(required=True)
    ts = DateTimeField(required=True)
    retweet = BooleanField()
    user_name = StringField(required=True)
    tokens = ListField(StringField())
    classification = EnumField(TweetClass)
    sentiment = EnumField(TweetSentiment)
    blob_polarity = FloatField()
    blob_objectivity = FloatField()
    meta = {'strict': False, 'indexes': ['ts', 'tweet_id']}


class Price(Document):
    ts = DateTimeField(required=True)
    open = DecimalField(required=True)
    close = DecimalField(required=True)
    low = DecimalField(required=True)
    high = DecimalField(required=True)
    volume = DecimalField(required=True)
    meta = {'indexes': ['ts']}


class DataPoint(Document):
    ts = DateTimeField(required=True)
    open = DecimalField(required=True)
    close = DecimalField(required=True)
    low = DecimalField(required=True)
    high = DecimalField(required=True)
    volume = DecimalField(required=True)

    polarity_avg = DecimalField(required=True, default=0.0)
    polarity_sum = DecimalField(required=True, default=0.0)
    polarity_scaled_sum = DecimalField(required=True, default=0.0)
    objectivity_avg = DecimalField(required=True, default=0.0)
    objectivity_sum = DecimalField(required=True, default=0.0)
    followers_sum = DecimalField(required=True, default=0.0)
    objectivity_scaled_sum = DecimalField(required=True, default=0.0)
    tweet_count = DecimalField(required=True, default=0.0)

    meta = {'indexes': ['ts']}
