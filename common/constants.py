import os
from datetime import date, datetime, timezone
from enum import IntEnum

from common.storage.file import File

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
MIN_DATE = datetime(2021, 4, 1, tzinfo=timezone.utc)
PRICE_INTERVAL_MINUTES = 1
DEAL_COEFFICIENT = 500  # (number of minutes) / DEAL_COEFFICIENT = expected deal count
COMMISSION = 0.001
DEFAULT_CURRENCY = 'btc'
TRAIN_FILE = File(ts=date(2021, 9, 1), days=90, ticker=DEFAULT_CURRENCY)
LABEL_COLUMNS = ['close', 'high', 'low']
CURRENCIES = ['btc', 'eth', 'bnb', 'sol', 'ada', 'xrp', 'doge']
INDICATOR_LENGTHS = [30, 60, 60 * 2, 60 * 4, 60 * 10, 60 * 24]
CURRENT_VERSION = 5


DB_CONNECTION = f"postgresql://postgres:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:5432/{os.getenv('POSTGRES_DB')}"
RMQ_CONNECTION = f"amqp://{os.getenv('RABBITMQ_USERNAME')}:{os.getenv('RABBITMQ_PASSWORD')}@{os.getenv('RABBITMQ_HOST')}"
ZK_CONNECTION = os.getenv('ZK_HOST')
GRAFANA_SERVER = "https://grafana.clrn.dev"

RMQ_EXCHANGE_NAME = 'crypto-ml'

TELEGRAM_USERS = [220583423, 365436333]
TELEGRAM_NOTIFY_CHATS = TELEGRAM_USERS


class EvaluationMethod(IntEnum):
    ClosePrice = 0      # use current close price only
    CloseLO = 1         # set up Limit Orders and check close price
    HighLowLO = 2       # set up Limit Orders and use high and low prices
    CleanLOPrice = 3    # clean next limit order price on each step
    CappedLOPrice = 4   # cap next limit order price
    OCO = 5             # use OCO order for selling


VERSION_TO_METHOD = {
    0: EvaluationMethod.CloseLO,
    1: EvaluationMethod.ClosePrice,
    2: EvaluationMethod.HighLowLO,
    3: EvaluationMethod.CleanLOPrice,
    4: EvaluationMethod.CappedLOPrice,
    5: EvaluationMethod.OCO,
}


class DealStatus(IntEnum):
    Open = 0
    Close = 1


class RealtimeStatus(IntEnum):
    Disabled = 0
    Enabled = 1


class HistoryStatus(IntEnum):
    Disabled = 0
    Enabled = 1


class OptimizedStatus(IntEnum):
    Disabled = 0
    Enabled = 1


class RunSumIntervalType(IntEnum):
    FromNow = 0
    FromTo = 1


class RunSumStatus(IntEnum):
    InProgress = 0
    Done = 1


class HistoryStatStatus(IntEnum):
    InProgress = 0
    Done = 1


class WriteAttributeHistoryStatus(IntEnum):
    Off = 0
    On = 1


class IsPortfolio(IntEnum):
    No = 0
    Yes = 1
