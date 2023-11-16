import re
from dataclasses import dataclass
from datetime import datetime, date, timezone
from typing import List


@dataclass
class File:
    ts: date
    days: int = 7
    ticker: str = 'btc'
    interval: str = '1min'
    ext: str = 'csv'

    @staticmethod
    def parse(file_name: str):
        m = re.match(r"(\S+)_last_(\d+)days_(\S+)_(\d{2})(\S+)\.(\S+)", file_name)
        groups = m.groups()
        ts = datetime.strptime(f"{datetime.now(tz=timezone.utc).year} {groups[3]} {groups[4]}", "%Y %d %B")
        return File(ts=ts.date(), days=int(groups[1]), ticker=groups[0], interval=groups[2], ext=groups[5])

    def name(self):
        date_num = str(self.ts.day).rjust(2, '0')
        month = self.ts.strftime('%B').lower()
        return f'{self.ticker}_last_{self.days}days_{self.interval}_{date_num}{month}.{self.ext}'


@dataclass
class ModelInputs:
    train: File
    validation: List[File]

