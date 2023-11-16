import pandas as pd
import talib


class MACDBasedPrediction:
    def __init__(self, df: pd.DataFrame, fast: int, slow: int, signal: int, commission: float = 0):
        self._df = df
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self._commission = commission

        self._initial_usd = 0
        self._initial_crypto = 1.0
        self._current_crypto = self._initial_crypto
        self._current_usd = 0.0
        self._deal_count = 0

    def buy_sell(self, x):
        if self._current_usd and x['buy_or_sell'] > 0:
            self._current_crypto = self._current_usd / x['close']
            self._current_usd = 0.0
        elif self._current_crypto and x['buy_or_sell'] < 0:
            self._current_usd = x['close'] * self._current_crypto - self._commission * (x['close'] * self._current_crypto)
            self._current_crypto = 0.0
            self._deal_count += 1

        return self._current_usd + x['close'] * self._current_crypto

    def run(self):
        macd, macdsignal, macdhist = talib.MACD(self._df['close'], fastperiod=self.fast, slowperiod=self.slow,
                                                signalperiod=self.signal)
        self._df['macd'] = macd
        self._df['macdsignal'] = macdsignal
        self._df['macdhist'] = macdhist
        self._df['buy_or_sell'] = self._df.apply(lambda x: x['macd'] - x['macdsignal'], axis=1)
        self._df['value'] = self._df.apply(self.buy_sell, axis=1)

        strat_profit = self._df['value'].iloc[-1] - self._df['close'].iloc[0]
        buy_hold_profit = self._df['close'].iloc[-1] - self._df['close'].iloc[0]
        print(f"{self.fast},{self.slow},{self.signal},"
              f"{int(strat_profit)},{int((strat_profit * 100) / self._df['close'].iloc[0])},"
              f"{int(buy_hold_profit)},{int((buy_hold_profit * 100) / self._df['close'].iloc[0])},"
              f"{self._deal_count},{int(strat_profit - buy_hold_profit)}")


def main():
    df = pd.read_csv('data/raw.csv', parse_dates=["ts"], index_col='ts')
    # df = df.loc['2020-12-01 00:00:00':'2021-02-22 00:00:00']
    df = df.loc['2021-01-28 00:00:00':'2021-01-30 00:00:00']

    print('fast,slow,signal,strategy_return,strategy_percent,buy_hold_return,buy_hold_percent,deals,diff')
    # MACDBasedPrediction(df.copy(), 12, 26, 9).run()
    for x in range(2, 18):
        for y in range(22, 34):
            for z in range(2, 20):
                MACDBasedPrediction(df.copy(), x, y, z).run()


if __name__ == '__main__':
    main()
