from itertools import cycle

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd


class TweetBasedPrediction:
    def __init__(self, df: pd.DataFrame, rolling_window: int, median_window: int, commission: float = 0):
        self._df = df
        self._rolling_window = rolling_window
        self._median_window = median_window
        self._commission = commission

        self._initial_usd = 0
        self._initial_crypto = 1.0
        self._current_crypto = self._initial_crypto
        self._current_usd = 0.0
        self._cycol = cycle('bgrcmk')
        self._deal_count = 0

    def plot_columns(self, y_names):
        fig, host = plt.subplots(figsize=(200, 20))

        host.set_xlabel('time')
        host.set_ylabel(y_names[0])

        p1, = host.plot(self._df[y_names[0]], color=next(self._cycol), label=y_names[0])
        host.yaxis.label.set_color(p1.get_color())
        main_patch = mpatches.Patch(color=p1.get_color(), label=y_names[0])

        position_cnt = 0.0
        color_cnt = 0.0
        patches = [main_patch]

        for col in y_names[1:]:
            color_cnt += 0.3
            position_cnt += 60

            graph = host.twinx()
            graph.set_ylabel(col)
            graph.spines['right'].set_position(('outward', position_cnt))

            if isinstance(col, list):
                for c in col:
                    color = next(self._cycol)
                    graph.yaxis.label.set_color(color)
                    graph.plot(self._df[c], color=color, label=c)
                    patches.append(mpatches.Patch(color=color, label=c))
            else:
                color = next(self._cycol)
                graph.yaxis.label.set_color(color)
                graph.plot(self._df[col], color=color, label=col)
                patches.append(mpatches.Patch(color=color, label=col))

        plt.legend(handles=patches)

    def buy_sell(self, x):
        if self._current_usd and x['buy_or_sell']:
            self._current_crypto = self._current_usd / x['close']
            self._current_usd = 0.0
        elif self._current_crypto and not x['buy_or_sell']:
            self._current_usd = x['close'] * self._current_crypto - self._commission * (x['close'] * self._current_crypto)
            self._current_crypto = 0.0
            self._deal_count += 1

        return self._current_usd + x['close'] * self._current_crypto

    def run(self):
        self._df['followers_sum_rolling'] = self._df['followers_sum'].rolling(window=self._rolling_window).mean()
        followers_max = self._df['followers_sum_rolling'].rolling(window=self._median_window).max()
        followers_min = self._df['followers_sum_rolling'].rolling(window=self._median_window).min()
        self._df['followers_median'] = followers_min + (followers_max - followers_min) / 2

        self._df['buy_or_sell'] = self._df.apply(lambda x: x['followers_sum_rolling'] >= x['followers_median'], axis=1)
        self._df['value'] = self._df.apply(self.buy_sell, axis=1)

        # plt.close()
        # plt.plot(self._df['close'], color='blue')
        # plt.plot(self._df['value'], color='red')
        #
        # try:
        #     plt.savefig(f'data/results/{self._rolling_window}x{self._median_window}_simple.png')
        # except:
        #     pass

        # plt.close()
        # self.plot_columns(['buy_or_sell',
        #                    ['followers_sum_rolling', 'followers_median'],
        #                    ['close', 'value'],
        #                    'followers_sum',
        #                    ])

        strat_profit = self._df['value'].iloc[-1] - self._df['close'].iloc[0]
        buy_hold_profit = self._df['close'].iloc[-1] - self._df['close'].iloc[0]
        print(f"{self._rolling_window},{self._median_window},"
              f"{int(strat_profit)},{int((strat_profit * 100) / self._df['close'].iloc[0])},"
              f"{int(buy_hold_profit)},{int((buy_hold_profit * 100) / self._df['close'].iloc[0])},"
              f"{self._deal_count},{int(strat_profit - buy_hold_profit)}")

        # try:
        #     plt.savefig(f'data/results/{self._rolling_window}x{self._median_window}_all.png')
        # except:
        #     pass


def main():
    df = pd.read_csv('data/raw.csv', parse_dates=["ts"], index_col='ts')
    df = df.loc['2020-12-01 00:00:00':'2021-02-22 00:00:00']

    print('x,y,strategy_return,strategy_percent,buy_hold_return,buy_hold_percent,deals,diff')
    # TweetBasedPrediction(df.copy(), 1440, 720, 0.001).run()
    for x in range(30, 1440 * 2, 30):
        for y in range(30, 1440 * 2, 30):
            TweetBasedPrediction(df.copy(), x, y, 0.001).run()


if __name__ == '__main__':
    main()
