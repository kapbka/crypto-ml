import matplotlib.pyplot as plt
import pandas as pd


def main():
    df = pd.read_csv('data/raw.csv', parse_dates=["ts"])
    data = df.tail(60 * 24 * 2)
    for feature_name in ['close', 'polarity_scaled_sum',
                         'polarity_avg', 'objectivity_scaled_sum',
                         'tweet_count', 'polarity_sum']:
        max_value = data[feature_name].max()
        min_value = data[feature_name].min()
        data[feature_name] = (data[feature_name] - min_value) / (max_value - min_value)
    # plt.title('120 -> 120')
    # plt.scatter(data['future_60_change'],
    #             data['past_60_word_bitcoin'])
    # plt.scatter(data['future_60_change'],
    #             data['past_60_word_buy'])
    # plt.scatter(data['future_60_change'],
    #             data['past_60_word_sell'])
    plt.plot(data['ts'], data['polarity_scaled_sum'])
    # plt.plot(data['ts'], data['tweet_count'])
    plt.plot(data['ts'], data['close'])
    plt.show()


if __name__ == '__main__':
    main()