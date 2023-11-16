import json
import logging
import sys

import pandas as pd

IGNORED_COLUMNS = ['polarity_avg', 'polarity_sum', 'polarity_scaled_sum', 'objectivity_avg', 'objectivity_sum',
                   'objectivity_scaled_sum', 'tweet_count', '_id']


def all_records():
    for line in sys.stdin:
        if line:
            record = json.loads(line)
            record["ts"] = record["ts"]["$date"][:-1]
            for c in IGNORED_COLUMNS:
                record.pop(c)
            yield record


def main():
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(name)s:%(message)s', level=logging.INFO)

    logging.info(f"Started")
    df = pd.DataFrame(all_records())

    logging.info(f"Sorting")
    df.sort_values(by=['ts'], inplace=True)

    logging.info(f"Saving to data/data_points.csv")
    df.to_csv('data/data_points.csv')

    logging.info(f"Completed: \n{df}")


if __name__ == '__main__':
    main()
