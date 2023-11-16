import logging
import time

from twitter import Api

from mining.db.api import connect
from mining.db.models import TwitterAccount


def get_api() -> Api:
    return Api(consumer_key='LSNo4FSxOSos5VwCYxaCVvL5p',
               consumer_secret='NdhaEPJ1Ve4qgXdG9FRXSAhd5AF0RRPZJHOkaUP6GiK1C8f7pv',
               access_token_key='195113015-uBqgboRPgL5cTIFrMJBNn5e1sJVSxkhw9F0XqquI',
               access_token_secret='ANkb1UggmoSqokOkrImjutV3wlTFxyCylX3SjNUIGcy0O',
               sleep_on_rate_limit=True)


def main():
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(name)s:%(message)s', level=logging.INFO)
    connect()

    api = get_api()
    while True:
        accs = [a for a in TwitterAccount.objects(followers=None).limit(100)]
        if not accs:
            time.sleep(60)
            continue

        success_count = 0
        failed_count = 0
        try:
            users = api.UsersLookup(screen_name=[a.name for a in accs])
            counts = {u.screen_name: u.followers_count for u in users}
            for acc in accs:
                number = counts.get(acc.name, 1)
                acc.followers = number
                acc.set()
                success_count += 1
        except:
            logging.error(f"Failed to fetch followers_count for {len(accs)} users")
            for acc in accs:
                acc.followers = 1
                acc.set()
                failed_count += 1

        if success_count or failed_count:
            logging.info(f"Updated {success_count} successfully, {failed_count} failed")


if __name__ == '__main__':
    main()