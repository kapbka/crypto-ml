import argparse
import sys

from common.log import setup_logging
from db.tools.run_sum import upload_run_sum


def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="This script runs running sum calculation")
    parser.add_argument("-v", "--verbosity", type=str, default='INFO', help="verbosity level: INFO/DEBUG")

    args = parser.parse_args(args)
    setup_logging(args.verbosity, filename='/tmp/running_sum.log')

    upload_run_sum(number_of_batches=3)


if __name__ == '__main__':
    main()
