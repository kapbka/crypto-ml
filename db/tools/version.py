import argparse
import sys

from common.log import setup_logging
from db.api import DB


def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="This script imports a new version to the DB")
    parser.add_argument("-v", "--verbosity", type=str, default='INFO', help="verbosity level: INFO/DEBUG")
    parser.add_argument("--comment", type=str, help=f"comment for the version used", default="")
    parser.add_argument("--version", type=int, help=f"version id to use", required=True)
    parser.add_argument("--name", type=str, help=f"name for the version used", required=True)

    args = parser.parse_args(args)
    setup_logging(args.verbosity)

    with DB() as db:
        db.version.set(version_num=args.version, name=args.name, comment=args.comment)


if __name__ == "__main__":
    main()
