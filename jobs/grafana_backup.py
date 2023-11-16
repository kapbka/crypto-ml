import sys
import logging
from common.log import setup_logging
from grafana.tools.backup import backup


if __name__ == '__main__':
    setup_logging(filename='/tmp/grafana_backup.log')
    logging.info('start grafana backup')
    backup(sys.argv[1])
    logging.info('end grafana backup')
