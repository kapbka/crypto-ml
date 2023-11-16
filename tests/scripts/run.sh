#!/bin/bash

set -ex

SRC_DIR="$(dirname $(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd))"
ROOT_DIR=$(dirname "${SRC_DIR}")
ENV_FILE="${ROOT_DIR}/.env-test"
COMPOSE="docker-compose -p crypto-ml-tests --env-file=${ENV_FILE}"

echo Using env file: "${ENV_FILE}"

function clean {
    rc=$?
    ${COMPOSE} logs
    [[ $rc == 0 ]] && ${COMPOSE} down -t 0 -v --remove-orphans
    [[ $rc == 0 ]] && echo "All tests passed!" || echo "Tests failed! Check logs above."
    exit $rc
}

trap clean EXIT

BOT_START_TIME=$(python3 -c "import datetime; print((datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y-%m-%d'))")
TEST_DATA_START_TIME=$(python3 -c "import datetime; print((datetime.datetime.now() - datetime.timedelta(days=2)).isoformat())")
export BOT_START_TIME

${COMPOSE} down -t 0 -v --remove-orphans

${COMPOSE} run -T unit-tests

# clean everything after running unit tests
${COMPOSE} down -t 0 -v --remove-orphans

${COMPOSE} run version-import

# run new trainer to produce 3 individuals
while true; do
  ${COMPOSE} run trainer /src/tools/trainer.py --currency btc --cells 1 --std-scaler --epoch 3 --offsets 0,1 --end "${TEST_DATA_START_TIME}" --save-profit -100 --labels "0.001,60" --crash-params "1,1,10"
  num=$(${COMPOSE} run psql -tc "SELECT count(1) FROM individual;" | tr -d '[:space:]')
  echo "Got $num individuals"
  [[ $num -ge 3 ]] && break
done

INDIVIDUAL_MD5=$(${COMPOSE} run psql -tc "SELECT md5 FROM individual ORDER BY id LIMIT 1;" | tr -d '[:space:]')

# drop production intervals to make sure we process only last week
${COMPOSE} run psql -tc "delete from run_sum_interval where code != 'week';"

# create portfolios and enable best one, that one waits for deal-worker to generate deals first
${COMPOSE} run rotate-portfolio

# this is to make sure actions left from previous deal worker are gone
${COMPOSE} stop deals-worker
${COMPOSE} run psql -tc "truncate table action;"

# everything is up and running
${COMPOSE} run all-ready

# run evaluator on trained individual
${COMPOSE} run local-evaluator /src/tools/local_eval.py "${INDIVIDUAL_MD5}" --version 5 --graphs --deals --realtime-offset 10

${COMPOSE} down

${COMPOSE} run coverage | grep -v -e CoverageWarning -e _warn | sed "s/\/src/${ROOT_DIR//\//\\/}/g" > coverage.xml
${COMPOSE} run coverage "cat .coverage" > coverage.coverage
${COMPOSE} run coverage "coverage report -i"
