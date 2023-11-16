#!/bin/bash

set -ex

IMAGE_NAME=clrn/ml:latest
SERVICES=${AUTO_UPDATE_SERVICES}

while true;
do
  docker pull ${IMAGE_NAME}
  docker run -i ${IMAGE_NAME} bash -c "cat ./docker-compose.yaml" > ./docker-compose.yaml
  docker-compose -p clrn-ml up -d "${SERVICES}"
  sleep 60
done
