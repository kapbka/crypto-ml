# current Makefile and its directory
MAKEFILE := $(abspath $(lastword $(MAKEFILE_LIST)))
ROOT := $(dir $(MAKEFILE))

include $(ROOT)/.env

REGISTRY := docker.clrn.dev
REQUIREMENTS_HASH := $(shell md5sum requirements.txt | cut -c 1-6)
MAIN_IMAGE := $(REGISTRY)/crypto-ml
GPU_IMAGE := $(REGISTRY)/crypto-ml-gpu
WAIT_IMAGE := $(REGISTRY)/waiter
PG_IMAGE := $(REGISTRY)/postgres

export REQUIREMENTS_HASH


build-base-tests:
	@echo Requirements hash is $(REQUIREMENTS_HASH), building if does not exist
	@docker pull docker.clrn.dev/python-base:$(REQUIREMENTS_HASH) || docker-compose build python-base
	@docker pull docker.clrn.dev/python-base-tests:$(REQUIREMENTS_HASH) || docker-compose build python-base-tests

build-base: build-base-tests
	@echo Requirements hash is $(REQUIREMENTS_HASH), building if does not exist
	@docker pull docker.clrn.dev/python-base-gpu:$(REQUIREMENTS_HASH) || docker-compose build python-base-gpu

push-base:
	@docker-compose push python-base
	@docker-compose push python-base-gpu
	@docker-compose push python-base-tests

build-tests: build-base-tests
	@docker-compose build postgres crypto-ml waiter
	@docker-compose build crypto-ml-tests
	@docker-compose build crypto-ml-coverage

build: build-base
	@docker-compose build postgres waiter crypto-ml crypto-ml-gpu
	@docker-compose build crypto-ml-tests
	@docker-compose build crypto-ml-coverage

test: build-tests
	@bash ./tests/scripts/run.sh

push: push-base
	@docker-compose push postgres waiter crypto-ml crypto-ml-gpu

deploy:
	@cd k8s && kustomize edit set image $(MAIN_IMAGE)=$(shell docker inspect --format='{{index .RepoDigests 0}}' $(MAIN_IMAGE))
	@cd k8s && kustomize edit set image $(WAIT_IMAGE)=$(shell docker inspect --format='{{index .RepoDigests 0}}' $(WAIT_IMAGE))
	@cd k8s && kustomize edit set image $(PG_IMAGE)=$(shell docker inspect --format='{{index .RepoDigests 0}}' $(PG_IMAGE))
	@cd k8s && kustomize edit set image $(GPU_IMAGE)=$(shell docker inspect --format='{{index .RepoDigests 0}}' $(GPU_IMAGE))
	@cd k8s && kustomize build | kubectl -n prod apply -f -
