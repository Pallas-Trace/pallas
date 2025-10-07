#!/bin/bash
REGISTRY="registry.gitlab.inria.fr"
PATH="pallas/pallas"

docker login "${REGISTRY}"

docker build -f Dockerfile_pallas . -t "${PATH}":latest
docker tag "${PATH}":latest "${REGISTRY}/${PATH}":latest
docker push "${REGISTRY}/${PATH}":latest

docker build -f Dockerfile_pallas_intel . -t "${PATH}"/intel:latest
docker tag "${PATH}"/intel:latest "${REGISTRY}/${PATH}"/intel:latest
docker push "${REGISTRY}/${PATH}"/intel:latest
