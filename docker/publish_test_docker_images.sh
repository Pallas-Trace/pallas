#!/bin/bash

REGISTRY="registry.gitlab.inria.fr"
URI="pallas/pallas"

docker login "${REGISTRY}"

docker build -f Dockerfile_pallas . -t "${REGISTRY}/${URI}":latest
docker push "${REGISTRY}/${URI}":latest

docker build -f Dockerfile_pallas_intel . -t "${REGISTRY}/${URI}"/intel:latest && \
docker push "${REGISTRY}/${URI}"/intel:latest