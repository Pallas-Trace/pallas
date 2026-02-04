#!/bin/bash

NAME=pallas_eztrace_example
docker build -f Dockerfile_eztrace_pallas . -t "$NAME":latest

docker login registry.gitlab.inria.fr
docker tag "$NAME":latest registry.gitlab.inria.fr/pallas/pallas/eztrace_example:latest
docker push registry.gitlab.inria.fr/pallas/pallas/eztrace_example:latest

docker login
docker tag "$NAME":latest eztrace/pallas:latest
docker push eztrace/pallas:latest