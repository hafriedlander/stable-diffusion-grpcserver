#!/bin/bash

git submodule update --init --recursive
docker run --rm -it -v `pwd`:/src $(docker build -q . -f Dockerfile.protoc) bash /build_protoc.sh
