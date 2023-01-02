#!/bin/bash

docker run --rm -it -v `pwd`:/src $(docker build -q . -f Dockerfile.protoc) bash /build_protoc.sh
