#!/bin/bash

git submodule update --init --recursive
python -m grpc_tools.protoc -Iapi-interfaces/src/proto --python_out=generated --grpc_python_out=generated api-interfaces/src/proto/engines.proto
python -m grpc_tools.protoc -Iapi-interfaces/src/proto --python_out=generated --grpc_python_out=generated api-interfaces/src/proto/dashboard.proto
python -m grpc_tools.protoc -Iapi-interfaces/src/proto --python_out=generated --grpc_python_out=generated api-interfaces/src/proto/generation.proto
python -m grpc_tools.protoc -Iapi-interfaces/src/proto --python_out=generated --grpc_python_out=generated api-interfaces/src/proto/completion.proto
