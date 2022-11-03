#!/bin/bash

python -m grpc_tools.protoc -Isdgrpcserver/proto -Iapi-interfaces/src/tensorizer/proto --python_out=sdgrpcserver/generated --grpc_python_out=sdgrpcserver/generated api-interfaces/src/tensorizer/proto/tensors.proto
python -m grpc_tools.protoc -Isdgrpcserver/proto -Iapi-interfaces/src/proto -Iapi-interfaces/src/tensorizer/proto --python_out=sdgrpcserver/generated --grpc_python_out=sdgrpcserver/generated api-interfaces/src/proto/generation.proto
python -m grpc_tools.protoc -Isdgrpcserver/proto -Iapi-interfaces/src/proto --python_out=sdgrpcserver/generated --grpc_python_out=sdgrpcserver/generated api-interfaces/src/proto/engines.proto
python -m grpc_tools.protoc -Isdgrpcserver/proto -Iapi-interfaces/src/proto --python_out=sdgrpcserver/generated --grpc_python_out=sdgrpcserver/generated api-interfaces/src/proto/dashboard.proto

