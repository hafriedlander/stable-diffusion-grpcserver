#!/bin/bash

# Generate the python bindings

python -m grpc_tools.protoc \
  -Iapi-interfaces/src/tensorizer/proto \
  -Iapi-interfaces/src/proto \
  --python_out=gyre/generated \
  --grpc_python_out=gyre/generated \
  --mypy_out=gyre/generated \
  api-interfaces/src/tensorizer/proto/tensors.proto \
  api-interfaces/src/proto/generation.proto \
  api-interfaces/src/proto/engines.proto \
  api-interfaces/src/proto/dashboard.proto

# Generate the OpenAPI 2.0 spec from the proto files

protoc \
  -Iapi-interfaces/src/tensorizer/proto -Iapi-interfaces/src/proto \
  --openapiv2_out gyre/generated \
  --openapiv2_opt logtostderr=true \
  --openapiv2_opt openapi_naming_strategy=simple,simple_operation_ids=true \
  --openapiv2_opt allow_merge=true,merge_file_name=stablecabal \
  --openapiv2_opt Mengines.proto=example.com/fix_package_bug \
  --openapiv2_opt grpc_api_configuration=openapiconfig.yaml \
   api-interfaces/src/proto/generation.proto api-interfaces/src/proto/engines.proto

# This opt would expose all Messages - generate_unbound_methods=true

# And convert to OpenAPI 3.0

swagger2openapi gyre/generated/stablecabal.swagger.json > gyre/generated/stablecabal.openapi.json