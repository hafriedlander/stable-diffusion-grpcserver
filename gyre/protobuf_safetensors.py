from sdgrpcserver.generated.generation_pb2 import (
    Safetensors,
    SafetensorsMeta,
    SafetensorsTensor,
)
from sdgrpcserver.protobuf_tensors import deserialize_tensor, serialize_tensor


def serialize_safetensor(safetensors):
    proto_safetensors = Safetensors()

    for k, v in safetensors.metadata().items():
        proto_safetensors.metadata.append(SafetensorsMeta(key=k, value=v))

    for k in safetensors.keys():
        proto_safetensors.tensors.append(
            SafetensorsTensor(key=k, tensor=serialize_tensor(safetensors.get_tensor(k)))
        )

    return proto_safetensors


class FakeSafetensors:
    def __init__(self, metadata, tensors):
        self._metadata = metadata
        self._tensors = tensors

    def metadata(self):
        return self._metadata

    def keys(self):
        return self._tensors.keys()

    def get_tensor(self, key):
        return self._tensors[key]


def deserialize_safetensors(proto_safetensors):
    metadata = {}
    tensors = {}

    for meta in proto_safetensors.metadata:
        metadata[meta.key] = meta.value

    for tensor in proto_safetensors.tensors:
        tensors[tensor.key] = deserialize_tensor(tensor.tensor)

    return FakeSafetensors(metadata, tensors)
