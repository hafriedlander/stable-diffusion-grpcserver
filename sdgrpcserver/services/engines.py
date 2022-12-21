import inspect

import engines_pb2
import engines_pb2_grpc
import generation_pb2

from sdgrpcserver.pipeline.samplers import sampler_properties
from sdgrpcserver.services.exception_to_grpc import exception_to_grpc


class EnginesServiceServicer(engines_pb2_grpc.EnginesServiceServicer):
    def __init__(self, manager):
        self._manager = manager

    @exception_to_grpc
    def ListEngines(self, request, context):
        engines = engines_pb2.Engines()

        status = self._manager.getStatus()
        for engine in self._manager.engines:
            if not (
                engine.get("id", False)
                and engine.get("enabled", False)
                and engine.get("visible", False)
            ):
                continue

            info = engines_pb2.EngineInfo()
            info.id = engine["id"]
            info.name = engine["name"]
            info.description = engine["description"]
            info.owner = "stable-diffusion-grpcserver"
            info.ready = status.get(engine["id"], False)
            info.type = engines_pb2.EngineType.PICTURE

            fqclass_name = engine.get("class", "UnifiedPipeline")
            class_obj = self._manager._import_class(fqclass_name)

            for sampler in sampler_properties(
                include_diffusers=getattr(class_obj, "_diffusers_capable", True),
                include_kdiffusion=getattr(class_obj, "_kdiffusion_capable", False),
            ):
                info.supported_samplers.append(engines_pb2.EngineSampler(**sampler))

            engines.engine.append(info)

        return engines
