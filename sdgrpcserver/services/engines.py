import inspect

import engines_pb2
import engines_pb2_grpc
import generation_pb2


class EnginesServiceServicer(engines_pb2_grpc.EnginesServiceServicer):
    def __init__(self, manager):
        self._manager = manager

    def ListEngines(self, request, context):
        engines = engines_pb2.Engines()

        all_noise_types = [
            generation_pb2.SAMPLER_NOISE_NORMAL,
            generation_pb2.SAMPLER_NOISE_BROWNIAN,
        ]
        normal_only = [generation_pb2.SAMPLER_NOISE_NORMAL]

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

            if info.ready:
                pipeline = self._manager._pipelines[engine["id"]]
                for k, v in pipeline.get_samplers().items():
                    if callable(v):
                        args = set(inspect.signature(v).parameters.keys())

                        info.supported_samplers.append(
                            engines_pb2.EngineSampler(
                                sampler=k,
                                supports_eta="eta" in args,
                                supports_churn="churn" in args,
                                supports_sigma_limits="sigmas" in args
                                or "sigma_min" in args,
                                supports_karras_rho="sigmas" in args,
                                supported_noise_types=all_noise_types
                                if "noise_sampler" in args
                                else normal_only,
                            )
                        )
                    else:
                        args = set(inspect.signature(v.step).parameters.keys())

                        info.supported_samplers.append(
                            engines_pb2.EngineSampler(
                                sampler=k, supports_eta="eta" in args
                            )
                        )

            engines.engine.append(info)

        return engines
