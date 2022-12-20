import generation_pb2
from google.protobuf import json_format as pb_json_format
from twisted.internet import reactor
from twisted.web import resource
from twisted.web.resource import EncodingResourceWrapper, NoResource
from twisted.web.server import NOT_DONE_YET, GzipEncoderFactory

from sdgrpcserver.http.json_api_controller import GRPCGatewayContext, JSONAPIController


class GrpcGateway_EnginesController(JSONAPIController):
    def add_EnginesServiceServicer(self, engines_servicer):
        self._engines_servicer = engines_servicer

    def handle_GET(self, request):
        if not self._engines_servicer:
            return NoResource().render(request)

        return self._engines_servicer.ListEngines(None, None)


class GrpcGateway_GenerateController(JSONAPIController):
    def add_GenerationServiceServicer(self, generation_servicer):
        self._generation_servicer = generation_servicer

    def handle_POST(self, request, input):
        if not self._generation_servicer:
            return NoResource().render(request)

        generation_request = generation_pb2.Request()
        pb_json_format.ParseDict(input, generation_request, ignore_unknown_fields=True)

        reactor.callInThread(self._generate, request, generation_request)
        return NOT_DONE_YET

    def _generate(self, request, generation_request):
        for result in self._generation_servicer.Generate(
            generation_request, GRPCGatewayContext(request)
        ):
            json_result = pb_json_format.MessageToJson(
                result, including_default_value_fields=True
            ).encode("utf-8")

            reactor.callFromThread(request.write, json_result)
            reactor.callFromThread(request.write, b"\n")

        reactor.callFromThread(
            lambda: request.finish() if not request._disconnected else None
        )


class GrpcGatewayController(resource.Resource):
    def __init__(self):
        super().__init__()

        self.engines_bridge = GrpcGateway_EnginesController()
        self.generate_bridge = GrpcGateway_GenerateController()

    def getChild(self, path, request):
        print(path)
        if path == b"engines":
            return EncodingResourceWrapper(self.engines_bridge, [GzipEncoderFactory()])
        if path == b"generate":
            return self.generate_bridge

        return NoResource()

    def render(self, request):
        return NoResource().render(request)

    def add_EnginesServiceServicer(self, engines_servicer):
        self.engines_bridge.add_EnginesServiceServicer(engines_servicer)

    def add_GenerationServiceServicer(self, generation_servicer):
        self.generate_bridge.add_GenerationServiceServicer(generation_servicer)
