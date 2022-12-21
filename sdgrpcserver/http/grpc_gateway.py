import generation_pb2
import grpc
from google.protobuf import json_format as pb_json_format
from twisted.internet import reactor
from twisted.web import resource
from twisted.web.resource import NoResource
from twisted.web.server import NOT_DONE_YET

from sdgrpcserver.http.grpc_gateway_controller import GRPCGatewayController


class GrpcGateway_EnginesController(GRPCGatewayController):
    def handle_GET(self, web_request, _, context):
        return self._servicer.ListEngines(None, context)


class GrpcGateway_GenerateController(GRPCGatewayController):
    input_class = generation_pb2.Request

    def handle_POST(self, web_request, generation_request, context):
        reactor.callInThread(self._generate, web_request, generation_request, context)
        return NOT_DONE_YET

    def _error(self, request, code, message):
        # We can't set the http status code during a stream, so just write
        # the status message into the body
        status = grpc.Status(code=code, details=message)
        json_result = pb_json_format.MessageToJson(status).encode("utf-8")

        reactor.callFromThread(request.write, json_result)
        reactor.callFromThread(request.write, b"\n")

    def _generate(self, request, generation_request, context):
        for result in self._servicer.Generate(generation_request, context):
            try:
                json_result = pb_json_format.MessageToJson(result).encode("utf-8")

                reactor.callFromThread(request.write, json_result)
                reactor.callFromThread(request.write, b"\n")
            except grpc.RpcError:
                self._error(request, context.code, context.message)
                break
            except BaseException:
                self._error(request, grpc.StatusCode.INTERNAL, "Internal Error")
                break

        reactor.callFromThread(
            lambda: request.finish() if not request._disconnected else None
        )


class GrpcGateway_AsyncGenerateController(GRPCGatewayController):
    input_class = generation_pb2.Request

    def handle_POST(self, web_request, generation_request, context):
        return self._servicer.AsyncGenerate(generation_request, context)


class GrpcGateway_AsyncResultController(GRPCGatewayController):
    input_class = generation_pb2.AsyncHandle

    def handle_POST(self, web_request, async_handle, context):
        return self._servicer.AsyncResult(async_handle, context)


class GrpcGateway_AsyncCancelController(GRPCGatewayController):
    input_class = generation_pb2.AsyncHandle

    def handle_POST(self, web_request, async_handle, context):
        return self._servicer.AsyncCancel(async_handle, context)


class GrpcGatewayRouter(resource.Resource):
    def __init__(self):
        super().__init__()

        self.engines_bridge = GrpcGateway_EnginesController()
        self.generate_bridge = GrpcGateway_GenerateController()
        self.async_generate_bridge = GrpcGateway_AsyncGenerateController()
        self.async_result_bridge = GrpcGateway_AsyncResultController()
        self.async_cancel_bridge = GrpcGateway_AsyncCancelController()

    def getChild(self, path, request):
        if path == b"engines":
            return self.engines_bridge
        if path == b"generate":
            return self.generate_bridge
        if path == b"asyncGenerate":
            return self.async_generate_bridge
        if path == b"asyncResult":
            return self.async_result_bridge
        if path == b"asyncCancel":
            return self.async_cancel_bridge

        return NoResource()

    def render(self, request):
        return NoResource().render(request)

    def add_EnginesServiceServicer(self, engines_servicer):
        self.engines_bridge.add_Servicer(engines_servicer)

    def add_GenerationServiceServicer(self, generation_servicer):
        self.generate_bridge.add_Servicer(generation_servicer)
        self.async_generate_bridge.add_Servicer(generation_servicer)
        self.async_result_bridge.add_Servicer(generation_servicer)
        self.async_cancel_bridge.add_Servicer(generation_servicer)
