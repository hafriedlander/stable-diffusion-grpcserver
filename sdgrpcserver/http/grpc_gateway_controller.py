from typing import Callable

import grpc
from google.protobuf import json_format as pb_json_format
from google.protobuf.message import Message
from twisted.web.error import Error as WebError

from sdgrpcserver.http.json_api_controller import JSONAPIController

GRPC_HTTP_CODES = {
    grpc.StatusCode.OK: 200,
    grpc.StatusCode.CANCELLED: 499,
    grpc.StatusCode.UNKNOWN: 500,
    grpc.StatusCode.INVALID_ARGUMENT: 400,
    grpc.StatusCode.DEADLINE_EXCEEDED: 504,
    grpc.StatusCode.NOT_FOUND: 404,
    grpc.StatusCode.ALREADY_EXISTS: 409,
    grpc.StatusCode.PERMISSION_DENIED: 403,
    grpc.StatusCode.UNAUTHENTICATED: 401,
    grpc.StatusCode.RESOURCE_EXHAUSTED: 429,
    grpc.StatusCode.FAILED_PRECONDITION: 400,
    grpc.StatusCode.ABORTED: 409,
    grpc.StatusCode.OUT_OF_RANGE: 400,
    grpc.StatusCode.UNIMPLEMENTED: 501,
    grpc.StatusCode.INTERNAL: 500,
    grpc.StatusCode.UNAVAILABLE: 503,
    grpc.StatusCode.DATA_LOSS: 500,
}


class GRPCGatewayContext:
    def __init__(self, request):
        self.request = request
        self.code = grpc.StatusCode.OK
        self.message = "OK"
        self.cancel_callback: Callable | None = None

        self.request.notifyFinish().addErrback(self._finishError)

    def _finishError(self, *args):
        print(*args)

        if self.cancel_callback:
            self.cancel_callback()

    def add_callback(self, callback):
        self.cancel_callback = callback

    def set_code(self, code):
        self.code = code

    @property
    def http_code(self):
        return GRPC_HTTP_CODES[self.code]

    def set_details(self, message):
        self.message = message

    @property
    def http_message(self):
        return self.message.encode("utf-8")

    def abort(self, code, message):
        if code == grpc.StatusCode.OK:
            raise ValueError("Abort called with OK as status code")

        self.set_code(code)
        self.set_details(message)
        raise grpc.RpcError()


class GRPCGatewayController(JSONAPIController):
    input_class = None

    def add_Servicer(self, servicer):
        self._servicer = servicer

    def _render_common(self, request, handler, input):
        def wrapped_handler(request, input):
            context = GRPCGatewayContext(request)

            try:
                param = None
                if self.input_class:
                    param = self.input_class()
                if param and input:
                    pb_json_format.ParseDict(input, param, ignore_unknown_fields=True)

                if not self._servicer:
                    context.abort(
                        grpc.StatusCode.UNAVAILABLE, "Service not available yet"
                    )

                response = handler(request, param, context)
            except grpc.RpcError:
                raise WebError(context.http_code, context.http_message)
            else:
                if context.code != grpc.StatusCode.OK:
                    raise WebError(context.http_code, context.http_message)

            if isinstance(response, Message):
                response = pb_json_format.MessageToJson(response)

            return response

        return super()._render_common(request, wrapped_handler, input)
