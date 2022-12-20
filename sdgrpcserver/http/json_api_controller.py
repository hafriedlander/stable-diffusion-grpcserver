import json

import grpc
from google.protobuf import json_format as pb_json_format
from twisted.web import resource
from twisted.web.resource import ErrorPage, NoResource
from twisted.web.server import NOT_DONE_YET


class NotAcceptableResource(ErrorPage):
    def __init__(
        self, message="Sorry, Accept header does not match a type we can serve"
    ):
        super().__init__(416, "Not Acceptable", message)


class UnsupportedMediaTypeResource(ErrorPage):
    def __init__(
        self, message="Sorry, Content-Type header does not match a type we can process"
    ):
        super().__init__(415, "Unsupported Media Type", message)


GRPC_HTTP_CODES = {
    grpc.StatusCode.OK: 200,
    grpc.StatusCode.NOT_FOUND: 404,
    grpc.StatusCode.UNIMPLEMENTED: 405,
    grpc.StatusCode.INTERNAL: 500,
}


class GRPCGatewayContext:
    def __init__(self, request):
        self.request = request
        self.code = 200
        self.message = b""

    def add_callback(self, callback):
        self.request.notifyFinish().addErrback(lambda *_: callback())

    def set_code(self, code):
        self.code = GRPC_HTTP_CODES[code]
        self.request.setResponseCode(self.code, self.message)

    def set_details(self, message):
        self.message = message.encode("utf-8") if isinstance(message, str) else message
        self.request.setResponseCode(self.code, self.message)


class JSONAPIController(resource.Resource):
    def _render_common(self, request, handler, *args, **kwargs):
        if not handler:
            return NoResource().render(request)

        accept_header = request.getHeader("accept")
        if not accept_header or accept_header != "application/json":
            return NotAcceptableResource().render(request)

        response = handler(request, *args, **kwargs)

        # Handle when a controller returns NOT_DONE_YET because it's
        # still working in the background
        if response is NOT_DONE_YET:
            return NOT_DONE_YET

        # Convert dict or object instances into json strings
        if isinstance(response, dict):
            response = json.dumps(response)
        elif not isinstance(response, str | bytes):
            response = pb_json_format.MessageToJson(response)

        # JSON is always encoded as utf-8
        if isinstance(response, str):
            response = response.encode("utf-8")

        # And return it
        request.setHeader("content-type", "application/json")
        return response

    def render_GET(self, request):
        handler = getattr(self, "handle_GET", None)
        return self._render_common(request, handler)

    def render_POST(self, request):
        handler = getattr(self, "handle_POST", None)

        content_type_header = request.getHeader("content-type")
        if not content_type_header or content_type_header != "application/json":
            return UnsupportedMediaTypeResource().render(request)

        return self._render_common(request, handler, json.load(request.content))
