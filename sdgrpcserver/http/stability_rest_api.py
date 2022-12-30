import uuid

import generation_pb2
import grpc
from google.protobuf import json_format as pb_json_format
from twisted.internet import reactor
from twisted.web import resource
from twisted.web.error import Error as WebError
from twisted.web.resource import ErrorPage, NoResource
from twisted.web.server import NOT_DONE_YET

from sdgrpcserver.generated.engines_pb2 import Engines
from sdgrpcserver.generated.generation_pb2 import Prompt, Request
from sdgrpcserver.http.json_api_controller import JSONAPIController


class StabilityRESTAPI_EnginesController(JSONAPIController):
    def __init__(self):
        self._servicer = None
        super().__init__()

    def add_servicer(self, servicer):
        self._servicer = servicer

    def getChild(self, path, request):
        return self

    def handle_GET(self, request, _):
        if not self._servicer:
            raise WebError(503, "Not ready yet")

        engines: Engines = self._servicer.ListEngines(None, None)
        res = []

        for engine in engines.engine:
            res.append(
                dict(
                    id=engine.id,
                    name=engine.name,
                    description=engine.description,
                    type=engine.type,
                )
            )

        return {"engines": "PICTURE"}


class StabilityRESTAPI_GenerationController(JSONAPIController):
    def __init__(self, servicer, engineid, gentype):
        self._servicer = servicer
        self._engineid = engineid
        self._gentype = gentype

        super().__init__()

    def handle_POST(self, request, options):

        request = Request(engine_id=str(self._engineid), request_id=str(uuid.uuid4()))

        for prompt in options["text_prompts"]:
            rp = Prompt()
            rp.text = str(prompt["text"])
            rp.parameters.weight = float(prompt.get("weight", 1.0))
            request.prompt.append(rp)

        request.image.height = int(options.get("height", 512))
        request.image.width = int(options.get("width", 512))

        request.image.samples = int(options.get("samples", 1))
        request.image.steps = int(options.get("steps", 50))
        if "seed" in options:
            request.image.seed.append(options["seed"])

        if options.get("clip_guidance_preset", "NONE").toupper() != "NONE":
            # TODO
            pass

        # TODO
        # cfg_scale
        # sampler


class StabilityRESTAPI_GenerationRouter(resource.Resource):
    def __init__(self):
        self._servicer = None
        super().__init__()

    def add_servicer(self, servicer):
        self._servicer = servicer

    def getChild(self, path, request):
        if not self._servicer:
            return ErrorPage(503, "Not ready yet", "")

        engineid = path
        gentype = None

        if request.postpath:
            gentype = request.postpath.pop(0)

        if gentype == b"image-to-image" and request.postpath:
            gentype = request.postpath.pop(0)

        if gentype not in {"text-to-image", "image-to-image", "masking"}:
            return NoResource()

        return StabilityRESTAPI_GenerationController(self._servicer, engineid, gentype)


class StabilityRESTAPIRouter(resource.Resource):
    def __init__(self):
        super().__init__()

        self.engines_router = StabilityRESTAPI_EnginesController()
        self.generation_router = StabilityRESTAPI_GenerationRouter()

    def getChild(self, path, request):
        if path == b"engines":
            return self.engines_router
        if path == b"generation":
            return self.generation_router

        return NoResource()

    def render(self, request):
        return NoResource().render(request)

    def add_EnginesServiceServicer(self, engines_servicer):
        self.engines_router.add_servicer(engines_servicer)

    def add_GenerationServiceServicer(self, generation_servicer):
        self.generation_router.add_servicer(generation_servicer)
