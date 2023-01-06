import json
import uuid
from base64 import b64encode

import grpc
import multipart
from engines_pb2 import Engines, EngineType
from generation_pb2 import (
    ARTIFACT_IMAGE,
    ARTIFACT_MASK,
    CHANNEL_A,
    CHANNEL_DISCARD,
    Artifact,
    DiffusionSampler,
    GuidanceInstanceParameters,
    ImageAdjustment,
    ImageAdjustment_Channels,
    ImageAdjustment_Invert,
    Prompt,
    PromptParameters,
    Request,
    StepParameter,
)
from google.protobuf import json_format as pb_json_format
from twisted.internet import reactor
from twisted.web import resource
from twisted.web.error import Error as WebError
from twisted.web.resource import ErrorPage, NoResource
from twisted.web.server import NOT_DONE_YET

from gyre.http.grpc_gateway_controller import GRPCGatewayContext
from gyre.http.json_api_controller import (
    JSONAPIController,
    UnsupportedMediaTypeResource,
)


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
                    type=EngineType.Name(engine.type),
                )
            )

        return {"engines": res}


class StabilityRESTAPI_GenerationController(JSONAPIController):
    return_types = {"application/json", "image/png"}

    def __init__(self, servicer, engineid, gentype):
        self._servicer = servicer
        self._engineid = engineid
        self._gentype = gentype

        super().__init__()

    def render_POST(self, request):
        handler = getattr(self, "handle_POST", None)

        content = None

        content_type_header = request.getHeader("content-type")
        if content_type_header:
            content_type, options = multipart.parse_options_header(content_type_header)

            if content_type == "application/json":
                content = {"options": json.load(request.content)}

            elif content_type == "multipart/form-data":
                parser = multipart.MultipartParser(request.content, options["boundary"])
                content = {}
                for part in parser:
                    if part.name == "init_image" and part.content_type == "image/png":
                        content["init_image"] = part.raw
                    elif part.name == "mask_image" and part.content.type == "image/png":
                        content["mask_image"] = part.raw
                    elif part.name == "options":
                        content["options"] = json.load(part.file)
                    else:
                        print("Ignoring unknown form part", part.name)

        if content is None:
            return UnsupportedMediaTypeResource().render(request)
        else:
            return self._render_common(request, handler, content)

    def _number(self, options, attr, type, default=None, minVal=None, maxVal=None):
        val = type(options.get(attr, default))
        if minVal is not None and val < minVal:
            raise ValueError(f"{attr} may not be less than {minVal}, but {val} passed")
        if maxVal is not None and val > maxVal:
            raise ValueError(f"{attr} may not be more than {maxVal}, but {val} passed")

        return val

    def _image_to_prompt(
        self, image, init: bool = False, mask: bool = False, adjustments=[]
    ) -> Prompt:
        if init and mask:
            raise ValueError("init and mask cannot both be True")

        artifact = Artifact(
            type=ARTIFACT_MASK if mask else ARTIFACT_IMAGE, binary=image
        )

        for adjustment in adjustments:
            artifact.adjustments.append(adjustment)

        return Prompt(
            artifact=artifact,
            parameters=PromptParameters(init=init),
        )

    def _mask_to_prompt(self, init_image, mask_image, mask_source):
        # With init_image_alpha, pull alpha from init_image into r,g,b then invert
        if mask_source == "INIT_IMAGE_ALPHA":
            return self._image_to_prompt(
                init_image,
                mask=True,
                adjustments=[
                    ImageAdjustment(
                        channels=ImageAdjustment_Channels(
                            r=CHANNEL_A,
                            g=CHANNEL_A,
                            b=CHANNEL_A,
                            a=CHANNEL_DISCARD,
                        )
                    ),
                    ImageAdjustment(invert=ImageAdjustment_Invert()),
                ],
            )

        # With mask_image_white we can just use mask as-is
        elif mask_source == "MASK_IMAGE_WHITE":
            if mask_image:
                return self._image_to_prompt(mask_image, mask=True)
            else:
                raise ValueError(
                    "mask_source is MASK_IMAGE_WHITE but no mask_image was provided"
                )

        # With mask_image_black, invert the mask
        elif mask_source == "MASK_IMAGE_BLACK":
            if mask_image:
                return self._image_to_prompt(
                    mask_image,
                    mask=True,
                    adjustments=[ImageAdjustment(invert=ImageAdjustment_Invert())],
                )
            else:
                raise ValueError(
                    "mask_source is MASK_IMAGE_BLACK but no mask_image was provided"
                )

        elif mask_source:
            raise ValueError(f"Unknown mask_source {mask_source}")
        else:
            raise ValueError("masking requires a mask_source parameter")

    def handle_POST(self, http_request, data):
        if not self._servicer:
            raise WebError(503, "Not ready yet")

        init_image = data.get("init_image")
        mask_image = data.get("mask_image")
        options = data.get("options")

        if not options:
            raise ValueError("No options provided. Please check API docs.")

        accept_header = http_request.getHeader("accept")
        is_png = accept_header == "image/png"

        request = Request(
            engine_id=self._engineid.decode("utf-8"), request_id=str(uuid.uuid4())
        )
        parameters = StepParameter()

        # -- init_image

        if self._gentype == b"text-to-image":
            if init_image:
                raise ValueError("Don't pass init_image to text-to-image")
        else:
            if not init_image:
                raise ValueError(f"{self._gentype} requires init_image")

            request.prompt.append(self._image_to_prompt(init_image, init=True))

        # -- cfg_scale

        parameters.sampler.cfg_scale = self._number(
            options, "cfg_scale", float, 7, 0, 35
        )

        # -- clip_guidance_preset

        if options.get("clip_guidance_preset", "NONE").upper() != "NONE":
            guidance_parameters = GuidanceInstanceParameters()
            guidance_parameters.guidance_strength = 0.333
            parameters.guidance.instances.append(guidance_parameters)

        # -- height

        request.image.height = self._number(options, "height", int, 512, 512, 2048)

        # -- mask_source

        mask_source = options.get("mask_source", "").upper()

        if self._gentype != b"masking":
            if mask_source:
                raise ValueError(f"Don't pass mask_source to {self._gentype}")
        else:
            request.prompt.append(
                self._mask_to_prompt(init_image, mask_image, mask_source)
            )

        # -- sampler

        sampler_str = "SAMPLER_K_DPMPP_SDE"
        if "sampler" in options:
            sampler_str = "SAMPLER_" + str(options["sampler"]).upper()

        request.image.transform.diffusion = DiffusionSampler.Value(sampler_str)

        # -- samples

        if is_png:
            request.image.samples = self._number(options, "samples", int, 1, 1, 1)
        else:
            request.image.samples = self._number(options, "samples", int, 1, 1, 10)

        # -- seed

        if "seed" in options:
            request.image.seed.append(
                self._number(options, "seed", int, 0, 0, 2147483647)
            )

        # -- step_schedule_end

        parameters.schedule.end = self._number(
            options, "step_schedule_end", float, 0, 0, 1
        )

        # -- step_schedule_start

        parameters.schedule.start = self._number(
            options, "step_schedule_start", float, 1, 0, 1
        )

        # -- steps

        request.image.steps = self._number(options, "steps", int, 50, 10, 150)

        # -- text_prompts

        for prompt in options["text_prompts"]:
            rp = Prompt()
            rp.text = str(prompt["text"])
            rp.parameters.weight = float(prompt.get("weight", 1.0))
            request.prompt.append(rp)

        # -- width

        request.image.width = self._number(options, "width", int, 512, 512, 2048)

        images = []
        finish = []
        seeds = []

        request.image.parameters.append(parameters)
        context = GRPCGatewayContext(http_request)
        try:
            for answer in self._servicer.Generate(request, context):
                for artifact in answer.artifacts:
                    if artifact.mime == "image/png":
                        images.append(artifact.binary)
                        finish.append(artifact.finish_reason)
                        seeds.append(artifact.seed)
        except grpc.RpcError:
            raise WebError(context.http_code, context.http_message)

        if is_png:
            http_request.setHeader("finish-reason", str(finish[0]))
            http_request.setHeader("seed", str(seeds[0]))
            return images[0]
        else:
            http_request.setHeader("finish-reason", json.dumps(finish))
            http_request.setHeader("seed", json.dumps(seeds))
            return json.dumps([b64encode(image).decode("ascii") for image in images])


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

        if gentype not in {b"text-to-image", b"image-to-image", b"masking"}:
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
