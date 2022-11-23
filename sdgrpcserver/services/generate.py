
from math import sqrt
import random, traceback, threading, json
from types import SimpleNamespace as SN
import torch

from google.protobuf import json_format as pb_json_format
import grpc
import generation_pb2, generation_pb2_grpc

from sdgrpcserver.utils import image_to_artifact, artifact_to_image

from sdgrpcserver import images, constants
from sdgrpcserver.debug_recorder import DebugNullRecorder

def buildDefaultMaskPostAdjustments():
    hardenMask = generation_pb2.ImageAdjustment()
    hardenMask.levels.input_low = 0
    hardenMask.levels.input_high = 0.05
    hardenMask.levels.output_low = 0
    hardenMask.levels.output_high = 1

    blur = generation_pb2.ImageAdjustment()
    blur.blur.sigma = 32
    blur.blur.direction = generation_pb2.DIRECTION_UP

    return [hardenMask, blur]

DEFAULT_POST_ADJUSTMENTS = buildDefaultMaskPostAdjustments();

debugCtr=0

class ParameterExtractor:
    """
    ParameterExtractor pulls fields out of a deeply nested GRPC structure.

    Every method that doesn't start with an "_" is a field that can be
    extracted from a Request object

    They shouldn't be called directly, but through "get", which will
    memo-ise the result.
    """

    def __init__(self, manager, request):
        self._manager = manager
        self._request = request
        self._result = {}

    def _save_debug_tensor(self, tensor):
        return
        global debugCtr
        debugCtr += 1 
        with open(f"{constants.debug_path}/debug-adjustments-{debugCtr}.png", "wb") as f:
            f.write(images.toPngBytes(tensor)[0])

    def _handleImageAdjustment(self, tensor, adjustments):
        if type(tensor) is bytes: tensor = images.fromPngBytes(tensor)

        self._save_debug_tensor(tensor)

        for adjustment in adjustments:
            which = adjustment.WhichOneof("adjustment")

            if which == "blur":
                sigma = adjustment.blur.sigma
                direction = adjustment.blur.direction

                if direction == generation_pb2.DIRECTION_DOWN or direction == generation_pb2.DIRECTION_UP:
                    orig = tensor
                    repeatCount=256
                    sigma /= sqrt(repeatCount)

                    for _ in range(repeatCount):
                        tensor = images.gaussianblur(tensor, sigma)
                        if direction == generation_pb2.DIRECTION_DOWN:
                            tensor = torch.minimum(tensor, orig)
                        else:
                            tensor = torch.maximum(tensor, orig)
                else:
                    tensor = images.gaussianblur(tensor, adjustment.blur.sigma)
            elif which == "invert":
                tensor = images.invert(tensor)
            elif which == "levels":
                tensor = images.levels(tensor, adjustment.levels.input_low, adjustment.levels.input_high, adjustment.levels.output_low, adjustment.levels.output_high)
            elif which == "channels":
                tensor = images.channelmap(tensor, [adjustment.channels.r,  adjustment.channels.g,  adjustment.channels.b,  adjustment.channels.a])
            elif which == "rescale":
                self.unimp("Rescale")
            elif which == "crop":
                tensor = images.crop(tensor, adjustment.crop.top, adjustment.crop.left, adjustment.crop.height, adjustment.crop.width)
            
            self._save_debug_tensor(tensor)
        
        return tensor

    def _image_stepparameter(self, field):
        if self._request.WhichOneof("params") != "image": return None

        for ctx in self._request.image.parameters:
            parts = field.split(".")

            while parts:
                if ctx.HasField(parts[0]):
                    ctx = getattr(ctx, parts.pop(0))
                else:
                    parts = ctx = None

            if ctx: return ctx

    def _image_parameter(self, field):
        if self._request.WhichOneof("params") != "image": return None
        if not self._request.image.HasField(field): return None
        return getattr(self._request.image, field)

    def _prompt_of_type(self, ptype):
        for prompt in self._request.prompt:
            which = prompt.WhichOneof("prompt")
            if which == ptype: 
                yield prompt

    def prompt(self):
        tokens = []

        for prompt in self._prompt_of_type("text"):
            weight = 1.0
            if prompt.HasField("parameters") and prompt.parameters.HasField("weight"): weight = prompt.parameters.weight
            if weight > 0: tokens.append((prompt.text, weight))

        return tokens if tokens else None

    def negative_prompt(self):
        tokens = []

        for prompt in self._prompt_of_type("text"):
            weight = 1.0
            if prompt.HasField("parameters") and prompt.parameters.HasField("weight"): weight = prompt.parameters.weight
            if weight < 0: tokens.append((prompt.text, -weight))

        return tokens if tokens else None

    def num_images_per_prompt(self):
        return self._image_parameter("samples")

    def height(self):
        image = self.get("init_image")
        if image is not None: return image.shape[2]
        return self._image_parameter("height")

    def width(self):
        image = self.get("init_image")
        if image is not None: return image.shape[3]
        return self._image_parameter("width")

    def seed(self):
        if self._request.WhichOneof("params") != "image": return None
        seed = list(self._request.image.seed)
        return seed if seed else None

    def guidance_scale(self):
        return self._image_stepparameter("sampler.cfg_scale")

    def clip_guidance_scale(self):
        if self._request.WhichOneof("params") != "image": return None

        for parameters in self._request.image.parameters:
            if parameters.HasField("guidance"):
                guidance = parameters.guidance
                for instance in guidance.instances:
                    if instance.HasField("guidance_strength"):
                        return instance.guidance_strength

    def sampler(self):
        if self._request.WhichOneof("params") != "image": return None
        if not self._request.image.HasField("transform"): return None
        if self._request.image.transform.WhichOneof("type") != "diffusion": return None
        return self._request.image.transform.diffusion

    def num_inference_steps(self):
        return self._image_parameter("steps")

    def eta(self):
        return self._image_stepparameter("sampler.eta")

    def churn(self):
        churn_settings = self._image_stepparameter("sampler.churn")
        return churn_settings.churn if churn_settings else None

    def churn_tmin(self):
        return self._image_stepparameter("sampler.churn.churn_tmin")

    def churn_tmax(self):
        return self._image_stepparameter("sampler.churn.churn_tmax")

    def sigma_min(self):
        return self._image_stepparameter("sampler.sigma.sigma_min")

    def sigma_max(self):
        return self._image_stepparameter("sampler.sigma.sigma_max")

    def karras_rho(self):
        return self._image_stepparameter("sampler.sigma.karras_rho")

    def scheduler_noise_type(self):
        noise_type = self._image_stepparameter("sampler.noise_type")

        if noise_type == generation_pb2.SAMPLER_NOISE_NORMAL: 
            return "normal"
        if noise_type == generation_pb2.SAMPLER_NOISE_BROWNIAN: 
            return "brownian"

        return None

    def init_image(self):
        for prompt in self._prompt_of_type("artifact"):
            if prompt.artifact.type == generation_pb2.ARTIFACT_IMAGE:
                image = images.fromPngBytes(prompt.artifact.binary).to(self._manager.mode.device)
                image = self._handleImageAdjustment(image, prompt.artifact.adjustments)
                return image

    def mask_image(self):
        for prompt in self._prompt_of_type("artifact"):
            if prompt.artifact.type == generation_pb2.ARTIFACT_MASK:
                mask = images.fromPngBytes(prompt.artifact.binary).to(self._manager.mode.device)
                return self._handleImageAdjustment(mask, prompt.artifact.adjustments)

    def outmask_image(self):
        for prompt in self._prompt_of_type("artifact"):
            if prompt.artifact.type == generation_pb2.ARTIFACT_MASK:
                mask = self.get("mask_image")
                post_adjustments = prompt.artifact.postAdjustments
                return self._handleImageAdjustment(mask, post_adjustments if post_adjustments else DEFAULT_POST_ADJUSTMENTS)

    def strength(self):
        return self._image_stepparameter("schedule.start")

    def get(self, field):
        if field not in self._result: 
            self._result[field] = getattr(self, field)()
        return self._result[field]

    def fields(self):
        return [
            key
            for key in dir(self)
            if key[0] != "_" and key != "get" and key != "fields"
        ]

class GenerationServiceServicer(generation_pb2_grpc.GenerationServiceServicer):
    def __init__(self, manager, supress_metadata = False, debug_recorder = DebugNullRecorder()):
        self._manager = manager
        self._supress_metadata = supress_metadata
        self._debug_recorder = debug_recorder

    def unimp(self, what):
        raise NotImplementedError(f"{what} not implemented")

    def batched_seeds(self, samples, seeds, batchmax):
        # If we weren't given any seeds at all, just start with a single -1
        if not seeds: seeds = [-1]

        # Replace any negative seeds with a randomly selected one
        seeds = [seed if seed >= 0 else random.randrange(0, 2**32-1) for seed in seeds]

        # Fill seeds up to params.samples if we didn't get passed enough
        if len(seeds) < samples:
            # Starting with the last seed we were given
            nextseed = seeds[-1]+1
            while len(seeds) < samples: 
                seeds.append(nextseed)
                nextseed += 1

        # Calculate the most even possible split across batchmax
        if samples <= batchmax:
            batches = [samples]
        elif samples % batchmax == 0:
            batches = [batchmax] * (samples // batchmax)
        else:
            d = samples // batchmax + 1
            batchsize = samples // d
            r = samples - batchsize * d
            batches = [batchsize+1]*r + [batchsize] * (d-r)        

        for batch in batches:
            batchseeds, seeds = seeds[:batch], seeds[batch:]
            yield batchseeds

    def Generate(self, request, context):
        with self._debug_recorder.record(request.request_id) as recorder:
            recorder.store('generate request', request)

            try:
                # Assume that "None" actually means "Image" (stability-sdk/client.py doesn't set it)
                if request.requested_type != generation_pb2.ARTIFACT_NONE and request.requested_type != generation_pb2.ARTIFACT_IMAGE:
                    self.unimp('Generation of anything except images')

                extractor = ParameterExtractor(self._manager, request)
                kwargs = {}

                for field in extractor.fields():
                    val = extractor.get(field)
                    if val is not None: kwargs[field] = val

                try:
                    pipe = self._manager.getPipe(request.engine_id)
                except KeyError as e:
                    context.set_code(grpc.StatusCode.NOT_FOUND)
                    context.set_details("Engine not found")
                    return

                stop_event = threading.Event()
                context.add_callback(lambda: stop_event.set())

                ctr = 0
                samples = kwargs.get("num_images_per_prompt", 1)
                seeds = kwargs.get("seed", None)
                batchmax = self._manager.batchMode.batchmax(kwargs["width"] * kwargs["height"])

                for seeds in self.batched_seeds(samples, seeds, batchmax):
                    batchargs = {
                        **kwargs,
                        "seed": seeds,
                        "num_images_per_prompt": len(seeds)
                    }

                    logargs = {**batchargs}
                    for field in ["init_image", "mask_image", "outmask_image"]:
                        if field in logargs: logargs[field] = "yes"

                    print()
                    print(f'Generating {repr(logargs)}')

                    recorder.store('pipe.generate calls', kwargs)

                    results = pipe.generate(**batchargs, stop_event=stop_event)

                    meta = pb_json_format.MessageToDict(request)
                    for prompt in meta['prompt']:
                        if 'artifact' in prompt:
                                del prompt['artifact']['binary']

                    for i, (result_image, nsfw) in enumerate(zip(results[0], results[1])):
                        answer = generation_pb2.Answer()
                        answer.request_id=request.request_id
                        answer.answer_id=f"{request.request_id}-{ctr}"

                        if self._supress_metadata:
                            artifact=image_to_artifact(result_image)
                        else:
                            meta["image"]["samples"] = 1
                            meta["image"]["seed"] = [seeds[i]]
                            artifact=image_to_artifact(result_image, meta={"generation_parameters": json.dumps(meta)})

                        artifact.finish_reason=generation_pb2.FILTER if nsfw else generation_pb2.NULL
                        artifact.index=ctr
                        artifact.seed=seeds[i]
                        answer.artifacts.append(artifact)

                        recorder.store('pipe.generate result', artifact)
                        
                        yield answer
                        ctr += 1
                
            except NotImplementedError as e:
                context.set_code(grpc.StatusCode.UNIMPLEMENTED)
                context.set_details(str(e))
                print(f"Unsupported request parameters: {e}")
            except Exception as e:
                traceback.print_exc()
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("Something went wrong")
