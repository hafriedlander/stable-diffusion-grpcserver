import functools
import json
import random
import threading
import time
import traceback
import uuid
from math import sqrt
from queue import Empty, Queue
from types import SimpleNamespace as SN
from typing import Callable, Iterable

import generation_pb2
import generation_pb2_grpc
import grpc
import torch
from google.protobuf import json_format as pb_json_format

from gyre import constants, images
from gyre.debug_recorder import DebugNullRecorder
from gyre.manager import EngineNotFoundError
from gyre.protobuf_safetensors import deserialize_safetensors
from gyre.protobuf_tensors import deserialize_tensor
from gyre.services.exception_to_grpc import exception_to_grpc
from gyre.utils import artifact_to_image, image_to_artifact


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


DEFAULT_POST_ADJUSTMENTS = buildDefaultMaskPostAdjustments()

debugCtr = 0


class AsyncContext:
    def __init__(self, deadline=None):
        self.queue = Queue()
        self.code = grpc.StatusCode.OK
        self.message = ""
        self.cancel_callback: Callable | None = None
        self.thread: threading.Thread | None = None
        self.deadline: float | None = None

        if deadline:
            self.deadline = time.monotonic() + deadline

    # These are methods for the async handlers

    def cancel(self):
        if self.cancel_callback:
            self.cancel_callback()

        self.code = grpc.StatusCode.CANCELLED
        self.message = "Cancelled"

    def set_deadline(self, deadline):
        new_deadline = time.monotonic() + deadline

        if self.deadline:
            self.deadline = min(new_deadline, self.deadline)
        else:
            self.deadline = new_deadline

    def clear_deadline(self):
        self.deadline = None

    def past_deadline(self):
        return self.deadline and time.monotonic() > self.deadline

    # These mirror methods from GRPC Context

    def add_callback(self, callback):
        self.cancel_callback = callback

    def set_code(self, code):
        self.code = code

    def set_details(self, message):
        self.message = message

    def abort(self, code, details):
        if code == grpc.StatusCode.OK:
            raise ValueError("Abort called with OK as status code")

        self.set_code(code)
        self.set_details(details)

        raise grpc.RpcError()


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
        # Add a cache to self.get to prevent multiple requests from recalculating
        # self.get = functools.cache(self.get)

    def _save_debug_tensor(self, tensor):
        return
        global debugCtr
        debugCtr += 1
        with open(
            f"{constants.debug_path}/debug-adjustments-{debugCtr}.png", "wb"
        ) as f:
            f.write(images.toPngBytes(tensor)[0])

    def _apply_image_adjustment(self, tensor, adjustments):
        if not adjustments:
            return tensor

        if type(tensor) is bytes:
            tensor = images.fromPngBytes(tensor)

        self._save_debug_tensor(tensor)

        for adjustment in adjustments:
            which = adjustment.WhichOneof("adjustment")

            if which == "blur":
                sigma = adjustment.blur.sigma
                direction = adjustment.blur.direction

                if direction == generation_pb2.DIRECTION_DOWN:
                    tensor = images.directionalblur(tensor, sigma, "down")
                elif direction == generation_pb2.DIRECTION_UP:
                    tensor = images.directionalblur(tensor, sigma, "up")
                else:
                    tensor = images.gaussianblur(tensor, sigma)
            elif which == "invert":
                tensor = images.invert(tensor)
            elif which == "levels":
                tensor = images.levels(
                    tensor,
                    adjustment.levels.input_low,
                    adjustment.levels.input_high,
                    adjustment.levels.output_low,
                    adjustment.levels.output_high,
                )
            elif which == "channels":
                tensor = images.channelmap(
                    tensor,
                    [
                        adjustment.channels.r,
                        adjustment.channels.g,
                        adjustment.channels.b,
                        adjustment.channels.a,
                    ],
                )
            elif which == "rescale":
                # Calculate fit mode
                if adjustment.rescale.mode == generation_pb2.RESCALE_STRICT:
                    fit = "strict"
                elif adjustment.rescale.mode == generation_pb2.RESCALE_COVER:
                    fit = "cover"
                else:
                    fit = "contain"

                # Calculate pad mode (should only be used for CONTAIN modes)
                pad_mode = "constant"
                if adjustment.rescale.mode == generation_pb2.RESCALE_CONTAIN_REPLICATE:
                    pad_mode = "replicate"
                elif adjustment.rescale.mode == generation_pb2.RESCALE_CONTAIN_REFLECT:
                    pad_mode = "reflect"

                tensor = images.rescale(
                    tensor,
                    adjustment.rescale.height,
                    adjustment.rescale.width,
                    fit,
                    pad_mode,
                )

            elif which == "crop":
                tensor = images.crop(
                    tensor,
                    adjustment.crop.top,
                    adjustment.crop.left,
                    adjustment.crop.height,
                    adjustment.crop.width,
                )

            elif which == "depth":
                with self._manager.with_engine(task="depth") as estimator:
                    tensor = estimator(tensor)

            else:
                raise ValueError(f"Unkown image adjustment {which}")

            self._save_debug_tensor(tensor)

        return tensor

    def _image_from_artifact_binary(self, artifact):
        return images.fromPngBytes(artifact.binary).to(self._manager.mode.device)

    def _image_from_artifact_reference(self, artifact):
        if artifact.ref.WhichOneof("reference") == "id":
            test = lambda x: x.id == artifact.ref.id
        else:
            test = lambda x: x.uuid == artifact.ref.uuid

        for prompt in self._prompt_of_type("artifact"):
            if test(prompt.artifact):
                return self._image_from_artifact(prompt.artifact, artifact.ref.stage)

    def _image_from_artifact(
        self,
        artifact: generation_pb2.Artifact,
        stage=generation_pb2.ARTIFACT_AFTER_ADJUSTMENTS,
    ):
        if artifact.WhichOneof("data") == "binary":
            image = self._image_from_artifact_binary(artifact)
        elif artifact.WhichOneof("data") == "ref":
            image = self._image_from_artifact_reference(artifact)
        else:
            raise ValueError(
                f"Can't convert Artifact of type {artifact.WhichOneof('data')} to an image"
            )

        if stage == generation_pb2.ARTIFACT_BEFORE_ADJUSTMENTS:
            return image

        image = self._apply_image_adjustment(image, artifact.adjustments)
        if stage == generation_pb2.ARTIFACT_AFTER_ADJUSTMENTS:
            return image

        image = self._apply_image_adjustment(image, artifact.postAdjustments)
        return image

    def _image_stepparameter(self, field):
        if self._request.WhichOneof("params") != "image":
            return None

        for ctx in self._request.image.parameters:
            parts = field.split(".")

            while parts:
                if ctx.HasField(parts[0]):
                    ctx = getattr(ctx, parts.pop(0))
                else:
                    parts = ctx = None

            if ctx:
                return ctx

    def _image_parameter(self, field):
        if self._request.WhichOneof("params") != "image":
            return None
        if not self._request.image.HasField(field):
            return None
        return getattr(self._request.image, field)

    def _prompt_of_type(self, ptype) -> Iterable[generation_pb2.Prompt]:
        for prompt in self._request.prompt:
            which = prompt.WhichOneof("prompt")
            if which == ptype:
                yield prompt

    def _prompt_of_artifact_type(self, atype) -> Iterable[generation_pb2.Prompt]:
        for prompt in self._prompt_of_type("artifact"):
            if prompt.artifact.type == atype:
                yield prompt

    def prompt(self):
        tokens = []

        for prompt in self._prompt_of_type("text"):
            weight = 1.0
            if prompt.HasField("parameters") and prompt.parameters.HasField("weight"):
                weight = prompt.parameters.weight
            if weight > 0:
                tokens.append((prompt.text, weight))

        return tokens if tokens else None

    def negative_prompt(self):
        tokens = []

        for prompt in self._prompt_of_type("text"):
            weight = 1.0
            if prompt.HasField("parameters") and prompt.parameters.HasField("weight"):
                weight = prompt.parameters.weight
            if weight < 0:
                tokens.append((prompt.text, -weight))

        return tokens if tokens else None

    def num_images_per_prompt(self):
        return self._image_parameter("samples")

    def height(self):
        image = self.get("init_image")
        if image is not None:
            return image.shape[2]
        return self._image_parameter("height")

    def width(self):
        image = self.get("init_image")
        if image is not None:
            return image.shape[3]
        return self._image_parameter("width")

    def seed(self):
        if self._request.WhichOneof("params") != "image":
            return None
        seed = list(self._request.image.seed)
        return seed if seed else None

    def guidance_scale(self):
        return self._image_stepparameter("sampler.cfg_scale")

    def clip_guidance_scale(self):
        if self._request.WhichOneof("params") != "image":
            return None

        for parameters in self._request.image.parameters:
            if parameters.HasField("guidance"):
                guidance = parameters.guidance
                for instance in guidance.instances:
                    if instance.HasField("guidance_strength"):
                        return instance.guidance_strength

    def sampler(self):
        if self._request.WhichOneof("params") != "image":
            return None
        if not self._request.image.HasField("transform"):
            return None
        if self._request.image.transform.WhichOneof("type") != "diffusion":
            return None
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
                return self._image_from_artifact(prompt.artifact)

    def mask_image(self):
        for prompt in self._prompt_of_type("artifact"):
            if prompt.artifact.type == generation_pb2.ARTIFACT_MASK:
                return self._image_from_artifact(prompt.artifact)

    def outmask_image(self):
        for prompt in self._prompt_of_type("artifact"):
            if prompt.artifact.type == generation_pb2.ARTIFACT_MASK:
                return self._image_from_artifact(
                    prompt.artifact, generation_pb2.ARTIFACT_AFTER_POSTADJUSTMENTS
                )

    def depth_map(self):
        for prompt in self._prompt_of_type("artifact"):
            if prompt.artifact.type == generation_pb2.ARTIFACT_DEPTH:
                return self._image_from_artifact(prompt.artifact)

    def lora(self):
        loras = []

        for prompt in self._prompt_of_artifact_type(generation_pb2.ARTIFACT_LORA):
            safetensors = deserialize_safetensors(prompt.artifact.lora.lora)
            weights = {}
            for weight in prompt.artifact.lora.weights:
                weights[weight.model_name] = weight.weight

            loras.append((safetensors, weights))

        return loras if loras else None

    def strength(self):
        return self._image_stepparameter("schedule.start")

    def hires_fix(self):
        hires = self._image_parameter("hires")
        return hires.enable if hires else None

    def hires_oos_fraction(self):
        hires = self._image_parameter("hires")
        if hires and hires.HasField("oos_fraction"):
            return hires.oos_fraction
        return None

    def tiling(self):
        return self._image_parameter("tiling")

    def get(self, field):
        return getattr(self, field)()

    def fields(self):
        return [
            key
            for key in dir(self)
            if key[0] != "_" and key != "get" and key != "fields"
        ]


class GenerationServiceServicer(generation_pb2_grpc.GenerationServiceServicer):
    def __init__(
        self,
        manager,
        supress_metadata=False,
        debug_recorder=DebugNullRecorder(),
        ram_monitor=None,
    ):
        self._manager = manager
        self._supress_metadata = supress_metadata
        self._debug_recorder = debug_recorder
        self._ram_monitor = ram_monitor

        # For async support
        self._async_contexts_lock = threading.Lock()
        self._async_contexts: dict[str, AsyncContext] = {}

    def unimp(self, what):
        raise NotImplementedError(f"{what} not implemented")

    def batched_seeds(self, samples, seeds, batchmax):
        # If we weren't given any seeds at all, just start with a single -1
        if not seeds:
            seeds = [-1]

        # Replace any negative seeds with a randomly selected one
        seeds = [
            seed if seed >= 0 else random.randrange(0, 2**32 - 1) for seed in seeds
        ]

        # Fill seeds up to params.samples if we didn't get passed enough
        if len(seeds) < samples:
            # Starting with the last seed we were given
            nextseed = seeds[-1] + 1
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
            batches = [batchsize + 1] * r + [batchsize] * (d - r)

        for batch in batches:
            batchseeds, seeds = seeds[:batch], seeds[batch:]
            yield batchseeds

    @exception_to_grpc(
        {
            EngineNotFoundError: grpc.StatusCode.NOT_FOUND,
            NotImplementedError: grpc.StatusCode.UNIMPLEMENTED,
        }
    )
    def Generate(self, request, context):
        with self._debug_recorder.record(request.request_id) as recorder:
            recorder.store("generate request", request)

            # Assume that "None" actually means "Image" (stability-sdk/client.py doesn't set it)
            if (
                request.requested_type != generation_pb2.ARTIFACT_NONE
                and request.requested_type != generation_pb2.ARTIFACT_IMAGE
            ):
                self.unimp("Generation of anything except images")

            extractor = ParameterExtractor(self._manager, request)
            kwargs = {}

            for field in extractor.fields():
                val = extractor.get(field)
                if val is not None:
                    kwargs[field] = val

            if self._ram_monitor:
                print("Arguments processed")
                self._ram_monitor.print()

            stop_event = threading.Event()
            context.add_callback(lambda: stop_event.set())

            ctr = 0
            samples = kwargs.get("num_images_per_prompt", 1)
            seeds = kwargs.get("seed", None)
            batchmax = self._manager.batchMode.batchmax(
                kwargs["width"] * kwargs["height"]
            )

            for seeds in self.batched_seeds(samples, seeds, batchmax):
                batchargs = {
                    **kwargs,
                    "seed": seeds,
                    "num_images_per_prompt": len(seeds),
                }

                logargs = {**batchargs}
                for field in [
                    "init_image",
                    "mask_image",
                    "outmask_image",
                    "depth_map",
                    "lora",
                ]:
                    if field in logargs:
                        value = logargs[field]
                        logargs[field] = (
                            f"[{len(value)}]" if isinstance(value, list) else "yes"
                        )

                print()
                print(f"Generating {repr(logargs)}")

                recorder.store("pipe.generate calls", kwargs)

                with self._manager.with_engine(request.engine_id) as engine:
                    results = engine.generate(**batchargs, stop_event=stop_event)

                meta = pb_json_format.MessageToDict(request)
                for prompt in meta["prompt"]:
                    if "artifact" in prompt:
                        if "binary" in prompt["artifact"]:
                            del prompt["artifact"]["binary"]
                        if "lora" in prompt["artifact"]:
                            del prompt["artifact"]["lora"]

                for i, (result_image, nsfw) in enumerate(zip(results[0], results[1])):
                    answer = generation_pb2.Answer()
                    answer.request_id = request.request_id
                    answer.answer_id = f"{request.request_id}-{ctr}"

                    img_seed = seeds[i] if i < len(seeds) else 0

                    if self._supress_metadata:
                        artifact = image_to_artifact(result_image)
                    else:
                        meta["image"]["samples"] = 1
                        meta["image"]["seed"] = [img_seed]
                        artifact = image_to_artifact(
                            result_image,
                            meta={"generation_parameters": json.dumps(meta)},
                        )

                    artifact.finish_reason = (
                        generation_pb2.FILTER if nsfw else generation_pb2.NULL
                    )
                    artifact.index = ctr
                    artifact.seed = img_seed
                    answer.artifacts.append(artifact)

                    recorder.store("pipe.generate result", artifact)

                    yield answer
                    ctr += 1

                if stop_event.is_set():
                    break

            if self._ram_monitor:
                self._ram_monitor.print()

    def _try_deleting_context(self, key):
        """
        Since multiple threads might be deleting contexts, we need to wrap
        it in a try block to avoid failing if we attempt to double-delete.
        """
        try:
            del self._async_contexts[key]
        except KeyError:
            pass

    def _check_deadlines(self):
        deadline_expired = [
            key for key, value in self._async_contexts.items() if value.past_deadline()
        ]

        for key in deadline_expired:
            self._try_deleting_context(key)

    @exception_to_grpc
    def AsyncGenerate(self, request: generation_pb2.Request, context):
        self._check_deadlines()

        async_context = AsyncContext()
        check_context = None
        handle = None

        # Find an unusued handle.
        while check_context is not async_context:
            handle = str(uuid.uuid4())
            # Done a slightly weird way to ensure dict access is atomic
            check_context = self._async_contexts.setdefault(handle, async_context)

        # Start the request in a thread.
        # TODO: Ideally this would be in a queue too rather than spawning threads

        def thread_function():
            try:
                for answer in self.Generate(request, async_context):
                    async_context.queue.put(answer)
            except grpc.RpcError:
                # RpcError will have set code and details already in context#abort
                pass
            finally:
                async_context.queue.put("DONE")
                # Remove queue after 10 minutes, to avoid queues that never get
                # emptied by clients from consuming memory
                async_context.set_deadline(60 * 10)

        async_context.thread = threading.Thread(target=thread_function)
        async_context.thread.start()

        return generation_pb2.AsyncHandle(
            request_id=request.request_id, async_handle=handle
        )

    @exception_to_grpc
    def AsyncResult(self, request, context):
        self._check_deadlines()

        async_context = self._async_contexts.get(request.async_handle)

        if not async_context:
            context.abort(grpc.StatusCode.NOT_FOUND, "No such async handle")
        assert async_context  # context.abort will raise an exception

        async_answer = generation_pb2.AsyncAnswer(complete=False)

        try:
            while True:
                answer = async_context.queue.get(timeout=2)
                if answer == "DONE":
                    async_answer.complete = True
                    async_answer.status.code = async_context.code
                    async_answer.status.message = async_context.message
                    break
                else:
                    async_answer.answer.append(answer)
        except Empty:
            pass

        if async_answer.complete:
            self._try_deleting_context(request.async_handle)

        return async_answer

    @exception_to_grpc
    def AsyncCancel(self, request, context):
        self._check_deadlines()

        async_context = self._async_contexts.get(request.async_handle)

        if not async_context:
            context.abort(grpc.StatusCode.NOT_FOUND, "No such async handle")
        assert async_context  # context.abort will raise an exception

        async_context.cancel()

        self._try_deleting_context(request.async_handle)

        return generation_pb2.AsyncCancelAnswer()
