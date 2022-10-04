
import random, traceback, threading
from types import SimpleNamespace as SN

import grpc
import generation_pb2, generation_pb2_grpc

from sdgrpcserver.utils import image_to_artifact, artifact_to_image

from sdgrpcserver import images

def buildDefaultMaskPostAdjustments():
    hardenMask = generation_pb2.ImageAdjustment()
    hardenMask.levels.input_low = 0
    hardenMask.levels.input_high = 0.05
    hardenMask.levels.output_low = 0
    hardenMask.levels.output_high = 1

    blur = generation_pb2.ImageAdjustment()
    blur.blur.sigma = 48

    levels = generation_pb2.ImageAdjustment()
    levels.levels.input_low = 0
    levels.levels.input_high = 0.5
    levels.levels.output_low = 0
    levels.levels.output_high = 1

    return [hardenMask, blur, levels]

defaultMaskPostAdjustments = buildDefaultMaskPostAdjustments();

class GenerationServiceServicer(generation_pb2_grpc.GenerationServiceServicer):
    def __init__(self, manager):
        self._manager = manager

    def unimp(self, what):
        raise NotImplementedError(f"{what} not implemented")

    def _handleImageAdjustment(self, tensor, adjustments):
        if type(tensor) is bytes: tensor = images.fromPngBytes(tensor)
        
        for adjustment in adjustments:
            which = adjustment.WhichOneof("adjustment")

            if which == "blur":
                tensor = images.gaussianblur(tensor, adjustment.blur.sigma)
            elif which == "invert":
                tensor = images.invert(tensor)
            elif which == "levels":
                tensor = images.levels(tensor, adjustment.levels.input_low, adjustment.levels.input_high, adjustment.levels.output_low, adjustment.levels.output_high)
            elif which == "channels":
                tensor = images.channelmap(tensor, [adjustment.channels.r,  adjustment.channels.g,  adjustment.channels.b,  adjustment.channels.a])
            elif which == "rescale":
                raise NotImplementedError("Rescale not currently implemented")
        
        return tensor

    def Generate(self, request, context):
        try:
            # Assume that "None" actually means "Image" (stability-sdk/client.py doesn't set it)
            if request.requested_type != generation_pb2.ARTIFACT_NONE and request.requested_type != generation_pb2.ARTIFACT_IMAGE:
                self.unimp('Generation of anything except images')

            # Extract prompt inputs
            image=None
            inMask=None
            outMask=None
            text=""
            negative=""

            for prompt in request.prompt:
                which = prompt.WhichOneof("prompt")
                if which == "text": 
                    print(prompt, prompt.HasField("parameters"), prompt.parameters.HasField("weight") if prompt.HasField("parameters") else "")
                    if prompt.HasField("parameters") and prompt.parameters.HasField("weight") and prompt.parameters.weight < 0:
                        negative += prompt.text
                    else:
                        text += prompt.text
                elif which == "sequence": 
                    self.unimp("Sequence prompts")
                else:
                    if prompt.artifact.type == generation_pb2.ARTIFACT_IMAGE:
                        image = images.fromPngBytes(prompt.artifact.binary).to(self._manager.device)
                        image = self._handleImageAdjustment(image, prompt.artifact.adjustments)
                    elif prompt.artifact.type == generation_pb2.ARTIFACT_MASK:
                        mask = images.fromPngBytes(prompt.artifact.binary).to(self._manager.device)
                        inMask = self._handleImageAdjustment(mask, prompt.artifact.adjustments)

                        postAdjustments = prompt.artifact.postAdjustments
                        if not postAdjustments: postAdjustments = defaultMaskPostAdjustments

                        outMask = self._handleImageAdjustment(inMask, postAdjustments)
                    else:
                        self.unimp(f"Artifact prompts of type {prompt.artifact.type}")

            params=SN(
                height=512,
                width=512,
                cfg_scale=7.5,
                eta=0,
                sampler=None,
                steps=50,
                seed=-1,
                samples=1,
                strength=0.8
            )

            for field in vars(params):
                try:
                    if request.image.HasField(field):
                        setattr(params, field, getattr(request.image, field))
                except Exception as e:
                    pass
            
            seeds = list(request.image.seed)

            for extras in request.image.parameters:
                if extras.HasField("sampler"):
                    if extras.sampler.HasField("cfg_scale"): params.cfg_scale = extras.sampler.cfg_scale
                    if extras.sampler.HasField("eta"): params.eta = extras.sampler.eta
                if extras.HasField("schedule"):
                    if extras.schedule.HasField("start"): params.strength = extras.schedule.start            
            
            if request.image.HasField("transform") and request.image.transform.WhichOneof("type") == "diffusion": params.sampler = request.image.transform.diffusion

            try:
                pipe = self._manager.getPipe(request.engine_id)
            except KeyError as e:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Engine not found")
                return

            ctr = 0
            last_seed = -1

            stop_event = threading.Event()
            context.add_callback(lambda: stop_event.set())

            for _ in range(params.samples):
                if seeds:
                    seed = seeds.pop(0)
                elif last_seed != -1:
                    seed = last_seed + 1
                else:
                    seed = -1

                if seed == -1: seed = random.randrange(0, 4294967295)

                params.seed = last_seed = seed
                print(f'Generating {repr(params)}, {"with Image" if image != None else ""}, {"with Mask" if inMask != None else ""}')
                results = pipe.generate(text=text, negative_text=negative, image=image, mask=inMask, outmask=outMask, params=params, stop_event=stop_event)

                for result_image, nsfw in zip(results[0], results[1]):
                    answer = generation_pb2.Answer()
                    answer.request_id=request.request_id
                    answer.answer_id=f"{request.request_id}-{ctr}"
                    artifact=image_to_artifact(result_image)
                    artifact.finish_reason=generation_pb2.FILTER if nsfw else generation_pb2.NULL
                    answer.artifacts.append(artifact)
 
                    yield answer
                    ctr += 1
            
        except NotImplementedError as e:
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details(str(e))
            print(f"Unsupported request parameters: {e}")
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Something went wrong")
            traceback.print_exc()
