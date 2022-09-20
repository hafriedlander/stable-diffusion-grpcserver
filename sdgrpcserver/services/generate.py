
from types import SimpleNamespace as SN

import grpc
from generated import generation_pb2, generation_pb2_grpc

from sdgrpcserver.utils import image_to_artifact, artifact_to_image

class GenerationServiceServicer(generation_pb2_grpc.GenerationServiceServicer):
    def __init__(self, manager):
        self._manager = manager

    def unimp(self):
        raise NotImplementedError('Generation of anything except images not implemented')
        
    def Generate(self, request, context):
        try:
            # Assume that "None" actually means "Image" (stability-sdk/client.py doesn't set it)
            if request.requested_type != generation_pb2.ARTIFACT_NONE and request.requested_type != generation_pb2.ARTIFACT_IMAGE:
                print(request.requested_type)
                context.set_code(grpc.StatusCode.UNIMPLEMENTED)
                context.set_details('Generation of anything except images not implemented')
                raise NotImplementedError('Generation of anything except images not implemented')

            image=None
            mask=None
            text=""
            for prompt in request.prompt:
                which = prompt.WhichOneof("prompt")
                if which == "text": text += prompt.text
                elif which == "sequence": self.unimp()
                else:
                  if prompt.artifact.type == generation_pb2.ARTIFACT_IMAGE:
                    image=artifact_to_image(prompt.artifact)
                  else:
                    self.unimp()


            params=SN(
                height=512,
                width=512,
                samples=1,
                steps=50
            )

            for field in vars(params):
                if request.image.HasField(field):
                    setattr(params, field, getattr(request.image, field))
            
            params.seed = -1
            for seed in request.image.seed: params.seed = seed

            pipe = self._manager.getPipe(request.engine_id)
            images = pipe.generate(text=text, image=image, params=params)

            answer = generation_pb2.Answer()
            answer.request_id="x" # TODO - not this, copy from request
            answer.answer_id="y"
            for image in images[0]: answer.artifacts.append(image_to_artifact(image))

            yield answer
        except Exception as e:
            print(e)
