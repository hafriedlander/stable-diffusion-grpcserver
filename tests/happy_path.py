import io
import re
from collections import OrderedDict

from PIL import Image, ImageOps
from test_harness import ALGORITHMS, TestHarness, VRAMUsageMonitor, generation_pb2

from sdgrpcserver import images


def load_masked_image(path):
    test_image = Image.open(path)

    # Split into 3 channels
    r, g, b, a = test_image.split()
    # Recombine back to RGB image
    test_image = Image.merge("RGB", (r, g, b))
    test_mask = Image.merge("RGB", (a, a, a))
    test_mask = ImageOps.invert(test_mask)

    with io.BytesIO() as output:
        test_image.save(output, format="PNG")
        test_image_png = output.getvalue()

    with io.BytesIO() as output:
        test_mask.save(output, format="PNG")
        test_mask_png = output.getvalue()

    return test_image_png, test_mask_png


args = OrderedDict()
args["sampler"] = [
    {"sampler": "ddim", "eta": 0},
    {"sampler": "ddim", "eta": 0.8},
    {"sampler": "plms"},
    {"sampler": "k_lms"},
    {"sampler": "k_euler"},
    {"sampler": "k_euler_ancestral"},
    {"sampler": "k_heun"},
    {"sampler": "k_dpm_2"},
    {"sampler": "k_dpm_2_ancestral"},
    {"sampler": "dpm_fast"},
    {"sampler": "dpm_adaptive"},
    {"sampler": "dpmspp_1"},
    {"sampler": "dpmspp_2"},
    {"sampler": "dpmspp_3"},
    {"sampler": "dpmspp_2s_ancestral"},
    {"sampler": "dpmspp_sde"},
    {"sampler": "dpmspp_2m"},
]
args["image"] = [
    {},
    {"image": True, "strength": 0.25},
    {"image": True, "strength": 0.5},
    {"image": True, "strength": 0.75},
    {"image": True, "mask": True, "strength": 0.5},
    {"image": True, "mask": True, "strength": 1},
    {
        "image": True,
        "mask": True,
        "strength": 1.5,
    },
]
args["engine"] = [{"engine": "sd1"}, {"engine": "sd2"}, {"engine": "sd2v"}]


image_by_size = {
    512: load_masked_image("happy_path.image_512.png"),
    768: load_masked_image("happy_path.image_768.png"),
}


class TestRunner(TestHarness):
    def __init__(self, combos, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.combos = combos

    def engine(self, item, request, prompt, parameters):
        request.engine_id = item["engine"]

        if item["engine"] == "sd2v":
            request.image.width = 768
            request.image.height = 768

    def sampler(self, item, request, prompt, parameters):
        request.image.transform.diffusion = ALGORITHMS[item["sampler"]]

        eta = item.get("eta", None)
        if eta != None:
            parameters.sampler.eta = eta

    def image(self, item, request, prompt, parameters):
        image, mask = image_by_size[request.image.height]

        if item.get("image", False):
            prompt.append(
                generation_pb2.Prompt(
                    parameters=generation_pb2.PromptParameters(init=True),
                    artifact=generation_pb2.Artifact(
                        type=generation_pb2.ARTIFACT_IMAGE, binary=image
                    ),
                )
            )

            parameters.schedule.start = item["strength"]
            parameters.schedule.end = 0.01

        if item.get("mask", False):
            prompt.append(
                generation_pb2.Prompt(
                    artifact=generation_pb2.Artifact(
                        type=generation_pb2.ARTIFACT_MASK, binary=mask
                    )
                )
            )

    def build_combinations(self, args, idx):
        if idx == len(args.keys()) - 1:
            key = list(args.keys())[idx]
            return [{key: item} for item in args[key]]

        key = list(args.keys())[idx]
        result = []

        for item in args[key]:
            result += [
                {**combo, key: item} for combo in self.build_combinations(args, idx + 1)
            ]

        return result

    def test(self):
        combinations = self.build_combinations(self.combos, 0)

        for combo in combinations:
            request_id = re.sub("[^\w]+", "_", repr(combo))
            request_id = request_id.strip("_")

            prompt = [
                generation_pb2.Prompt(
                    text="Award wining DSLR photo of a shark in the deep ocean, f2/8 35mm Portra 400, highly detailed, trending on artstation"
                )
            ]

            parameters = generation_pb2.StepParameter()

            request = generation_pb2.Request(
                engine_id="testengine",
                request_id=request_id,
                prompt=[],
                image=generation_pb2.ImageParameters(
                    height=512,
                    width=512,
                    seed=[420420420],  # It's the funny number
                    steps=50,
                    samples=1,
                    parameters=[],
                ),
            )

            if (
                combo["sampler"]["sampler"] == "plms"
                and combo["engine"]["engine"] == "sd2v"
            ):
                continue

            for key, item in combo.items():
                getattr(self, key)(item, request, prompt, parameters)

            for part in prompt:
                request.prompt.append(part)

            request.image.parameters.append(parameters)

            self.save_output(request_id, self.call_generator(request))


monitor = VRAMUsageMonitor()
monitor.start()

stats = {}

for vramO in range(4):
    instance = TestRunner(
        engine_path="happy_path.engines.yaml",
        combos=args,
        prefix=f"hp_{vramO}",
        vramO=vramO,
        monitor=monitor,
    )
    stats[f"run vram-optimisation-level={vramO}"] = instance.run()

monitor.stop()

print("Stats")
print(repr(stats))
