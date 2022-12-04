import os

from sdgrpcserver import images

DEFAULT_ENABLED = set(
    [
        # "initial",
        # "step",
        # "mask",
        # "shapednoise",
        # "initnoise",
        # "blendin",
        # "blendout",
        # "small",
        # "hires_lo",
        # "hires_hi",
    ]
)

DEFAULT_OUTPUT_PATH = "/tests/debug-out/"


class LatentDebugger:
    def __init__(self, vae, output_path=DEFAULT_OUTPUT_PATH, enabled=None, prefix=""):
        self.vae = vae
        self.output_path = output_path
        self.enabled = enabled if enabled is not None else DEFAULT_ENABLED
        self.prefix = prefix

        self.counters = {}

    def log(self, label, i, latents):
        if label not in self.enabled:
            return

        prefix = "debug" if not self.prefix else f"debug-{self.prefix}"

        self.counters[label] = i = self.counters.get(label, 0) + 1

        stage_latents = 1 / 0.18215 * latents
        stage_image = self.vae.decode(stage_latents).sample
        stage_image = (stage_image / 2 + 0.5).clamp(0, 1).cpu()

        for j, pngBytes in enumerate(images.toPngBytes(stage_image)):
            path = os.path.join(self.output_path, f"{prefix}-{label}-{j}-{i}.png")
            with open(path, "wb") as f:
                f.write(pngBytes)
