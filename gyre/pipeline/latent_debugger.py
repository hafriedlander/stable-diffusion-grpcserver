import os

from gyre import images

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
        # "natinit_image",
        # "init_image",
        # "natmask_image",
        # "mask_image",
        # "natdepth_map",
        # "depth_map",
        # "hires_in",
        # "hires_lopre",
        # "hires_hipre",
        # "hires",
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

    def log(self, label, i, latents=None, pixels=None):
        if label not in self.enabled:
            return

        prefix = "debug" if not self.prefix else f"debug-{self.prefix}"

        self.counters[label] = i = self.counters.get(label, 0) + 1

        if latents is not None:
            stage_latents = 1 / 0.18215 * latents
            stage_image = self.vae.decode(stage_latents).sample
            pixels = (stage_image / 2 + 0.5).clamp(0, 1).cpu()

        for j, pngBytes in enumerate(images.toPngBytes(pixels)):
            path = os.path.join(self.output_path, f"{prefix}-{label}-{j}-{i}.png")
            with open(path, "wb") as f:
                f.write(pngBytes)
