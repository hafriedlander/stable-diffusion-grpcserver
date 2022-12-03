"""
isort:skip_file
"""
from test_harness import TestHarness, ALGORITHMS
from sdgrpcserver import images
from PIL import Image, ImageOps


class TestRunner(TestHarness):
    """
    Tests to ensure grafted inpaint works
    """

    def params(self, **extra):
        return {
            "height": 768,
            "width": 768,
            "guidance_scale": 7.5,
            "sampler": ALGORITHMS["k_euler_ancestral"],
            "churn": 0.4,
            "karras_rho": 7,
            "num_inference_steps": 64,
            "seed": 420420420,
            "strength": 1.0,
            **extra,
        }

    def testres(self, width, height):
        test_image = Image.open(f"graftedpaint.image_{width}_{height}.png")

        # Split into 3 channels
        r, g, b, a = test_image.split()
        # Recombine back to RGB image
        test_image = Image.merge("RGB", (r, g, b))
        test_mask = Image.merge("RGB", (a, a, a))
        test_mask = ImageOps.invert(test_mask)

        image = images.fromPIL(test_image).to(self.manager.mode.device)
        mask = images.fromPIL(test_mask).to(self.manager.mode.device)

        def gen(args, engine, grafted, tag):
            pipeline = self.get_pipeline(engine)
            pipeline._pipeline.set_options({"grafted_inpaint": grafted})
            self.save_output(f"{tag}_{width}_{height}", pipeline.generate(**args)[0])

        kwargs = self.params(
            width=width,
            height=height,
            init_image=image,
            mask_image=mask,
            prompt=["An nvinkpunk cat wearing a spacesuit stares at a large moon"],
            seed=[420420420, 420420421, 420420422, 420420423],
            num_images_per_prompt=4,
        )

        gen(kwargs, "justinkpunk", False, "ink")
        gen(kwargs, "withsd2inpaint", False, "sd2")
        gen(kwargs, "withsd2inpaint", True, "graft")

        clipargs = dict(**kwargs, clip_guidance_scale=0.5)
        clipargs["num_inference_steps"] = 96

        gen(clipargs, "justinkpunk", False, "clip_ink")
        gen(clipargs, "withsd2inpaint", False, "clip_sd2")
        gen(clipargs, "withsd2inpaint", True, "clip_graft")

    def test(self):
        self.testres(512, 512)
        self.testres(768, 600)
        self.testres(768, 768)


runner = TestRunner(
    engine_path="graftedpaint.engine.yaml", prefix="graftedpaint", vramO=3
)
runner.run()
