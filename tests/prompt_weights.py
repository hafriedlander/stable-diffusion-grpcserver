import numpy as np
from test_harness import ALGORITHMS, TestHarness


class TestRunner(TestHarness):
    """
    Tests to ensure grafted inpaint works
    """

    def params(self, **extra):
        return {
            "height": 512,
            "width": 512,
            "guidance_scale": 7.5,
            "sampler": ALGORITHMS["k_euler"],
            "num_inference_steps": 64,
            "seed": 420420420,
            **extra,
        }

    def test(self):
        def gen(args, tag):
            pipeline = self.get_pipeline()
            self.save_output(f"{tag}", pipeline.generate(**args)[0])

        for i in np.linspace(-0.5, 0.5, 5):
            prompt_tokens = [
                (
                    "So let me tell you a story. One day I was walking under the summer sun. "
                    "I had decided I wanted to take the evening air, and so had left the house around 5pm. "
                    "It had been raining earlier, but in this golden hour the sun was gently warm against my skin. "
                    "As I rounded a corner I had not walked around before I came across a wonderful sight. ",
                    1.0,
                ),
                ("A DSLR photo of a meadow filled with ", 1.0),
                ("daisies", 1.0 + i),
                (" and ", 1.0),
                ("tulips", 1.0 - i),
                (", f/2.8 35mm Portra 400", 1.0),
            ]

            kwargs = self.params(
                prompt=[prompt_tokens],
            )

            gen(kwargs, f"{i}")

            clipargs = dict(**kwargs, clip_guidance_scale=0.5)
            clipargs["num_inference_steps"] = 96

            gen(clipargs, f"clip_{i}")


runner = TestRunner(
    engine_path="prompt_weights.engine.yaml", prefix="prompt_weights", vramO=3
)
runner.run()
