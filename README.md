An implementation of a server for the Stability AI API

Working:

- Txt2Img and Img2Img from Stability-AI/Stability-SDK, specifying a prompt
- Can load multiple pipelines, such as Stable and Waifu Diffusion, and swap between them as needed

Core API functions not working yet:

- Most parameters not yet passed through to pipeline
- Inpainting included but not exposed over API
- Most samplers (like euler_a) are not currently supported in Diffusers

Extensions not done yet:

- Negative prompting included but not exposed over API
- Very low VRAM mode (unet only in CUDA) not working yet
- Embedding params in png
- Enable / disable NSFW filter
- Extra APIs: Image resizing, aspect ratio shifting, asset management
- Community features: Prompt calculation, prompt suggestion, seamless outpainting, prompt compositing
