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
- CLIP guided generation https://github.com/huggingface/diffusers/pull/561
- Community features: 
  - Prompt calculation https://github.com/pharmapsychotic/clip-interrogator/blob/main/clip_interrogator.ipynb
  - Prompt suggestion https://huggingface.co/spaces/Gustavosta/MagicPrompt-Stable-Diffusion
  - Seamless outpainting https://github.com/parlance-zz/g-diffuser-bot/tree/g-diffuser-bot-beta2
  - Prompt compositing https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch
  - Automasking https://github.com/ThstereforeGames/txt2mask
  - Other schedulers (needs custom pipeline for some). https://github.com/huggingface/diffusers/commit/489894e4d9272dec88fa3a64a8161aeab471fc18
  - Huge seeds
