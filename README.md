An implementation of a server for the Stability AI API

Working:

- Txt2Img and Img2Img from Stability-AI/Stability-SDK, specifying a prompt
- Can load multiple pipelines, such as Stable and Waifu Diffusion, and swap between them as needed
- Mid and Low VRAM modes for larger generated images at the expense of some performance
- Adjustable NSFW behaviour
- Masked painting uses diffusers inpainting when strength < 1, and Parlance's seamless outpainting when strength >= 1

Core API functions not working yet:

- Most samplers (like euler_a) are not currently supported in Diffusers

Thanks to / Credits:

- Seamless outpainting https://github.com/parlance-zz/g-diffuser-bot/tree/g-diffuser-bot-beta2

Extensions not done yet:

- Cancel and progress over API
- Negative prompting included but not exposed over API
- Embedding params in png
- Extra APIs
  - Image resizing
  - Aspect ratio shifting
  - Asset management
  - Extension negotiation so we can:
    - Ping back progress notices
    - Allow cancellation requests
    - Specify negative prompts
- CLIP guided generation https://github.com/huggingface/diffusers/pull/561
- Community features: 
  - Prompt calculation https://github.com/pharmapsychotic/clip-interrogator/blob/main/clip_interrogator.ipynb
  - Prompt suggestion https://huggingface.co/spaces/Gustavosta/MagicPrompt-Stable-Diffusion
  - Prompt compositing https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch
  - Automasking https://github.com/ThstereforeGames/txt2mask
  - Other schedulers (needs custom pipeline for some). https://github.com/huggingface/diffusers/commit/489894e4d9272dec88fa3a64a8161aeab471fc18
  - Huge seeds
- Other thoughts
  - Figure out how to just suppress NSFW filtering altogether (takes VRAM, if you're not interested)