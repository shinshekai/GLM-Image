# GLM-Image Pinokio App

Generate images using the [zai-org/GLM-Image](https://huggingface.co/zai-org/GLM-Image) model with a Gradio web interface.

## Features

- **Text-to-Image**: Generate images from text descriptions
- **Image-to-Image**: Edit images with text prompts (style transfer, background replacement, etc.)
- Auto-saves generated images to `output/` folder
- Adjustable parameters: dimensions, inference steps, guidance scale, seed

## Requirements

- CUDA-capable GPU with 16GB+ VRAM (recommended)
- The model (~30-40GB) will be downloaded on first use

## Usage

1. Click **Install** to set up dependencies
2. Click **Start** to launch the web UI
3. Open the Web UI link when it appears

### Text-to-Image

1. Enter a detailed prompt describing your desired image
2. Adjust width, height, steps, and guidance scale
3. Click **Generate**

### Image-to-Image

1. Upload an input image
2. Describe the modifications (e.g., "Replace the background with a beach")
3. Click **Generate**

## API Usage

### Python

```python
from diffusers.pipelines.glm_image import GlmImagePipeline
import torch

pipe = GlmImagePipeline.from_pretrained(
    "zai-org/GLM-Image",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

# Text-to-Image
image = pipe(
    prompt="A cute robot in a garden",
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=1.5,
).images[0]
image.save("output.png")
```

### cURL (when server is running)

```bash
curl -X POST http://localhost:7860/api/predict \
  -H "Content-Type: application/json" \
  -d '{"data": ["A cute robot in a garden", 42, true, 1024, 1024, 50, 1.5]}'
```

## License

This app uses the GLM-Image model. Please refer to the [model page](https://huggingface.co/zai-org/GLM-Image) for license information.
