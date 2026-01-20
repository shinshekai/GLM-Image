import gradio as gr
import numpy as np
import random
import torch
from diffusers.pipelines.glm_image import GlmImagePipeline
from datetime import datetime
from PIL import Image
import devicetorch
import os
import sys
from huggingface_hub import snapshot_download, hf_hub_download
from huggingface_hub.utils import tqdm as hf_tqdm
from tqdm import tqdm
import logging

# Enable verbose logging for huggingface_hub downloads
logging.getLogger("huggingface_hub").setLevel(logging.INFO)

# Enable progress bars globally
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # Use requests-based download for better progress
os.environ["TQDM_POSITION"] = "0"

# Configuration
dtype = torch.bfloat16
device = devicetorch.get(torch)
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048

# Global pipeline
pipe = None

css = """
nav {
  text-align: center;
}
#logo {
  width: 50px;
  display: inline;
}
.container {
  max-width: 1200px;
  margin: auto;
}
"""

def format_size(size_bytes):
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def download_with_progress(repo_id="zai-org/GLM-Image"):
    """Download model files with detailed progress tracking."""
    from huggingface_hub import HfApi, hf_hub_url
    from huggingface_hub.file_download import hf_hub_download
    import requests
    
    print("\n" + "=" * 60)
    print("📥 Checking/Downloading model files...")
    print("=" * 60 + "\n")
    
    # Get list of files in the repository
    api = HfApi()
    try:
        repo_info = api.repo_info(repo_id=repo_id, repo_type="model")
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    except Exception as e:
        print(f"⚠️ Could not fetch repo info: {e}")
        print("Proceeding with default download...")
        return None
    
    # Calculate total size and get file info
    total_size = 0
    file_infos = []
    
    print("📋 Fetching file information...")
    for filename in files:
        try:
            # Get file metadata
            file_url = hf_hub_url(repo_id=repo_id, filename=filename)
            response = requests.head(file_url, allow_redirects=True, timeout=10)
            file_size = int(response.headers.get('content-length', 0))
            total_size += file_size
            file_infos.append((filename, file_size))
        except Exception:
            file_infos.append((filename, 0))
    
    print(f"\n📦 Total model size: {format_size(total_size)}")
    print(f"📁 Number of files: {len(files)}")
    print("-" * 60 + "\n")
    
    # Download each file with progress
    downloaded_size = 0
    for i, (filename, file_size) in enumerate(file_infos, 1):
        size_str = format_size(file_size) if file_size > 0 else "Unknown size"
        print(f"[{i}/{len(files)}] Downloading: {filename}")
        print(f"         Size: {size_str}")
        
        try:
            # Download with progress bar
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                resume_download=True,
                force_download=False,
            )
            downloaded_size += file_size
            progress_pct = (downloaded_size / total_size * 100) if total_size > 0 else 0
            print(f"         ✅ Complete! Overall progress: {format_size(downloaded_size)}/{format_size(total_size)} ({progress_pct:.1f}%)\n")
        except Exception as e:
            print(f"         ⚠️ Error downloading {filename}: {e}\n")
    
    print("=" * 60)
    print("✅ All model files ready!")
    print("=" * 60 + "\n")
    
    return True


def load_pipeline():
    """Load the GLM-Image pipeline if not already loaded."""
    global pipe
    if pipe is None:
        # First, download all files with progress tracking
        download_with_progress("zai-org/GLM-Image")
        
        print("\n🔄 Loading GLM-Image pipeline into memory...")
        print("   (This may take a moment...)\n")
        
        pipe = GlmImagePipeline.from_pretrained(
            "zai-org/GLM-Image",
            torch_dtype=dtype,
            device_map="cuda"
        )
        print("✅ Pipeline loaded successfully!")
    return pipe


def save_image(image):
    """Save generated image to output folder with unique name."""
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"glm_image_{timestamp}.png"
    filepath = os.path.join(output_folder, filename)
    image.save(filepath)
    
    return filepath


def text_to_image(
    prompt,
    negative_prompt="",
    seed=42,
    randomize_seed=True,
    width=1024,
    height=1024,
    num_inference_steps=50,
    guidance_scale=1.5
):
    """Generate image from text prompt."""
    pipeline = load_pipeline()
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    print(f"Generating image with prompt: {prompt[:100]}...")
    
    # Ensure dimensions are multiples of 32
    width = (width // 32) * 32
    height = (height // 32) * 32
    
    try:
        # Try with negative prompt first
        kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "generation_config": None,
        }
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
            
        try:
            image = pipeline(**kwargs).images[0]
        except TypeError as e:
            if "negative_prompt" in str(e):
                print(f"⚠️ Pipeline does not support negative_prompt, ignoring it.")
                del kwargs["negative_prompt"]
                image = pipeline(**kwargs).images[0]
            else:
                raise e
                
    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)}")
    
    print("Image generated!")
    devicetorch.empty_cache(torch)
    
    saved_path = save_image(image)
    print(f"Image saved to: {saved_path}")
    
    return image, seed, saved_path


def image_to_image(
    input_image,
    prompt,
    negative_prompt="",
    seed=42,
    randomize_seed=True,
    width=1024,
    height=1024,
    num_inference_steps=50,
    guidance_scale=1.5
):
    """Generate image from input image and text prompt."""
    if input_image is None:
        raise gr.Error("Please upload an input image.")
    
    pipeline = load_pipeline()
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    # Convert input image to PIL if needed
    if not isinstance(input_image, Image.Image):
        input_image = Image.fromarray(input_image).convert("RGB")
    else:
        input_image = input_image.convert("RGB")
    
    print(f"Generating image with prompt: {prompt[:100]}...")
    
    # Ensure dimensions are multiples of 32
    width = (width // 32) * 32
    height = (height // 32) * 32
    
    try:
        # Try with negative prompt first
        kwargs = {
            "prompt": prompt,
            "image": [input_image],
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "generation_config": None,
        }
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
            
        try:
            image = pipeline(**kwargs).images[0]
        except TypeError as e:
            if "negative_prompt" in str(e):
                print(f"⚠️ Pipeline does not support negative_prompt, ignoring it.")
                del kwargs["negative_prompt"]
                image = pipeline(**kwargs).images[0]
            else:
                raise e
                
    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)}")
    
    print("Image generated!")
    devicetorch.empty_cache(torch)
    
    saved_path = save_image(image)
    print(f"Image saved to: {saved_path}")
    
    return image, seed, saved_path


# Build the Gradio interface
# Build the Gradio interface
# Build the Gradio interface
# Build the Gradio interface
# Build the Gradio interface
with gr.Blocks(title="GLM-Image", css=css) as demo:
    gr.Markdown(
        """
        # 🎨 GLM-Image Generator
        Generate high-quality images using the **zai-org/GLM-Image** model.
        """
    )
    
    with gr.Tabs():
        # Text-to-Image Tab
        with gr.TabItem("Text to Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    t2i_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe the image you want to generate...",
                        lines=3
                    )
                    t2i_negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="What to exclude (e.g., blurry, bad quality, distorted)...",
                        lines=2
                    )
                    
                    with gr.Accordion("Advanced Settings", open=False):
                        with gr.Row():
                            t2i_width = gr.Slider(
                                label="Width",
                                minimum=256,
                                maximum=MAX_IMAGE_SIZE,
                                step=32,
                                value=1024
                            )
                            t2i_height = gr.Slider(
                                label="Height",
                                minimum=256,
                                maximum=MAX_IMAGE_SIZE,
                                step=32,
                                value=1024
                            )
                        with gr.Row():
                            t2i_steps = gr.Slider(
                                label="Inference Steps",
                                minimum=10,
                                maximum=100,
                                step=1,
                                value=50
                            )
                            t2i_guidance = gr.Slider(
                                label="Guidance Scale",
                                minimum=0,
                                maximum=20,
                                step=0.1,
                                value=1.5
                            )
                        with gr.Row():
                            t2i_seed = gr.Number(
                                label="Seed",
                                value=42,
                                precision=0,
                                step=1
                            )
                            t2i_randomize = gr.Checkbox(
                                label="Randomize Seed",
                                value=True
                            )
                            
                    t2i_generate = gr.Button("✨ Generate Image", variant="primary", scale=1)
                
                with gr.Column(scale=1):
                    t2i_output = gr.Image(label="Generated Image", type="pil", interactive=False)
                    with gr.Row():
                        t2i_seed_output = gr.Number(label="Used Seed", interactive=False)
                        t2i_path_output = gr.Textbox(label="Saved Path", interactive=False)
            
            t2i_generate.click(
                fn=text_to_image,
                inputs=[
                    t2i_prompt, 
                    t2i_negative_prompt, 
                    t2i_seed, 
                    t2i_randomize, 
                    t2i_width, 
                    t2i_height, 
                    t2i_steps, 
                    t2i_guidance
                ],
                outputs=[t2i_output, t2i_seed_output, t2i_path_output]
            )
        
        # Image-to-Image Tab
        with gr.TabItem("Image to Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    i2i_input = gr.Image(label="Input Image", type="pil", height=300)
                    i2i_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe how you want to modify the image...",
                        lines=3
                    )
                    i2i_negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="What to exclude (e.g., blurry, bad quality)...",
                        lines=2
                    )
                    
                    with gr.Accordion("Advanced Settings", open=False):
                        with gr.Row():
                            i2i_width = gr.Slider(
                                label="Width",
                                minimum=256,
                                maximum=MAX_IMAGE_SIZE,
                                step=32,
                                value=1024
                            )
                            i2i_height = gr.Slider(
                                label="Height",
                                minimum=256,
                                maximum=MAX_IMAGE_SIZE,
                                step=32,
                                value=1024
                            )
                        with gr.Row():
                            i2i_steps = gr.Slider(
                                label="Inference Steps",
                                minimum=10,
                                maximum=100,
                                step=1,
                                value=50
                            )
                            i2i_guidance = gr.Slider(
                                label="Guidance Scale",
                                minimum=0,
                                maximum=20,
                                step=0.1,
                                value=1.5
                            )
                        with gr.Row():
                            i2i_seed = gr.Number(
                                label="Seed",
                                value=42,
                                precision=0,
                                step=1
                            )
                            i2i_randomize = gr.Checkbox(
                                label="Randomize Seed",
                                value=True
                            )
                            
                    i2i_generate = gr.Button("✨ Generate Image", variant="primary", scale=1)
                
                with gr.Column(scale=1):
                    i2i_output = gr.Image(label="Generated Image", type="pil", interactive=False)
                    with gr.Row():
                        i2i_seed_output = gr.Number(label="Used Seed", interactive=False)
                        i2i_path_output = gr.Textbox(label="Saved Path", interactive=False)
            
            i2i_generate.click(
                fn=image_to_image,
                inputs=[
                    i2i_input, 
                    i2i_prompt, 
                    i2i_negative_prompt, 
                    i2i_seed, 
                    i2i_randomize, 
                    i2i_width, 
                    i2i_height, 
                    i2i_steps, 
                    i2i_guidance
                ],
                outputs=[i2i_output, i2i_seed_output, i2i_path_output]
            )

        # About Tab
        with gr.TabItem("About"):
            gr.Markdown(
                """
                # ℹ️ About GLM-Image
                
                This application is a user interface for the **GLM-Image** generation model.
                
                ## Open Source Licenses
                This software uses the following third-party open-source components:
                
                *   **GLM-Image** (MIT / Apache 2.0 Components): [Hugging Face](https://huggingface.co/zai-org/GLM-Image)
                *   **Gradio** (Apache 2.0): [GitHub](https://github.com/gradio-app/gradio)
                *   **Diffusers** (Apache 2.0): [GitHub](https://github.com/huggingface/diffusers)
                *   **Transformers** (Apache 2.0): [GitHub](https://github.com/huggingface/transformers)
                *   **Hugging Face Hub** (Apache 2.0): [GitHub](https://github.com/huggingface/huggingface_hub)
                *   **PyTorch** (BSD-style): [Website](https://pytorch.org/)
                *   **NumPy** (BSD): [Website](https://numpy.org/)
                *   **Pillow** (HPND): [GitHub](https://github.com/python-pillow/Pillow)
                
                Please refer to their respective repositories for full license text.
                """
            )

demo.launch(theme=gr.themes.Base(), server_name="127.0.0.1")
