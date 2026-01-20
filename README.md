# GLM-Image

<div align="center">
  <img src="icon.png" width="128" height="128" alt="GLM-Image Icon" />
  <h1>GLM-Image for Pinokio</h1>
  <p>
    <strong>A high-quality, professional-grade image generation UI powered by <a href="https://huggingface.co/zai-org/GLM-Image">GLM-Image</a>.</strong>
  </p>
  <p>
    Run it locally with one click.
  </p>
</div>

---

## 🚀 About

This application brings the power of **GLM-Image** to your desktop. GLM-Image is an open-source image generation model capable of producing stunning, high-resolution visuals from text prompts or existing images.

Built for **Pinokio**, this app offers a fully-featured **Gradio** interface that handles model downloading, environment setup, and inference automatically.

## ✨ Features

*   **🎨 Text-to-Image Generation**: Create detailed, high-resolution (up to 2048x2048) images from simple text descriptions.
*   **🖼️ Image-to-Image Editing**: Transform existing images using text prompts. Change styles, backgrounds, or details easily.
*   **🎛️ Advanced Controls**: Full control over generation parameters:
    *   **Resolution**: Adjustable width and height (up to 2K).
    *   **Quality**: Tune `Inference Steps` and `Guidance Scale`.
    *   **Seed Control**: Randomize or lock seeds for reproducible results.
*   **💾 Auto-Saving**: All generated images are automatically saved to the `output/` directory with timestamps.
*   **🧹 Disk Space Optimizer**: Includes a built-in "Save Disk Space" tool to deduplicate redundant library files, saving gigabytes of storage.

## 🛠️ System Requirements

*   **OS**: Windows, Linux, or macOS
*   **GPU**: NVIDIA GPU with **16GB+ VRAM** recommended (The model is large ~33GB).
*   **Storage**: At least **60GB** of free disk space (for model weights and environment).

## 📦 How to Run

### Option 1: One-Click Install (Pinokio)

1.  Download and install [Pinokio](https://pinokio.computer/).
2.  Navigate to the **Discover** page or paste this repository URL into the Pinokio browser:
    ```
    https://github.com/shinshekai/GLM-Image
    ```
3.  Click **Install**.
4.  Once installed, click **Start**.

### Option 2: Manual Update / Reset

*   **Update**: If a new version is released, click "Update" in the dashboard.
*   **Reset**: If things break, use the "Reset" button to reinstall the environment (this won't delete your `output/` images).
*   **Save Space**: Click "Save Disk Space" in the dashboard menu to optimize storage usage.

## ⚖️ License

**Software License**: This application logic (UI and launcher scripts) is licensed under the terms found in the `LICENSE` file.

**Model License**: The **GLM-Image** model weights are released by **Zhipu AI (zai-org)** under the **MIT License**.
*   **Note**: The tokenizer and specific weights derived from `X-Omni-En` are subject to **Apache 2.0**.
*   Please verify your usage complies with the model's official license at [Hugging Face](https://huggingface.co/zai-org/GLM-Image).

## 🤝 Attribution

This project is built using:
*   [GLM-Image](https://huggingface.co/zai-org/GLM-Image)
*   [Gradio](https://gradio.app/)
*   [Diffusers](https://github.com/huggingface/diffusers)
*   [PyTorch](https://pytorch.org/)

---
<div align="center">
  <sub>Created for the Pinokio Community</sub>
</div>
