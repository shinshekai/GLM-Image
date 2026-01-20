module.exports = {
    run: [
        // Install Python dependencies
        {
            method: "shell.run",
            params: {
                venv: "../env",
                path: "app",
                env: {
                    HF_HUB_ENABLE_HF_TRANSFER: "1",
                    PIP_PROGRESS_BAR: "on"
                },
                message: [
                    "uv pip install -r requirements.txt"
                ],
            }
        },
        // Install PyTorch with CUDA support
        {
            method: "script.start",
            params: {
                uri: "torch.js",
                params: {
                    venv: "env"
                }
            }
        }
    ]
}
