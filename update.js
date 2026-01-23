module.exports = {
    run: [{
        method: "shell.run",
        params: {
            message: "git pull"
        }
    }, {
        // Update PyTorch for correct GPU (RTX 5090 support)
        method: "script.start",
        params: {
            uri: "torch.js",
            params: {
                venv: "env",
                path: "app"
            }
        }
    }]
}
