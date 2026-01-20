module.exports = {
    daemon: true,
    run: [
        {
            method: "shell.run",
            params: {
                venv: "../env",
                path: "app",
                env: {
                    HF_TOKEN: "{{env.HF_TOKEN}}",
                    HF_HUB_ENABLE_HF_TRANSFER: "1"
                },
                message: [
                    "python app.py",
                ],
                on: [{
                    "event": "/(http:\\/\\/\\S+)/",
                    "done": true
                }]
            }
        },
        {
            method: "local.set",
            params: {
                url: "{{input.event[1]}}"
            }
        }
    ]
}
