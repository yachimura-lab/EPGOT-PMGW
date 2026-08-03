#/bin/bash
# Pin the uv version
UV_VERSION=0.5.9
curl -LsSf https://astral.sh/uv/$UV_VERSION/install.sh | sh
echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
