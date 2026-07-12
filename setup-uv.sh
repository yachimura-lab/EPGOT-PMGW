#/bin/bash
# uv のバージョン固定
UV_VERSION=0.5.9
curl -LsSf https://astral.sh/uv/$UV_VERSION/install.sh | sh
echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
