#!/usr/bin/env bash
set -euo pipefail

# This is the version that generated uv.lock.
UV_VERSION=0.12.3
curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
