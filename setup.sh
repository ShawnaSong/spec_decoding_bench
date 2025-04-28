#!/bin/bash

curl -Ls https://astral.sh/uv/install.sh | sh

export PATH="$HOME/.local/bin:$PATH"

uv venv myenv --python 3.10 --seed
source myenv/bin/activate
uv pip install vllm

echo ""
echo "Virtual environment 'myenv' has been created successfully!"
echo "Please manually activate it by running:"
echo ""
echo "    source myenv/bin/activate"
echo ""
echo "and to install vllm, run:"
echo ""
echo "    uv pip install vllm"
echo ""