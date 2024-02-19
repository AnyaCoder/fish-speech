#!/bin/bash

# Setting up the environment variables
export no_proxy="localhost,127.0.0.1,0.0.0.0"
export PYTHONPATH=$(dirname "$0")
export PYTHON_FOLDERPATH="~/miniconda/envs/vits/"
export SCRIPT_FOLDERPATH="~/miniconda/envs/vits/Scripts/"
export CARGO="~/.cargo/bin/"
export RUST_LOG=info
export PATH="$CARGO:$PYTHON_FOLDERPATH:$SCRIPT_FOLDERPATH:$PATH"

# Echo the PATH to ensure it's set correctly
echo $PATH

# Run the Python script
${PYTHON_FOLDERPATH}python fish_speech/webui/manage.py

# Wait for a key press before exiting
read -p "Press any key to continue..." -n1 -s