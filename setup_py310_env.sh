#!/bin/bash

# Exit immediately if a command fails
set -e

# Name of the new environment
ENV_NAME="py310_env"

echo "Creating Conda environment '$ENV_NAME' with Python 3.10..."
conda create -y -n $ENV_NAME python=3.10

echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "Installing required packages..."
pip install numpy scipy matplotlib psutil memory_profiler jupyter

echo "✅ Environment '$ENV_NAME' is ready."

echo "To activate this environment in the future, run:"
echo "    conda activate $ENV_NAME"