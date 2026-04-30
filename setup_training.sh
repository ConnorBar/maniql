#!/usr/bin/env bash

set -e

ENV_NAME="torch-maniql"
TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/torch-maniql" && pwd)"

echo "======================================"
echo " Setting up training environment"
echo "======================================"

if ! command -v conda &> /dev/null; then
    echo "Conda not found. Install Anaconda/Miniconda first."
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo " Conda env '$ENV_NAME' already exists, skipping create..."
else
    echo " Creating conda environment from $TRAIN_DIR/environment.yml..."
    conda env create -f "$TRAIN_DIR/environment.yml"
fi

echo " Activating environment..."
conda activate $ENV_NAME

echo " Installing r3m..."
pip install -e "$TRAIN_DIR/r3m"

echo ""
echo "======================================"
echo " Training environment ready!"
echo " Activate with: conda activate $ENV_NAME"
echo " Train with:    python torch-maniql/train_iql.py"
echo "======================================"
