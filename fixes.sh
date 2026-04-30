#!/usr/bin/env bash
# fixes.sh — run this inside the active conda env (tacsl) if you hit
# pysdf / lxml wheel build errors on setup.

set -e

sudo apt-get update
sudo apt-get install -y build-essential gcc g++ make \
    python3-dev libxml2-dev libxslt-dev

# pysdf==0.1.9 sometimes fails from a binary wheel; force source build.
pip install --no-binary pysdf pysdf==0.1.9

# Expose the conda env's libpython for Isaac Gym (avoids the common
# "cannot open shared object file" error at import time).
CONDA_PREFIX="${CONDA_PREFIX:-$(conda info --base)/envs/tacsl}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
echo "export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:\$LD_LIBRARY_PATH" \
    >> ~/.bashrc
echo "[fixes.sh] done — restart shell or 'source ~/.bashrc' for LD_LIBRARY_PATH to persist."

