#!/usr/bin/env bash

set -e  # Exit on any error

ENV_NAME="tacsl"
PYTHON_VERSION="3.8"
ISAAC_DIR="IsaacGym_Preview_TacSL_Package"
REPO_URL="git@github.com:quan-luu/manifeel-isaacgymenvs.git"
REPO_DIR="manifeel-isaacgymenvs"

echo "======================================"
echo " Setting up TacSL environment"
echo "======================================"

# -------------------------------
# Check conda exists
# -------------------------------
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Install Anaconda/Miniconda first."
    exit 1
fi

# -------------------------------
# Create environment if needed
# -------------------------------
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo " Conda env '$ENV_NAME' already exists"
else
    echo " Creating conda environment..."
    conda create --name $ENV_NAME python=$PYTHON_VERSION -y
fi

# Activate environment
echo " Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# -------------------------------
# Download Isaac Gym prompt
# -------------------------------
# echo ""
# echo " Download TacSL-specific Isaac Gym from:"
# echo " https://purdue0-my.sharepoint.com/:f:/r/personal/tamosa_purdue_edu/Documents/Marslab%20Capstone%20Project/Codebase/IsaacGym_Preview_TacSL_Package?csf=1&web=1&e=OasHLX"
# echo ""
#
# while true; do
#     read -p "Have you downloaded and placed it in this directory? (y/n): " yn
#     case $yn in
#         [Yy]* )
#             if [ -d "$ISAAC_DIR" ]; then
#                 echo " Found $ISAAC_DIR. Continuing..."
#                 break
#             else
#                 echo " Folder '$ISAAC_DIR' not found. Please place it here."
#             fi
#             ;;
#         [Nn]* )
#             echo " Please download it before continuing."
#             ;;
#         * )
#             echo "Please answer y or n."
#             ;;
#     esac
# done

if [ -d "$ISAAC_DIR" ]; then
    echo ""
    echo "======================================"
    echo " Found $ISAAC_DIR. Skipping download steps..."
    echo "======================================"
else
    echo ""
    echo "  $ISAAC_DIR not found."
    echo "Please download TacSL-specific Isaac Gym from:"
    echo "https://purdue0-my.sharepoint.com/:f:/r/personal/tamosa_purdue_edu/Documents/Marslab%20Capstone%20Project/Codebase/IsaacGym_Preview_TacSL_Package?csf=1&web=1&e=OasHLX"
    echo ""

    # Optional: Automatically open the link for yourself
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open "https://purdue0-my.sharepoint.com/:f:/r/personal/tamosa_purdue_edu/Documents/Marslab%20Capstone%20Project/Codebase/IsaacGym_Preview_TacSL_Package?csf=1&web=1&e=OasHLX" > /dev/null 2>&1
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        open "https://purdue0-my.sharepoint.com/:f:/r/personal/tamosa_purdue_edu/Documents/Marslab%20Capstone%20Project/Codebase/IsaacGym_Preview_TacSL_Package?csf=1&web=1&e=OasHLX"
    fi

    # 2. Entry into the manual confirmation loop
    while true; do
        read -p "Have you downloaded and placed it in this directory? (y/n): " yn
        case $yn in
            [Yy]* )
                if [ -d "$ISAAC_DIR" ]; then
                    echo " Verified. Continuing setup..."
                    break
                else
                    echo " Folder '$ISAAC_DIR' still not found in the current directory. Please check the folder name and try again."
                fi
                ;;
            [Nn]* )
                echo "Exiting. Please download the package to continue the setup."
                exit 1
                ;;
            * )
                echo "Please answer y or n."
                ;;
        esac
    done
fi

# The rest of your script follows here...

# -------------------------------
# Install Isaac Gym
# -------------------------------
echo " Installing Isaac Gym..."
pip install -e $ISAAC_DIR/isaacgym/python/

# -------------------------------
# Clone repo if needed
# -------------------------------
if [ -d "$REPO_DIR" ]; then
    echo " Repo already cloned"
else
    echo " Cloning manifeel repo..."
    git clone $REPO_URL
fi

cd $REPO_DIR

# -------------------------------
# Checkout branch safely
# -------------------------------
if git branch --list tacsl-manifeel-rl; then
    echo " Branch exists, switching..."
    git checkout tacsl-manifeel-rl
else
    echo " Creating branch..."
    git checkout -b tacsl-manifeel-rl
fi

# -------------------------------
# Install dependencies
# -------------------------------
echo " Installing repo..."

if grep -q "warp-lang==0.10.1" setup.py; then
    echo " Fixing broken warp dependency..."
    sed -i' ' 's/warp-lang==0.10.1/warp-lang==0.11.0/g' setup.py
    echo " Patched warp-lang to 0.11.0"
fi


pip install -e .

echo " Installing additional requirements..."
# pip install -r blob/manifeel-tacff/isaacgymenvs/tacsl_sensors/install/requirements.txt
pip install -r isaacgymenvs/tacsl_sensors/install/requirements.txt

echo ""
echo "======================================"
echo " Setup complete!"
echo "======================================"






