FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.8 python3.8-dev python3-pip \
    git wget curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /workspace

# PyTorch in its own layer so rebuilds on code changes are fast
RUN pip install --no-cache-dir \
    torch==2.2.0+cu121 \
    torchvision==0.17.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    numpy tqdm "pillow>=9.0.1" "imageio<3" \
    wandb "gdown>=4.4.0" \
    omegaconf "hydra-core>=1.1.1"

COPY r3m/ ./r3m/
RUN pip install -e ./r3m

COPY *.py ./

ENTRYPOINT ["python", "train_iql.py"]
