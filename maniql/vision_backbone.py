"""Flax-native ResNet vision backbone with R3M pretrained weight loading.

Key design decisions:
  * BatchNorm is **folded** into conv weights at load time.  This means the
    Flax ResNet has no BatchNorm layers -- just Conv + ReLU -- and is fully
    compatible with the existing ``Model`` wrapper (params only, no batch_stats).
  * Weight conversion transposes PyTorch OIHW kernels to Flax HWIO.
  * The module is architecture-agnostic: pass ``arch="resnet18"`` (or 34/50)
    and the correct block layout is selected automatically.
"""

import os
from typing import Any, Dict, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN = jnp.array([0.485, 0.456, 0.406])
IMAGENET_STD = jnp.array([0.229, 0.224, 0.225])

RESNET_CONFIGS: Dict[str, Tuple[Sequence[int], bool]] = {
    "resnet18": ([2, 2, 2, 2], False),
    "resnet34": ([3, 4, 6, 3], False),
    "resnet50": ([3, 4, 6, 3], True),
}

RESNET_OUT_DIM: Dict[str, int] = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
}

FORCE_FIELD_DIM = 420
FORCE_GRID_SHAPE = (14, 10, 3)

# ---------------------------------------------------------------------------
#  Preprocessing helpers (run inside jit)
# ---------------------------------------------------------------------------

def r3m_preprocess(images: jnp.ndarray) -> jnp.ndarray:
    """Normalize images for an ImageNet-pretrained backbone.

    Args:
        images: ``(B, 224, 224, 3)`` in ``[0, 255]`` (uint8 or float).
    Returns:
        ``(B, 224, 224, 3)`` float32, ImageNet-normalised.
    """
    x = images.astype(jnp.float32) / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def force_to_image(force: jnp.ndarray) -> jnp.ndarray:
    """Reshape a flat force-field vector to a pseudo-image for ResNet.

    ``(B, 420)`` -> reshape ``(B, 14, 10, 3)`` -> bilinear resize to
    ``(B, 224, 224, 3)``.  Values are standardised per sample so the
    distribution is roughly zero-mean / unit-variance (comparable to
    ImageNet normalisation).
    """
    B = force.shape[0]
    x = force.reshape(B, *FORCE_GRID_SHAPE)
    mean = jnp.mean(x, axis=(1, 2, 3), keepdims=True)
    std = jnp.std(x, axis=(1, 2, 3), keepdims=True) + 1e-8
    x = (x - mean) / std
    return jax.image.resize(x, (B, 224, 224, 3), method="bilinear")


# ---------------------------------------------------------------------------
#  Flax ResNet blocks (BN already folded -- just Conv + ReLU)
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    """Two-conv residual block (ResNet-18 / 34)."""
    planes: int
    stride: int = 1
    downsample: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x

        x = nn.Conv(self.planes, (3, 3), strides=(self.stride, self.stride),
                     padding=[(1, 1), (1, 1)], use_bias=True, name="conv1")(x)
        x = nn.relu(x)

        x = nn.Conv(self.planes, (3, 3), strides=(1, 1),
                     padding=[(1, 1), (1, 1)], use_bias=True, name="conv2")(x)

        if self.downsample:
            residual = nn.Conv(self.planes, (1, 1),
                               strides=(self.stride, self.stride),
                               padding=[(0, 0), (0, 0)], use_bias=True,
                               name="downsample")(residual)

        return nn.relu(x + residual)


class Bottleneck(nn.Module):
    """Three-conv residual block (ResNet-50)."""
    planes: int
    stride: int = 1
    downsample: bool = False
    expansion: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        out_features = self.planes * self.expansion
        residual = x

        x = nn.Conv(self.planes, (1, 1), use_bias=True, name="conv1")(x)
        x = nn.relu(x)

        x = nn.Conv(self.planes, (3, 3), strides=(self.stride, self.stride),
                     padding=[(1, 1), (1, 1)], use_bias=True, name="conv2")(x)
        x = nn.relu(x)

        x = nn.Conv(out_features, (1, 1), use_bias=True, name="conv3")(x)

        if self.downsample:
            residual = nn.Conv(out_features, (1, 1),
                               strides=(self.stride, self.stride),
                               padding=[(0, 0), (0, 0)], use_bias=True,
                               name="downsample")(residual)

        return nn.relu(x + residual)


# ---------------------------------------------------------------------------
#  Full ResNet feature extractor
# ---------------------------------------------------------------------------

class FlaxResNet(nn.Module):
    """ResNet feature extractor (18 / 34 / 50).

    Input:  ``(B, 224, 224, 3)`` float32, already normalised.
    Output: ``(B, out_dim)`` -- 512 for resnet18/34, 2048 for resnet50.
    """
    arch: str = "resnet18"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        layers_config, use_bottleneck = RESNET_CONFIGS[self.arch]
        expansion = 4 if use_bottleneck else 1

        # stem: conv1 -> relu -> maxpool
        x = nn.Conv(64, (7, 7), strides=(2, 2), padding=[(3, 3), (3, 3)],
                     use_bias=True, name="conv1")(x)
        x = nn.relu(x)
        # match PyTorch MaxPool2d(3, stride=2, padding=1)
        x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)),
                     constant_values=float("-inf"))
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="VALID")

        # residual layers
        plane_sizes = [64, 128, 256, 512]
        in_channels = 64

        for layer_idx, num_blocks in enumerate(layers_config):
            planes = plane_sizes[layer_idx]
            out_channels = planes * expansion

            for block_idx in range(num_blocks):
                stride = 2 if layer_idx > 0 and block_idx == 0 else 1
                needs_ds = (block_idx == 0
                            and (stride != 1 or in_channels != out_channels))

                if use_bottleneck:
                    x = Bottleneck(planes=planes, stride=stride,
                                   downsample=needs_ds,
                                   name=f"layer{layer_idx+1}_{block_idx}")(x)
                else:
                    x = BasicBlock(planes=planes, stride=stride,
                                   downsample=needs_ds,
                                   name=f"layer{layer_idx+1}_{block_idx}")(x)

                in_channels = out_channels

        # global average pool
        return jnp.mean(x, axis=(1, 2))


# ---------------------------------------------------------------------------
#  R3M PyTorch -> Flax weight conversion
# ---------------------------------------------------------------------------

def _fold_bn(conv_w: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
             mean: np.ndarray, var: np.ndarray,
             eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    """Fold BatchNorm into a preceding conv and return Flax-format arrays.

    PyTorch conv weight shape: ``(O, I, kH, kW)``
    Returns ``(kernel, bias)`` in Flax layout ``(kH, kW, I, O)`` / ``(O,)``.
    """
    inv_std = 1.0 / np.sqrt(var + eps)
    scale = gamma * inv_std
    folded_w = conv_w * scale.reshape(-1, 1, 1, 1)
    folded_b = beta - mean * scale
    kernel = np.transpose(folded_w, (2, 3, 1, 0)).astype(np.float32)
    return kernel, folded_b.astype(np.float32)


def _to_numpy(state_dict: dict) -> dict:
    """Convert a PyTorch state-dict to plain numpy, stripping common prefixes."""
    out: dict = {}
    for k, v in state_dict.items():
        key = k
        for prefix in ("module.", "convnet."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        out[key] = v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
    return out


def load_r3m_to_flax(pytorch_state_dict: dict,
                     arch: str = "resnet18") -> dict:
    """Convert an R3M PyTorch state-dict to a Flax params dict.

    The returned dict can be injected directly into the ``FlaxResNet``
    sub-tree of any Flax model's params.

    Args:
        pytorch_state_dict: R3M state dict (may have ``module.convnet.``
            prefix from ``DataParallel``).
        arch: One of ``"resnet18"``, ``"resnet34"``, ``"resnet50"``.

    Returns:
        Plain (unfrozen) dict matching the ``FlaxResNet`` parameter tree.
    """
    sd = _to_numpy(pytorch_state_dict)
    params: dict = {}

    # --- stem: conv1 + bn1 ---
    k, b = _fold_bn(sd["conv1.weight"], sd["bn1.weight"], sd["bn1.bias"],
                     sd["bn1.running_mean"], sd["bn1.running_var"])
    params["conv1"] = {"kernel": k, "bias": b}

    # --- residual layers ---
    layers_config, use_bottleneck = RESNET_CONFIGS[arch]
    n_convs = 3 if use_bottleneck else 2

    for layer_idx, num_blocks in enumerate(layers_config):
        for block_idx in range(num_blocks):
            pt = f"layer{layer_idx+1}.{block_idx}"
            fx = f"layer{layer_idx+1}_{block_idx}"
            block: dict = {}

            for ci in range(1, n_convs + 1):
                k, b = _fold_bn(
                    sd[f"{pt}.conv{ci}.weight"],
                    sd[f"{pt}.bn{ci}.weight"],
                    sd[f"{pt}.bn{ci}.bias"],
                    sd[f"{pt}.bn{ci}.running_mean"],
                    sd[f"{pt}.bn{ci}.running_var"],
                )
                block[f"conv{ci}"] = {"kernel": k, "bias": b}

            ds_key = f"{pt}.downsample.0.weight"
            if ds_key in sd:
                k, b = _fold_bn(
                    sd[ds_key],
                    sd[f"{pt}.downsample.1.weight"],
                    sd[f"{pt}.downsample.1.bias"],
                    sd[f"{pt}.downsample.1.running_mean"],
                    sd[f"{pt}.downsample.1.running_var"],
                )
                block["downsample"] = {"kernel": k, "bias": b}

            params[fx] = block

    return params


def load_r3m_checkpoint(checkpoint_path: str) -> dict:
    """Load an R3M checkpoint and return the raw vision state-dict.

    * ``*.npz`` — NumPy arrays only (**no PyTorch**). Produce with
      ``python maniql/convert_r3m_checkpoint_to_npz.py``.
    * ``*.pt`` / ``*.pth`` — PyTorch ``torch.load`` (needs ``torch`` installed).
      Training is still pure JAX; PyTorch is only used for this I/O if you
      keep the original R3M file format.
    """
    path = os.path.expanduser(checkpoint_path)
    if path.endswith(".npz"):
        with np.load(path, mmap_mode="r") as z:
            return {
                k: np.asarray(z[k])
                for k in z.files
                if "lang_enc" not in k and "lang_rew" not in k
            }

    import torch

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    sd = ckpt["r3m"] if "r3m" in ckpt else ckpt
    return {k: v for k, v in sd.items()
            if "lang_enc" not in k and "lang_rew" not in k}


# ---------------------------------------------------------------------------
#  Utility: inject pretrained backbone weights into a Flax model's params
# ---------------------------------------------------------------------------

_BACKBONE_NAMES = frozenset({"wrist_backbone", "tactile_backbone",
                              "force_backbone"})


def inject_r3m_weights(params, r3m_backbone_params: dict):
    """Replace all vision-backbone sub-trees with R3M pretrained weights.

    Works recursively so it finds backbones regardless of nesting depth
    (e.g. inside ``MMCritic_0/encoder/wrist_backbone``).

    Args:
        params: Flax frozen-dict (or plain dict) model params.
        r3m_backbone_params: Dict returned by ``load_r3m_to_flax``.

    Returns:
        New params with backbone sub-trees replaced.
    """
    from flax.core import freeze, unfreeze

    def _replace(d: dict) -> dict:
        out = {}
        for k, v in d.items():
            if k in _BACKBONE_NAMES:
                out[k] = r3m_backbone_params
            elif isinstance(v, dict):
                out[k] = _replace(v)
            else:
                out[k] = v
        return out

    return freeze(_replace(unfreeze(params)))
