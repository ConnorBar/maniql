"""One-time R3M checkpoint conversion: PyTorch ``.pt`` -> NumPy ``.npz``.

After conversion you can ``pip uninstall torch torchvision torchaudio`` (and
often ``nvidia-cudnn-cu12``) from your **JAX training** environment and pass
``--r3m_checkpoint …/model.npz`` to ``train_iql.py`` — no PyTorch needed there.

Run once (any env that has torch + the R3M file):

    python maniql/convert_r3m_checkpoint_to_npz.py \\
        --input ~/.r3m/r3m_18/model.pt \\
        --output ~/.r3m/r3m_18/model.npz
"""

from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string("input", "", "Path to R3M .pt checkpoint.")
flags.DEFINE_string("output", "", "Path to write .npz (vision tensors only).")


def main(_):
    if not FLAGS.input or not FLAGS.output:
        raise SystemExit("--input and --output are required.")
    import numpy as np
    import torch

    path_in = FLAGS.input
    path_out = FLAGS.output
    try:
        ckpt = torch.load(path_in, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path_in, map_location="cpu")
    if isinstance(ckpt, dict) and "r3m" in ckpt:
        sd = ckpt["r3m"]
    else:
        sd = ckpt
    sd = {
        k: v.detach().cpu().numpy().astype("float32", copy=False)
        for k, v in sd.items()
        if "lang_enc" not in k and "lang_rew" not in k and hasattr(v, "detach")
    }
    np.savez_compressed(path_out, **sd)
    print(f"[OK] Wrote {path_out} ({len(sd)} tensors).")


if __name__ == "__main__":
    app.run(main)
