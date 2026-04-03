import argparse
import importlib
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from obs_modality import SPLIT_KEYS

# Avoid OpenMP shared-memory init failures in restricted environments.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

#--- lowk not sure what this is for ^^^^


# ----- parsing args -----
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Chunked preprocessor for raw transition files -> one final pickle of named arrays. "
            "Keeps RAM bounded by flushing chunks, then merges chunks."
        )
    )
    parser.add_argument("--input-dir", type=str, default="/home/purduerm/cap/data")
    parser.add_argument("--input-glob", type=str, default="*_transitions.pkl")
    parser.add_argument(
        "--output",
        type=str,
        default="/home/purduerm/cap/data/preprocessed/all_transitions_preprocessed.pkl",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="wrist,state",
        help="Comma-separated: wrist,tactile,forcefield,state",
    )
    parser.add_argument("--wrist-encoder", type=str, default="r3m", choices=["raw", "r3m", "vip"])
    parser.add_argument("--wrist-model", type=str, default="resnet50", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--r3m-repo", type=str, default="/home/purduerm/cap/work/furniture-bench/r3m")
    parser.add_argument("--vip-repo", type=str, default="/home/purduerm/cap/work/furniture-bench/vip")
    parser.add_argument(
        "--chunk-size-files",
        type=int,
        default=3,
        help="Flush a chunk every N input files.",
    )
    parser.add_argument(
        "--chunk-dir",
        type=str,
        default="",
        help="Optional chunk directory. Default is <output_stem>_chunks next to output.",
    )
    parser.add_argument(
        "--keep-chunks",
        action="store_true",
        help="Keep chunk files after successful merge.",
    )
    return parser.parse_args()


# Stored as (H, W, C); must match multimodal_nets.TactileEncoder input.
TACTILE_HWC = (160, 120, 3)


def multimodal_split_storage(selected_features: Sequence[str]) -> bool:
    """When all four modalities are selected, write split arrays (no flat concat)."""
    return set(selected_features) == {"wrist", "tactile", "forcefield", "state"}


def parse_features(features: str) -> List[str]:
    valid = {"wrist", "tactile", "forcefield", "state"}
    parsed = [f.strip() for f in features.split(",") if f.strip()]
    invalid = [f for f in parsed if f not in valid]
    if invalid:
        raise ValueError(f"Invalid feature names: {invalid}. Valid: {sorted(valid)}")
    if not parsed:
        raise ValueError("At least one feature must be selected.")
    return parsed


def try_append_repo_path(repo_path: str):
    p = str(Path(repo_path).resolve())
    if p not in sys.path:
        sys.path.append(p)


def _import_attr(module_name: str, attr_name: str, repo_path: str = ""):
    """Import an attribute from module, retrying after repo path injection."""
    try:
        mod = importlib.import_module(module_name)
        if hasattr(mod, attr_name):
            return getattr(mod, attr_name)
    except Exception:
        pass

    if repo_path:
        try_append_repo_path(repo_path)

    # Force a clean re-import in case an unrelated module was cached.
    importlib.invalidate_caches()
    for k in list(sys.modules.keys()):
        if k == module_name or k.startswith(f"{module_name}."):
            del sys.modules[k]

    mod = importlib.import_module(module_name)
    if not hasattr(mod, attr_name):
        raise AttributeError(
            f"Module '{module_name}' does not expose '{attr_name}'. "
            f"Loaded from: {getattr(mod, '__file__', 'unknown')}"
        )
    return getattr(mod, attr_name)

######################################################
# -------------- R3M and VIP loading -------------- #
######################################################

def load_wrist_encoder(encoder_name: str, model_name: str, r3m_repo: str, vip_repo: str, device_str: str):
    if encoder_name == "raw":
        return None

    import torch

    if encoder_name == "r3m":
        load_r3m = _import_attr("r3m", "load_r3m", repo_path=r3m_repo)
        model = load_r3m(model_name)
    else:
        load_vip = _import_attr("vip", "load_vip", repo_path=vip_repo)
        model = load_vip(model_name)

    device = torch.device(device_str)
    model = model.to(device)
    model.eval()
    return model

######################################################
# ---- ENSURE IMAGES ARE IN THE CORRECT FORMAT ----- #
######################################################

def to_uint8_hwc(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.dtype != np.uint8:
        maxv = float(np.max(arr)) if arr.size else 0.0
        if maxv <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _squeeze_to_float32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).squeeze()


######################################################
# -------------- ENCODING WRIST IMAGES -------------- #
######################################################

def encode_wrist_images(
    model,
    encoder_name: str,
    wrist_images: Sequence[np.ndarray],
    batch_size: int,
    device_str: str,
) -> np.ndarray:
    """
    batch encodes the wrist images into a single vector using the given model. 
    
    Returns [N, wrist_dim], i.e [N, 2048] for resnet50
    """
    import torch
    import torchvision.transforms as T

    if encoder_name == "raw":
        return np.stack([np.asarray(img, dtype=np.float32).squeeze().reshape(-1) for img in wrist_images], axis=0)

    device = torch.device(device_str)
    tensor = torch.from_numpy(np.stack([to_uint8_hwc(img) for img in wrist_images], axis=0)).permute(0, 3, 1, 2).float()
    tensor = T.Resize((224, 224))(tensor)

    # ----- batch encodes the wrist images, returns [N, wrist_dim], i.e [N, 2048] for resnet50
    outputs = []
    with torch.no_grad():
        for i in range(0, tensor.shape[0], batch_size):
            batch = tensor[i : i + batch_size].to(device)
            feat = model(batch).detach().cpu().numpy().astype(np.float32)
            outputs.append(feat)
    return np.concatenate(outputs, axis=0)

######################################################
# --------- CONCATENATING OBSERVATIONS ------------- #
######################################################

def flatten_obs_transition(
    tr: Dict,
    idx: int,
    selected_features: Sequence[str],
    wrist_obs_encoded: np.ndarray,
    wrist_next_obs_encoded: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    called at each step of the transition to flatten the obs at timestep idx into a single vector.
    """
    obs_parts = []
    next_parts = []

    if "wrist" in selected_features:
        obs_parts.append(np.asarray(wrist_obs_encoded[idx], dtype=np.float32).reshape(-1))
        next_parts.append(np.asarray(wrist_next_obs_encoded[idx], dtype=np.float32).reshape(-1))
    if "tactile" in selected_features:
        obs_parts.append(_squeeze_to_float32(tr["obs"]["right_tactile_camera_taxim"]).reshape(-1))
        next_parts.append(_squeeze_to_float32(tr["next_obs"]["right_tactile_camera_taxim"]).reshape(-1))
    if "forcefield" in selected_features:
        obs_parts.append(_squeeze_to_float32(tr["obs"]["tactile_force_field_right"]).reshape(-1))
        next_parts.append(_squeeze_to_float32(tr["next_obs"]["tactile_force_field_right"]).reshape(-1))
    if "state" in selected_features:
        obs_parts.append(_squeeze_to_float32(tr["obs"]["state"]).reshape(-1))
        next_parts.append(_squeeze_to_float32(tr["next_obs"]["state"]).reshape(-1))

    return np.concatenate(obs_parts, axis=0), np.concatenate(next_parts, axis=0)


def passthrough_vectors_for_transition(
    tr: Dict,
    i: int,
    selected_features: Sequence[str],
    wrist_obs_encoded: np.ndarray,
    wrist_next_obs_encoded: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Wrist + state only, in the same order as ``selected_features``."""
    obs_parts, next_parts = [], []
    for feat in selected_features:
        if feat == "wrist":
            obs_parts.append(np.asarray(wrist_obs_encoded[i], dtype=np.float32).reshape(-1))
            next_parts.append(np.asarray(wrist_next_obs_encoded[i], dtype=np.float32).reshape(-1))
        elif feat == "state":
            obs_parts.append(_squeeze_to_float32(tr["obs"]["state"]).reshape(-1))
            next_parts.append(_squeeze_to_float32(tr["next_obs"]["state"]).reshape(-1))
    return np.concatenate(obs_parts, axis=0), np.concatenate(next_parts, axis=0)


def tactile_volume(tr: Dict, *, next: bool = False) -> np.ndarray:
    key = "next_obs" if next else "obs"
    flat = _squeeze_to_float32(tr[key]["right_tactile_camera_taxim"]).reshape(-1)
    return flat.reshape(TACTILE_HWC).astype(np.float32, copy=False)


def forcefield_vector(tr: Dict, *, next: bool = False) -> np.ndarray:
    key = "next_obs" if next else "obs"
    return _squeeze_to_float32(tr[key]["tactile_force_field_right"]).reshape(-1).astype(np.float32)


######################################################
# -------------- PREPROCESSING FILE TO ARRAYS -------------- #
######################################################

def preprocess_file_to_arrays(
    file_path: Path,
    selected_features: Sequence[str],
    wrist_encoder_name: str,
    wrist_model,
    batch_size: int,
    device_str: str,
    assigned_seed: int,
) -> Dict[str, np.ndarray]:
    """
    primary function for converting the raw transition into expected format for the pickle file for IQL.
    """
    # ----- loading file and getting transitions -----
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    transitions = data.get("transitions", data) if isinstance(data, dict) else data
    if not isinstance(transitions, list):
        raise ValueError(f"Expected transitions list in {file_path}. Got {type(transitions)}")
    if len(transitions) == 0:
        raise ValueError(f"No transitions in {file_path}")

    # Trim sticky done tail: keep active portion + one terminal transition.
    # The done flag turns on at task completion and stays on for a trailing
    # block.  Find the first done=True and keep up to (and including) it.
    first_done_idx = None
    for idx, tr in enumerate(transitions):
        if bool(np.asarray(tr["done"]).reshape(-1)[0]):
            first_done_idx = idx
            break
    if first_done_idx is not None:
        transitions = transitions[: first_done_idx + 1]

    # ----- encoding wrist images -----
    needs_wrist = "wrist" in selected_features
    wrist_obs_encoded = wrist_next_obs_encoded = None
    if needs_wrist:
        wrist_obs = [np.asarray(tr["obs"]["wrist"]) for tr in transitions]
        wrist_next_obs = [np.asarray(tr["next_obs"]["wrist"]) for tr in transitions]
        wrist_obs_encoded = encode_wrist_images(wrist_model, wrist_encoder_name, wrist_obs, batch_size, device_str)
        wrist_next_obs_encoded = encode_wrist_images(wrist_model, wrist_encoder_name, wrist_next_obs, batch_size, device_str)

    split_mode = multimodal_split_storage(selected_features)

    # ----- observations -----
    obs_rows = []
    next_obs_rows = []
    passthrough_rows = []
    next_passthrough_rows = []
    tactile_rows = []
    next_tactile_rows = []
    force_rows = []
    next_force_rows = []
    actions = []
    rewards = []
    dones = []
    success = []
    timeouts = []
    seeds = []
    source_files = []

    for i, tr in enumerate(transitions):
        if split_mode:
            pv, npv = passthrough_vectors_for_transition(
                tr, i, selected_features, wrist_obs_encoded, wrist_next_obs_encoded)
            passthrough_rows.append(pv.astype(np.float32))
            next_passthrough_rows.append(npv.astype(np.float32))
            tactile_rows.append(tactile_volume(tr, next=False))
            next_tactile_rows.append(tactile_volume(tr, next=True))
            force_rows.append(forcefield_vector(tr, next=False))
            next_force_rows.append(forcefield_vector(tr, next=True))
        else:
            obs_vec, next_obs_vec = flatten_obs_transition(
                tr, i, selected_features, wrist_obs_encoded, wrist_next_obs_encoded)
            obs_rows.append(obs_vec.astype(np.float32))
            next_obs_rows.append(next_obs_vec.astype(np.float32))
        actions.append(_squeeze_to_float32(tr["action"]).reshape(-1))
        rewards.append(float(np.asarray(tr["reward"]).reshape(-1)[0]))         # very small rewards
        dones.append(float(bool(np.asarray(tr["done"]).reshape(-1)[0])))       # NOTE assuming is the correct flag for success
        success.append(float(bool(np.asarray(tr["success"]).reshape(-1)[0])))  # NOTE is all zeros
        timeouts.append(float(bool(np.asarray(tr["timeout"]).reshape(-1)[0]))) # NOTE is all zeros
        seeds.append(int(assigned_seed)) # this will be used for the seed of the environment in isaac gym
        source_files.append(file_path.name)

    # ----- returning the arrays -----
    # NOTE that we are going to treat DONE as the success flag for IQL.
    if split_mode:
        k_pass, k_tact, k_ff = SPLIT_KEYS
        return {
            "obs": {
                k_pass: np.stack(passthrough_rows, axis=0).astype(np.float32),
                k_tact: np.stack(tactile_rows, axis=0).astype(np.float32),
                k_ff: np.stack(force_rows, axis=0).astype(np.float32),
            },
            "next_obs": {
                k_pass: np.stack(next_passthrough_rows, axis=0).astype(np.float32),
                k_tact: np.stack(next_tactile_rows, axis=0).astype(np.float32),
                k_ff: np.stack(next_force_rows, axis=0).astype(np.float32),
            },
            "actions": np.stack(actions, axis=0).astype(np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.float32),
            "success": np.asarray(success, dtype=np.float32),
            "timeouts": np.asarray(timeouts, dtype=np.float32),
            "seed": np.asarray(seeds, dtype=np.int32),
            "source_file": np.asarray(source_files, dtype=object),
        }
    return {
        "obs": np.stack(obs_rows, axis=0).astype(np.float32),
        "next_obs": np.stack(next_obs_rows, axis=0).astype(np.float32),
        "actions": np.stack(actions, axis=0).astype(np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.float32),
        "success": np.asarray(success, dtype=np.float32),
        "timeouts": np.asarray(timeouts, dtype=np.float32),
        "seed": np.asarray(seeds, dtype=np.int32),
        "source_file": np.asarray(source_files, dtype=object),
    }


######################################################
#  SCAFFOLDING OBSERVATION LAYOUT FOR PICKLE FILE  #
######################################################

def build_obs_layout(selected_features: Sequence[str], sample_tr: Dict, sample_obs_encoded: np.ndarray) -> List[Dict]:
    """
    based on first transition, build the observation layout for the pickle file.
    """
    layout = []
    offset = 0
    for feature in selected_features:
        if feature == "wrist":
            dim = int(np.asarray(sample_obs_encoded).reshape(-1).shape[0])
            name = "wrist"
        elif feature == "tactile":
            dim = int(_squeeze_to_float32(sample_tr["obs"]["right_tactile_camera_taxim"]).reshape(-1).shape[0])
            name = "right_tactile_camera_taxim"
        elif feature == "forcefield":
            dim = int(_squeeze_to_float32(sample_tr["obs"]["tactile_force_field_right"]).reshape(-1).shape[0])
            name = "tactile_force_field_right"
        else:
            dim = int(_squeeze_to_float32(sample_tr["obs"]["state"]).reshape(-1).shape[0])
            name = "state"
        layout.append({"name": name, "start": offset, "end": offset + dim, "dim": dim})
        offset += dim
    return layout


######################################################
# ------------ FLUSHING CHUNKS TO FILES ------------ #
######################################################

def flush_chunk(chunk_idx: int, chunk_dir: Path, file_records: List[Dict], buffers: Dict):
    is_split = isinstance(buffers.get("obs"), dict)
    if is_split:
        if len(buffers["obs"][SPLIT_KEYS[0]]) == 0:
            return
    elif len(buffers["obs"]) == 0:
        return
    chunk_path = chunk_dir / f"chunk_{chunk_idx:05d}.pkl"
    if is_split:
        payload = {
            "obs": {
                k: np.concatenate(buffers["obs"][k], axis=0).astype(np.float32)
                for k in SPLIT_KEYS
            },
            "next_obs": {
                k: np.concatenate(buffers["next_obs"][k], axis=0).astype(np.float32)
                for k in SPLIT_KEYS
            },
            "actions": np.concatenate(buffers["actions"], axis=0).astype(np.float32),
            "rewards": np.concatenate(buffers["rewards"], axis=0).astype(np.float32),
            "dones": np.concatenate(buffers["dones"], axis=0).astype(np.float32),
            "success": np.concatenate(buffers["success"], axis=0).astype(np.float32),
            "timeouts": np.concatenate(buffers["timeouts"], axis=0).astype(np.float32),
            "seed": np.concatenate(buffers["seed"], axis=0).astype(np.int32),
            "source_file": np.concatenate(buffers["source_file"], axis=0),
            "file_records": file_records,
        }
    else:
        payload = {
            "obs": np.concatenate(buffers["obs"], axis=0).astype(np.float32),
            "next_obs": np.concatenate(buffers["next_obs"], axis=0).astype(np.float32),
            "actions": np.concatenate(buffers["actions"], axis=0).astype(np.float32),
            "rewards": np.concatenate(buffers["rewards"], axis=0).astype(np.float32),
            "dones": np.concatenate(buffers["dones"], axis=0).astype(np.float32),
            "success": np.concatenate(buffers["success"], axis=0).astype(np.float32),
            "timeouts": np.concatenate(buffers["timeouts"], axis=0).astype(np.float32),
            "seed": np.concatenate(buffers["seed"], axis=0).astype(np.int32),
            "source_file": np.concatenate(buffers["source_file"], axis=0),
            "file_records": file_records,
        }
    with open(chunk_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


######################################################
# --------- MERGING CHUNKS TO FINAL FILE ----------- #
######################################################

def merge_chunks_to_final(chunk_dir: Path, output_path: Path, metadata: Dict):
    chunk_files = sorted(chunk_dir.glob("chunk_*.pkl"))
    if not chunk_files:
        raise RuntimeError(f"No chunk files found in {chunk_dir}")

    with open(chunk_files[0], "rb") as f:
        first = pickle.load(f)
    split = isinstance(first.get("obs"), dict) and SPLIT_KEYS[0] in first["obs"]

    all_file_records = []
    total = 0
    act_dim = None

    for chunk_file in chunk_files:
        with open(chunk_file, "rb") as f:
            chunk = pickle.load(f)
        n = int(chunk["actions"].shape[0])
        total += n
        act_dim = int(chunk["actions"].shape[1]) if act_dim is None else act_dim
        all_file_records.extend(chunk["file_records"])

    tmp_dir = output_path.parent / f".{output_path.stem}_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    actions_mm = np.memmap(tmp_dir / "actions.dat", dtype=np.float32, mode="w+", shape=(total, act_dim))
    rewards_mm = np.memmap(tmp_dir / "rewards.dat", dtype=np.float32, mode="w+", shape=(total,))
    dones_mm = np.memmap(tmp_dir / "dones.dat", dtype=np.float32, mode="w+", shape=(total,))
    success_mm = np.memmap(tmp_dir / "success.dat", dtype=np.float32, mode="w+", shape=(total,))
    timeouts_mm = np.memmap(tmp_dir / "timeouts.dat", dtype=np.float32, mode="w+", shape=(total,))
    seed_mm = np.memmap(tmp_dir / "seed.dat", dtype=np.int32, mode="w+", shape=(total,))
    source_file_arr = np.empty((total,), dtype=object)

    if split:
        k_pass, k_tact, k_ff = SPLIT_KEYS
        ptp_dim = int(first["obs"][k_pass].shape[1])
        tac_shape = tuple(int(x) for x in first["obs"][k_tact].shape[1:])
        ff_dim = int(first["obs"][k_ff].shape[1])
        obs_ptp_mm = np.memmap(tmp_dir / "obs_pass.dat", dtype=np.float32, mode="w+", shape=(total, ptp_dim))
        obs_tac_mm = np.memmap(
            tmp_dir / "obs_tact.dat", dtype=np.float32, mode="w+", shape=(total, *tac_shape))
        obs_ff_mm = np.memmap(tmp_dir / "obs_forcefield.dat", dtype=np.float32, mode="w+", shape=(total, ff_dim))
        nptp_mm = np.memmap(tmp_dir / "next_obs_pass.dat", dtype=np.float32, mode="w+", shape=(total, ptp_dim))
        ntac_mm = np.memmap(
            tmp_dir / "next_obs_tact.dat", dtype=np.float32, mode="w+", shape=(total, *tac_shape))
        nff_mm = np.memmap(tmp_dir / "next_obs_forcefield.dat", dtype=np.float32, mode="w+", shape=(total, ff_dim))

        cursor = 0
        for chunk_file in tqdm(chunk_files, desc="Merging chunks"):
            with open(chunk_file, "rb") as f:
                chunk = pickle.load(f)
            n = int(chunk["actions"].shape[0])
            sl = slice(cursor, cursor + n)
            obs_ptp_mm[sl] = chunk["obs"][k_pass]
            obs_tac_mm[sl] = chunk["obs"][k_tact]
            obs_ff_mm[sl] = chunk["obs"][k_ff]
            nptp_mm[sl] = chunk["next_obs"][k_pass]
            ntac_mm[sl] = chunk["next_obs"][k_tact]
            nff_mm[sl] = chunk["next_obs"][k_ff]
            actions_mm[sl] = chunk["actions"]
            rewards_mm[sl] = chunk["rewards"]
            dones_mm[sl] = chunk["dones"]
            success_mm[sl] = chunk["success"]
            timeouts_mm[sl] = chunk["timeouts"]
            seed_mm[sl] = chunk["seed"]
            source_file_arr[sl] = chunk["source_file"]
            cursor += n

        file_index = []
        run_start = 0
        for rec in all_file_records:
            n_tr = int(rec["num_transitions"])
            file_index.append(
                {
                    "source_file": rec["source_file"],
                    "seed": int(rec["seed"]),
                    "num_transitions": n_tr,
                    "start_idx": run_start,
                    "end_idx_exclusive": run_start + n_tr,
                }
            )
            run_start += n_tr

        final_payload = {
            "metadata": metadata,
            "file_index": file_index,
            "obs": {
                k_pass: np.asarray(obs_ptp_mm),
                k_tact: np.asarray(obs_tac_mm),
                k_ff: np.asarray(obs_ff_mm),
            },
            "next_obs": {
                k_pass: np.asarray(nptp_mm),
                k_tact: np.asarray(ntac_mm),
                k_ff: np.asarray(nff_mm),
            },
            "actions": np.asarray(actions_mm),
            "rewards": np.asarray(rewards_mm),
            "dones": np.asarray(dones_mm),
            "terminals": np.clip(np.asarray(dones_mm) - np.asarray(timeouts_mm), 0.0, 1.0).astype(np.float32),
            "success": np.asarray(success_mm),
            "timeouts": np.asarray(timeouts_mm),
            "seed": np.asarray(seed_mm),
            "source_file": source_file_arr,
        }
        with open(output_path, "wb") as f:
            pickle.dump(final_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        del (obs_ptp_mm, obs_tac_mm, obs_ff_mm, nptp_mm, ntac_mm, nff_mm)
        del actions_mm, rewards_mm, dones_mm, success_mm, timeouts_mm, seed_mm
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    obs_dim = int(first["obs"].shape[1])
    obs_mm = np.memmap(tmp_dir / "obs.dat", dtype=np.float32, mode="w+", shape=(total, obs_dim))
    next_obs_mm = np.memmap(tmp_dir / "next_obs.dat", dtype=np.float32, mode="w+", shape=(total, obs_dim))

    cursor = 0
    for chunk_file in tqdm(chunk_files, desc="Merging chunks"):
        with open(chunk_file, "rb") as f:
            chunk = pickle.load(f)
        n = int(chunk["obs"].shape[0])
        sl = slice(cursor, cursor + n)
        obs_mm[sl] = chunk["obs"]
        next_obs_mm[sl] = chunk["next_obs"]
        actions_mm[sl] = chunk["actions"]
        rewards_mm[sl] = chunk["rewards"]
        dones_mm[sl] = chunk["dones"]
        success_mm[sl] = chunk["success"]
        timeouts_mm[sl] = chunk["timeouts"]
        seed_mm[sl] = chunk["seed"]
        source_file_arr[sl] = chunk["source_file"]
        cursor += n

    file_index = []
    run_start = 0
    for rec in all_file_records:
        n = int(rec["num_transitions"])
        file_index.append(
            {
                "source_file": rec["source_file"],
                "seed": int(rec["seed"]),
                "num_transitions": n,
                "start_idx": run_start,
                "end_idx_exclusive": run_start + n,
            }
        )
        run_start += n

    final_payload = {
        "metadata": metadata,
        "file_index": file_index,
        "obs": np.asarray(obs_mm),
        "next_obs": np.asarray(next_obs_mm),
        "actions": np.asarray(actions_mm),
        "rewards": np.asarray(rewards_mm),
        "dones": np.asarray(dones_mm),
        "terminals": np.clip(np.asarray(dones_mm) - np.asarray(timeouts_mm), 0.0, 1.0).astype(np.float32),
        "success": np.asarray(success_mm),
        "timeouts": np.asarray(timeouts_mm),
        "seed": np.asarray(seed_mm),
        "source_file": source_file_arr,
    }
    with open(output_path, "wb") as f:
        pickle.dump(final_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    del obs_mm, next_obs_mm, actions_mm, rewards_mm, dones_mm, success_mm, timeouts_mm, seed_mm
    shutil.rmtree(tmp_dir, ignore_errors=True)


######################################################
# -------------- MAIN FUNCTION -------------- #
######################################################

def main():
    # ----- parsing args -----
    args = parse_args()
    selected_features = parse_features(args.features)
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ----- sorting files and creating chunk directory  -----
    all_files = sorted(input_dir.glob(args.input_glob), key=lambda p: p.name)
    if not all_files:
        raise FileNotFoundError(f"No files matching '{args.input_glob}' in {input_dir}")

    chunk_dir = Path(args.chunk_dir) if args.chunk_dir else (output_path.parent / f"{output_path.stem}_chunks")
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # ----- loading wrist model -----
    wrist_model = None
    if "wrist" in selected_features and args.wrist_encoder != "raw":
        wrist_model = load_wrist_encoder(args.wrist_encoder, args.wrist_model, args.r3m_repo, args.vip_repo, args.device)

    # ----- loading first file to get sample observation -----
    # this is used to build the observation layout
    with open(all_files[0], "rb") as f:
        first_data = pickle.load(f)
    first_transitions: dict = first_data.get("transitions", first_data) if isinstance(first_data, dict) else first_data
    first_tr = first_transitions[0]
    if "wrist" in selected_features:
        sample_wrist = np.asarray(first_tr["obs"]["wrist"])
        if args.wrist_encoder == "raw":
            sample_emb = sample_wrist.squeeze().reshape(-1)
        else:
            sample_emb = encode_wrist_images(wrist_model, args.wrist_encoder, [sample_wrist], args.batch_size, args.device)[0]
    else:
        sample_emb = np.array([], dtype=np.float32)
    # ----- building observation layout based on the first file -----
    obs_layout = build_obs_layout(selected_features, first_tr, sample_emb)
    use_split = multimodal_split_storage(selected_features)

    # ----- initializing buffers for chunking -----
    scalar_buffer_keys = ["actions", "rewards", "dones", "success", "timeouts", "seed", "source_file"]
    if use_split:
        buffers = {
            "obs": {k: [] for k in SPLIT_KEYS},
            "next_obs": {k: [] for k in SPLIT_KEYS},
            **{k: [] for k in scalar_buffer_keys},
        }
    else:
        buffers = {k: [] for k in (["obs", "next_obs"] + scalar_buffer_keys)}
    chunk_idx = 0
    files_in_chunk = 0
    file_records = [] # used for metadata and bookkeeping
    total_transitions = 0

    # ----- for each file, process the transitions into IQL format arrays, chunking as we go -----
    for file_idx, file_path in enumerate(tqdm(all_files, desc="Preprocessing files")):
        assigned_seed = int(args.seed_start) + file_idx
        # ----- preprocess file to arrays -----
        arrs = preprocess_file_to_arrays(
            file_path=file_path,
            selected_features=selected_features,
            wrist_encoder_name=args.wrist_encoder,
            wrist_model=wrist_model,
            batch_size=args.batch_size,
            device_str=args.device,
            assigned_seed=assigned_seed,
        )

        if use_split:
            for k in SPLIT_KEYS:
                buffers["obs"][k].append(arrs["obs"][k])
                buffers["next_obs"][k].append(arrs["next_obs"][k])
            for k in scalar_buffer_keys:
                buffers[k].append(arrs[k])
        else:
            for k in buffers:
                buffers[k].append(arrs[k])
        # ----- appending file records -----
        n_here = int(arrs["obs"][SPLIT_KEYS[0]].shape[0]) if use_split else int(arrs["obs"].shape[0])
        file_records.append(
            {
                "source_file": file_path.name,
                "seed": assigned_seed,
                "num_transitions": n_here,
            }
        )
        total_transitions += n_here
        files_in_chunk += 1

        # ----- flush buffers to file -----
        if files_in_chunk >= args.chunk_size_files:
            flush_chunk(chunk_idx, chunk_dir, file_records, buffers)
            chunk_idx += 1
            files_in_chunk = 0
            file_records = []
            if use_split:
                buffers = {
                    "obs": {k: [] for k in SPLIT_KEYS},
                    "next_obs": {k: [] for k in SPLIT_KEYS},
                    **{k: [] for k in scalar_buffer_keys},
                }
            else:
                buffers = {k: [] for k in (["obs", "next_obs"] + scalar_buffer_keys)}

    # ----- flush any remaining chunks -----
    if files_in_chunk > 0:
        flush_chunk(chunk_idx, chunk_dir, file_records, buffers)

    metadata = {
        "input_dir": str(input_dir),
        "input_glob": args.input_glob,
        "num_files": len(all_files),
        "num_transitions": total_transitions,
        "selected_features": selected_features,
        "obs_layout": obs_layout,
        "modality_storage": "split" if use_split else "concat",
        "wrist_encoder": args.wrist_encoder if "wrist" in selected_features else None,
        "wrist_model": args.wrist_model if ("wrist" in selected_features and args.wrist_encoder != "raw") else None,
        "seed_start": int(args.seed_start),
        "file_sort": "filename ascending",
    }
    if use_split:
        metadata["obs_modality_spec"] = {
            "passthrough_order": [f for f in selected_features if f in ("wrist", "state")],
            "tactile_shape": list(TACTILE_HWC),
            "forcefield_dim": int(
                _squeeze_to_float32(first_tr["obs"]["tactile_force_field_right"]).reshape(-1).shape[0]),
        }
    merge_chunks_to_final(chunk_dir, output_path, metadata)

    if not args.keep_chunks:
        shutil.rmtree(chunk_dir, ignore_errors=True)

    print(f"[done] wrote merged dataset: {output_path}")
    print(f"[done] files merged: {len(all_files)} transitions: {total_transitions}")


if __name__ == "__main__":
    main()
