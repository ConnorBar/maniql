"""Offline IQL training on ManiFeel datasets with R3M backbone finetuning.

The vision backbone (FlaxResNet initialised from R3M pretrained weights)
is finetuned end-to-end via IQL loss gradients.  Raw images are stored in
the preprocessed pickle; encoding happens on-the-fly during training.

Usage
-----
    # wrist + state (resnet18)
    python train_iql.py \\
        --dataset_path data/preprocessed/raw_wrist_state.pkl \\
        --backbone resnet18 \\
        --r3m_checkpoint ~/.r3m/r3m_18/model.pt \\
        --save_dir runs/iql_ws_r3m18 \\
        --max_steps 200000

    # full multimodal
    python train_iql.py \\
        --dataset_path data/preprocessed/raw_full.pkl \\
        --backbone resnet18 \\
        --r3m_checkpoint ~/.r3m/r3m_18/model.pt \\
        --save_dir runs/iql_full_r3m18 \\
        --max_steps 200000 \\
        --batch_size 64

    Lower ``--batch_size`` if you hit CUDA OOM -- each head has its own
    ResNet backbone(s).

    GPU / cuDNN
    ~~~~~~~~~~~
    If you see ``CUDNN_STATUS_INTERNAL_ERROR`` or
    ``DNN library initialization failed`` even when ``nvidia-smi`` shows free
    memory, check for **two cuDNN majors** in one env. Example: ``jaxlib`` tag
    ``+cuda12.cudnn89`` needs cuDNN **8.9**, but pip ``nvidia-cudnn-cu12`` **9.x**
    (often pulled in with PyTorch) prepends another ``libcudnn`` and JAX breaks.
    Fix: ``pip uninstall nvidia-cudnn-cu12`` (PyTorch manylinux wheels usually
    bundle their own cuDNN), **or** upgrade ``jax``+``jaxlib`` to a wheel built
    for cuDNN 9 (see JAX install docs; may need Python >= 3.9). A null
    ``LD_LIBRARY_PATH`` does not prevent this — NVIDIA pip metapackages use
    ``.pth`` entries. Or train on CPU::

        python maniql/train_iql.py ... --jax_platform=cpu

    Same effect: ``export JAX_PLATFORMS=cpu`` before launching (CPU is slow for
    vision but verifies the rest of the pipeline).
"""

import os
import sys
import textwrap


def _jax_platform_from_argv():
    """Read ``--jax_platform`` before ``import jax`` (absl parses flags later)."""
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--jax_platform="):
            return a.split("=", 1)[1].strip()
        if a == "--jax_platform":
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                return argv[i + 1].strip()
            return ""
        i += 1
    return None


_jax_platform = _jax_platform_from_argv()
if _jax_platform is not None and _jax_platform != "":
    os.environ["JAX_PLATFORMS"] = _jax_platform

# JAX defaults to grabbing most GPU VRAM up front; on WSL / smaller GPUs that
# often causes CUDA OOM during peaks (multimodal IQL uses large tactile batches).
# Override with XLA_PYTHON_CLIENT_PREALLOCATE=true if you prefer the default.
# this is the main reason that fixed the OOM issue
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Can avoid brittle cuDNN init when many CUDA libs are visible on LD_LIBRARY_PATH.
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

# gets access to critic and learner modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                "implicit_q_learning"))

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

from critic import loss as expectile_loss
from manifeel_iql import ManiFeelDataset
from multimodal_nets import MultiModalLearner


def _is_jax_gpu_dnn_init_failure(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        s in text
        for s in ("dnn library", "cudnn_status", "cuda_dnn", "cudnn"))


def _print_jax_gpu_dnn_help() -> None:
    print(
        textwrap.dedent(
            """
            [HINT] GPU cuDNN init failed (not an application bug in train_iql).
              1) Compare driver vs JAX:  nvidia-smi  (CUDA version line)  and
                 pip show jax jaxlib  — reinstall from the official table so the
                 pip wheel matches CUDA 11 vs 12:
                 https://github.com/google/jax#installation
              2) cuDNN version clash: jaxlib tag cudnn89 vs pip nvidia-cudnn-cu12 9.x.
                 conda list | grep -iE 'cudnn|jaxlib'
                 If you see both, try:  pip uninstall nvidia-cudnn-cu12
                 (PyTorch often still works; if not, upgrade jax+jaxlib to a cuDNN9
                 wheel or use a separate env for JAX training.)
              3) Even if LD_LIBRARY_PATH is empty, RPATH / .pth can load another
                 libcudnn from $CONDA_PREFIX/lib. Remove mismatched cuda/cudnn
                 packages or use a clean venv.
              4) See which libcudnn jaxlib loads (Linux): run
                   python -c "import jaxlib, os; print(os.path.dirname(jaxlib.__file__))"
                 then in that directory  ldd xla_extension*.so  | grep -i cudnn
                 (if the .so name differs, ldd whichever extension jaxlib ships.)
              5) Reinstall jax + jaxlib together from the JAX install table.
              6) Unblock training without GPU:  --jax_platform=cpu
            """
        ).strip(),
        file=sys.stderr,
    )


FLAGS = flags.FLAGS

flags.DEFINE_string("dataset_path", "", "Path to preprocessed pickle.")
flags.DEFINE_string("save_dir", "./runs/manifeel_iql/",
                    "Tensorboard + checkpoint dir.")
flags.DEFINE_string("backbone", "resnet18",
                    "Vision backbone architecture (resnet18 / resnet34 / resnet50).")
flags.DEFINE_string("r3m_checkpoint", "",
                    "Path to R3M .pt checkpoint for backbone init. "
                    "Leave empty to train from scratch.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 2000, "Test-set evaluation interval.")
flags.DEFINE_string("env_name", "",
                    "Optional gym env for rollout eval. "
                    "If empty, rollout evaluation is skipped.")
flags.DEFINE_integer("eval_episodes", 10, "Rollout episodes.")
flags.DEFINE_integer("batch_size", 128, "Mini-batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Gradient steps.")
flags.DEFINE_integer("save_interval", 50000,
                     "Checkpoint interval (0 = final only).")
flags.DEFINE_float("test_ratio", 0.1, "Episode fraction held out for test.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("normalize_rewards", False,
                     "Normalize rewards by trajectory return range.")
flags.DEFINE_boolean("clip_actions", True, "Clip actions to [-1+eps, 1-eps].")
flags.DEFINE_boolean("validate", True, "Run data sanity checks.")
flags.DEFINE_string(
    "jax_platform",
    "",
    "If set (e.g. cpu), exported as JAX_PLATFORMS before JAX loads CUDA/cuDNN. "
    "Must be passed on the command line (not only in --config); "
    "or use export JAX_PLATFORMS=cpu.",
)

config_flags.DEFINE_config_file(
    "config",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "implicit_q_learning", "configs", "mujoco_config.py"),
    "Training hyperparameter config.",
    lock_config=False,
)


# ---------------------------------------------------------------------------
#  Eval helpers
# ---------------------------------------------------------------------------

@jax.jit
def _eval_losses(actor, critic, value, target_critic, batch,
                 discount, expectile, temperature, rng):
    q1, q2 = target_critic(batch.observations, batch.actions)
    q = jnp.minimum(q1, q2)
    v = value(batch.observations)

    value_loss = expectile_loss(q - v, expectile).mean()

    next_v = value(batch.next_observations)
    target_q = batch.rewards + discount * batch.masks * next_v
    q1_pred, q2_pred = critic(batch.observations, batch.actions)
    critic_loss = ((q1_pred - target_q) ** 2
                   + (q2_pred - target_q) ** 2).mean()

    dist = actor.apply_fn.apply({"params": actor.params}, batch.observations)
    log_probs = dist.log_prob(batch.actions)
    exp_a = jnp.minimum(jnp.exp((q - v) * temperature), 100.0)
    actor_loss = -(exp_a * log_probs).mean()

    return {
        "critic_loss": critic_loss,
        "value_loss": value_loss,
        "actor_loss": actor_loss,
        "v": v.mean(),
        "q1": q1_pred.mean(),
        "q2": q2_pred.mean(),
        "adv": (q - v).mean(),
    }


def eval_on_dataset(agent, dataset, batch_size, n_batches=10):
    """Average eval losses over multiple batches from dataset."""
    accum = None
    for _ in range(n_batches):
        batch = dataset.sample(batch_size)
        info = _eval_losses(
            agent.actor, agent.critic, agent.value, agent.target_critic,
            batch, agent.discount, agent.expectile, agent.temperature,
            agent.rng,
        )
        info = jax.device_get(info)
        if accum is None:
            accum = {k: float(np.asarray(v)) for k, v in info.items()}
        else:
            for k, v in info.items():
                accum[k] += float(np.asarray(v))
    return {k: v / n_batches for k, v in accum.items()}


# ---------------------------------------------------------------------------
#  Reward normalisation
# ---------------------------------------------------------------------------

def normalize_rewards(dataset: ManiFeelDataset):
    episode_returns = []
    cur = 0.0
    for i in range(dataset.size):
        cur += float(dataset.rewards[i])
        if dataset.dones_float[i] == 1.0:
            episode_returns.append(cur)
            cur = 0.0
    if len(episode_returns) < 2:
        print("[WARN] <2 trajectories, skipping reward normalisation.")
        return
    ret_range = max(episode_returns) - min(episode_returns)
    if ret_range < 1e-8:
        print("[WARN] Flat returns, skipping normalisation.")
        return
    dataset.rewards /= ret_range
    dataset.rewards *= 1000.0
    print(f"[INFO] Rewards normalised (range {ret_range:.4f} -> 1000).")


# ---------------------------------------------------------------------------
#  Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(agent, save_dir: str, step: int):
    ckpt = os.path.join(save_dir, f"checkpoint_{step}")
    os.makedirs(ckpt, exist_ok=True)
    agent.actor.save(os.path.join(ckpt, "actor.flax"))
    agent.critic.save(os.path.join(ckpt, "critic.flax"))
    agent.value.save(os.path.join(ckpt, "value.flax"))


def _maybe_make_eval_env(env_name, seed):
    if not env_name:
        print(f"[WARN] No environment name provided, skipping rollout evaluation.")
        return None, None
    try:
        import gym
        import wrappers
        from evaluation import evaluate
    except Exception as e:
        print(f"[WARN] Rollout eval disabled: {e}")
        return None, None

    try:
        env = gym.make(env_name)
        env = wrappers.EpisodeMonitor(env)
        env = wrappers.SinglePrecision(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env, evaluate
    except Exception as e:
        print(f"[WARN] Failed to create environment {env_name!r}: {e}")
        return None, None


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main(_):
    if not FLAGS.dataset_path:
        raise ValueError("--dataset_path is required.")

    os.makedirs(FLAGS.save_dir, exist_ok=True)
    summary_writer = SummaryWriter(
        os.path.join(FLAGS.save_dir, "tb", str(FLAGS.seed)),
        write_to_disk=True,
    )
    np.random.seed(FLAGS.seed)

    # ---- load dataset ----
    full_ds = ManiFeelDataset(
        FLAGS.dataset_path, clip_actions=FLAGS.clip_actions)

    mode = full_ds.mode
    arch = FLAGS.backbone
    r3m_ckpt = FLAGS.r3m_checkpoint or None

    print(f"[INFO] mode={mode}  backbone={arch}  "
          f"r3m_checkpoint={r3m_ckpt or '(none, random init)'}")

    train_ds, test_ds = full_ds.train_test_split(
        test_ratio=FLAGS.test_ratio, seed=FLAGS.seed)
    print("=== Train ===")
    print(train_ds.summary())
    print("=== Test ===")
    print(test_ds.summary())

    if FLAGS.validate:
        print("--- Train validation ---")
        train_ds.validate()
        print("--- Test validation ---")
        test_ds.validate()
        print()

    if FLAGS.normalize_rewards:
        normalize_rewards(train_ds)
        normalize_rewards(test_ds)

    # ---- build agent ----
    kwargs = dict(FLAGS.config)
    kwargs["max_steps"] = FLAGS.max_steps
    try:
        agent = MultiModalLearner(
            FLAGS.seed,
            train_ds.observation_example(),
            train_ds.actions[:1],
            arch=arch,
            mode=mode,
            r3m_checkpoint=r3m_ckpt,
            **kwargs,
        )
    except Exception as e:
        if _is_jax_gpu_dnn_init_failure(e):
            _print_jax_gpu_dnn_help()
        raise

    # passthrough until i implement isaac gym connection
    eval_env, eval_fn = _maybe_make_eval_env(FLAGS.env_name, FLAGS.seed)
    eval_returns = []

    print(f"[START] Training IQL (backbone={arch}, mode={mode}) "
          f"for {FLAGS.max_steps:,} steps "
          f"(batch={FLAGS.batch_size}, train={train_ds.size:,}, "
          f"test={test_ds.size:,})")

    # ---- training loop ----
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1, disable=not FLAGS.tqdm):
        batch = train_ds.sample(FLAGS.batch_size)
        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            info_host = jax.device_get(update_info)
            for k, v in info_host.items():
                v = np.asarray(v)
                if v.ndim == 0:
                    summary_writer.add_scalar(f"train/{k}", float(v), i)
                else:
                    summary_writer.add_histogram(f"train/{k}", v, i)
            summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            test_info = eval_on_dataset(agent, test_ds, FLAGS.batch_size)
            train_info = eval_on_dataset(agent, train_ds, FLAGS.batch_size)
            for k, v in test_info.items():
                summary_writer.add_scalar(f"test/{k}", v, i)
            for k, v in train_info.items():
                summary_writer.add_scalar(f"train_eval/{k}", v, i)
            summary_writer.flush()

            if eval_env is not None and eval_fn is not None:
                try:
                    stats = eval_fn(agent, eval_env, FLAGS.eval_episodes)
                    for k, v in stats.items():
                        summary_writer.add_scalar(
                            f"evaluation/average_{k}s", v, i)
                    summary_writer.flush()
                    eval_returns.append(
                        (i, float(stats.get("return", np.nan))))
                    np.savetxt(
                        os.path.join(FLAGS.save_dir, f"{FLAGS.seed}.txt"),
                        eval_returns, fmt=["%d", "%.1f"])
                except Exception as e:
                    print(f"[WARN] Rollout eval failed at step {i}: {e}")

        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_checkpoint(agent, FLAGS.save_dir, i)

    save_checkpoint(agent, FLAGS.save_dir, FLAGS.max_steps)
    summary_writer.close()
    print(f"\n[DONE] Checkpoints in {FLAGS.save_dir}")


if __name__ == "__main__":
    app.run(main)
