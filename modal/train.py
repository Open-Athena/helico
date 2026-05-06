"""Train Helico on Modal — DDP over 8×H100, with W&B logging and volume-backed checkpoints.

Configure via env vars before `modal run`:
    HELICO_TRAIN_GPU=H100:8            # GPU type:count (default H100:8)
    HELICO_TRAIN_RUN_NAME=foo          # W&B run name + checkpoint subdir
    HELICO_TRAIN_MAX_STEPS=10000       # hard step cap
    HELICO_TRAIN_CROP=384              # token crop size
    HELICO_TRAIN_BATCH=1               # per-GPU batch (fine-tune stays at 1)
    HELICO_TRAIN_LR=5e-5               # LR (use 5e-5 when fine-tuning from v1)
    HELICO_TRAIN_WARMUP=200
    HELICO_TRAIN_SAVE_EVERY=250
    HELICO_TRAIN_LOG_EVERY=10
    HELICO_TRAIN_VAL_EVERY=0           # 0 disables; e.g. 500 runs val every 500 steps
    HELICO_TRAIN_VAL_SAMPLES=32
    HELICO_TRAIN_N_DIFFUSION_SAMPLES=8 # Diffusion noise samples per trunk forward (gh#6)
    HELICO_TRAIN_DIFFUSION_PAIR_SOURCE=z   # "z" or "distogram_logits" (gh#9)
    HELICO_TRAIN_FREEZE_TRUNK=0            # 1 = freeze trunk, train only diffusion (gh#9)
    HELICO_TRAIN_RESUME=               # /ckpts/<run>/step_<N>.pt to resume
    HELICO_TRAIN_PROTENIX_INIT=1       # warm-start from Protenix v1 weights
    HELICO_TRAIN_CUTOFF=2021-09-30     # train = release_date < this (AF3/Protenix/OF3 shared cutoff)
    HELICO_VAL_START=2022-05-01        # val window lower bound (AF3 Recent PDB)
    HELICO_VAL_END=2023-01-12          # val window upper bound (AF3 Recent PDB)
    HELICO_TRAIN_WANDB_PROJECT=helico

Example (proof run, 1×H100, 500 steps):
    HELICO_TRAIN_GPU=H100:1 HELICO_TRAIN_MAX_STEPS=500 HELICO_TRAIN_CROP=256 \
        HELICO_TRAIN_RUN_NAME=proof-v1 modal run modal/train.py

Example (full fine-tune, 8×H100):
    HELICO_TRAIN_GPU=H100:8 HELICO_TRAIN_MAX_STEPS=10000 HELICO_TRAIN_CROP=384 \
        HELICO_TRAIN_RUN_NAME=v1-finetune-01 modal run modal/train.py
"""

import os
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent

PROTENIX_URL_DEFAULT = "https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_base_default_v1.0.0.pt"
PROTENIX_URL = os.environ.get("HELICO_PROTENIX_URL", PROTENIX_URL_DEFAULT)
PROTENIX_CKPT_PATH = "/root/helico/checkpoints/" + PROTENIX_URL.rsplit("/", 1)[-1]

# Static Modal decorator values — set via env vars at `modal run` time.
GPU = os.environ.get("HELICO_TRAIN_GPU", "H100:8")
RUN_NAME = os.environ.get("HELICO_TRAIN_RUN_NAME", "helico-run")
WANDB_PROJECT = os.environ.get("HELICO_TRAIN_WANDB_PROJECT", "helico")

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "curl", "git")
    .pip_install(
        # Pin torch<2.11: torch 2.11 ships cuDNN 13 wheels whose flash-attn
        # kernel is missing for n_tokens>=128 in eval+no_grad mode (gh#3).
        # 2.10 ships cuDNN 9.x which handles those shapes correctly.
        "torch>=2.10,<2.11",
        "cuequivariance-torch>=0.8",
        "cuequivariance-ops-torch-cu12>=0.8",
        "biopython>=1.80",
        "numpy",
        "scipy",
        "pyyaml>=6.0",
        "huggingface_hub>=0.20",
        "requests",
        "tqdm",
        "wandb",
    )
    # Protenix v1 checkpoint — used for warm-start initialization.
    .run_commands(
        f"mkdir -p /root/helico/checkpoints && "
        f"curl -fL --retry 5 --retry-delay 5 --retry-connrefused "
        f"--connect-timeout 30 --max-time 900 "
        f"-o {PROTENIX_CKPT_PATH} {PROTENIX_URL} && "
        f"ls -lh {PROTENIX_CKPT_PATH}"
    )
    .add_local_dir(str(ROOT / "src"), remote_path="/root/helico/src")
    .add_local_file(str(ROOT / "pyproject.toml"), remote_path="/root/helico/pyproject.toml")
    .add_local_file(str(ROOT / "README.md"), remote_path="/root/helico/README.md")
)

app = modal.App("helico-train", image=train_image)

# Volumes:
#   helico-train-data — bulk training data (populated by modal/sync_train_data.py
#     or by uploading helico-processed/ from local).
#   helico-checkpoints — durable checkpoint storage across runs/restarts.
data_volume = modal.Volume.from_name("helico-train-data", create_if_missing=True)
ckpt_volume = modal.Volume.from_name("helico-checkpoints", create_if_missing=True)
DATA_ROOT = "/cache/helico-data"
CKPT_ROOT = "/ckpts"


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v else default


# Collected at module import so they're visible to the remote function.
TRAIN_ARGS = {
    "run_name": RUN_NAME,
    "max_steps": _env_int("HELICO_TRAIN_MAX_STEPS", 10_000),
    "crop_size": _env_int("HELICO_TRAIN_CROP", 384),
    "batch_size": _env_int("HELICO_TRAIN_BATCH", 1),
    "lr": _env_float("HELICO_TRAIN_LR", 5e-5),
    "warmup_steps": _env_int("HELICO_TRAIN_WARMUP", 200),
    "save_every": _env_int("HELICO_TRAIN_SAVE_EVERY", 250),
    "log_every": _env_int("HELICO_TRAIN_LOG_EVERY", 10),
    "val_every": _env_int("HELICO_TRAIN_VAL_EVERY", 0),
    "val_samples": _env_int("HELICO_TRAIN_VAL_SAMPLES", 32),
    "n_diffusion_samples": _env_int("HELICO_TRAIN_N_DIFFUSION_SAMPLES", 8),
    "diffusion_pair_source": os.environ.get("HELICO_TRAIN_DIFFUSION_PAIR_SOURCE", "z"),
    "freeze_trunk": os.environ.get("HELICO_TRAIN_FREEZE_TRUNK", "0") == "1",
    "resume_from": os.environ.get("HELICO_TRAIN_RESUME", ""),
    "protenix_init": os.environ.get("HELICO_TRAIN_PROTENIX_INIT", "1") == "1",
    "train_cutoff": os.environ.get("HELICO_TRAIN_CUTOFF", "2021-09-30"),
    "val_cutoff_start": os.environ.get("HELICO_VAL_START", "2022-05-01"),
    "val_cutoff_end": os.environ.get("HELICO_VAL_END", "2023-01-12"),
    "wandb_project": WANDB_PROJECT,
    "gpu": GPU,
}


@app.function(
    gpu=GPU,
    timeout=24 * 3600,
    volumes={DATA_ROOT: data_volume, CKPT_ROOT: ckpt_volume},
    secrets=[modal.Secret.from_name("helico-wandb-modal")],
    ephemeral_disk=600 * 1024,
)
def train_remote(args: dict) -> dict:
    """Run DDP training inside a single multi-GPU container via torchrun."""
    import subprocess
    import shlex

    # Install helico first (editable). wandb goes into the venv too so
    # rank-0 logging works — the image-level wandb lives in a different
    # Python that helico-train doesn't use.
    subprocess.run(
        "cd /root/helico && uv venv --python 3.11 && uv pip install -e . wandb",
        check=True, shell=True,
    )

    n_gpus = int(args["gpu"].split(":")[-1]) if ":" in args["gpu"] else 1

    run_name = args["run_name"]
    run_ckpt_dir = Path(CKPT_ROOT) / run_name
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Expected layout on the data volume (populated by sync_train_data.py or a
    # HF snapshot of our own processed data).
    processed_dir = Path(DATA_ROOT) / "processed"
    manifest_path = processed_dir / "manifest.json"
    msa_dir = Path(DATA_ROOT) / "msas"  # optional, extracted MSA tree
    assert manifest_path.exists(), f"manifest not found: {manifest_path}"

    # If warm-starting from Protenix v1 and there's no existing checkpoint,
    # create step_0.pt from Protenix weights so the training loop resumes from it.
    resume_from = args["resume_from"]
    if not resume_from and args["protenix_init"]:
        seed_path = run_ckpt_dir / "protenix_v1_seed.pt"
        if not seed_path.exists():
            _seed_from_protenix(seed_path)
            ckpt_volume.commit()
        resume_from = str(seed_path)

    # Environment for subprocess.
    env = os.environ.copy()
    env["HELICO_DATA_DIR"] = DATA_ROOT
    env["WANDB_PROJECT"] = args["wandb_project"]
    env["WANDB_RUN_NAME"] = run_name
    env["HELICO_WANDB_ENABLE"] = "1"
    # For cuEquivariance kernel caches.
    env["PYTHONUNBUFFERED"] = "1"

    # Build the training command. Multi-GPU uses torchrun; single-GPU just calls
    # helico-train directly (no DDP init).
    venv_bin = "/root/helico/.venv/bin"
    base_cli = [
        "--processed-dir", str(processed_dir),
        "--manifest", str(manifest_path),
        "--crop-size", str(args["crop_size"]),
        "--batch-size", str(args["batch_size"]),
        "--lr", str(args["lr"]),
        "--max-steps", str(args["max_steps"]),
        "--warmup-steps", str(args["warmup_steps"]),
        "--save-every", str(args["save_every"]),
        "--log-every", str(args["log_every"]),
        "--val-every", str(args["val_every"]),
        "--val-samples", str(args["val_samples"]),
        "--n-diffusion-samples", str(args["n_diffusion_samples"]),
        "--diffusion-pair-source", args["diffusion_pair_source"],
        "--checkpoint-dir", str(run_ckpt_dir),
        "--train-cutoff", args["train_cutoff"],
        "--val-cutoff-start", args["val_cutoff_start"],
        "--val-cutoff-end", args["val_cutoff_end"],
    ]
    if msa_dir.exists():
        base_cli += ["--msa-dir", str(msa_dir)]
    if resume_from:
        base_cli += ["--resume", resume_from]
    if args.get("freeze_trunk"):
        base_cli += ["--freeze-trunk"]

    if n_gpus > 1:
        cmd = [
            f"{venv_bin}/torchrun",
            "--standalone", f"--nproc_per_node={n_gpus}",
            "-m", "helico.train",
            "--distributed",
        ] + base_cli
    else:
        cmd = [f"{venv_bin}/helico-train"] + base_cli

    print(f"[train] command: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    # Stream output so Modal logs show progress live.
    result = subprocess.run(cmd, env=env, check=False)
    # Commit checkpoint volume even on failure so we can inspect partial state.
    ckpt_volume.commit()
    return {
        "run_name": run_name,
        "returncode": result.returncode,
        "checkpoint_dir": str(run_ckpt_dir),
    }


def _seed_from_protenix(out_path: Path) -> None:
    """Save a Helico checkpoint initialized from Protenix v1, compatible with load_checkpoint."""
    import sys
    sys.path.insert(0, "/root/helico/src")
    from collections import OrderedDict
    import torch
    from helico.model import Helico
    from helico.load_protenix import infer_protenix_config, load_protenix_state_dict

    print(f"[seed] loading Protenix checkpoint from {PROTENIX_CKPT_PATH}", flush=True)
    ckpt = torch.load(PROTENIX_CKPT_PATH, map_location="cpu", weights_only=False)
    ptx_sd = ckpt["model"]
    ptx_sd = OrderedDict((k.removeprefix("module."), v) for k, v in ptx_sd.items())
    config = infer_protenix_config(ptx_sd)
    print(f"[seed] inferred config: d_pair={config.d_pair}, d_msa={config.d_msa}", flush=True)
    model = Helico(config)
    stats = load_protenix_state_dict(ptx_sd, model)
    print(f"[seed] transferred {stats['n_transferred']} params", flush=True)

    torch.save(
        {"step": 0, "model_state_dict": model.state_dict()},
        out_path,
    )
    print(f"[seed] wrote {out_path}", flush=True)


@app.local_entrypoint()
def main():
    print(f"Launching Helico training run: {TRAIN_ARGS['run_name']}")
    print(f"  gpu        = {TRAIN_ARGS['gpu']}")
    print(f"  max_steps  = {TRAIN_ARGS['max_steps']}")
    print(f"  crop_size  = {TRAIN_ARGS['crop_size']}")
    print(f"  lr         = {TRAIN_ARGS['lr']}")
    print(f"  resume     = {TRAIN_ARGS['resume_from'] or '(none)'}")
    print(f"  protenix_init = {TRAIN_ARGS['protenix_init']}")
    result = train_remote.remote(TRAIN_ARGS)
    print(f"\n=== Result ===\n{result}")
