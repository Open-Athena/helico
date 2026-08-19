"""Fold MarinFold's four evaluation classes on Modal, and score lDDT.

`modal/bench.py` benchmarks FoldBench, whose targets, ground truths and
categories it is built around. This app answers a different question on a
different target set: does the by-class *contact* accuracy difference
(`.agents/project/figures/contact_accuracy_by_dataset.py`) survive into folding
accuracy? MarinFold supplies better contacts than Protenix v2 single sequence on
`foldbench100` and worse on de novo designs -- so does Helico win and lose in
the same places?

Targets come from `experiments/marinfold_contacts/byclass/` by default, built by
that directory's `build_targets.py` (ground truths converted to mmCIF) and
`export_contacts.py` (MarinFold exp199 contacts in Helico token indices). That
set is decontaminated: every target Helico could have trained on is dropped.

`HELICO_TARGETS_DIR` points the app at a different set with the same layout --
`targets.csv`, `gt/<target_id>.cif.gz`, `arms/<arm>.json` -- and results are
written next to it, in `<targets-dir>/../results`.

Everything downstream of "parse the ground truth" is the same code the FoldBench
bench uses -- `structure_to_chains`, `predict_target`, `match_atoms`,
`score_monomer` -- so the numbers are comparable to the monomer arms there.

Arms, selected by environment variable:

    HELICO_BYCLASS_CONTACTS_ARM=marinfold_L   MarinFold's predicted contacts
    HELICO_BYCLASS_ORACLE=1                   ground-truth contacts (the ceiling)
    (neither)                                 contacts withheld

Run:
    HELICO_BYCLASS_CONTACTS_ARM=marinfold_L modal run --detach \\
        modal/bench_byclass.py --checkpoint /ckpts/contacts-msafree-01/final.pt \\
        --out-tag mf_L

`helico.experiment.ensure_byclass_run` wraps this with idempotent caching and a
cost estimate, and is what notebooks should call.
"""

import os
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent

# The target set is a directory of `targets.csv` + `gt/<target_id>.cif.gz` +
# `arms/<arm>.json`. It defaults to the by-class set this app was written for;
# any experiment with the same layout points at its own via the env var, which
# is how exp14 runs exp245's FoldBench monomer sets through this same runner
# rather than forking it.
TARGETS_DIR = Path(os.environ.get(
    "HELICO_TARGETS_DIR", str(ROOT / "experiments/marinfold_contacts/byclass/data")))

N_WORKERS = int(os.environ.get("HELICO_BENCH_WORKERS", "8"))
GPU_TYPE = os.environ.get("HELICO_BENCH_GPU", "H100")

# Module-level env reads happen twice -- once locally, once when Modal imports
# this module inside the container, where the launching shell's env is gone.
# Both values are baked into the image below so the two agree. bench.py learned
# this the hard way: a bench that silently ran without contacts and reported
# success.
CONTACTS_ARM = os.environ.get("HELICO_BYCLASS_CONTACTS_ARM", "")
ORACLE = os.environ.get("HELICO_BYCLASS_ORACLE", "0") == "1"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "curl")
    .pip_install(
        "torch>=2.7",
        "cuequivariance-torch>=0.8,<0.9",
        "cuequivariance-ops-torch-cu12>=0.8,<0.9",
        "biopython>=1.80",
        "numpy>=2.0",
        "scipy",
        "pyyaml>=6.0",
        "huggingface_hub>=0.20",
        "requests",
        "tqdm",
        # helico.contacts imports pyconfind for the oracle arm.
        "pyconfind>=0.6",
        # score_monomer computes TM-score alongside lDDT. Scoring happens in
        # this container rather than a separate CPU one, so tmtools belongs
        # here; without it every target predicts fine and then fails to score.
        "tmtools",
    )
    .run_commands(
        "python -c 'from pyconfind import cached_rotamer_library;"
        " print(cached_rotamer_library())'"
    )
    .env({"HELICO_BYCLASS_CONTACTS_ARM": CONTACTS_ARM,
          "HELICO_BYCLASS_ORACLE": "1" if ORACLE else "0"})
    .add_local_dir(str(TARGETS_DIR), remote_path="/root/byclass")
    .add_local_dir(str(ROOT / "src"), remote_path="/root/helico/src")
    .add_local_file(str(ROOT / "pyproject.toml"), remote_path="/root/helico/pyproject.toml")
    .add_local_file(str(ROOT / "README.md"), remote_path="/root/helico/README.md")
)

app = modal.App("helico-bench-byclass", image=image)

data_volume = modal.Volume.from_name("helico-bench-data", create_if_missing=True)
DATA_CACHE = "/cache/helico-data"
ckpt_volume = modal.Volume.from_name("helico-checkpoints", create_if_missing=True)
CKPT_MOUNT = "/ckpts"


@app.cls(image=image, gpu=GPU_TYPE, timeout=3600, max_containers=N_WORKERS,
         volumes={DATA_CACHE: data_volume, CKPT_MOUNT: ckpt_volume},
         secrets=[modal.Secret.from_name("helico-hf-modal")])
class Predictor:
    checkpoint_path: str = modal.parameter(default="")

    @modal.enter()
    def setup(self):
        import json
        import os
        import subprocess
        import sys

        os.environ["HELICO_DATA_DIR"] = DATA_CACHE
        os.makedirs(DATA_CACHE, exist_ok=True)
        subprocess.run("cd /root/helico && uv venv --python 3.11 && uv pip install -e .",
                       check=True, shell=True)
        sys.path.insert(0, "/root/helico/src")

        # Fail loudly on a missing arm. A silently absent contact map looks
        # exactly like "predicted contacts do not help".
        self.contact_map = None
        if CONTACTS_ARM:
            arm = Path("/root/byclass/arms") / f"{CONTACTS_ARM}.json"
            if not arm.exists():
                raise FileNotFoundError(
                    f"contact arm {CONTACTS_ARM!r} not found at {arm}; available: "
                    f"{[p.stem for p in arm.parent.glob('*.json')]}")
            self.contact_map = json.loads(arm.read_text())
            print(f"contact arm {CONTACTS_ARM}: {len(self.contact_map)} targets, "
                  f"{sum(len(v) for v in self.contact_map.values())} pairs", flush=True)
        print(f"oracle contacts: {'ON' if ORACLE else 'OFF'}", flush=True)

        import torch
        from huggingface_hub import snapshot_download
        from helico.data import parse_ccd
        from helico.model import Helico

        # Only the CCD cache is needed -- the ground truths are mounted, not
        # downloaded, so none of FoldBench is pulled here.
        for attempt in range(5):
            try:
                snapshot_download("timodonnell/helico-data", repo_type="dataset",
                                  local_dir=DATA_CACHE,
                                  allow_patterns=["processed/ccd_cache.pkl"],
                                  max_workers=8, etag_timeout=30)
                break
            except Exception as e:  # noqa: BLE001 - transient CDN stalls
                print(f"ccd download attempt {attempt+1} failed: {e}", flush=True)
        else:
            raise RuntimeError("ccd_cache.pkl download failed after 5 attempts")
        data_volume.commit()

        from helico.model import HelicoConfig

        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if "model_state_dict" not in ckpt:
            raise ValueError(f"{self.checkpoint_path} is not a Helico checkpoint")
        # The saved config is a TrainConfig dict, so it carries training-only
        # fields (lr, warmup, ...) alongside the model ones. Take the
        # intersection rather than a hardcoded subset -- picking a subset is
        # how bench.py once silently benched a different model than the
        # checkpoint specified.
        saved = ckpt.get("config") or {}
        overrides = {k: v for k, v in saved.items() if hasattr(HelicoConfig, k)}
        cfg = HelicoConfig(**overrides)
        print(f"checkpoint step {ckpt.get('step')} ({len(overrides)} config fields), "
              f"use_contacts={cfg.use_contacts}, use_msa={cfg.use_msa}", flush=True)
        import platform
        import socket
        import time

        load_started = time.monotonic()
        self.model = Helico(cfg).cuda().to(torch.bfloat16).eval()
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.ccd = parse_ccd()
        self.model_load_seconds = time.monotonic() - load_started

        # Recorded per run so a number can always be traced back to the model,
        # the sampling settings and the hardware that produced it. Without this
        # a results CSV is uninterpretable a month later, and re-running to find
        # out costs what the original run cost.
        properties = torch.cuda.get_device_properties(0)
        self.run_meta = {
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_step": ckpt.get("step"),
            "use_contacts": bool(cfg.use_contacts),
            "use_msa": bool(cfg.use_msa),
            "dtype": "bfloat16",
            "contacts_arm": CONTACTS_ARM,
            "oracle_contacts": ORACLE,
            "n_contact_arm_targets": len(self.contact_map or {}),
            "gpu_name": str(properties.name),
            "gpu_total_memory_gb": round(properties.total_memory / 1e9, 2),
            "gpu_compute_capability": f"{properties.major}.{properties.minor}",
            "gpu_count": int(torch.cuda.device_count()),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            # str(): torch.__version__ is a TorchVersion, a str subclass living
            # in torch.torch_version. Returning it unwrapped pickles that module
            # reference into the result, and the launching client has no torch
            # to unpickle it with -- every call then dies in deserialization
            # *after* the GPU work is done, and with --detach the containers
            # keep running with nobody collecting the results.
            "torch_version": str(torch.__version__),
            "model_load_seconds": round(self.model_load_seconds, 3),
        }
        print(f"run metadata: {self.run_meta}", flush=True)

    @modal.method()
    def predict(self, target_id: str, n_samples: int = 3, n_cycles: int = 6,
                max_tokens: int = 2048) -> dict:
        import logging

        import torch
        from helico.bench import match_atoms, predict_target, score_monomer, structure_to_chains
        from helico.data import parse_mmcif

        import time

        logging.basicConfig(level=logging.INFO)
        row = {"target_id": target_id, "status": "error",
               "n_samples": n_samples, "n_cycles": n_cycles,
               "n_seeds": 1, "seed": 42, "max_tokens": max_tokens,
               **self.run_meta}
        started = time.monotonic()
        try:
            gt_path = Path("/root/byclass/gt") / f"{target_id}.cif.gz"
            gt = parse_mmcif(gt_path, max_resolution=float("inf"))
            assert gt is not None, f"failed to parse {gt_path}"
            chains = structure_to_chains(gt)

            # Address contacts by the chain id actually parsed, not the "A"
            # default. These ground truths come from gemmi, which names the
            # subchain "Axp"; against the default every pair silently fell out
            # of range and the arm scored as contacts-withheld.
            prot = [c for c in chains if c["type"] == "protein"]
            pairs = (self.contact_map or {}).get(target_id)
            if CONTACTS_ARM and not pairs:
                # A target the arm does not cover must not quietly fold without
                # contacts: that scores exactly like the contacts-withheld arm
                # and is indistinguishable from "predicted contacts do not
                # help" once it reaches a mean. Report it and let the analysis
                # drop the target from every arm.
                row["status"] = "no_contacts"
                return row
            if pairs:
                if len(prot) != 1:
                    raise ValueError(
                        f"{target_id}: expected one protein chain, got "
                        f"{[c['id'] for c in prot]}")
                cid = prot[0]["id"]
                pairs = [(cid, int(i), cid, int(j)) for i, j in pairs]

            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            predict_started = time.monotonic()
            pred = predict_target(
                self.model, chains, self.ccd, target_name=target_id,
                n_samples=n_samples, max_tokens=max_tokens,
                # Helico here is MSA-free by construction; passing no server
                # keeps it that way and skips the alignment fetch entirely.
                msa_server_url=None, single_sequence=True, n_cycles=n_cycles,
                oracle_contacts_from=gt if ORACLE else None,
                contact_pairs=pairs,
            )
            if pred is None:
                row["status"] = "too_large"
                row["elapsed_seconds"] = round(time.monotonic() - started, 3)
                return row
            tokenized, res = pred
            row["predict_seconds"] = round(time.monotonic() - predict_started, 3)
            row["n_tokens"] = int(tokenized.n_tokens)

            # Scoring runs here rather than in a separate CPU class: these are
            # monomers, so score_monomer's lDDT is all that is needed and it
            # costs nothing next to the diffusion sampling.
            coords = res["coords"][0].cpu().float().numpy()
            matched = match_atoms(tokenized, coords, gt)
            if len(matched.pred_coords) == 0:
                row["status"] = "no_match"
                return row
            row.update(score_monomer(matched))
            row["n_matched_atoms"] = len(matched.pred_coords)
            row["n_contacts"] = len(pairs or [])
            row["status"] = "ok"

            # The structure itself, returned rather than written to a volume:
            # eight containers committing concurrently is a conflict waiting to
            # happen, and a gzipped PDB of a few hundred residues is tens of
            # kilobytes. Re-scoring, re-plotting, or computing a metric nobody
            # thought of yet then costs nothing instead of a full re-run.
            import gzip

            from helico.train import coords_to_pdb

            pdb = coords_to_pdb(res["coords"][0], res["plddt"][0], tokenized)
            row["pdb_gz"] = gzip.compress(pdb.encode())
            row["mean_plddt"] = float(res["plddt"][0].mean())
            row["elapsed_seconds"] = round(time.monotonic() - started, 3)
            return row
        except Exception as e:  # noqa: BLE001 - one bad target must not kill the fan-out
            logging.exception(f"{target_id} failed")
            row["error"] = f"{type(e).__name__}: {e}"
            row["elapsed_seconds"] = round(time.monotonic() - started, 3)
            return row


@app.local_entrypoint()
def run(checkpoint: str, out_tag: str, n_samples: int = 3, n_cycles: int = 6,
        max_tokens: int = 2048, datasets: str = "", limit: int = 0):
    import csv

    with (TARGETS_DIR / "targets.csv").open() as f:
        targets = list(csv.DictReader(f))
    if datasets:
        keep = set(datasets.split(","))
        targets = [t for t in targets if t["dataset"] in keep]
    if limit:
        targets = targets[:limit]
    meta = {t["target_id"]: t for t in targets}
    print(f"{len(targets)} targets  arm={CONTACTS_ARM or ('oracle' if ORACLE else 'off')}  "
          f"tag={out_tag}")

    predictor = Predictor(checkpoint_path=checkpoint)
    rows = list(predictor.predict.map(
        [t["target_id"] for t in targets],
        kwargs={"n_samples": n_samples, "n_cycles": n_cycles,
                "max_tokens": max_tokens},
    ))

    import gzip
    import json
    import shutil

    out = TARGETS_DIR.parent / "results"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{out_tag}.csv"
    # score_monomer returns four metrics; the earlier field list kept only
    # lDDT and silently dropped the rest, which meant re-running to answer
    # "what was the TM-score?".
    fields = ["target_id", "dataset", "stem", "status", "lddt", "tm_score",
              "gdt_ts", "rmsd", "n_matched_atoms", "n_contacts", "n_tokens",
              "mean_plddt", "error"]
    with dest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            m = meta.get(r["target_id"], {})
            w.writerow({**r, "dataset": m.get("dataset", ""), "stem": m.get("stem", "")})

    # Predicted structures, one gzipped PDB per target. Re-scoring against a
    # different metric, or a reviewer looking at a specific prediction, then
    # costs nothing rather than another full run.
    # Cleared, not merged: a re-run over a smaller target list would otherwise
    # leave the previous run's structures in place, and they would be cached,
    # published and read back as this run's output -- possibly from a different
    # checkpoint.
    structures = out / "predictions" / out_tag
    if structures.exists():
        shutil.rmtree(structures)
    structures.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for r in rows:
        blob = r.get("pdb_gz")
        if not blob:
            continue
        (structures / f"{r['target_id']}.pdb.gz").write_bytes(blob)
        n_written += 1

    # Per-target timing and the hardware it ran on, in exp245's timings shape.
    timing_fields = ["target_id", "dataset", "n_tokens", "status",
                     "elapsed_seconds", "predict_seconds", "model_load_seconds",
                     "gpu_name", "gpu_total_memory_gb", "gpu_compute_capability",
                     "hostname", "platform", "torch_version"]
    with (out / f"{out_tag}.timings.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=timing_fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            m = meta.get(r["target_id"], {})
            w.writerow({**r, "dataset": m.get("dataset", "")})

    # One run manifest: what was run, on what, with which sampling settings.
    # Taken from the workers rather than from the launch arguments, so it
    # records what actually executed.
    sample = next((r for r in rows if r.get("gpu_name")), {})
    elapsed = [r["elapsed_seconds"] for r in rows if r.get("elapsed_seconds")]
    manifest = {
        "tag": out_tag,
        "targets_dir": str(TARGETS_DIR),
        "n_targets": len(targets),
        "n_ok": sum(1 for r in rows if r.get("status") == "ok"),
        "n_structures": n_written,
        "arm": CONTACTS_ARM or ("oracle" if ORACLE else "off"),
        "sampling": {
            "n_diffusion_samples": n_samples,
            "n_trunk_recycles": n_cycles,
            "n_trunk_runs": 1,
            "seed": 42,
            "single_sequence": True,
            "msa": False,
        },
        "model": {k: sample.get(k) for k in
                  ("checkpoint_path", "checkpoint_step", "use_contacts",
                   "use_msa", "dtype")},
        "hardware": {k: sample.get(k) for k in
                     ("gpu_name", "gpu_total_memory_gb", "gpu_compute_capability",
                      "hostname", "platform", "python_version", "torch_version")},
        "timing_seconds": {
            "total_gpu": round(sum(elapsed), 1),
            "per_target_mean": round(sum(elapsed) / max(len(elapsed), 1), 2),
            "per_target_max": round(max(elapsed), 2) if elapsed else None,
            "model_load": sample.get("model_load_seconds"),
        },
        "workers": N_WORKERS,
        "gpu_type": GPU_TYPE,
    }
    (out / f"{out_tag}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{n_written} structures -> {structures}")

    ok = [r for r in rows if r.get("status") == "ok"]
    print(f"ok {len(ok)}/{len(rows)} -> {dest}")
    by: dict[str, list] = {}
    for r in ok:
        by.setdefault(meta[r["target_id"]]["dataset"], []).append(r["lddt"])
    for ds, vs in sorted(by.items()):
        print(f"  {ds:14s} n={len(vs):4d}  lDDT {sum(vs)/len(vs):.4f}")
    for r in rows:
        if r.get("status") != "ok":
            print(f"  FAILED {r['target_id']}: {r.get('status')} {r.get('error', '')}")
