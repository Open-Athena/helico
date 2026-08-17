"""Fold MarinFold's four evaluation classes on Modal, and score lDDT.

`modal/bench.py` benchmarks FoldBench, whose targets, ground truths and
categories it is built around. This app answers a different question on a
different target set: does the by-class *contact* accuracy difference
(`.agents/project/figures/contact_accuracy_by_dataset.py`) survive into folding
accuracy? MarinFold supplies better contacts than Protenix v2 single sequence on
`foldbench100` and worse on de novo designs -- so does Helico win and lose in
the same places?

Targets come from `experiments/marinfold_contacts/byclass/`, built by that
directory's `build_targets.py` (ground truths converted to mmCIF) and
`export_contacts.py` (MarinFold exp199 contacts in Helico token indices). That
set is decontaminated: every target Helico could have trained on is dropped.

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
"""

import os
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
BYCLASS = ROOT / "experiments/marinfold_contacts/byclass/data"

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
    .add_local_dir(str(BYCLASS), remote_path="/root/byclass")
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
        self.model = Helico(cfg).cuda().to(torch.bfloat16).eval()
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.ccd = parse_ccd()

    @modal.method()
    def predict(self, target_id: str, n_samples: int = 3, n_cycles: int = 6,
                max_tokens: int = 2048) -> dict:
        import logging

        import torch
        from helico.bench import match_atoms, predict_target, score_monomer, structure_to_chains
        from helico.data import parse_mmcif

        logging.basicConfig(level=logging.INFO)
        row = {"target_id": target_id, "status": "error"}
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
            if pairs:
                if len(prot) != 1:
                    raise ValueError(
                        f"{target_id}: expected one protein chain, got "
                        f"{[c['id'] for c in prot]}")
                cid = prot[0]["id"]
                pairs = [(cid, int(i), cid, int(j)) for i, j in pairs]

            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
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
                return row
            tokenized, res = pred

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
            return row
        except Exception as e:  # noqa: BLE001 - one bad target must not kill the fan-out
            logging.exception(f"{target_id} failed")
            row["error"] = f"{type(e).__name__}: {e}"
            return row


@app.local_entrypoint()
def run(checkpoint: str, out_tag: str, n_samples: int = 3, n_cycles: int = 6,
        datasets: str = "", limit: int = 0):
    import csv

    with (BYCLASS / "targets.csv").open() as f:
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
        kwargs={"n_samples": n_samples, "n_cycles": n_cycles},
    ))

    out = ROOT / "experiments/marinfold_contacts/byclass/results"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{out_tag}.csv"
    fields = ["target_id", "dataset", "stem", "status", "lddt", "n_matched_atoms",
              "n_contacts", "error"]
    with dest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            m = meta.get(r["target_id"], {})
            w.writerow({**r, "dataset": m.get("dataset", ""), "stem": m.get("stem", "")})

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
