"""Protenix v2 baseline on the MarinFold contact-experiment targets.

Runs the **official ByteDance implementation** (`protenix==2.0.0`, model
`protenix-v2`) rather than Helico's reimplementation: v2 changes the
architecture (c_z=256, ~464M params) and reimplementing it just to benchmark it
would put any discrepancy in our code rather than in the comparison.

Two arms, both on the same 98 FoldBench monomer targets used everywhere else in
helico#11: `--use_msa true` and `--use_msa false` (single sequence).

Protenix runs at its own recommended defaults (N_cycle/N_sample/N_step chosen by
`--use_default_params`), which is more inference compute than the Helico arms
get. That is deliberate -- a baseline should be given its best shot, so a win
against it is conservative.

Usage:
    modal run modal/bench_protenix_v2.py --use-msa true  --out-tag v2_msa
    modal run modal/bench_protenix_v2.py --use-msa false --out-tag v2_singleseq
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent

PROTENIX_MODEL_NAME = "protenix-v2"
PROTENIX_VERSION = "2.0.0"

upstream_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11",
    )
    .apt_install("wget", "curl", "git", "build-essential", "ninja-build")
    .pip_install("ninja")
    # Let pip resolve protenix 2.0.0's own pinned dependency set rather than
    # freezing a hand-copied list that would drift from what upstream expects.
    .pip_install(f"protenix=={PROTENIX_VERSION}")
    .pip_install("huggingface_hub>=0.20")
    # ByteDance's CDN now returns 403 for protenix-v2.pt specifically -- every
    # other asset in dependency_url.py still resolves. Pre-place the community
    # mirror where protenix looks (CHECKPOINT_DIR = $PROTENIX_ROOT_DIR/checkpoint,
    # default $HOME) so get_model() finds it and never attempts the download.
    # The SHA256 is checked against the mirror's model card: a silently wrong
    # checkpoint would produce a plausible-but-meaningless baseline.
    .run_commands(
        "mkdir -p /root/checkpoint && python - <<'PY'\n"
        "import hashlib, shutil\n"
        "from huggingface_hub import hf_hub_download\n"
        "EXPECT = '8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599'\n"
        "p = hf_hub_download('TMF001/protenix-v2-weights', 'protenix-v2.pt')\n"
        "h = hashlib.sha256()\n"
        "with open(p, 'rb') as f:\n"
        "    for chunk in iter(lambda: f.read(1 << 22), b''):\n"
        "        h.update(chunk)\n"
        "assert h.hexdigest() == EXPECT, f'checkpoint sha256 mismatch: {h.hexdigest()}'\n"
        "shutil.copy(p, '/root/checkpoint/protenix-v2.pt')\n"
        "print('protenix-v2.pt staged and verified')\n"
        "PY"
    )
    .run_commands(
        "python -c 'from protenix.web_service.dependency_url import URL; print(len(URL))' || true"
    )
)

app = modal.App("helico-protenix-v2", image=upstream_image)

# Shared data volume (same one Helico's bench uses). Upstream Protenix
# writes output CIFs under /cache/helico-data/upstream_protenix/<pdb_id>/.
data_volume = modal.Volume.from_name("helico-bench-data", create_if_missing=True)
DATA_CACHE = "/cache/helico-data"


@app.cls(image=upstream_image, gpu="H100", timeout=5400,
         max_containers=10,
         volumes={DATA_CACHE: data_volume})
class UpstreamPredictor:
    @modal.method()
    def predict(
        self,
        pdb_id: str,
        input_json_relpath: str,   # relative to DATA_CACHE
        dump_relpath: str,         # relative to DATA_CACHE
        seeds_csv: str = "42",
        model_name: str = PROTENIX_MODEL_NAME,
        use_msa: str = "true",
    ) -> dict:
        """Run Protenix 1.0.9 inference. Uses the `protenix pred` CLI.

        With `--use_default_params=true` (the default), Protenix picks its
        recommended N_cycle / N_sample / N_step for the chosen model, which
        for protenix_base_default_v1.0.0 matches the published protocol
        (5 samples, 200 steps, 10 cycles).
        """
        import logging
        import subprocess

        logger = logging.getLogger(__name__)

        input_path = Path(DATA_CACHE) / input_json_relpath
        dump_dir = Path(DATA_CACHE) / dump_relpath
        dump_dir.mkdir(parents=True, exist_ok=True)

        if not input_path.exists():
            return {"pdb_id": pdb_id, "status": "error",
                    "error": f"missing input json at {input_path}"}

        # Per Protenix docs, the CLI auto-downloads the named model into
        # ~/.cache/protenix on first use.
        cmd = [
            "protenix", "pred",
            "-i", str(input_path),
            "-o", str(dump_dir),
            "-n", model_name,
            "--seeds", seeds_csv,
            "--use_msa", use_msa,
            "--use_template", "false",
            "--use_rna_msa", "false",
        ]
        logger.info(f"[{pdb_id}] running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, capture_output=False)
            data_volume.commit()
            produced = sorted(str(p.relative_to(dump_dir)) for p in dump_dir.rglob("*.cif"))
            return {
                "pdb_id": pdb_id,
                "status": "ok",
                "dump_relpath": dump_relpath,
                "n_cifs": len(produced),
                "cif_paths": produced,
            }
        except subprocess.CalledProcessError as e:
            return {"pdb_id": pdb_id, "status": "error",
                    "error": f"returncode={e.returncode}"}
        except Exception as e:
            return {"pdb_id": pdb_id, "status": "error", "error": repr(e)}


@app.local_entrypoint()
def run_v2(
    use_msa: str = "true",
    out_tag: str = "v2_msa",
    targets_csv: str = "",
    staging_dir: str = "/tmp/protenix-v2-staging",
    targets_file: str = "",
    gt_dir: str = "",
):
    """Stage the 98 contact-experiment targets, run Protenix v2, pull outputs.

    Scoring is a separate step (scripts/pm/score_upstream.py) so the same lDDT
    path scores these as scores every other arm.
    """
    import csv
    import shutil
    import subprocess
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from helico.upstream_protenix import build_protenix_input

    if targets_file:
        # A CSV with a target_id column and ground truths in --gt-dir. Used for
        # the by-class set (experiments/marinfold_contacts/byclass), whose
        # targets are not FoldBench targets and whose CIFs live elsewhere.
        with open(targets_file) as f:
            target_list = [r["target_id"] for r in csv.DictReader(f)]
    elif targets_csv:
        target_list = [t.strip() for t in targets_csv.split(",") if t.strip()]
    else:
        # Default: exactly the target list every other arm in helico#11 uses.
        with (ROOT / "experiments/marinfold_contacts/arms/targets.csv").open() as f:
            target_list = [r["pdb_id"] for r in csv.DictReader(f)]
    print(f"targets: {len(target_list)}  use_msa={use_msa}  tag={out_tag}")

    foldbench_local = Path.home() / ".cache/helico/data/benchmarks/FoldBench"
    gt_dir_local = Path(gt_dir) if gt_dir else foldbench_local / "examples/ground_truths"
    msa_local = foldbench_local / "foldbench-msas"

    staging_root = Path(staging_dir) / out_tag
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)
    out_root = ROOT / f"experiments/marinfold_contacts/upstream/{out_tag}"
    out_root.mkdir(parents=True, exist_ok=True)

    staged = []
    for pdb_id in target_list:
        stage = staging_root / pdb_id
        stage.mkdir()
        remote_base = f"/protenix_v2/{out_tag}/{pdb_id}"
        try:
            info = build_protenix_input(
                pdb_id=pdb_id,
                gt_cif_path=gt_dir_local / f"{pdb_id}.cif.gz",
                foldbench_msa_dir=msa_local,
                out_dir=stage,
                remote_msa_prefix=f"{DATA_CACHE}{remote_base}/msa",
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {pdb_id}: {type(e).__name__}: {e}")
            continue
        staged.append({"pdb_id": pdb_id, "stage_local": stage,
                       "remote_base": remote_base, **info})
    print(f"staged {len(staged)}/{len(target_list)}")

    print("uploading staging tree ...")
    subprocess.run(["modal", "volume", "put", "--force", "helico-bench-data",
                    str(staging_root), f"/protenix_v2/{out_tag}"], check=True)

    print(f"dispatching {len(staged)} runs ...")
    predictor = UpstreamPredictor()
    results = list(predictor.predict.map(
        [s["pdb_id"] for s in staged],
        [f"{s['remote_base'].lstrip('/')}/inputs.json" for s in staged],
        [f"{s['remote_base'].lstrip('/')}/predictions" for s in staged],
        kwargs={"use_msa": use_msa},
    ))
    ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"predictions ok: {ok}/{len(results)}")
    for r in results:
        if r.get("status") != "ok":
            print(f"  FAILED {r.get('pdb_id')}: {r.get('error')}")

    print("pulling outputs ...")
    subprocess.run(["modal", "volume", "get", "--force", "helico-bench-data",
                    f"/protenix_v2/{out_tag}", str(out_root)], check=False)
    print(f"done -> {out_root}")
