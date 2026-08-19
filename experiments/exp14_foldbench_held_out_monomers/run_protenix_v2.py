"""Step 4b -- run Protenix-v2 on all 333 units, in both modes.

`modal/bench_protenix_v2.py` runs ByteDance's own `protenix==2.0.0` at its
recommended defaults, which is more inference compute than the Helico arms get.
That is deliberate: a baseline should be given its best shot, so a Helico win
against it is conservative.

Both modes are run here rather than reusing exp245's Protenix predictions. 294
of the 333 units are in exp245's Modal volume and 39 are not -- those were
reused from older published runs -- so reusing would mix two sampling
provenances inside one arm and leave 39 units to be run at settings matched by
hand. Running all 333 at one setting costs more and removes the question.

The +MSA mode reads FoldBench's shipped alignments; `gen_missing_msas.py` has to
have filled the 16 gaps first, because `build_protenix_input` raises on a
missing a3m in *either* mode.

Run:
    uv run python experiments/exp14_foldbench_held_out_monomers/run_protenix_v2.py
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import upstream as U  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PRED_ROOT = U.CACHE / "protenix_v2"
MODES = {"single_seq": "false", "msa": "true"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", action="append", choices=sorted(MODES),
                        help="repeatable; defaults to both")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    modes = args.mode or sorted(MODES)

    with (U.DATA / "targets.csv").open() as handle:
        targets = list(csv.DictReader(handle))
    missing = [t["target_id"] for t in targets if t["msa_available"] == "0"]
    if missing:
        raise SystemExit(
            f"{len(missing)} targets still have no a3m ({missing[:5]}). "
            f"build_protenix_input raises on a missing alignment in both modes, "
            f"so run gen_missing_msas.py and rebuild targets.csv first."
        )

    PRED_ROOT.mkdir(parents=True, exist_ok=True)
    env = {**os.environ,
           "HELICO_UPSTREAM_DIR": str(PRED_ROOT),
           "HELICO_BENCH_WORKERS": str(args.workers)}

    for mode in modes:
        # Invoke Modal through this interpreter, not the `modal` on PATH: that
        # one is Anaconda's and has no torch, and this app's *local* entrypoint
        # imports helico (via build_protenix_input) before it dispatches
        # anything. bench_byclass's entrypoint does not, which is why it runs
        # fine either way.
        cmd = [
            sys.executable, "-m", "modal", "run", "--detach",
            "modal/bench_protenix_v2.py",
            "--use-msa", MODES[mode],
            "--out-tag", mode,
            "--targets-file", str(U.DATA / "targets.csv"),
            "--gt-dir", str(U.GT_DIR),
        ]
        print(f"[{mode}] {len(targets)} targets: {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True, env=env, cwd=str(REPO_ROOT))
        print(f"[{mode}] -> {PRED_ROOT / mode}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
