"""Run the `mf_L` arm again on contacts from MarinFold #232's step-363000 checkpoint.

The published arms condition on the #232 *sweep* final (step 145,199), which was
the best decontaminated MarinFold checkpoint when this experiment ran. #232 then
continued that run to step 363,000 and it predicts contacts better (legacy-554
R-precision 0.6051 against 0.5916), so MarinFold #250 scored all 333 monomers
with it and kept the dense matrices.

Only the MarinFold arm changes. `off`, `oracle`, `v2ss` and `v2msa` do not
depend on which MarinFold checkpoint produced anything, so their published
results stand and are not re-run — which also makes the comparison paired: same
targets, same Helico checkpoint, same sampling, one input changed.

    uv run python experiments/exp14_foldbench_held_out_monomers/export_marinfold_contacts.py \
        --dense-dir <dir of foldbench_monomer__*.npz> \
        --precision-csv <that run's concatenated metrics> --suffix _363k
    uv run python experiments/exp14_foldbench_held_out_monomers/run_step363000_arm.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from helico.experiment import ensure_byclass_run, experiment_dir, set_experiment  # noqa: E402

#: The Helico checkpoint every published arm used. Changing it would make the new
#: arm incomparable with the ones that are not being re-run.
CHECKPOINT = "/ckpts/contacts-msafree-01/final.pt"
ARM = "mf_L_363k"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default=ARM)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpu", default="H100")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    set_experiment("exp14_foldbench_held_out_monomers")
    data = experiment_dir() / "data"
    arm_file = data / "arms" / f"{arguments.arm}.json"
    if not arm_file.exists():
        raise SystemExit(f"{arm_file} does not exist — run export_marinfold_contacts.py "
                         f"--dense-dir ... --suffix _363k first")
    arm = json.loads(arm_file.read_text())
    print(f"{arguments.arm}: {len(arm)} targets, "
          f"{sum(len(v) for v in arm.values())} contact pairs")
    if arguments.dry_run:
        return 0

    run = ensure_byclass_run(
        arguments.arm,
        targets_dir=data,
        checkpoint=CHECKPOINT,
        contacts_arm=arguments.arm,
        workers=arguments.workers,
        gpu=arguments.gpu,
        n_samples=3,      # every published arm's sampling, unchanged
        n_cycles=6,
        est_wall_hours=1.0,
    )
    print(f"cached={run.cached} arm={run.arm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
