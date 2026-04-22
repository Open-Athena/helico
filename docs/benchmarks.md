# Benchmarks

Cross-experiment comparison tables computed from the bench summary CSVs
committed under each `experiments/exp<N>_*/data/` directory.

!!! note
    This page will be auto-generated from the committed CSVs in a future
    rev (`scripts/pm/plot_bench.py --all`). For now it's a placeholder —
    results will appear here as experiments land.

Methodology:

- **FoldBench** default subset: `cutoff_date > 2024-01-01`, `max_tokens <= 2048`.
- Scored with **LDDT** (all categories) and **DockQ** (interface categories);
  **LDDT-PLI** + **LRMSD** for `interface_protein_ligand`.
- Success threshold: `DockQ >= 0.23` (non-ligand interfaces), `LDDT-PLI > 0.8 && LRMSD < 2.0 Å` (ligand).
