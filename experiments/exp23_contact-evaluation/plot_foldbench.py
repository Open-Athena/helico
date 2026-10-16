"""Plot official all-denominator scores and paired assembly-bootstrap differences."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CATEGORIES = {
    'interface_protein_protein': 'Protein–protein',
    'interface_protein_ligand': 'Protein–ligand',
    'interface_antibody_antigen': 'Antibody–antigen',
    'interface_protein_peptide': 'Protein–peptide',
    'interface_protein_dna': 'Protein–DNA',
    'interface_protein_rna': 'Protein–RNA',
    'monomer_protein': 'Protein monomer',
    'monomer_rna': 'RNA monomer',
    'monomer_dna': 'DNA monomer',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = pd.read_csv(args.data / 'summary.csv').set_index(['category', 'model'])
    paired = pd.read_csv(args.data / 'paired_differences.csv').set_index('category')
    if set(paired.index) != set(CATEGORIES):
        raise ValueError('Unexpected or incomplete official benchmark categories')
    names = list(CATEGORIES); y = np.arange(len(names))
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 6.3), sharey=True,
                                    gridspec_kw={'width_ratios': [1.3, 1]})
    for model, label, color, offset in [('protenix', 'Protenix initialization', '#8b94a3', -.17),
                                         ('ema', 'Helico EMA', '#247ba0', .17)]:
        values = [summary.loc[(name, model),
                  'success_rate_all' if name.startswith('interface') else 'mean_lddt_all']
                  for name in names]
        left.barh(y + offset, 100 * np.array(values), height=.32, color=color, label=label)
    labels = [f"{CATEGORIES[name]}  (n={int(paired.loc[name, 'denominator'])})" for name in names]
    left.set_yticks(y, labels); left.invert_yaxis(); left.set_xlim(0, 100)
    left.set_xlabel('Interface success (%) / monomer lDDT (0–100)')
    left.set_title('Confidence-selected, four samples')
    fig.legend(*left.get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.57, .93), ncol=2, frameon=False, fontsize=9)
    values = paired.loc[names]
    mid = 100 * values['mean'].to_numpy()
    error = 100 * np.stack([values['mean'] - values['low'], values['high'] - values['mean']])
    right.errorbar(mid, y, xerr=error, fmt='o', color='#247ba0', capsize=3)
    right.axvline(0, color='#7c828c', lw=1)
    right.set_xlabel('Helico − initialization (points)')
    right.set_title('Paired difference, 95% interval')
    for axis in [left, right]:
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='x', alpha=.15); axis.set_axisbelow(True)
    fig.suptitle('FoldBench: matched full-input inference', fontsize=15, y=.98)
    fig.text(.02, .02, 'Every official target remains in the denominator; unavailable or failed predictions score zero.\n'
             'Intervals resample whole assemblies (4,000 draws). One recycle, 200 diffusion steps, up to 512 MSA rows; no templates.',
             fontsize=9, color='#444444')
    fig.tight_layout(rect=[0, .09, 1, .95])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ['.png', '.svg']:
        fig.savefig(args.output.with_suffix(suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
