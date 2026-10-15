"""Render the committed validation measurements without rerunning inference."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 180})


def conditioning():
    data = pd.read_csv(ROOT/'data/conditioning-v1/conditioning_summary.csv')
    groups = ['intra', 'pp', 'pl']
    labels = ['Within protein', 'Protein–protein', 'Protein–ligand*']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    specs = [('rescue_positive4', 'requested_contact_satisfaction',
              'Missed native contacts recovered', 'Requested contacts satisfied (%)', 100),
             ('corrective_negative4', 'requested_no_contact_satisfaction',
              'False contacts removed', 'Requested absences satisfied (%)', 100),
             ('full', 'change_observed_lddt',
              'Effect of full native contact matrix', 'Change in observed-atom lDDT', 1)]
    for ax, (mode, metric, title, ylabel, scale) in zip(axes, specs):
        for msa, shift, color in [(True, -.18, '#1e5c93'), (False, .18, '#b07732')]:
            rows = data[(data['mode']==mode)&(data.metric==metric)&(data.msa==msa)].set_index('group').loc[groups]
            mean = rows['mean'].to_numpy()*scale
            errors = np.stack([rows['mean']-rows.low, rows.high-rows['mean']])*scale
            ax.bar(np.arange(3)+shift, mean, .34, color=color,
                   label='MSA on' if msa else 'MSA off', yerr=errors, capsize=3)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(3), labels, rotation=20, ha='right')
        ax.axhline(0, color='.5', linewidth=.7)
    axes[0].legend(frameon=False)
    fig.suptitle('Contact conditioning: structural control remains weak', fontsize=14, x=.51)
    fig.text(.02, .015, 'Oracle validation diagnostic; 8 targets/group, 2 seeds; targets without eligible corrections omitted. 95% target-bootstrap intervals.\n'
             'First two panels request up to four corrections. *Ligand group includes ions/glycans. lDDT uses fixed atom/chain correspondence.', fontsize=9)
    fig.tight_layout(rect=(0,.11,1,.94))
    fig.savefig(ROOT/'plots/contact-conditioning.png');plt.close(fig)


def search():
    data = pd.read_csv(ROOT/'data/search-v1/search_paired_differences.csv')
    data = data[data.horizon=='unmask']
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
    for ax, metric, title in zip(axes, ['selected_observed_lddt','oracle_observed_lddt'],
                                ['Confidence-selected structure', 'Best generated structure (oracle)']):
        rows = data[data.metric==metric].set_index('contrast').loc[['tree-minus-ordinary','unmask-minus-ordinary']]
        ax.errorbar(rows['mean'], [1,0], xerr=np.stack([rows['mean']-rows.low,rows.high-rows['mean']]),
                    fmt='o', color='#1e5c93', capsize=5)
        ax.axvline(0,color='.5',linewidth=.8)
        ax.set_yticks([1,0],['Contact tree','Progressive unmasking'])
        ax.set_ylim(-.5,1.5);ax.set_xlim(-.006,.012)
        ax.set_xlabel('Observed-atom lDDT difference\nfrom ordinary sampling')
        ax.set_title(title,fontsize=11)
    fig.suptitle('No resolved search advantage on the 24-target panel',fontsize=14)
    fig.text(.02,.025,'Same model-compute allowance per target/seed; 200 diffusion steps. 95% paired target-bootstrap intervals.\n'
             'Fixed atom/chain geometry; ordinary sampling batches 4 structures.\n'
             'This unmasking variant generates coordinates at every revelation step.',fontsize=8)
    fig.tight_layout(rect=(0,.18,1,.93))
    fig.savefig(ROOT/'plots/contact-search.png');plt.close(fig)


if __name__ == '__main__':
    conditioning()
    if (ROOT/'data/search-v1/search_paired_differences.csv').exists():search()
