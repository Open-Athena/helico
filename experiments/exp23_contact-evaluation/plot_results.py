"""Render the committed validation measurements without rerunning inference."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / 'plots/corrected'
PLOTS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 180})


def conditioning():
    data = pd.read_csv(ROOT/'data/conditioning-v2/conditioning_summary.csv')
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
    fig.suptitle('Structural response to contact conditioning', fontsize=14, x=.51)
    fig.text(.02, .015, 'Oracle validation diagnostic; 8 targets/group, 2 seeds; targets without eligible corrections omitted. 95% target-bootstrap intervals.\n'
             'First two panels request up to four corrections. *Ligand group includes ions/glycans. lDDT uses fixed atom/chain correspondence.', fontsize=9)
    fig.tight_layout(rect=(0,.11,1,.94))
    fig.savefig(PLOTS/'contact-conditioning.png');fig.savefig(PLOTS/'contact-conditioning.svg');plt.close(fig)


def search():
    data = pd.read_csv(ROOT/'data/search-v2/search_paired_differences.csv')
    data = data[data.horizon=='unmask']
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
    for ax, metric, title in zip(axes, ['selected_observed_lddt','oracle_observed_lddt'],
                                ['Confidence-selected structure', 'Best generated structure (oracle)']):
        rows = data[data.metric==metric].set_index('contrast').loc[['tree-minus-ordinary','unmask-minus-ordinary']]
        ax.errorbar(rows['mean'], [1,0], xerr=np.stack([rows['mean']-rows.low,rows.high-rows['mean']]),
                    fmt='o', color='#1e5c93', capsize=5)
        ax.axvline(0,color='.5',linewidth=.8)
        ax.set_yticks([1,0],['Contact tree','Progressive unmasking'])
        ax.set_ylim(-.5,1.5)
        ax.set_xlabel('Observed-atom lDDT difference\nfrom ordinary sampling')
        ax.set_title(title,fontsize=11)
    fig.suptitle('Matched-time search on the 24-target panel',fontsize=14)
    fig.text(.02,.025,'Same model-compute allowance per target/seed; 200 diffusion steps. 95% paired target-bootstrap intervals.\n'
             'Fixed atom/chain geometry; ordinary sampling batches 4 structures.\n'
             'This unmasking variant generates coordinates at every revelation step.',fontsize=8)
    fig.tight_layout(rect=(0,.18,1,.93))
    fig.savefig(PLOTS/'contact-search.png');fig.savefig(PLOTS/'contact-search.svg');plt.close(fig)


def full_validation():
    root=ROOT/'data/validation-structures-v3'
    paired=pd.read_csv(root/'structure_paired_differences.csv').set_index('metric')
    calibration=pd.read_csv(root/'discrepancy_calibration.csv')
    calibration=calibration[calibration.weights=='ema'].set_index('group').loc[['intra','pp','pl']]
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),gridspec_kw={'width_ratios':[1,1.3]})
    rows=paired.loc[['observed_lddt','interface_lddt']]
    axes[0].errorbar(rows['mean'],[1,0],xerr=np.stack([rows['mean']-rows.low,rows.high-rows['mean']]),
                    fmt='o',color='#1e5c93',capsize=5)
    axes[0].axvline(0,color='.5',linewidth=.8)
    axes[0].set_yticks([1,0],['Full structure\n384 complexes','Interface\n331 evaluable'])
    axes[0].set_ylim(-.5,1.5)
    axes[0].set_xlabel('EMA minus initialization lDDT')
    axes[0].set_title('Structure quality after contact training',fontsize=11)
    for offset,column,label,color in [(-.18,'mean_predicted_probability','Predicted probability','#1e5c93'),
                                      (.18,'native_contact_fraction','Observed reference frequency','#b07732')]:
        axes[1].bar(np.arange(3)+offset,100*calibration[column],.34,label=label,color=color)
    axes[1].set_xticks(range(3),['Within protein','Protein–protein','Protein–ligand*'])
    axes[1].set_ylabel('Contact probability / frequency (%)')
    axes[1].set_ylim(0,100);axes[1].legend(frameon=False,fontsize=9)
    axes[1].set_title('Disagreement proposal calibration\nagainst the observed reference',fontsize=11)
    fig.suptitle('Full 384-complex validation',fontsize=14)
    fig.text(.02,.025,'Left: confidence-selected best of 2 structures; 95% paired assembly-bootstrap intervals; fixed atom/chain correspondence.\n'
             'Right: top eligible missing-contact proposal per group/sample, generated distance ≥7 Å; unobserved labels excluded after selection.\n'
             '*Includes ions/glycans. One reference structure cannot exclude a physically valid alternative contact.',fontsize=8)
    fig.tight_layout(rect=(0,.17,1,.92))
    fig.savefig(PLOTS/'full-validation.png');fig.savefig(PLOTS/'full-validation.svg');plt.close(fig)


def contact_hypotheses():
    data=pd.read_csv(ROOT/'data/contact-hypotheses-v2/search_paired_differences.csv')
    fig,axes=plt.subplots(1,3,figsize=(12,4.2))
    specs=[('selected_observed_lddt','Confidence-selected full structure'),
           ('selected_interface_lddt','Confidence-selected interface'),
           ('oracle_observed_lddt','Best generated structure (oracle)')]
    for ax,(metric,title) in zip(axes,specs):
        rows=pd.concat([data[(data.horizon==method)&(data.contrast==method+'-minus-ordinary')&
                             (data.metric==metric)] for method in ['contact7','contact32']])
        ax.errorbar(rows['mean'],[1,0],xerr=np.stack([rows['mean']-rows.low,rows.high-rows['mean']]),
                    fmt='o',color='#1e5c93',capsize=5)
        ax.axvline(0,color='.5',linewidth=.8);ax.set_ylim(-.5,1.5)
        ax.set_yticks([1,0],['7 contact steps','32 contact steps'])
        ax.set_title(title,fontsize=10);ax.set_xlabel('lDDT difference from\nordinary sampling')
    fig.suptitle('Contact-only diffusion at matched sampling time',fontsize=14)
    fig.text(.02,.025,'Four independent contact hypotheses, one 200-step coordinate sample each. Ordinary sampling uses each method’s own model-time budget.\n'
             '24 validation targets (20 with evaluable interfaces), 2 seeds; 95% paired target-bootstrap intervals. Fixed atom/chain correspondence.\n'
             'Candidate counts and timing for each target are retained in the accompanying CSV tables.',fontsize=8)
    fig.tight_layout(rect=(0,.18,1,.92))
    fig.savefig(PLOTS/'contact-hypotheses.png');fig.savefig(PLOTS/'contact-hypotheses.svg');plt.close(fig)


if __name__ == '__main__':
    if (ROOT/'data/conditioning-v2/conditioning_summary.csv').exists():conditioning()
    if (ROOT/'data/search-v2/search_paired_differences.csv').exists():search()
    if (ROOT/'data/validation-structures-v3/structure_paired_differences.csv').exists():full_validation()
    if (ROOT/'data/contact-hypotheses-v2/search_paired_differences.csv').exists():contact_hypotheses()
