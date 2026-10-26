"""
Three-solution comparison: noVel (noTAO) vs TSOnly (TAO T/S) vs Vel (TAO T/S/U/V).

Same prior, same {theta,salt,uwind,swdown} fixed-increment factorial, three different
data constraints. Shows (A) how much each full solution improves on the prior per
target, and (B) how the control attribution (signed projection share of d_full) shifts
as you add more obs. Reads the per-family anova CSVs + the shared metrics.csv.

Env: ~/miniforge3/envs/tpose/bin/python
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tiw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f'{HERE}/figs/comparison'
CACHE = tiw.CACHE
# family -> (anchor run in metrics.csv, label, color)
FAM = {'novel':  ('noVel',  'noVel (noTAO)',   'tab:orange'),
       'tsonly': ('TSOnly', 'TSOnly (TAO T/S)', 'tab:green'),
       'vel':    ('Vel',    'Vel (TAO T/S/U/V)', 'tab:blue')}
PRODS = [('argo_T', 'Argo T'), ('argo_S', 'Argo S'), ('tao_U', 'TAO U'),
         ('tao_V', 'TAO V'), ('sst', 'SST'), ('ssh', 'SSH')]
MAINS = ['θ', 'S', 'u wind', 'sw']


def _budget(family, projection=False):
    f = 'factorial_budget_projection.csv' if projection else 'factorial_budget.csv'
    return pd.read_csv(f'{HERE}/figs/{family}/anova/{f}')


def skill_table():
    """% RMSE reduction vs prior for each full solution, per target (negative = worse)."""
    m = pd.read_csv(f'{CACHE}/metrics.csv')
    r = m[m.metric == 'rmse'].pivot(index='product', columns='run', values='value')
    rows = {}
    for prod, _ in PRODS:
        p = r.loc[prod, 'prior']
        rows[prod] = {lbl: 100 * (p - r.loc[prod, anc]) / p
                      for anc, lbl, _ in FAM.values()}
    return pd.DataFrame(rows).T  # index=product, cols=family label


def fig_skill(df):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(PRODS)); w = 0.26
    for i, (_, lbl, col) in enumerate(FAM.values()):
        ax.bar(x + (i - 1) * w, [df.loc[p, lbl] for p, _ in PRODS], w, label=lbl, color=col)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in PRODS])
    ax.set_ylabel('% RMSE reduction vs prior')
    ax.set_title('Full-solution skill: how much each data constraint improves on the prior\n'
                 '(TSOnly sits between noVel and Vel; only the velocity constraint fixes TAO U)')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig(f'{OUT}/three_solutions_skill.png', dpi=140); plt.close(fig)


def fig_attribution():
    """Signed main-effect projection share of each family's own d_full, per target."""
    proj = {fam: _budget(fam, projection=True).set_index('term') for fam in FAM}
    fig, axs = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    x = np.arange(len(MAINS)); w = 0.26
    for ax, (prod, plabel) in zip(axs.ravel(), PRODS):
        for i, (fam, (_, lbl, col)) in enumerate(FAM.items()):
            vals = [proj[fam].loc[c, plabel] for c in MAINS]
            ax.bar(x + (i - 1) * w, vals, w, label=lbl, color=col)
        res = {fam: proj[fam].loc['residual r', plabel] for fam in FAM}
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(MAINS)
        ax.set_title(f'{plabel}   (residual: ' +
                     ', '.join(f'{FAM[f][1].split()[0]} {res[f]:.0f}%' for f in FAM) + ')',
                     fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    axs[0, 0].set_ylabel('% of d_full (signed projection)')
    axs[1, 0].set_ylabel('% of d_full (signed projection)')
    axs[0, 0].legend(fontsize=9)
    fig.suptitle('Main-effect attribution across data constraints: signed projection share '
                 'of each family’s own increment d_full = m(solution) − m(prior)\n'
                 '(how the role of θ/S/u-wind/sw shifts as TAO T/S then velocity obs are added)')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f'{OUT}/three_solutions_attribution.png', dpi=140); plt.close(fig)


def combined_csv(skill):
    """One tidy CSV: full-solution RMSE + skill% + main-effect projection shares + residual."""
    m = pd.read_csv(f'{CACHE}/metrics.csv')
    r = m[m.metric == 'rmse'].pivot(index='product', columns='run', values='value')
    proj = {fam: _budget(fam, projection=True).set_index('term') for fam in FAM}
    rows = []
    for prod, plabel in PRODS:
        for fam, (anc, lbl, _) in FAM.items():
            d = dict(target=plabel, family=lbl, rmse=r.loc[prod, anc],
                     rmse_prior=r.loc[prod, 'prior'], skill_pct=skill.loc[prod, lbl],
                     residual_pct=proj[fam].loc['residual r', plabel])
            d.update({c: proj[fam].loc[c, plabel] for c in MAINS})
            rows.append(d)
    pd.DataFrame(rows).to_csv(f'{OUT}/three_solutions_comparison.csv', index=False)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    skill = skill_table()
    fig_skill(skill)
    fig_attribution()
    combined_csv(skill)
    print('wrote three_solutions_{skill,attribution}.png + three_solutions_comparison.csv to', OUT)
    print(skill.round(1))
