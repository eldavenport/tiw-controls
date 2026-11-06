"""
Pooled depth-error profiles across ALL FOUR windows (sep2012, nov2012, jan2013,
mar2013), restricted to |lat|<=17. prior / noVel / Vel (no TSOnly — it only
exists for sep2012). RMSE vs depth for Argo T/S and TAO U/V, combining every
window's obs into a single average. Caches per-depth SSE/N so re-plot is cheap.

Env: ~/miniforge3/envs/tpose/bin/python.  Run after the misfit recompute.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tiw
import misfit_cost as mc

LAT = mc.LAT_LIMIT
SOLS = [('prior', '#7a7a7a', 'prior (no DA)'),
        ('noVel', '#D55E00', 'noVel (noTAO)'),
        ('Vel',   '#0072B2', 'Vel (TAO T/S/U/V)')]
# product -> obs file per year + variables + axis label/limits
PROD = [
    ('argo', 'T', 'Argo θ', '°C', -500, lambda y: f'{mc._C}/USGO_TP_{y}_PFL_D_w.nc'),
    ('argo', 'S', 'Argo S', 'psu', -500, lambda y: f'{mc._C}/USGO_TP_{y}_PFL_D_w.nc'),
    ('tao',  'U', 'TAO U', 'm/s', -315, lambda y: f'{mc._TAO}/TAO_WO_{y}_ADCP{"_v2" if y == 2012 else ""}.nc'),
    ('tao',  'V', 'TAO V', 'm/s', -315, lambda y: f'{mc._TAO}/TAO_WO_{y}_ADCP{"_v2" if y == 2012 else ""}.nc'),
]
CACHE = f'{tiw.CACHE}/profiles_allwin.npz'


def compute():
    # accumulate SSE[depth] and N[depth] per (solution, product, var), pooled over
    # all windows/years; depth grid is constant per product (Argo 97, ADCP 62).
    acc, depth = {}, {}
    for win in mc.WINDOWS:
        state = mc._load(win, 'diag_state')          # prior/noVel/Vel (skips unreadable)
        for sol, _, _ in SOLS:
            if sol not in state:
                continue
            ds, prep = state[sol], {}
            for prod, v, *_rest, pathf in PROD:
                for y in mc.WINDOWS[win]['years']:
                    p = pathf(y)
                    if not os.path.exists(p):
                        continue
                    obs = tiw.open_obs(p)
                    depth[prod] = obs['depth'].values
                    if v not in prep:
                        prep[v] = mc._prep_field(ds, v)
                    man = mc._sample(prep[v], obs)
                    o = obs[f'prof_{v}'].values
                    w = obs[f'prof_{v}weight'].values
                    latok = (np.abs(obs['lat'].values) <= LAT)[:, None]
                    valid = (o != -9999.) & (w > 0) & np.isfinite(man) & latok
                    sse = np.where(valid, (o - man) ** 2, 0.0).sum(0)
                    nn = valid.sum(0)
                    k = (sol, prod, v)
                    if k not in acc:
                        acc[k] = [np.zeros_like(sse, float), np.zeros_like(nn)]
                    acc[k][0] += sse
                    acc[k][1] += nn
            del prep
            print(f'  sampled {win}/{sol}', flush=True)
    out = {'depth_argo': depth['argo'], 'depth_tao': depth['tao']}
    for (sol, prod, v), (sse, nn) in acc.items():
        out[f'{sol}_{prod}_{v}_sse'] = sse
        out[f'{sol}_{prod}_{v}_n'] = nn
    np.savez(CACHE, **out)
    print('wrote', CACHE, flush=True)


def plot():
    d = np.load(CACHE)
    fig, axs = plt.subplots(1, 4, figsize=(15, 6.5))
    for ax, (prod, v, label, unit, zlim, _) in zip(axs, PROD):
        depth = d[f'depth_{prod}']
        for sol, c, rlab in SOLS:
            sse, nn = d[f'{sol}_{prod}_{v}_sse'], d[f'{sol}_{prod}_{v}_n']
            rmse = np.sqrt(np.where(nn > 0, sse / np.maximum(nn, 1), np.nan))
            ax.plot(rmse, depth, color=c, lw=2.4 if sol == 'Vel' else 1.9,
                    ls='--' if sol == 'prior' else '-', label=rlab,
                    marker='o' if sol == 'prior' else None, ms=3, markevery=4)
        ntot = int(d[f'Vel_{prod}_{v}_n'].max())
        ax.set_title(f'{label}  (n≈{ntot})', fontsize=13, fontweight='bold')
        ax.set_xlabel(f'RMSE ({unit})'); ax.set_ylim(zlim, 0); ax.set_xlim(left=0)
        ax.grid(alpha=0.25, lw=0.6); ax.spines[['top', 'right']].set_visible(False)
    axs[0].set_ylabel('depth (m)', fontsize=12)
    axs[0].legend(fontsize=10, loc='lower right', framealpha=0.9)
    fig.suptitle('Depth error profiles pooled over all 4 windows, |lat|≤17°  '
                 '(prior → noVel → Vel; TAO velocity is the discriminator)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = 'figs/comparison/error_profiles_depth_17_allwindows.png'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print('wrote', out, flush=True)


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['plot']:
        plot()
    else:
        compute(); plot()
