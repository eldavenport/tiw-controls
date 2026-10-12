"""
Pairwise interaction (fixed-increment ANOVA) analysis for a combination run.

For a pair run m_ij (e.g. theta x uwind = run_initialT_uwind) and its two solo
runs m_i, m_j, in model state space:
    d_i   = m_i  - m_prior            (main effect of control i)
    d_j   = m_j  - m_prior            (main effect of control j)
    d_ij  = m_ij - m_prior            (full combined increment)
    a_ij  = m_ij - m_i - m_j + m_prior = d_ij - (d_i + d_j)   (interaction)
a_ij is the part of the combined solution that is NOT the sum of the two
single-control effects -- the nonlinear coupling between the controls.

Figures (figs/anova/<pair>_*.png):
  _euc : time-mean equatorial zonal-velocity section (lon x depth) decomposition
  _tiv : Hovmoller of near-surface meridional velocity, 2-5N (lon x time) -- TIWs
  _budget : obs-space RMSE per target + relative interaction magnitude

Env: ~/miniforge3/envs/tpose/bin/python
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tiw

HERE = os.path.dirname(__file__)
CACHE = tiw.CACHE
FIGS = f'{HERE}/figs/anova'
EUC_LON = (130, 282)        # EUC spans the basin
TIW_LON = (200, 282)        # TIWs live in the eastern cold tongue (~160W-80W)
TIW_LAT = (2.0, 5.0)        # off-equatorial band where TIW meridional velocity peaks
TIW_ZTOP = 15               # top ~50 m (see tiw grid: 15 levels above -50 m)


def _mask(ds, v):
    return ds[v].where(tiw.wet_mask(ds, ds[v]))


def load_pair_fields(runs):
    """One streaming pass per run: time-mean EUC U section + TIW V Hovmoller."""
    euc, tiv = {}, {}
    for run in runs:
        ds = tiw.load_run(run)
        u = _mask(ds, 'UVEL').sel(YC=slice(-0.5, 0.5)).mean('YC').mean('time')  # (Z,XG)
        v = (_mask(ds, 'VVEL').sel(YG=slice(*TIW_LAT)).isel(Z=slice(0, TIW_ZTOP))
             .mean('YG').mean('Z'))                                            # (time,XC)
        out = xr.Dataset({'euc': u, 'tiv': v}).compute()
        euc[run], tiv[run] = out['euc'], out['tiv']
        print(f'  loaded {run}', flush=True)
    return euc, tiv


def _anova(f, i, j, ij, prior):
    """Return main effects, full increment and interaction for field dict f."""
    di = f[i] - f[prior]; dj = f[j] - f[prior]; dij = f[ij] - f[prior]
    inter = f[ij] - f[i] - f[j] + f[prior]
    return di, dj, dij, inter


def _rms(a):
    return float(np.sqrt(np.nanmean(np.asarray(a) ** 2)))


def fig_euc(pair, i, j, euc):
    prior = 'prior'; ij = pair
    di, dj, dij, inter = _anova(euc, i, j, ij, prior)
    X, Z = euc[prior].XG.values, euc[prior].Z.values
    xl = (X >= EUC_LON[0]) & (X <= EUC_LON[1])
    xi, abs_fields = X[xl], [euc[prior], euc[i], euc[j], euc[ij]]
    abs_lab = ['prior', tiw.CONTROL_XX[i], tiw.CONTROL_XX[j], f'{tiw.CONTROL_XX[i]}+{tiw.CONTROL_XX[j]}']
    inc_fields = [di, dj, dij, inter]
    inc_lab = [f'd[{tiw.CONTROL_XX[i]}]', f'd[{tiw.CONTROL_XX[j]}]',
               'd[combined]', 'interaction a_ij']

    fig, axs = plt.subplots(2, 4, figsize=(19, 7), sharex=True, sharey=True)
    av = np.nanpercentile(np.abs(euc[ij].values[:, xl]), 99)
    for ax, F, lab in zip(axs[0], abs_fields, abs_lab):
        pc = ax.pcolormesh(xi, Z, F.values[:, xl], cmap='RdBu_r', vmin=-av, vmax=av)
        ax.set_title(f'U: {lab}'); fig.colorbar(pc, ax=ax, shrink=.8)
    iv = np.nanpercentile(np.abs(dij.values[:, xl]), 99)            # mains+combined share scale
    av2 = np.nanpercentile(np.abs(inter.values[:, xl]), 99)         # interaction its own
    for k, (ax, F, lab) in enumerate(zip(axs[1], inc_fields, inc_lab)):
        lim = av2 if k == 3 else iv
        pc = ax.pcolormesh(xi, Z, F.values[:, xl], cmap='RdBu_r', vmin=-lim, vmax=lim)
        ax.set_title(f'{lab}  (RMS={_rms(F.values[:, xl]):.3f} m/s)'); fig.colorbar(pc, ax=ax, shrink=.8)
    for ax in axs[:, 0]:
        ax.set_ylabel('depth (m)')
    for ax in axs[1]:
        ax.set_xlabel('lon (deg E)')
    axs[0, 0].set_ylim(-300, 0)
    fig.suptitle(f'EUC (time-mean equatorial U): {tiw.CONTROL_XX[i]} x {tiw.CONTROL_XX[j]} '
                 f'fixed-increment decomposition\nbottom-right = nonlinear interaction '
                 f'(combined minus sum of singles)')
    fig.tight_layout(); fig.savefig(f'{FIGS}/{pair}_euc.png', dpi=140); plt.close(fig)


def fig_tiw(pair, i, j, tiv):
    prior = 'prior'; ij = pair
    di, dj, dij, inter = _anova(tiv, i, j, ij, prior)
    X = tiv[prior].XC.values; t = np.arange(tiv[prior].sizes['time'])
    xl = (X >= TIW_LON[0]) & (X <= TIW_LON[1]); xi = X[xl]
    panels = [(tiv[ij], f'V: {tiw.CONTROL_XX[i]}+{tiw.CONTROL_XX[j]} (combined)'),
              (di, f'main d[{tiw.CONTROL_XX[i]}]'),
              (dj, f'main d[{tiw.CONTROL_XX[j]}]'),
              (di + dj, 'sum of mains'),
              (inter, 'interaction a_ij')]
    fig, axs = plt.subplots(1, 5, figsize=(20, 6), sharey=True)
    av = np.nanpercentile(np.abs(tiv[ij].values[:, xl]), 98)
    for k, (ax, (F, lab)) in enumerate(zip(axs, panels)):
        A = F.values[:, xl] if F.ndim == 2 else F.values[xl]
        lim = np.nanpercentile(np.abs(inter.values[:, xl]), 98) if k == 4 else av
        pc = ax.pcolormesh(xi, t, A, cmap='RdBu_r', vmin=-lim, vmax=lim, shading='auto')
        ax.set_title(f'{lab}\n(RMS={_rms(A):.4f})'); ax.set_xlabel('lon (deg E)')
        fig.colorbar(pc, ax=ax, shrink=.8)
    axs[0].set_ylabel('day of run')
    fig.suptitle(f'TIW meridional velocity (2-5N, top 50 m), lon-time: '
                 f'{tiw.CONTROL_XX[i]} x {tiw.CONTROL_XX[j]} decomposition  '
                 f'(interaction = how the pair reshapes TIW phase/amplitude)')
    fig.tight_layout(); fig.savefig(f'{FIGS}/{pair}_tiw.png', dpi=140); plt.close(fig)


# ---- obs-space budget per target -------------------------------------------
PRODS = [('argo_T', 'Argo T'), ('argo_S', 'Argo S'), ('tao_U', 'TAO U'),
         ('tao_V', 'TAO V'), ('sst', 'SST'), ('ssh', 'SSH')]


def _load_obsspace(prodkey, run):
    if prodkey in ('sst', 'ssh'):
        return xr.open_dataarray(f'{CACHE}/grid_{run}_{prodkey}.nc').values
    p, var = prodkey.split('_')
    return np.load(f'{CACHE}/model_{run}_{p}_{var}.npy')


def budget(pair, i, j):
    prior, ij = 'prior', pair
    r = pd.read_csv(f'{CACHE}/metrics.csv')
    rp = r[r.metric == 'rmse'].pivot(index='product', columns='run', values='value')
    rows = []
    for key, lab in PRODS:
        if key not in rp.index:
            continue
        mp, mi, mj, mij = (_load_obsspace(key, run) for run in (prior, i, j, ij))
        valid = np.all([np.isfinite(x) for x in (mp, mi, mj, mij)], axis=0)
        dij = (mij - mp)[valid]
        inter = (mij - mi - mj + mp)[valid]
        rows.append(dict(
            target=lab,
            rmse_prior=rp.loc[key, prior], rmse_i=rp.loc[key, i], rmse_j=rp.loc[key, j],
            rmse_combined=rp.loc[key, ij], rmse_Vel=rp.loc[key, 'Vel'],
            interaction_frac_pct=100 * _rms(inter) / _rms(dij) if _rms(dij) else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(f'{FIGS}/{pair}_budget.csv', index=False)

    fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
    x = np.arange(len(df)); w = 0.16
    for k, (col, c, l) in enumerate([
            ('rmse_prior', '0.5', 'prior'), ('rmse_i', 'tab:green', tiw.CONTROL_XX[i]),
            ('rmse_j', 'tab:purple', tiw.CONTROL_XX[j]), ('rmse_combined', 'tab:red', 'combined'),
            ('rmse_Vel', 'tab:blue', 'Vel (full)')]):
        axs[0].bar(x + (k - 2) * w, df[col], w, color=c, label=l)
    axs[0].set_xticks(x); axs[0].set_xticklabels(df['target']); axs[0].set_ylabel('RMSE')
    axs[0].set_title('(a) misfit: singles vs combined vs full'); axs[0].legend(fontsize=8)
    axs[1].bar(x, df['interaction_frac_pct'], 0.6, color='darkorange')
    axs[1].set_xticks(x); axs[1].set_xticklabels(df['target'])
    axs[1].set_ylabel('%')
    axs[1].set_title(r'(b) interaction size:  $\|a_{ij}\| / \|d_{combined}\|$')
    fig.suptitle(f'{tiw.CONTROL_XX[i]} x {tiw.CONTROL_XX[j]} obs-space budget per target')
    fig.tight_layout(); fig.savefig(f'{FIGS}/{pair}_budget.png', dpi=140); plt.close(fig)
    return df


if __name__ == '__main__':
    os.makedirs(FIGS, exist_ok=True)
    import sys
    pairs = sys.argv[1:] or list(tiw.COMBOS)
    for pair in pairs:
        i, j = tiw.COMBOS[pair]
        euc, tiv = load_pair_fields([pair, i, j, 'prior'])
        fig_euc(pair, i, j, euc)
        fig_tiw(pair, i, j, tiv)
        print(f'[{pair}] EUC + TIW figs done')
        try:
            df = budget(pair, i, j)
            print(df.to_string(index=False))
        except Exception as e:
            print(f'[{pair}] budget skipped (metrics not ready?): {type(e).__name__}: {e}')
    print(f'done -> {FIGS}')
