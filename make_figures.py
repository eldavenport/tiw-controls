"""
Figures for the withheld-controls experiments.

Per-experiment figures (one set per single-control run) go to figs/<control>/;
cross-experiment comparisons go to figs/intercomparison/.
Reads tiw.CACHE (from compute_metrics.py) + metrics.csv.

Run:  ~/miniforge3/envs/tpose/bin/python make_figures.py
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
CACHE = tiw.CACHE                 # intermediate data on /data
FIGS = f'{HERE}/figs'            # figures stay in the repo (deliverables)

# fixed context runs (same in every experiment)
FIX_LABEL = {'prior': 'prior (no DA)', 'noVel': 'TPOSE6-noVel', 'Vel': 'TPOSE6-Vel (full)'}
FIX_COLOR = {'prior': '0.5', 'noVel': 'tab:orange', 'Vel': 'tab:blue'}
CTRL_COLOR = {'initialT': 'tab:green', 'initialS': 'tab:red', 'zonalWind': 'tab:purple',
              'meridionalWind': 'tab:brown', 'swdown': 'tab:olive',
              'atmosT': 'tab:cyan', 'atmosQ': 'tab:pink', 'lwdown': 'gold',
              'precip': 'tab:blue', 'uvel': 'darkgray', 'vvel': 'slategray'}
# all single-control runs, kept + null, ordered for the summary ranking figure.
ALL_CONTROLS = tiw.CONTROLS + tiw.CONTROLS_NULL

PRODS = [('argo_T', 'Argo T', '°C'), ('argo_S', 'Argo S', 'psu'),
         ('tao_U', 'TAO U', 'm/s'), ('tao_V', 'TAO V', 'm/s'),
         ('sst', 'SST', '°C'), ('ssh', 'SSH', 'm')]
# 165E dropped: TAO ADCP there has no overlap with the model window.
MOORINGS = {190.0: '170°W', 220.0: '140°W', 250.0: '110°W'}
SEP1 = 244

plt.rcParams.update({'font.size': 11})


def order_of(control):
    return ['prior', 'noVel', control, 'Vel']


def label_of(run, control):
    return FIX_LABEL.get(run) or f'TPOSE6-Vel ({tiw.CONTROL_XX[control]})'


def color_of(run, control):
    return FIX_COLOR.get(run) or CTRL_COLOR[control]


def _metrics():
    return pd.read_csv(f'{CACHE}/metrics.csv')


def _rmse_pivot(df):
    return df[df.metric == 'rmse'].pivot(index='product', columns='run', values='value')


def _baseline(r, prod):
    """prior (iter0) where available, else noVel."""
    return 'prior' if (prod in r.index and np.isfinite(r.loc[prod, 'prior'])) else 'noVel'


# ---- Fig 1: headline (per experiment) --------------------------------------
def fig_headline(control, outdir):
    r = _rmse_pivot(_metrics())
    prods = [p for p in PRODS if p[0] in r.index]
    xx = tiw.CONTROL_XX[control]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

    # (a) % RMSE improvement of this control vs prior (iter0) and noVel
    x = np.arange(len(prods)); w = 0.38
    for j, (base, hatch, lbl) in enumerate([('prior', None, 'vs prior (iter0)'),
                                            ('noVel', '//', 'vs TPOSE6-noVel')]):
        vals = [100 * (r.loc[k, base] - r.loc[k, control]) / r.loc[k, base]
                if base in r.columns else np.nan for k, _, _ in prods]
        bars = ax[0].bar(x + (j - 0.5) * w, [v if np.isfinite(v) else 0 for v in vals],
                         w, label=lbl, color=color_of(control, control), hatch=hatch,
                         edgecolor='k', linewidth=0.4)
        for bar, v in zip(bars, vals):
            ax[0].text(bar.get_x() + bar.get_width() / 2, (v if np.isfinite(v) else 0),
                       'n/a' if not np.isfinite(v) else f'{v:.0f}%', ha='center',
                       va='bottom' if (np.isfinite(v) and v >= 0) else 'top', fontsize=8)
    ax[0].set_xticks(x); ax[0].set_xticklabels([p[1] for p in prods])
    ax[0].axhline(0, color='k', lw=.8)
    ax[0].set_ylabel(f'% RMSE improvement of TPOSE6-Vel ({xx})')
    ax[0].set_title(f'(a) {xx}-only RMSE improvement vs prior (iter0) & noVel')
    ax[0].legend(fontsize=8, loc='upper right')

    # (b) RMSE relative to prior (iter0)
    x = np.arange(len(prods)); w = 0.2
    runs = order_of(control)
    for i, run in enumerate(runs):
        vals = [r.loc[k, run] / r.loc[k, _baseline(r, k)] if run in r.columns else np.nan
                for k, _, _ in prods]
        ax[1].bar(x + (i - 1.5) * w, vals, w, label=label_of(run, control), color=color_of(run, control))
    ax[1].set_xticks(x); ax[1].set_xticklabels([p[1] for p in prods])
    ax[1].axhline(1, color='k', lw=.8, ls=':'); ax[1].set_ylim(0, 1.3)
    ax[1].set_ylabel('RMSE / prior (iter0) RMSE')
    ax[1].set_title('(b) Relative misfit by run')
    ax[1].legend(fontsize=7, loc='upper center', ncol=4)
    fig.tight_layout(); fig.savefig(f'{outdir}/fig1_headline.png', dpi=140); plt.close(fig)


# ---- Fig 2: depth-resolved RMSE (per experiment) ---------------------------
def fig_depth(control, outdir):
    fig, axs = plt.subplots(1, 4, figsize=(15, 5), sharey=True)
    for ax, (prod, var) in zip(axs, [('argo', 'T'), ('argo', 'S'), ('tao', 'U'), ('tao', 'V')]):
        obs = np.load(f'{CACHE}/obs_{prod}.npz')
        depth = obs['depth']; o = obs[f'obs_{var}']; w = obs[f'w_{var}']
        for run in order_of(control):
            fp = f'{CACHE}/model_{run}_{prod}_{var}.npy'
            if os.path.exists(fp):
                ax.plot(tiw.rmse_profile(o, np.load(fp), w), depth,
                        color=color_of(run, control), label=label_of(run, control))
        ax.set_title(f'{prod.upper()} {var}'); ax.set_xlabel('RMSE'); ax.grid(alpha=.3)
        ax.legend(fontsize=7, loc='lower right')
    axs[0].set_ylabel('depth (m)'); axs[0].set_ylim(-500, 0)
    fig.suptitle('Depth-resolved RMSE vs observations')
    fig.tight_layout(); fig.savefig(f'{outdir}/fig2_depth_rmse.png', dpi=140); plt.close(fig)


# ---- Fig 3: time evolution of misfit (per experiment) ----------------------
def _daily_rmse_profile(o, m, w, day):
    out = np.full(tiw.NT, np.nan)
    e2 = np.where((o != -9999.) & (w > 0) & np.isfinite(m), (o - m) ** 2, np.nan)
    for d in range(tiw.NT):
        sel = day == d
        if sel.any() and np.isfinite(e2[sel]).any():
            out[d] = np.sqrt(np.nanmean(e2[sel]))
    return out


def fig_time(control, outdir):
    fig, axs = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    panels = [('argo', 'T'), ('argo', 'S'), ('tao', 'U'), ('tao', 'V'), ('sst', None), ('ssh', None)]
    for ax, (prod, var) in zip(axs.ravel(), panels):
        if var is not None:
            obs = np.load(f'{CACHE}/obs_{prod}.npz')
            o = obs[f'obs_{var}']; w = obs[f'w_{var}']; day = obs['day']
            for run in order_of(control):
                fp = f'{CACHE}/model_{run}_{prod}_{var}.npy'
                if os.path.exists(fp):
                    ax.plot(tiw.DAYS, _daily_rmse_profile(o, np.load(fp), w, day),
                            color=color_of(run, control), label=label_of(run, control))
            ax.set_title(f'{prod.upper()} {var}')
        else:
            obs_grid = _gridded_obs(prod)
            for run in order_of(control):
                fp = f'{CACHE}/grid_{run}_{prod}.nc'
                if os.path.exists(fp):
                    d = xr.open_dataarray(fp).values - obs_grid
                    ax.plot(tiw.DAYS, np.sqrt(np.nanmean(d.reshape(tiw.NT, -1) ** 2, axis=1)),
                            color=color_of(run, control), label=label_of(run, control))
            ax.set_title(prod.upper())
        ax.grid(alpha=.3)
        if ax.get_legend_handles_labels()[0]:
            y0, y1 = ax.get_ylim()
            ax.set_ylim(y0, y1 + 0.28 * (y1 - y0))
            ax.legend(fontsize=7, loc='upper left')
    for ax in axs[-1]:
        ax.tick_params(axis='x', rotation=30)
    fig.suptitle('Misfit evolution over the assimilation window')
    fig.tight_layout(); fig.savefig(f'{outdir}/fig3_time.png', dpi=140); plt.close(fig)


# ---- Fig 4: EUC at equatorial TAO moorings (per experiment) -----------------
def _euc(uprof, depth):
    u = np.where(depth > -300, uprof, np.nan)
    if not np.isfinite(u).any():
        return np.nan, np.nan
    i = np.nanargmax(u)
    return float(depth[i]), float(u[i])


def fig_euc(control, outdir):
    obs = np.load(f'{CACHE}/obs_tao.npz')
    depth = obs['depth']; lon = obs['lon']; lat = obs['lat']
    o = np.where(obs['obs_U'] == -9999., np.nan, obs['obs_U'])
    models = {run: np.load(f'{CACHE}/model_{run}_tao_U.npy')
              for run in order_of(control) if os.path.exists(f'{CACHE}/model_{run}_tao_U.npy')}
    fig, axs = plt.subplots(1, len(MOORINGS), figsize=(12, 5), sharey=True)
    stats = []
    for ax, (mlon, name) in zip(axs, MOORINGS.items()):
        sel = (np.abs(lat) < 0.5) & (np.abs(lon - mlon) < 0.5)
        if sel.sum() == 0:
            ax.set_title(f'{name} (none)'); continue
        valid = ~np.isnan(o[sel])                 # only obs+model overlap samples
        for m in models.values():
            valid &= ~np.isnan(m[sel])
        omean = np.nanmean(np.where(valid, o[sel], np.nan), axis=0)
        ax.plot(omean, depth, 'k', lw=2, label='TAO (model-matched)')
        stats.append(('TAO', name, *_euc(omean, depth)))
        for run, m in models.items():
            mmean = np.nanmean(np.where(valid, m[sel], np.nan), axis=0)
            ax.plot(mmean, depth, color=color_of(run, control), label=label_of(run, control))
            stats.append((run, name, *_euc(mmean, depth)))
        ax.set_title(f'EUC at {name}'); ax.set_xlabel('U (m/s)'); ax.grid(alpha=.3)
        ax.axvline(0, color='0.7', lw=.8)
    axs[0].set_ylabel('depth (m)'); axs[0].set_ylim(-300, 0)
    axs[0].legend(fontsize=7, loc='lower right')
    fig.suptitle('Mean zonal velocity (EUC) at equatorial TAO moorings (model-matched)')
    fig.tight_layout(); fig.savefig(f'{outdir}/fig4_euc.png', dpi=140); plt.close(fig)
    pd.DataFrame(stats, columns=['run', 'mooring', 'EUC_core_depth_m', 'EUC_core_speed_ms']
                 ).to_csv(f'{outdir}/euc_stats.csv', index=False)


# ---- Fig 5: SST / SSH misfit-reduction maps (per experiment) ----------------
def _gridded_obs(prod):
    ref = xr.open_dataarray(f'{CACHE}/grid_Vel_{prod}.nc')
    ny, nx = ref.sizes['YC'], ref.sizes['XC']
    if prod == 'sst':
        sst, _ = tiw.load_sst(nx, ny)
        return np.moveaxis(sst[:, :, SEP1:SEP1 + tiw.NT], 2, 0)
    rads = tiw.load_rads(nx, ny)
    arrs = []
    for sat in rads:
        a = np.moveaxis(rads[sat][:, :, SEP1:SEP1 + tiw.NT], 2, 0)
        arrs.append(a - np.nanmean(a, axis=0, keepdims=True))
    return np.nanmean(np.stack(arrs), axis=0)


def _rmse_map(prod, run, obs_grid):
    m = xr.open_dataarray(f'{CACHE}/grid_{run}_{prod}.nc').values
    return np.sqrt(np.nanmean((m - obs_grid) ** 2, axis=0))


def fig_maps(control, outdir):
    ref = xr.open_dataarray(f'{CACHE}/grid_Vel_sst.nc')
    XC, YC = ref.XC.values, ref.YC.values
    for prod in ('sst', 'ssh'):
        base = 'prior'
        if not (os.path.exists(f'{CACHE}/grid_{base}_{prod}.nc')
                and os.path.exists(f'{CACHE}/grid_{control}_{prod}.nc')):
            continue
        obs_grid = _gridded_obs(prod)
        rb = _rmse_map(prod, base, obs_grid)
        fig, axs = plt.subplots(1, 3, figsize=(16, 3.6))
        axs[0].pcolormesh(XC, YC, rb, vmin=0, vmax=np.nanpercentile(rb, 95), cmap='viridis')
        axs[0].set_title(f'{prod.upper()} RMSE: {label_of(base, control)}')
        for ax, run in zip(axs[1:], (control, 'Vel')):
            red = rb - _rmse_map(prod, run, obs_grid)       # +ve = improvement
            lim = np.nanpercentile(np.abs(red), 95)
            pc = ax.pcolormesh(XC, YC, red, vmin=-lim, vmax=lim, cmap='RdBu_r')
            ax.set_title(f'RMSE reduction: {label_of(base, control)}→{label_of(run, control)}')
            fig.colorbar(pc, ax=ax, shrink=.8)
        for ax in axs:
            ax.set_xlabel('lon'); ax.set_ylabel('lat')
        fig.tight_layout(); fig.savefig(f'{outdir}/fig5_{prod}_maps.png', dpi=140); plt.close(fig)


# ---- Fig 6: the applied control / its model fingerprint (per experiment) ----
# IC controls: day-0 increment (run-prior). Forcing controls: cumulative
# model response (run-prior) at the end of the window (the control field itself
# isn't carried in the run dir for forcing).
_FINGERPRINT = {
    'initialT':       ('THETA', 0,  'initial-T increment', '°C'),
    'initialS':       ('SALT',  0,  'initial-S increment', 'psu'),
    'zonalWind':      ('UVEL', -1,  'cumulative U response (day 121)', 'm/s'),
    'meridionalWind': ('VVEL', -1,  'cumulative V response (day 121)', 'm/s'),
    'swdown':         ('THETA', -1, 'cumulative T response (day 121)', '°C'),
    'atmosT':         ('THETA', -1, 'cumulative T response (day 121)', '°C'),
    'atmosQ':         ('THETA', -1, 'cumulative T response (day 121)', '°C'),
    'lwdown':         ('THETA', -1, 'cumulative T response (day 121)', '°C'),
    'precip':         ('SALT', -1,  'cumulative S response (day 121)', 'psu'),
    'uvel':           ('UVEL', 0,   'initial-U increment', 'm/s'),
    'vvel':           ('VVEL', 0,   'initial-V increment', 'm/s'),
}


def fig_control(control, outdir):
    var, t, desc, unit = _FINGERPRINT[control]
    run = tiw.load_run(control); pr = tiw.load_run('prior')
    d = (run[var].isel(time=t) - pr[var].isel(time=t)).where(tiw.wet_mask(run, run[var]))
    xdim = next(c for c in d.dims if c in ('XC', 'XG'))
    ydim = next(c for c in d.dims if c in ('YC', 'YG'))
    X, Y, Z = d[xdim].values, d[ydim].values, d['Z'].values
    arr = d.values
    xx = tiw.CONTROL_XX[control]

    fig, axs = plt.subplots(1, 2, figsize=(14, 4.2))
    rms = np.sqrt(np.nanmean(arr ** 2, axis=0))
    pc = axs[0].pcolormesh(X, Y, rms, vmin=0,
                           vmax=np.nanpercentile(rms[np.isfinite(rms)], 99), cmap='magma')
    axs[0].set_title(f'{xx}: {desc} ({unit}), depth-RMS')
    axs[0].set_xlabel('lon'); axs[0].set_ylabel('lat'); fig.colorbar(pc, ax=axs[0], shrink=.8)

    sec = arr[:, np.argmin(np.abs(Y)), :]
    lim = np.nanpercentile(np.abs(sec[np.isfinite(sec)]), 99)
    pc = axs[1].pcolormesh(X, Z, sec, vmin=-lim, vmax=lim, cmap='RdBu_r')
    axs[1].set_ylim(-400, 0); axs[1].set_title(f'{xx}: {desc} ({unit}), equatorial section')
    axs[1].set_xlabel('lon'); axs[1].set_ylabel('depth (m)'); fig.colorbar(pc, ax=axs[1], shrink=.8)
    fig.tight_layout(); fig.savefig(f'{outdir}/fig6_control_{var}.png', dpi=140); plt.close(fig)


# ============================================================================
# Intercomparison across experiments
# ============================================================================
def fig_intercomp_bars(controls, outdir):
    """% RMSE improvement of each control vs prior (iter0) and vs noVel, per product."""
    r = _rmse_pivot(_metrics())
    prods = [p for p in PRODS if p[0] in r.index]
    x = np.arange(len(prods)); w = 0.8 / len(controls)
    fig, axs = plt.subplots(2, 1, figsize=(13, 9))
    for ax, base, ttl in [(axs[0], 'prior', '(a) vs prior (iter0)'),
                          (axs[1], 'noVel', '(b) vs TPOSE6-noVel')]:
        for ci, c in enumerate(controls):
            vals = [100 * (r.loc[k, base] - r.loc[k, c]) / r.loc[k, base]
                    if base in r.columns else np.nan for k, _, _ in prods]
            ax.bar(x + (ci - (len(controls) - 1) / 2) * w,
                   [v if np.isfinite(v) else 0 for v in vals], w,
                   label=tiw.CONTROL_XX[c], color=CTRL_COLOR[c])
        ax.set_xticks(x); ax.set_xticklabels([p[1] for p in prods])
        ax.axhline(0, color='k', lw=.8); ax.grid(axis='y', alpha=.3)
        ax.set_ylabel('% RMSE improvement'); ax.set_title(ttl)
        ax.legend(fontsize=8, ncol=len(controls), loc='upper right')
    fig.suptitle('Which control drives which improvement (single-control runs)')
    fig.tight_layout(); fig.savefig(f'{outdir}/intercomp_improvement.png', dpi=140); plt.close(fig)


def fig_intercomp_depth(controls, outdir):
    """Depth-resolved RMSE with every control overlaid (+ prior, full)."""
    fig, axs = plt.subplots(1, 4, figsize=(15, 5), sharey=True)
    for ax, (prod, var) in zip(axs, [('argo', 'T'), ('argo', 'S'), ('tao', 'U'), ('tao', 'V')]):
        obs = np.load(f'{CACHE}/obs_{prod}.npz')
        depth = obs['depth']; o = obs[f'obs_{var}']; w = obs[f'w_{var}']
        series = [('prior', '0.5', 'prior (no DA)', 2, '--'),
                  ('Vel', 'tab:blue', 'TPOSE6-Vel (full)', 2, '-')]
        series += [(c, CTRL_COLOR[c], tiw.CONTROL_XX[c], 1.3, '-') for c in controls]
        for run, col, lab, lw, ls in series:
            fp = f'{CACHE}/model_{run}_{prod}_{var}.npy'
            if os.path.exists(fp):
                ax.plot(tiw.rmse_profile(o, np.load(fp), w), depth, col, lw=lw, ls=ls, label=lab)
        ax.set_title(f'{prod.upper()} {var}'); ax.set_xlabel('RMSE'); ax.grid(alpha=.3)
    axs[0].set_ylabel('depth (m)'); axs[0].set_ylim(-500, 0)
    axs[-1].legend(fontsize=7, loc='lower right')
    fig.suptitle('Depth-resolved RMSE: all single-control runs')
    fig.tight_layout(); fig.savefig(f'{outdir}/intercomp_depth_rmse.png', dpi=140); plt.close(fig)


# ============================================================================
# Increment decomposition: "work" each control does toward the full solution.
#   d_i = m_i - m_prior ;  d_full = m_Vel - m_prior
#   projection share  beta_i = <d_i, d_full> / <d_full, d_full>
#   sum_i beta_i + (interaction+untested residual) = 1
# Done in model-equivalent / state space (NOT RMSE), the quantity that actually
# decomposes. All products reference prior (iter0).
# ============================================================================
def _incr_vectors(prodkey, controls):
    base = 'prior'
    if prodkey in ('sst', 'ssh'):
        load = lambda r: xr.open_dataarray(f'{CACHE}/grid_{r}_{prodkey}.nc').values
    else:
        p, var = prodkey.split('_')
        load = lambda r: np.load(f'{CACHE}/model_{r}_{p}_{var}.npy')
    mp, mv = load(base), load('Vel')
    mc = {c: load(c) for c in controls}
    v = np.isfinite(mp) & np.isfinite(mv)
    for c in controls:
        v &= np.isfinite(mc[c])
    return base, (mv - mp)[v], {c: (mc[c][v] - mp[v]) for c in controls}


def fig_decomp_bars(controls, outdir):
    prods = [p for p in PRODS if os.path.exists(f'{CACHE}/grid_Vel_{p[0]}.nc')
             or os.path.exists(f"{CACHE}/model_Vel_{p[0].replace('_','_')}.npy")]
    x = np.arange(len(prods)); w = 0.8 / (len(controls) + 1)
    fig, ax = plt.subplots(figsize=(13, 5))
    shares = {}
    for ci, c in enumerate(controls + ['interaction']):
        vals = []
        for key, _, _ in prods:
            base, df, di = _incr_vectors(key, controls)
            den = float(np.dot(df, df))
            betas = {cc: float(np.dot(di[cc], df)) / den for cc in controls}
            shares[key] = betas
            vals.append(100 * (betas[c] if c in controls else 1 - sum(betas.values())))
        col = CTRL_COLOR[c] if c in controls else '0.6'
        lab = tiw.CONTROL_XX[c] if c in controls else 'interaction + untested'
        ax.bar(x + (ci - len(controls) / 2) * w, vals, w, label=lab, color=col)
    ax.set_xticks(x); ax.set_xticklabels([p[1] for p in prods])
    ax.axhline(0, color='k', lw=.8); ax.axhline(100, color='k', lw=.6, ls=':')
    ax.grid(axis='y', alpha=.3); ax.set_ylabel('% of full increment (projection share)')
    ax.set_title('Work done by each control toward the full solution '
                 r'($\beta_i=\langle d_i,d_{full}\rangle/\langle d_{full},d_{full}\rangle$)')
    ax.legend(fontsize=8, ncol=len(controls) + 1, loc='upper right')
    fig.tight_layout(); fig.savefig(f'{outdir}/decomp_work_shares.png', dpi=140); plt.close(fig)


def fig_decomp_depth(controls, outdir):
    fig, axs = plt.subplots(1, 4, figsize=(15, 5), sharey=True)
    for ax, (prod, var) in zip(axs, [('argo', 'T'), ('argo', 'S'), ('tao', 'U'), ('tao', 'V')]):
        L = lambda r: np.load(f'{CACHE}/model_{r}_{prod}_{var}.npy')
        mp, mv = L('prior'), L('Vel'); mc = {c: L(c) for c in controls}
        obs = np.load(f'{CACHE}/obs_{prod}.npz'); depth = obs['depth']
        valid = np.isfinite(mp) & np.isfinite(mv)
        for c in controls:
            valid &= np.isfinite(mc[c])
        prof = lambda d: np.nanmean(np.where(valid, d, np.nan), axis=0)
        dfull = prof(mv - mp); dsum = sum(prof(mc[c] - mp) for c in controls)
        for c in controls:
            ax.plot(prof(mc[c] - mp), depth, color=CTRL_COLOR[c], lw=1.3, label=tiw.CONTROL_XX[c])
        ax.plot(dfull, depth, 'k', lw=2, label='full (Vel−prior)')
        ax.plot(dsum, depth, '0.5', lw=1.5, ls='--', label='Σ single controls')
        ax.axvline(0, color='0.8', lw=.8)
        ax.set_title(f'{prod.upper()} {var}'); ax.set_xlabel('mean increment'); ax.grid(alpha=.3)
    axs[0].set_ylabel('depth (m)'); axs[0].set_ylim(-500, 0)
    axs[-1].legend(fontsize=7, loc='lower right')
    fig.suptitle('Increment decomposition by depth (mean model change from prior)')
    fig.tight_layout(); fig.savefig(f'{outdir}/decomp_depth.png', dpi=140); plt.close(fig)


def fig_decomp_maps(controls, outdir):
    for prod in ('sst', 'ssh'):
        base = 'prior'
        if not os.path.exists(f'{CACHE}/grid_{base}_{prod}.nc'):
            continue
        ref = xr.open_dataarray(f'{CACHE}/grid_Vel_{prod}.nc')
        XC, YC = ref.XC.values, ref.YC.values
        gb = xr.open_dataarray(f'{CACHE}/grid_{base}_{prod}.nc').values
        rmsmap = lambda run: np.sqrt(np.nanmean(
            (xr.open_dataarray(f'{CACHE}/grid_{run}_{prod}.nc').values - gb) ** 2, axis=0))
        panels = [('Vel', 'full (Vel−%s)' % base)] + [(c, tiw.CONTROL_XX[c]) for c in controls]
        full = rmsmap('Vel'); vmax = np.nanpercentile(full, 98)
        fig, axs = plt.subplots(2, 3, figsize=(16, 7))
        for ax, (run, lab) in zip(axs.ravel(), panels):
            pc = ax.pcolormesh(XC, YC, rmsmap(run), vmin=0, vmax=vmax, cmap='magma')
            ax.set_title(lab); fig.colorbar(pc, ax=ax, shrink=.8)
        for ax in axs.ravel()[len(panels):]:
            ax.axis('off')
        fig.suptitle(f'{prod.upper()} increment amplitude (RMS over time of run−{base})')
        fig.tight_layout(); fig.savefig(f'{outdir}/decomp_{prod}_maps.png', dpi=140); plt.close(fig)


# ============================================================================
# High-level summary: rank each single control's influence on skill per target.
# Two RMSE views, both straight from metrics.csv (prior=iter0 baseline, Vel=full):
#   (a) % RMSE improvement vs prior  = 100*(prior-ctrl)/prior
#   (b) share of the full-DA improvement captured = (prior-ctrl)/(prior-Vel)
# Rows sorted by mean (a) over targets => the ranking.
# ============================================================================
def fig_summary(controls, outdir):
    r = _rmse_pivot(_metrics())
    prods = [p for p in PRODS if p[0] in r.index and 'prior' in r.columns
             and np.isfinite(r.loc[p[0], 'prior'])]
    controls = [c for c in controls if c in r.columns]
    keys = [p[0] for p in prods]; labels = [p[1] for p in prods]

    pct = np.array([[100 * (r.loc[k, 'prior'] - r.loc[k, c]) / r.loc[k, 'prior']
                     for k in keys] for c in controls])
    share = np.array([[100 * (r.loc[k, 'prior'] - r.loc[k, c])
                       / (r.loc[k, 'prior'] - r.loc[k, 'Vel']) for k in keys]
                      for c in controls])
    order = np.argsort(-np.nanmean(pct, axis=1))          # rank by mean % improvement
    controls = [controls[i] for i in order]; pct = pct[order]; share = share[order]
    ylab = [tiw.CONTROL_XX[c] for c in controls]

    fig, axs = plt.subplots(1, 2, figsize=(15, 1.0 + 0.55 * len(controls)))
    for ax, M, ttl, lim in [
            (axs[0], pct, '(a) % RMSE improvement vs prior (iter0)', 10),
            (axs[1], share, '(b) % of full-DA (Vel) improvement captured', 100)]:
        im = ax.imshow(M, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_yticks(range(len(controls))); ax.set_yticklabels(ylab)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, 'n/a' if not np.isfinite(v) else f'{v:.0f}',
                        ha='center', va='center', fontsize=8,
                        color='k' if abs(v) < lim * 0.6 else 'w')
        ax.set_title(ttl); fig.colorbar(im, ax=ax, shrink=.7)
    fig.suptitle('Single-control influence on skill, ranked (top = most skill improvement)')
    fig.tight_layout(); fig.savefig(f'{outdir}/summary_ranking.png', dpi=140); plt.close(fig)
    # dump the numbers so claims trace to data, not the picture
    pd.DataFrame(pct, index=ylab, columns=labels).to_csv(f'{outdir}/summary_pct_vs_prior.csv')
    pd.DataFrame(share, index=ylab, columns=labels).to_csv(f'{outdir}/summary_share_of_Vel.csv')


if __name__ == '__main__':
    controls = [c for c in ALL_CONTROLS
                if os.path.exists(f'{CACHE}/model_{c}_argo_T.npy')]
    for c in controls:
        d = f'{FIGS}/{c}'; os.makedirs(d, exist_ok=True)
        for fn in (fig_headline, fig_depth, fig_time, fig_euc, fig_maps, fig_control):
            try:
                fn(c, d)
            except Exception as e:
                print(f'[{c}] {fn.__name__} FAILED: {type(e).__name__}: {e}')
        print(f'[{c}] done -> {d}')
    di = f'{FIGS}/intercomparison'; os.makedirs(di, exist_ok=True)
    for fn in (fig_intercomp_bars, fig_intercomp_depth):
        try:
            fn(controls, di)
        except Exception as e:
            print(f'[intercomp] {fn.__name__} FAILED: {type(e).__name__}: {e}')
    print(f'[intercomp] done -> {di}  (controls: {controls})')
    # decomposition on the KEPT controls only (uvel/vvel excluded from the
    # projection budget / combos per TODO; they fall into the untested residual).
    kept = [c for c in tiw.CONTROLS if c in controls]
    dd = f'{FIGS}/decomposition'; os.makedirs(dd, exist_ok=True)
    for fn in (fig_decomp_bars, fig_decomp_depth, fig_decomp_maps):
        try:
            fn(kept, dd)
        except Exception as e:
            print(f'[decomp] {fn.__name__} FAILED: {type(e).__name__}: {e}')
    print(f'[decomp] done -> {dd}')
    ds = f'{FIGS}/summary'; os.makedirs(ds, exist_ok=True)
    try:
        fig_summary(controls, ds)
        print(f'[summary] done -> {ds}')
    except Exception as e:
        print(f'[summary] fig_summary FAILED: {type(e).__name__}: {e}')
