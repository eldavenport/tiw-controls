"""
Fixed-increment ANOVA decomposition of the {theta,salt,uwind,swdown} factorial.

Every combo run has the named controls ON at their FULL-solution optimized
increments, all else at background (Stein-Alpert factor separation, NO
re-optimization), so the Mobius/ANOVA decomposition is exact per obs point:

    a_S = sum_{T subset S} (-1)^(|S|-|T|) m_T          (m_{} = m_prior)

is the irreducible |S|-way interaction of factor set S (main effect when
|S|=1). Fundamental identity (checked in __main__):
    sum_{nonempty S subset F} a_S == m_F - m_prior          (F = top4 set)
    m_Vel - m_prior == (m_top4 - m_prior) + r,  r = m_Vel - m_top4
so r (the excluded-controls residual) closes the budget to the full solution.

CANONICAL size = share of the full-solution increment, ||a_S|| / ||d_full||,
d_full = m_Vel - m_prior. ONE shared baseline for every term/order => directly
comparable. ||d_comb|| = ||m_S - m_prior|| is context, not the normalizer.

Outputs (figs/anova/):
  factorial_budget.csv   every term + residual, ||a||, %d_full, per target
  heatmap.png            controls x controls per target (diag=main, off=pair)
  budget_bars.png        all orders + residual per target
  map_<term>_<field>.png gridded SST/SSH interaction field (time-mean), any order
  <run>_{euc,tiw}.png    EUC U / TIW V field decomposition, ANY order (pair/triad/4-way)
                         from cached euctiw_<run>.nc (see cache3d)

Env: ~/miniforge3/envs/tpose/bin/python
"""
import itertools
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

FACTORS = 'TSUW'                                   # theta, salt, uwind, swdown
SYM = {'T': 'θ', 'S': 'S', 'U': 'u wind', 'W': 'sw'}  # 'u wind' not 'u': avoid uvel confusion
ORDER_NAME = {1: 'main', 2: 'pair', 3: 'triad', 4: 'tetrad'}
# run name for any subset of factors (frozenset of keys); prior = empty set.
RUN = {
    frozenset(): 'prior',
    frozenset('T'): 'initialT', frozenset('S'): 'initialS',
    frozenset('U'): 'zonalWind', frozenset('W'): 'swdown',
    frozenset('TU'): 'initialT_uwind', frozenset('TS'): 'theta_salt',
    frozenset('TW'): 'theta_swdown', frozenset('SU'): 'salt_uwind',
    frozenset('SW'): 'salt_swdown', frozenset('UW'): 'uwind_swdown',
    frozenset('TSU'): 'theta_salt_uwind', frozenset('TSW'): 'theta_salt_swdown',
    frozenset('TUW'): 'theta_uwind_swdown', frozenset('SUW'): 'salt_uwind_swdown',
    frozenset('TSUW'): 'top4',
}
PRODS = [('argo_T', 'Argo T'), ('argo_S', 'Argo S'), ('tao_U', 'TAO U'),
         ('tao_V', 'TAO V'), ('sst', 'SST'), ('ssh', 'SSH')]
EUC_LON, TIW_LON, TIW_LAT, TIW_ZTOP = (130, 282), (200, 282), (2.0, 5.0), 15


def term_label(S):
    return '×'.join(SYM[f] for f in sorted(S, key='TSUW'.index))


def subsets(order=None):
    """Non-empty subsets of FACTORS as frozensets, sorted by (size, TSUW order)."""
    out = []
    for k in range(1, len(FACTORS) + 1):
        if order and k != order:
            continue
        out += [frozenset(c) for c in itertools.combinations(FACTORS, k)]
    return sorted(out, key=lambda S: (len(S), sorted(S, key='TSUW'.index)))


# ---- obs-space engine -------------------------------------------------------
def _load_obs(prodkey, run):
    if prodkey in ('sst', 'ssh'):
        return xr.open_dataarray(f'{CACHE}/grid_{run}_{prodkey}.nc').values
    p, var = prodkey.split('_')
    return np.load(f'{CACHE}/model_{run}_{p}_{var}.npy')


def _rms(a):
    return float(np.sqrt(np.nanmean(a ** 2)))


def anova_term(S, prodkey):
    """Clean |S|-way interaction a_S, own increment d_S, shared d_full=m_Vel-m_prior,
    all in obs space on a common valid mask. S='residual' -> r=m_Vel-m_top4."""
    mp = _load_obs(prodkey, 'prior')
    full = _load_obs(prodkey, 'Vel')
    if S == 'residual':
        top4 = _load_obs(prodkey, RUN[frozenset(FACTORS)])
        valid = np.isfinite(mp) & np.isfinite(full) & np.isfinite(top4)
        return (full - top4)[valid], (full - top4)[valid], (full - mp)[valid]
    S = frozenset(S)
    fields = {T: _load_obs(prodkey, RUN[T]) for k in range(len(S) + 1)
              for T in map(frozenset, itertools.combinations(S, k))}
    valid = np.all([np.isfinite(v) for v in (*fields.values(), full)], axis=0)
    a = sum((-1) ** (len(S) - len(T)) * fields[T] for T in fields)
    return a[valid], (fields[S] - mp)[valid], (full - mp)[valid]


# ---- (1) full factorial budget table ----------------------------------------
def budget_table():
    terms = subsets() + ['residual']
    rows = []
    for S in terms:
        lab = 'residual r' if S == 'residual' else term_label(S)
        order = 'resid' if S == 'residual' else len(S)
        row = dict(term=lab, order=order)
        for key, name in PRODS:
            try:
                a, _, dfull = anova_term(S, key)
                row[name] = 100 * _rms(a) / _rms(dfull) if _rms(dfull) else np.nan
            except FileNotFoundError:
                row[name] = np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f'{FIGS}/factorial_budget.csv', index=False)
    return df


def budget_projection_table():
    """Signed share of the final increment each term accounts for:
    beta_S = <a_S, d_full> / <d_full, d_full>.  By the identity Sum_S a_S + r = d_full
    these shares sum to ~100% (how much of d_full this increment explains)."""
    terms = subsets() + ['residual']
    rows = []
    for S in terms:
        lab = 'residual r' if S == 'residual' else term_label(S)
        order = 'resid' if S == 'residual' else len(S)
        row = dict(term=lab, order=order)
        for key, name in PRODS:
            try:
                a, _, dfull = anova_term(S, key)
                den = float(np.dot(dfull, dfull))
                row[name] = 100 * float(np.dot(a, dfull)) / den if den else np.nan
            except FileNotFoundError:
                row[name] = np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f'{FIGS}/factorial_budget_projection.csv', index=False)
    return df


# ---- (2) controls x controls heatmap (main effects + pairwise) --------------
def fig_heatmap():
    n = len(FACTORS)
    mats = {}
    for key, name in PRODS:
        M = np.full((n, n), np.nan)
        for i, fi in enumerate(FACTORS):
            for j, fj in enumerate(FACTORS):
                S = frozenset(fi) if i == j else frozenset(fi + fj)
                try:
                    a, _, dfull = anova_term(S, key)
                    M[i, j] = 100 * _rms(a) / _rms(dfull) if _rms(dfull) else np.nan
                except FileNotFoundError:
                    pass
        mats[name] = M
    gmax = np.nanmax([np.nanmax(M) for M in mats.values()])   # shared scale across panels
    fig, axs = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for ax, (key, name) in zip(axs.flat, PRODS):
        M = mats[name]
        im = ax.imshow(M, cmap='viridis', vmin=0, vmax=gmax)
        ax.set_xticks(range(n)); ax.set_xticklabels([SYM[f] for f in FACTORS])
        ax.set_yticks(range(n)); ax.set_yticklabels([SYM[f] for f in FACTORS])
        for i in range(n):
            for j in range(n):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f'{M[i, j]:.0f}', ha='center', va='center',
                            color='w' if M[i, j] < 0.6 * gmax else 'k', fontsize=10)
        ax.set_title(f'{name}')
    cb = fig.colorbar(im, ax=axs, shrink=.8)
    cb.set_label('interaction strength  ||a_S|| / ||d_full||  (% of full increment)')
    fig.suptitle('Interaction matrix  ||a_S|| / ||d_full||  (%)   —   diagonal = main effect, '
                 'off-diagonal = pairwise interaction\n(triad / tetrad / residual in budget_bars.png)')
    fig.savefig(f'{FIGS}/heatmap.png', dpi=140); plt.close(fig)


# ---- (2b) all-orders + residual budget bars ---------------------------------
def fig_budget_bars(df, ylabel='% of ||d_full||',
                    suptitle=('Full factorial budget: every ANOVA term + excluded-controls residual, '
                              'as share of the full-solution increment ||d_full|| = ||m_Vel - m_prior||\n'
                              '(how much unique, irreducible signal does each interaction order carry)'),
                    fname='budget_bars.png', ylim=None):
    cmap = {1: 'tab:green', 2: 'tab:orange', 3: 'tab:red', 4: 'tab:purple', 'resid': '0.4'}
    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    for ax, (_, name) in zip(axs.flat, PRODS):
        x = np.arange(len(df)); colors = [cmap[o] for o in df['order']]
        ax.bar(x, df[name], color=colors)
        ax.set_xticks(x); ax.set_xticklabels(df['term'], rotation=90, fontsize=7)
        ax.axhline(0, color='k', lw=.6)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(name); ax.set_ylabel(ylabel)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[o]) for o in [1, 2, 3, 4, 'resid']]
    fig.legend(handles, ['main', 'pair', 'triad', 'tetrad', 'residual'],
               loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(f'{FIGS}/{fname}', dpi=140); plt.close(fig)


# ---- (2c) targeted EUC scalars: factor-separate a PHYSICAL diagnostic ---------
# Instead of global RMSE, reduce the cached time-mean equatorial U section to EUC
# scalars and run the same inclusion-exclusion (signed, in physical units): a_S =
# M(S's run) incl-excl, so a_θ = M(θ)-M(prior) = how much θ speeds/shoals the jet.
EUC_WIN = dict(lon=(180, 270), ztop=0, zbot=-300)   # core search window (dateline–90°W, top 300 m)
EUC_TRANS_LON = 220.0                                # transport reference longitude = 140°W
# TAO equatorial current-meter sites (obs longitudes), model 0–360°E.
TAO_LONS = {'170°W': 190.0, '140°W': 220.0, '110°W': 250.0}
METRICS = ['core_speed', 'core_depth', 'transport']
UNITS = {'core_speed': 'm/s', 'core_depth': 'm', 'transport': 'Sv'}
CMAP = {1: 'tab:green', 2: 'tab:orange', 3: 'tab:red', 4: 'tab:purple', 'resid': '0.4'}


def _col_scalars(col):
    """EUC scalars from a single-longitude U profile (Z,) already sliced to 0..-300 m."""
    pos = col.clip(min=0)
    dz = np.abs(np.gradient(col.Z.values))
    return dict(
        core_speed=float(col.max()),
        core_depth=float((pos * pos.Z).sum() / pos.sum()) if float(pos.sum()) > 0 else np.nan,
        transport=float(np.nansum(np.clip(col.values, 0, None) * dz) * 111e3 / 1e6))


def euc_scalars(e):
    """Basin-window scalars: core_speed = peak eastward U over lon 180–270°E & top 300 m,
    core_depth = eastward-U-weighted centroid over that window, transport at 140°W."""
    w = e.sel(XG=slice(*EUC_WIN['lon'])).sel(Z=slice(EUC_WIN['ztop'], EUC_WIN['zbot']))
    pos = w.clip(min=0)
    col = e.sel(XG=EUC_TRANS_LON, method='nearest').sel(Z=slice(0, -300))
    return dict(
        core_speed=float(w.max()),
        core_depth=float((pos * pos.Z).sum() / pos.sum()),
        transport=_col_scalars(col)['transport'])


def euc_scalars_at(e, lon):
    """EUC scalars in the U profile AT a longitude (the obs sites)."""
    return _col_scalars(e.sel(XG=lon, method='nearest').sel(Z=slice(0, -300)))


def _order(S):
    return 'resid' if S == 'residual' else len(S)


def _incl_excl(v, S):
    """Clean |S|-way factor-separation of a scalar response v (dict run->value)."""
    return sum((-1) ** (len(S) - len(T)) * v[RUN[frozenset(T)]]
               for k in range(len(S) + 1) for T in itertools.combinations(S, k))


def _scalar_rows(v, tag):
    """ANOVA rows (every subset + residual) for one scalar response v, exact budget.
    tag = extra columns (e.g. metric, longitude). Returns (rows, identity_err)."""
    rows, asum = [], 0.0
    for S in subsets():
        a = _incl_excl(v, S)
        asum += a
        rows.append(dict(**tag, term=term_label(S), order=_order(S), a=a))
    r = v['Vel'] - v[RUN[frozenset(FACTORS)]]
    rows.append(dict(**tag, term='residual r', order='resid', a=r))
    dfull = v['Vel'] - v['prior']
    for row in rows:
        row['pct_of_full'] = 100 * row['a'] / dfull if dfull else np.nan
    return rows, abs((asum + r) - dfull)


def _bars(ax, d, unit, title, ref=None, col='a', ylabel=None):
    ax.bar(np.arange(len(d)), d[col], color=[CMAP[o] for o in d['order']])
    ax.axhline(0, color='k', lw=.6)
    if ref is not None:                               # final Vel change = Σ all bars
        ax.axhline(ref, color='k', ls='--', lw=1.3, zorder=5)
    ax.set_xticks(np.arange(len(d))); ax.set_xticklabels(d['term'], rotation=90, fontsize=7)
    ax.set_title(title); ax.set_ylabel(ylabel or f'signed contribution ({unit})')


def _legend(fig):
    h = [plt.Rectangle((0, 0), 1, 1, color=CMAP[o]) for o in [1, 2, 3, 4, 'resid']]
    h += [plt.Line2D([0], [0], color='k', ls='--', lw=1.3)]
    fig.legend(h, ['main', 'pair', 'triad', 'tetrad', 'residual', 'Vel−prior (total)'],
               loc='lower center', ncol=6, bbox_to_anchor=(0.5, 0.0))


def euc_metric_anova():
    """Basin-window EUC scalar decomposition (unchanged analysis)."""
    runs = ['prior', 'Vel'] + [RUN[S] for S in subsets()]
    M = {r: euc_scalars(xr.open_dataset(f'{CACHE}/euctiw_{r}.nc')['euc']) for r in runs}
    pd.DataFrame({m: {r: M[r][m] for r in runs} for m in METRICS}).to_csv(
        f'{FIGS}/euc_metrics_raw.csv')
    rows, ident = [], {}
    for m in METRICS:
        rr, ident[m] = _scalar_rows({r: M[r][m] for r in runs}, dict(metric=m))
        rows += rr
    df = pd.DataFrame(rows)
    df.to_csv(f'{FIGS}/euc_anova.csv', index=False)
    print('EUC basin-window budget identity max|Σa+r-d_full|:',
          {k: f'{val:.2e}' for k, val in ident.items()})
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, m in zip(axs, METRICS):
        full = M['Vel'][m] - M['prior'][m]
        _bars(ax, df[df.metric == m], UNITS[m],
              f'{m}   (prior {M["prior"][m]:.3g} → Vel {M["Vel"][m]:.3g} {UNITS[m]}; d_full {full:+.3g})',
              ref=full)
    _legend(fig)
    fig.suptitle('EUC physical-scalar factor separation, BASIN WINDOW (signed, physical units): '
                 'how each control + interaction changes the equatorial undercurrent')
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(f'{FIGS}/euc_anova.png', dpi=140); plt.close(fig)

    # projection framing: each term as % of the final increment d_full (bars sum to ~100%)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, m in zip(axs, METRICS):
        full = M['Vel'][m] - M['prior'][m]
        _bars(ax, df[df.metric == m], UNITS[m],
              f'{m}  (d_full {full:+.3g} {UNITS[m]} = 100%)',
              ref=100, col='pct_of_full', ylabel='% of final increment')
    _legend(fig)
    fig.suptitle('EUC physical-scalar factor separation, BASIN WINDOW — projection framing: '
                 '% of the final increment d_full = M(Vel) - M(prior) each term accounts for (sum ~100%)')
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(f'{FIGS}/euc_anova_projection.png', dpi=140); plt.close(fig)
    return df


def euc_metric_anova_bylon():
    """Same scalars decomposed AT each TAO obs longitude (170°W/140°W/110°W)."""
    runs = ['prior', 'Vel'] + [RUN[S] for S in subsets()]
    E = {r: xr.open_dataset(f'{CACHE}/euctiw_{r}.nc')['euc'] for r in runs}
    M = {(L, r): euc_scalars_at(E[r], lon) for L, lon in TAO_LONS.items() for r in runs}
    pd.DataFrame({(L, m): {r: M[(L, r)][m] for r in runs}
                  for L in TAO_LONS for m in METRICS}).to_csv(f'{FIGS}/euc_metrics_bylon_raw.csv')
    rows, ident = [], {}
    for L in TAO_LONS:
        for m in METRICS:
            rr, ident[(L, m)] = _scalar_rows({r: M[(L, r)][m] for r in runs},
                                             dict(longitude=L, metric=m))
            rows += rr
    df = pd.DataFrame(rows)
    df.to_csv(f'{FIGS}/euc_anova_bylon.csv', index=False)
    print('EUC by-longitude budget identity max|Σa+r-d_full|:',
          f'{max(ident.values()):.2e} (all {len(ident)} metric×lon)')
    # rows = metric, cols = longitude; share x (same terms) + y per row (comparable across lon).
    fig, axs = plt.subplots(len(METRICS), len(TAO_LONS), figsize=(18, 13),
                            sharex=True, sharey='row')
    for i, m in enumerate(METRICS):
        for j, L in enumerate(TAO_LONS):
            ax = axs[i, j]
            full = M[(L, 'Vel')][m] - M[(L, 'prior')][m]
            _bars(ax, df[(df.metric == m) & (df.longitude == L)], UNITS[m],
                  f'{m} @ {L}  (prior {M[(L,"prior")][m]:.3g} → Vel {M[(L,"Vel")][m]:.3g}; '
                  f'd_full {full:+.3g})', ref=full)
            if i < len(METRICS) - 1:
                ax.set_xlabel('')
            if j > 0:                       # y shared across the row -> ylabel only on the left
                ax.set_ylabel('')
    _legend(fig)
    fig.suptitle('EUC physical-scalar factor separation AT TAO obs longitudes '
                 '(rows=metric, cols=170°W/140°W/110°W; signed physical units)')
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(f'{FIGS}/euc_anova_bylon.png', dpi=140); plt.close(fig)

    # projection framing: % of the final increment d_full at each longitude (sum ~100%)
    fig, axs = plt.subplots(len(METRICS), len(TAO_LONS), figsize=(18, 13),
                            sharex=True, sharey='row')
    for i, m in enumerate(METRICS):
        for j, L in enumerate(TAO_LONS):
            ax = axs[i, j]
            full = M[(L, 'Vel')][m] - M[(L, 'prior')][m]
            _bars(ax, df[(df.metric == m) & (df.longitude == L)], UNITS[m],
                  f'{m} @ {L}  (d_full {full:+.3g} = 100%)',
                  ref=100, col='pct_of_full', ylabel='% of final increment')
            if i < len(METRICS) - 1:
                ax.set_xlabel('')
            if j > 0:
                ax.set_ylabel('')
    _legend(fig)
    fig.suptitle('EUC physical-scalar factor separation AT TAO obs longitudes — projection framing: '
                 '% of final increment d_full each term accounts for (rows=metric, cols=lon; sum ~100%)')
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(f'{FIGS}/euc_anova_bylon_projection.png', dpi=140); plt.close(fig)
    return df


# ---- (2d) TIW phase + amplitude decomposition (vs Vel) ------------------------
# Band-pass each run's TIW Hovmöller to the westward TIW band, take the analytic
# (complex) signal z(x,t), project onto Vel's normalized TIW field: c = <ẑ_Vel, z>.
# Band-pass+Hilbert+projection are LINEAR, so c factor-separates EXACTLY:
#   Re(a_S) = amplitude contribution IN PHASE with Vel; Im(a_S) = quadrature (phase shift).
# Complemented by a nonlinear total-TIW-energy scalar (net wave activity incl. cancellation).
TIW_BAND = dict(per=(15, 40), wav=(700, 1600))   # days, km (period, zonal wavelength)
TIW_REGION = (180, 265)                            # cold-tongue lon window (all-ocean)
TIW_SST_LAT = (0, 5)                               # SST TIW band (cusps on cold-tongue N flank)
TIW_MAP_TERMS = ['TU', 'TS', 'TSU']                # dominant terms for envelope Hovmöllers


def _tiw_hov(run, channel):
    """TIW Hovmöller (time, XC) over the cold-tongue window: V (cached tiv) or SST/SSH band-mean."""
    if channel == 'V':
        h = xr.open_dataset(f'{CACHE}/euctiw_{run}.nc')['tiv']
    else:                                   # SST or SSH gridded field, mean over TIW lat band
        field = 'sst' if channel == 'SST' else 'ssh'
        s = xr.open_dataarray(f'{CACHE}/grid_{run}_{field}.nc')
        h = s.sel(YC=slice(*TIW_SST_LAT)).mean('YC')
    return h.sel(XC=slice(*TIW_REGION))


SEP1 = 244   # Sep-1 index into the full-year obs arrays (matches make_figures._gridded_obs)


def _tiw_src(run, channel):
    """Does the cached source for this run+channel exist? (noVel has SST/SSH grids but no V.)"""
    f = (f'{CACHE}/euctiw_{run}.nc' if channel == 'V'
         else f'{CACHE}/grid_{run}_{"sst" if channel == "SST" else "ssh"}.nc')
    return os.path.exists(f)


def _tiw_hov_obs(channel):
    """Observed TIW Hovmöller (time, XC), same band/region as the model (SST=mw_fusion,
    SSH=RADS multi-sat anomaly). No gridded V obs (TAO is sparse) -> SST/SSH only."""
    field = 'sst' if channel == 'SST' else 'ssh'
    ref = xr.open_dataarray(f'{CACHE}/grid_Vel_{field}.nc')
    ny, nx = ref.sizes['YC'], ref.sizes['XC']
    if channel == 'SST':
        sst, _ = tiw.load_sst(nx, ny)
        o = np.moveaxis(sst[:, :, SEP1:SEP1 + tiw.NT], 2, 0)
    else:
        rads = tiw.load_rads(nx, ny)
        arrs = [np.moveaxis(rads[s][:, :, SEP1:SEP1 + tiw.NT], 2, 0) for s in rads]
        o = np.nanmean(np.stack([a - np.nanmean(a, axis=0, keepdims=True) for a in arrs]), axis=0)
    da = xr.DataArray(o, dims=('time', 'YC', 'XC'), coords={'YC': ref.YC, 'XC': ref.XC})
    return da.sel(YC=slice(*TIW_SST_LAT)).mean('YC').sel(XC=slice(*TIW_REGION))


def _analytic_tiw(h, band=None):
    """2D-FFT band-pass to the westward TIW band, single spectral lobe -> complex analytic field."""
    band = band or TIW_BAND
    f = h.values.astype(float)
    f = np.nan_to_num(f - np.nanmean(f, axis=0, keepdims=True))   # demean per lon, fill NaN
    nt, nx = f.shape
    nu = np.fft.fftfreq(nt, 1.0)[:, None]                         # cpd
    ka = np.fft.fftfreq(nx, float(np.mean(np.diff(h.XC))) * 111.32)[None, :]   # cpkm
    (plo, phi), (wlo, whi) = band['per'], band['wav']
    band = ((np.abs(nu) >= 1 / phi) & (np.abs(nu) <= 1 / plo) &
            (np.abs(ka) >= 1 / whi) & (np.abs(ka) <= 1 / wlo))
    west = (nu > 0) & (ka > 0)                                    # westward phase speed, single lobe
    z = np.fft.ifft2(np.fft.fft2(f) * (band & west))
    return xr.DataArray(z, dims=h.dims, coords=h.coords)


def _tiw_period(z):
    """Energy-weighted TIW period (days) of a complex field, for deg->days phase conversion."""
    nt = z.sizes['time']
    P = (np.abs(np.fft.fft(z.values, axis=0)) ** 2).sum(1)
    nu = np.fft.fftfreq(nt, 1.0)
    m = nu > 0
    return float(1.0 / (np.sum(nu[m] * P[m]) / np.sum(P[m])))


def tiw_phase_amp(channel):
    runs = ['prior', 'Vel'] + [RUN[S] for S in subsets()]
    Z = {r: _analytic_tiw(_tiw_hov(r, channel)) for r in runs}
    zv = Z['Vel'].values.ravel()
    uvel = zv / np.sqrt(np.vdot(zv, zv).real)                    # unit Vel TIW field
    c = {r: complex(np.vdot(uvel, Z[r].values.ravel())) for r in runs}   # <ẑ_Vel, z_r>
    E = {r: float(np.sqrt(np.mean(np.abs(Z[r].values) ** 2))) for r in runs}   # total TIW energy (RMS)
    cv = c['Vel'].real                                           # = ||z_Vel|| (real, positive)
    Tdom = _tiw_period(Z['Vel'])
    lam = np.mean((TIW_BAND['wav']))                             # nominal wavelength for deg->lon

    rows, phase_rows = [], []
    amp_sum = pha_sum = 0 + 0j
    for S in subsets():
        a = _incl_excl(c, S)                                     # complex contribution
        amp_sum += a
        rows.append(dict(term=term_label(S), order=_order(S),
                         amp_pct=100 * a.real / cv, phase_pct=100 * a.imag / cv))
    rc = c['Vel'] - c[RUN[frozenset(FACTORS)]]
    rows.append(dict(term='residual r', order='resid', amp_pct=100 * rc.real / cv,
                     phase_pct=100 * rc.imag / cv))
    dfull = c['Vel'] - c['prior']
    ident_c = abs((amp_sum + rc) - dfull)
    # energy scalar (nonlinear) ANOVA
    erows, eident = _scalar_rows({r: E[r] for r in runs}, {})
    # per-run phase lag vs Vel (deg, days) + coherence (cos-similarity to Vel's TIW field)
    for r in runs:
        lag_deg = np.degrees(np.angle(c[r]))
        znorm = np.sqrt(np.vdot(Z[r].values.ravel(), Z[r].values.ravel()).real)
        coh = abs(c[r]) / znorm if znorm > 0 else np.nan   # uvel is unit-norm -> in [0,1]
        phase_rows.append(dict(run=r, lag_deg=lag_deg, lag_days=lag_deg / 360 * Tdom,
                               coherence=coh, energy=E[r]))
    df = pd.DataFrame(rows)
    df.to_csv(f'{FIGS}/tiw_phaseamp_{channel}.csv', index=False)
    pd.DataFrame(phase_rows).to_csv(f'{FIGS}/tiw_phase_{channel}.csv', index=False)
    print(f'[{channel}] complex identity |Σa+r-dfull|={ident_c:.2e}, energy identity={eident:.2e}, '
          f'Tdom={Tdom:.1f} d, ||z_Vel||={cv:.4g}')

    ed = pd.DataFrame(erows)
    efull = E['Vel'] - E['prior']
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.5))
    _bars(axs[0], df.assign(a=df.amp_pct), '% of Vel amp',
          f'{channel}: amplitude IN PHASE with Vel  (Re)', ref=100 * dfull.real / cv)
    _bars(axs[1], df.assign(a=df.phase_pct), '% of Vel amp',
          f'{channel}: PHASE-shift / quadrature  (Im)', ref=100 * dfull.imag / cv)
    _bars(axs[2], ed, 'm/s' if channel == 'V' else '°C',
          f'{channel}: total TIW energy  (prior {E["prior"]:.3g} → Vel {E["Vel"]:.3g}; d {efull:+.3g})',
          ref=efull)
    _legend(fig)
    fig.suptitle(f'TIW {channel} phase+amplitude factor separation vs Vel  '
                 f'(amplitude = reinforces Vel wave; quadrature = shifts its phase; '
                 f'energy = net wave activity).  Region {TIW_REGION}°E, band {TIW_BAND["per"]} d / '
                 f'{TIW_BAND["wav"]} km westward')
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(f'{FIGS}/tiw_phaseamp_{channel}.png', dpi=140); plt.close(fig)

    # projection framing: term as % of the *increment* d_full (not of the full Vel amp).
    # amp/phase normalize by the real/imag increment respectively, so each panel sums to
    # ~100%. amp_pct = 100*a.real/cv -> a.real = amp_pct*cv/100, hence /dfull.real below.
    df['amp_proj'] = df.amp_pct * cv / dfull.real
    df['phase_proj'] = df.phase_pct * cv / dfull.imag
    print(f'[{channel}] increment d_full: amp(Re)={dfull.real:+.4g}, phase(Im)={dfull.imag:+.4g}')
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.5))
    _bars(axs[0], df, '%', f'{channel}: amplitude in-phase (Re) — % of increment',
          ref=100, col='amp_proj', ylabel='% of final increment')
    _bars(axs[1], df, '%', f'{channel}: phase quadrature (Im) — % of increment',
          ref=100, col='phase_proj', ylabel='% of final increment')
    _bars(axs[2], ed, '%', f'{channel}: total TIW energy — % of increment',
          ref=100, col='pct_of_full', ylabel='% of final increment')
    _legend(fig)
    fig.suptitle(f'TIW {channel} phase+amplitude factor separation vs Vel — projection framing: '
                 f'% of the final increment d_full = c(Vel) - c(prior) each term accounts for (sum ~100%)')
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(f'{FIGS}/tiw_phaseamp_{channel}_projection.png', dpi=140); plt.close(fig)

    # envelope Hovmöllers |a_S^z| for dominant terms + the Vel-prior wave
    terms = TIW_MAP_TERMS
    fig, axs = plt.subplots(1, len(terms) + 1, figsize=(4.3 * (len(terms) + 1), 5), sharey=True)
    dfield = Z['Vel'] - Z['prior']
    X = dfield.XC.values; t = np.arange(dfield.sizes['time'])
    lim0 = np.nanpercentile(np.abs(dfield.values), 99)
    axs[0].pcolormesh(X, t, np.abs(dfield.values), vmin=0, vmax=lim0, cmap='magma')
    axs[0].set_title(f'|z_Vel − z_prior|\nRMS={_rms(np.abs(dfield.values)):.3g}')
    axs[0].set_ylabel('day of run')
    for ax, S in zip(axs[1:], terms):
        aS = _incl_excl(Z, S)
        env = np.abs(aS.values)
        ax.pcolormesh(X, t, env, vmin=0, vmax=np.nanpercentile(env, 99), cmap='magma')
        ax.set_title(f'|a[{term_label(S)}]|\nRMS={_rms(env):.3g}')
    axs[0].invert_yaxis()               # Hovmöller: time increases downward
    for ax in axs:
        ax.set_xlabel('lon (°E)')
    fig.suptitle(f'TIW {channel}: injected-wave envelope |a_S^z| (where/when each term adds TIW amplitude)')
    fig.tight_layout(); fig.savefig(f'{FIGS}/tiw_envelope_{channel}.png', dpi=140); plt.close(fig)
    return df


# ---- (2e) phase & amplitude TIME SERIES over the window (hypothesis test) ------
# Band-pass each run's Hovmöller, take the analytic field z(t,x), then per DAY:
#   amplitude(t) = RMS over lon of |z|  (band-passed envelope; the amplitude metric)
#   phase lag(t) = arg<ẑ_Vel(t), z_run(t)>_x  in days  (phase metric vs full solution)
#   coherence(t) = |<ẑ_Vel,z_run>| / ||z_run||  in [0,1]  (how wave-like/aligned)
# Resolves the hypothesis in time: θ should lock phase early; u wind should drive
# the amplitude growth; θ+u wind should amplify an already-correctly-phased wave.
PA_RUNS = [('prior', '0.5', 'prior (no DA)'), ('noVel', 'tab:orange', 'TPOSE6-noVel'),
           ('initialT', 'tab:green', 'θ (xx_theta)'),
           ('zonalWind', 'tab:purple', 'u wind (xx_uwind)'),
           ('initialT_uwind', 'tab:brown', 'θ+u wind'),
           ('top4', 'magenta', 'θ+S+u wind+sw'), ('Vel', 'k', 'Full (TPOSE6-Vel)')]


def tiw_pa_timeseries(channel, ref='Vel'):
    """Per-day TIW amplitude, phase lag & coherence vs a reference field.
    ref='Vel' (full-solution proxy, model-model) or 'obs' (band-passed observations)."""
    runs = [t for t in PA_RUNS if _tiw_src(t[0], channel)]   # noVel has no V field -> skipped
    Z = {r: _analytic_tiw(_tiw_hov(r, channel)) for r, _, _ in runs}
    zref = _analytic_tiw(_tiw_hov_obs(channel)) if ref == 'obs' else Z['Vel']
    Tdom = _tiw_period(zref)
    zR = zref.values                                           # (nt, nx)
    normR = np.sqrt((np.abs(zR) ** 2).sum(1))                  # ||z_ref(t)|| per day
    days = tiw.DAYS
    unit = 'm/s' if channel == 'V' else ('°C' if channel == 'SST' else 'm')
    out = {'date': days}
    fig, axs = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    if ref == 'obs':                                           # obs not a run -> draw it too
        axs[0].plot(days, np.sqrt(np.mean(np.abs(zR) ** 2, axis=1)), 'k', ls='--', lw=2.4,
                    label='observations', zorder=5)
    for r, color, lab in runs:
        z = Z[r].values
        amp = np.sqrt(np.mean(np.abs(z) ** 2, axis=1))         # envelope RMS over lon
        c = (np.conj(zR) * z).sum(1) / np.where(normR > 0, normR, np.nan)
        lag = np.angle(c) / (2 * np.pi) * Tdom                 # phase lag vs ref (days)
        znorm = np.sqrt((np.abs(z) ** 2).sum(1))
        coh = np.abs(c) / np.where(znorm > 0, znorm, np.nan)
        lw = 2.0 if r == 'Vel' else 1.3
        axs[0].plot(days, amp, color=color, label=lab, lw=lw)
        axs[1].plot(days, lag, color=color, lw=lw)
        axs[2].plot(days, coh, color=color, lw=lw)
        out[f'amp_{r}'] = amp; out[f'lag_{r}'] = lag; out[f'coh_{r}'] = coh
    axs[0].set_ylabel(f'TIW amplitude ({unit})\nband-passed envelope RMS')
    axs[1].axhline(0, color='k', lw=.6); axs[1].set_ylabel(f'phase lag vs {ref} (days)')
    axs[2].set_ylabel(f'coherence with {ref}  [0–1]'); axs[2].set_xlabel('date')
    for ax in axs:
        ax.grid(alpha=.3); ax.tick_params(axis='x', rotation=30)
    axs[0].legend(fontsize=8, ncol=3, loc='upper left')
    fig.suptitle(f'TIW {channel}: amplitude & phase evolution over the window vs {ref}  '
                 f'(band {TIW_BAND["per"]} d / {TIW_BAND["wav"]} km, Tdom={Tdom:.1f} d)\n'
                 'hypothesis: θ locks phase (low lag) early; u wind drives amplitude growth; '
                 'θ×u wind amplifies a correctly-phased wave')
    fig.tight_layout()
    d = f'{FIGS}/controlling_phase_amp'; os.makedirs(d, exist_ok=True)
    tag = '' if ref == 'Vel' else '_vsobs'
    fig.savefig(f'{d}/tiw_{channel}_amp_phase_timeseries{tag}.png', dpi=140); plt.close(fig)
    pd.DataFrame(out).to_csv(f'{d}/tiw_{channel}_amp_phase_timeseries{tag}.csv', index=False)


# runs scored against the observed TIW field, ordered prior/noVel -> controls -> combos -> Vel.
VS_OBS_RUNS = ['prior', 'noVel', 'initialT', 'initialS', 'zonalWind', 'swdown',
               'initialT_uwind', 'theta_salt', 'theta_salt_uwind', 'top4', 'Vel']


def _run_label(r):
    return {'prior': 'prior (no DA)', 'noVel': 'TPOSE6-noVel',
            'Vel': 'Full (Vel)'}.get(r) or tiw.CONTROL_XX.get(r, r)


def tiw_skill_vs_obs(channel):
    """Each run's TIW skill vs the band-passed OBSERVED field (SST=mw_fusion, SSH=RADS):
    coherence (complex pattern+phase match, 0-1), amplitude ratio ||z_run||/||z_obs||, and
    phase lag (days). Places noVel among the single/combo controls on a real obs metric."""
    zo = _analytic_tiw(_tiw_hov_obs(channel))
    Tdom = _tiw_period(zo)
    o = zo.values.ravel(); no = np.linalg.norm(o)
    runs = [r for r in VS_OBS_RUNS if _tiw_src(r, channel)]
    rows = []
    for r in runs:
        z = _analytic_tiw(_tiw_hov(r, channel)).values.ravel()
        c = np.vdot(o, z)                                      # <z_obs, z_run>
        rows.append(dict(run=r, label=_run_label(r),
                         coherence=abs(c) / (no * np.linalg.norm(z)),
                         amp_ratio=np.linalg.norm(z) / no,
                         lag_days=np.angle(c) / (2 * np.pi) * Tdom))
    df = pd.DataFrame(rows)
    d = f'{FIGS}/controlling_phase_amp'; os.makedirs(d, exist_ok=True)
    df.to_csv(f'{d}/tiw_{channel}_skill_vs_obs.csv', index=False)
    x = np.arange(len(df))
    colors = ['0.5' if r == 'prior' else 'tab:orange' if r == 'noVel'
              else 'k' if r == 'Vel' else 'tab:blue' for r in df.run]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, col, ttl, ref in [
            (axs[0], 'coherence', 'coherence with obs TIW  [0–1]', None),
            (axs[1], 'amp_ratio', 'amplitude ratio to obs  (1 = obs)', 1.0),
            (axs[2], 'lag_days', f'phase lag vs obs (days; Tdom={Tdom:.1f})', 0.0)]:
        ax.bar(x, df[col], color=colors)
        if ref is not None:
            ax.axhline(ref, color='k', ls='--', lw=1)
        ax.set_xticks(x); ax.set_xticklabels(df.label, rotation=90, fontsize=7)
        ax.set_title(ttl); ax.grid(axis='y', alpha=.3)
    src = 'mw_fusion SST' if channel == 'SST' else 'RADS SSH'
    fig.suptitle(f'TIW {channel} skill vs band-passed OBSERVATIONS ({src}; '
                 f'band {TIW_BAND["per"]} d / {TIW_BAND["wav"]} km, region {TIW_REGION}°E)')
    fig.tight_layout()
    fig.savefig(f'{d}/tiw_{channel}_skill_vs_obs.png', dpi=140); plt.close(fig)
    return df


# ---- (2g) depth-resolved TIW phase/amp vs the TAO current meters ---------------
TAO_PA_MOORINGS = {190.0: '170°W', 220.0: '140°W', 250.0: '110°W'}


def _tao_daily(arr, day, sel, nt):
    """Average selected profiles into a daily (nt, ndepth) grid; -9999/absent -> NaN."""
    out = np.full((nt, arr.shape[1]), np.nan)
    a = np.where(arr == -9999., np.nan, arr)
    for t in range(nt):
        m = sel & (day == t)
        if m.any():
            out[t] = np.nanmean(a[m], axis=0)
    return out


def _bandpass_time(x):
    """1D temporal band-pass to the TIW period band -> single-lobe analytic signal, per
    column. NaNs filled with 0 after per-column demean (gaps -> no injected TIW power)."""
    f = np.nan_to_num(x - np.nanmean(x, axis=0, keepdims=True))
    nu = np.fft.fftfreq(f.shape[0], 1.0)[:, None]
    plo, phi = TIW_BAND['per']
    keep = (np.abs(nu) >= 1 / phi) & (np.abs(nu) <= 1 / plo) & (nu > 0)
    return np.fft.ifft(np.fft.fft(f, axis=0) * keep, axis=0)


def _period_days(z):
    """Energy-weighted period (days) of a (nt, ncol) complex band-passed field."""
    P = (np.abs(np.fft.fft(z, axis=0)) ** 2).sum(1)
    nu = np.fft.fftfreq(z.shape[0], 1.0); m = nu > 0
    return 1.0 / (np.sum(nu[m] * P[m]) / np.sum(P[m]))


def tiw_tao_depth():
    """TIW meridional-velocity phase & amplitude vs the TAO current meters, resolved in
    DEPTH at each equatorial mooring (170/140/110°W). Temporal band-pass V(t) per depth,
    analytic signal, then amplitude(z), coherence(z) & phase lag(z) of each run vs TAO."""
    o = np.load(f'{CACHE}/obs_tao.npz')
    lon, lat, day, depth = o['lon'], o['lat'], o['day'], o['depth']
    wv = o['w_V']; obs_masked = np.where(wv > 0, o['obs_V'], -9999.)
    nt = tiw.NT
    pa = [t for t in PA_RUNS if t[0] != 'top4']          # top4 dropped from depth panels
    models = {r: np.load(f'{CACHE}/model_{r}_tao_V.npy') for r, _, _ in pa}
    rows = []
    fig, axs = plt.subplots(len(TAO_PA_MOORINGS), 3, figsize=(15, 12),
                            sharex='col', sharey=True)
    for i, (mlon, name) in enumerate(TAO_PA_MOORINGS.items()):
        sel = (np.abs(lon - mlon) < 0.6) & (np.abs(lat) < 0.6) & (day >= 0) & (day < nt)
        zo = _bandpass_time(_tao_daily(obs_masked, day, sel, nt))
        Tdom = _period_days(zo)
        no = np.sqrt((np.abs(zo) ** 2).sum(0))                     # ||z_obs(z)|| per depth
        axs[i, 0].plot(np.sqrt(np.mean(np.abs(zo) ** 2, 0)), depth, 'k', ls='--', lw=2.4,
                       label='TAO obs', zorder=5)
        for r, color, lab in pa:
            zm = _bandpass_time(_tao_daily(models[r], day, sel, nt))
            c = (np.conj(zo) * zm).sum(0)                          # <z_obs, z_run> per depth
            nm = np.sqrt((np.abs(zm) ** 2).sum(0))
            coh = np.abs(c) / np.where((no * nm) > 0, no * nm, np.nan)
            lag = np.angle(c) / (2 * np.pi) * Tdom
            lw = 2.0 if r == 'Vel' else 1.3
            axs[i, 0].plot(np.sqrt(np.mean(np.abs(zm) ** 2, 0)), depth, color=color, label=lab, lw=lw)
            axs[i, 1].plot(coh, depth, color=color, lw=lw)
            axs[i, 2].plot(lag, depth, color=color, lw=lw)
            for k, zd in enumerate(depth):
                rows.append(dict(mooring=name, depth=zd, run=r,
                                 amp=float(np.sqrt(np.mean(np.abs(zm[:, k]) ** 2))),
                                 coherence=float(coh[k]), lag_days=float(lag[k])))
        axs[i, 1].axvline(1, color='k', lw=.5, ls=':'); axs[i, 2].axvline(0, color='k', lw=.6)
        axs[i, 0].set_ylabel(f'{name}\ndepth (m)')
        axs[i, 0].set_title(f'{name}: TIW-V amplitude (m/s)'); axs[i, 1].set_title('coherence w/ TAO [0–1]')
        axs[i, 2].set_title(f'phase lag vs TAO (days; Tdom={Tdom:.1f})')
    axs[0, 0].set_ylim(-300, 0)
    for ax in axs[-1]:
        pass
    axs[0, 0].legend(fontsize=7, loc='lower right')
    fig.suptitle('TIW meridional-velocity phase & amplitude vs TAO current meters, by depth & longitude '
                 f'(temporal band {TIW_BAND["per"]} d)')
    fig.tight_layout()
    d = f'{FIGS}/controlling_phase_amp'; os.makedirs(d, exist_ok=True)
    fig.savefig(f'{d}/tiw_tao_depth_phase_amp.png', dpi=140); plt.close(fig)
    pd.DataFrame(rows).to_csv(f'{d}/tiw_tao_depth_phase_amp.csv', index=False)


# ---- (2h) wavelength-band sensitivity of the vs-obs phase skill ---------------
# DISJOINT bands (not nested): long wave = classic ~33-day TIW, short wave = the
# shorter-scale structure. Lets us see which scale regime each control governs.
SENS_BANDS = {'long 1100–1600 km': (1100, 1600), 'short 600–1100 km': (600, 1100)}


def tiw_band_sensitivity():
    """Per-run TIW skill vs obs in disjoint long- vs short-wave bands, SST + SSH:
    phase coherence (top row) AND amplitude ratio ||z||/||z_obs|| (bottom row)."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 9), sharex='col')
    rows = []
    for j, ch in enumerate(('SST', 'SSH')):
        hobs = _tiw_hov_obs(ch)
        runs = [r for r in VS_OBS_RUNS if _tiw_src(r, ch) and r != 'top4']
        hov = {r: _tiw_hov(r, ch) for r in runs}
        x = np.arange(len(runs)); w = 0.8 / len(SENS_BANDS)
        for bi, (blab, wav) in enumerate(SENS_BANDS.items()):
            band = dict(per=TIW_BAND['per'], wav=wav)
            o = _analytic_tiw(hobs, band).values.ravel(); no = np.linalg.norm(o)
            coh, amp = [], []
            for r in runs:
                z = _analytic_tiw(hov[r], band).values.ravel()
                coh.append(abs(np.vdot(o, z)) / (no * np.linalg.norm(z)))
                amp.append(np.linalg.norm(z) / no)
                rows.append(dict(channel=ch, band=blab, run=r, coherence=coh[-1], amp_ratio=amp[-1]))
            off = (bi - (len(SENS_BANDS) - 1) / 2) * w
            axs[0, j].bar(x + off, coh, w, label=blab)
            axs[1, j].bar(x + off, amp, w, label=blab)
        axs[0, j].set_title(f'{ch}: phase coherence with obs'); axs[0, j].set_ylabel('coherence [0–1]')
        axs[1, j].set_title(f'{ch}: amplitude ratio to obs'); axs[1, j].set_ylabel('amp ratio (1 = obs)')
        axs[1, j].axhline(1, color='k', ls='--', lw=1)
        for ax in (axs[0, j], axs[1, j]):
            ax.grid(axis='y', alpha=.3); ax.legend(fontsize=8)
            ax.set_xticks(x); ax.set_xticklabels([_run_label(r) for r in runs], rotation=90, fontsize=7)
    fig.suptitle(f'Wavelength-band sensitivity vs obs (disjoint bands, period {TIW_BAND["per"]} d): '
                 'phase coherence (top) + amplitude ratio (bottom)')
    fig.tight_layout()
    d = f'{FIGS}/controlling_phase_amp'; os.makedirs(d, exist_ok=True)
    fig.savefig(f'{d}/tiw_band_sensitivity.png', dpi=140); plt.close(fig)
    pd.DataFrame(rows).to_csv(f'{d}/tiw_band_sensitivity.csv', index=False)


def controlling_phase_amp():
    """Build the curated folder for the phase/amplitude-control hypothesis: generate the
    new time-series + vs-obs skill figures and copy in the existing TIW/EUC figures."""
    import shutil
    d = f'{FIGS}/controlling_phase_amp'; os.makedirs(d, exist_ok=True)
    for ch in ('V', 'SST'):
        tiw_pa_timeseries(ch, ref='Vel')
    for ch in ('SST', 'SSH'):                 # gridded obs only (TAO V too sparse for fields)
        tiw_pa_timeseries(ch, ref='obs')
        tiw_skill_vs_obs(ch)
    tiw_tao_depth()                            # depth-resolved V vs the TAO current meters
    tiw_band_sensitivity()                     # long- vs short-wave phase skill vs obs
    # existing figures that already address the hypothesis (window phase/amp split,
    # space-time envelopes, the θ×u-wind TIW field, and the u-wind→EUC-shear evidence)
    curated = [f'tiw_phaseamp_{c}.png' for c in ('V', 'SST')] \
        + [f'tiw_phaseamp_{c}_projection.png' for c in ('V', 'SST')] \
        + [f'tiw_envelope_{c}.png' for c in ('V', 'SST')] \
        + ['initialT_uwind_tiw.png', 'theta_salt_uwind_tiw.png', 'initialT_uwind_euc.png',
           'euc_anova.png', 'euc_anova_projection.png', 'euc_anova_bylon.png']
    copied = []
    for f in curated:
        src = f'{FIGS}/{f}'
        if os.path.exists(src):
            shutil.copy(src, f'{d}/{f}'); copied.append(f)
    print(f'-> controlling_phase_amp: timeseries vsVel(V,SST) + vsObs(SST,SSH) + '
          f'skill_vs_obs(SST,SSH) + tao_depth + copied {len(copied)} existing figs')
    return copied


# ---- (3) gridded interaction field map (any order, incl. residual) ----------
def _grid_timemean(S, field):
    """Inclusion-exclusion over the cached gridded field, time-averaged -> (YC,XC)."""
    if S == 'residual':
        full = xr.open_dataarray(f'{CACHE}/grid_Vel_{field}.nc').mean('time')
        top4 = xr.open_dataarray(f'{CACHE}/grid_{RUN[frozenset(FACTORS)]}_{field}.nc').mean('time')
        return full - top4, xr.open_dataarray(f'{CACHE}/grid_Vel_{field}.nc').mean('time') - \
            xr.open_dataarray(f'{CACHE}/grid_prior_{field}.nc').mean('time')
    S = frozenset(S)
    g = {T: xr.open_dataarray(f'{CACHE}/grid_{RUN[T]}_{field}.nc').mean('time')
         for k in range(len(S) + 1) for T in map(frozenset, itertools.combinations(S, k))}
    a = sum((-1) ** (len(S) - len(T)) * g[T] for T in g)
    dfull = xr.open_dataarray(f'{CACHE}/grid_Vel_{field}.nc').mean('time') - g[frozenset()]
    return a, dfull


def fig_grid_map(terms, field='sst'):
    unit = '°C' if field == 'sst' else 'm'
    _, dfull = _grid_timemean(terms[0], field)
    panels = [(dfull, 'd_full  m_Vel-m_prior')]
    for S in terms:
        a, _ = _grid_timemean(S, field)
        panels.append((a, 'residual r' if S == 'residual' else f'a[{term_label(S)}]'))
    lim = max(np.nanpercentile(np.abs(A.values), 99) for A, _ in panels)   # shared scale
    fig, axs = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.2),
                            sharex=True, sharey=True, constrained_layout=True)
    for ax, (A, lab) in zip(axs, panels):
        pc = ax.pcolormesh(A.XC, A.YC, A.values, cmap='RdBu_r', vmin=-lim, vmax=lim)
        ax.set_title(f'{lab}\n(RMS={_rms(A.values):.3f} {unit})')
        ax.set_xlabel('lon'); ax.axhline(0, color='k', lw=.5, ls=':')
    axs[0].set_ylabel('lat')
    fig.colorbar(pc, ax=list(axs), shrink=.8, label=f'{field.upper()} increment ({unit})')
    fig.suptitle(f'{field.upper()} time-mean interaction fields (lat×lon)')
    name = '_'.join('resid' if S == 'residual'
                    else term_label(S).replace('×', '').replace(' ', '') for S in terms)
    fig.savefig(f'{FIGS}/map_{name}_{field}.png', dpi=140); plt.close(fig)


# ---- (4) 3D field decomposition of a dominant pair (EUC U, TIW V) -----------
def _mask(ds, v):
    return ds[v].where(tiw.wet_mask(ds, ds[v]))


def _has_3d(run):
    import glob
    return bool(glob.glob(f'{tiw.RUNS[run][0]}/diag_state.*.data'))


def _euc_tiv_from_disk(run):
    """Reduce a run's 3D diag_state to the two ANOVA fields (heavy: streams UVEL/VVEL)."""
    ds = tiw.load_run(run)
    u = _mask(ds, 'UVEL').sel(YC=slice(-0.5, 0.5)).mean('YC').mean('time')          # (Z,XG)
    v = (_mask(ds, 'VVEL').sel(YG=slice(*TIW_LAT)).isel(Z=slice(0, TIW_ZTOP))
         .mean('YG').mean('Z'))                                                      # (time,XC)
    out = xr.Dataset({'euc': u, 'tiv': v}).compute()
    return out['euc'], out['tiv']


def cache_3d_fields(runs):
    """Persist each run's reduced EUC/TIW fields so diag_state can be deleted after.
    These are tiny (lon×depth, lon×time) vs the full 3D output."""
    for run in runs:
        if not _has_3d(run):
            print(f'  skip {run}: no diag_state on disk', flush=True)
            continue
        euc, tiv = _euc_tiv_from_disk(run)
        xr.Dataset({'euc': euc, 'tiv': tiv}).to_netcdf(f'{CACHE}/euctiw_{run}.nc')
        print(f'  cached {run} -> euctiw_{run}.nc', flush=True)


def _load_3d(runs):
    """EUC/TIW field dicts, preferring the persistent cache over 3D diag_state."""
    euc, tiv = {}, {}
    for run in runs:
        c = f'{CACHE}/euctiw_{run}.nc'
        if os.path.exists(c):
            ds = xr.open_dataset(c)
            euc[run], tiv[run] = ds['euc'], ds['tiv']
            print(f'  loaded {run} (cache)', flush=True)
        else:
            euc[run], tiv[run] = _euc_tiv_from_disk(run)
            print(f'  loaded {run} (diag_state)', flush=True)
    return euc, tiv


# field kind -> plotting config. euc field dims (Z,XG); tiw field dims (time,XC).
KIND = {
    'euc': dict(xc='XG', yc='Z',   lon=EUC_LON, ylim=(-300, 0), ylab='depth (m)',
                fmt='.3f', name='EUC (time-mean equatorial U)'),
    'tiw': dict(xc='XC', yc='time', lon=TIW_LON, ylim=None,     ylab='day of run',
                fmt='.4f', name='TIW V (2–5°N, top 50 m), lon×time'),
}


def _field_anova(S, f):
    """EUC/TIW field decomposition for factor set S (clean incl-exclusion, any order).
    Returns d_S (combined incr), {i: main d_i}, a_S (irreducible |S|-way interaction)."""
    prior = f['prior']
    dS = f[RUN[frozenset(S)]] - prior
    mains = {i: f[RUN[frozenset(i)]] - prior for i in S}
    aS = sum((-1) ** (len(S) - len(T)) * f[RUN[frozenset(T)]]
             for k in range(len(S) + 1) for T in itertools.combinations(S, k))
    return dS, mains, aS


def _slice(F, cfg):
    """Return (x, y, 2D array) for a field restricted to the kind's lon window."""
    X = F[cfg['xc']].values
    xl = (X >= cfg['lon'][0]) & (X <= cfg['lon'][1])
    Y = F[cfg['yc']].values if cfg['yc'] == 'Z' else np.arange(F.sizes['time'])
    return X[xl], Y, F.values[:, xl]


def fig_decomp(S, f, kind):
    """Order-agnostic fixed-increment field decomposition (pair/triad/4-way)."""
    cfg = KIND[kind]
    S = ''.join(sorted(set(S), key='TSUW'.index))
    dS, mains, aS = _field_anova(S, f)
    summ = sum(mains.values())
    # combined = this term's factors on together; additive = what you'd get if they
    # didn't interact (sum of each-alone); interaction = combined minus that.
    inc = ([(dS, f'combined ({term_label(S)})')]
           + [(mains[i], f'{SYM[i]} alone') for i in S]
           + [(summ, 'additive (sum of each alone)')])
    if len(S) >= 3:
        inc += [(dS - summ, 'interaction total (combined − additive)')]
    aS_lab = f'{ORDER_NAME[len(S)]} interaction' if len(S) > 1 else 'main'
    panels = inc + [(aS, aS_lab)]
    cols = [(_slice(F, cfg)[2], lab) for F, lab in panels]  # (2D array, label)
    xi, Y, _ = _slice(dS, cfg)
    lim = max(np.nanpercentile(np.abs(A), 99) for A, _ in cols)   # shared across all panels
    n = len(panels)
    fig, axs = plt.subplots(1, n, figsize=(3.7 * n, 4.8), sharey=True, constrained_layout=True)
    for ax, (A, lab) in zip(axs, cols):
        pc = ax.pcolormesh(xi, Y, A, cmap='RdBu_r', vmin=-lim, vmax=lim, shading='auto')
        ax.set_title(f'{lab}\n(RMS={_rms(A):{cfg["fmt"][1:]}} m/s)'); ax.set_xlabel('lon (deg E)')
    if cfg['ylim']:
        axs[0].set_ylim(*cfg['ylim'])
    if cfg['yc'] == 'time':
        axs[0].invert_yaxis()           # Hovmöller: time increases downward
    axs[0].set_ylabel(cfg['ylab'])
    fig.colorbar(pc, ax=list(axs), shrink=.8, label='m/s')
    fig.suptitle(f'{cfg["name"]}: {term_label(S)} fixed-increment decomposition  '
                 f'(rightmost = {ORDER_NAME[len(S)]} interaction)')
    fig.savefig(f'{FIGS}/{RUN[frozenset(S)]}_{kind}.png', dpi=140); plt.close(fig)


if __name__ == '__main__':
    import sys
    os.makedirs(FIGS, exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'budget'

    if cmd in ('budget', 'all'):
        df = budget_table()
        print(df.to_string(index=False))
        fig_heatmap(); fig_budget_bars(df)
        dfp = budget_projection_table()
        print('\nprojection shares (% of d_full, sum to ~100):')
        print(dfp.to_string(index=False))
        print('column sums:', dfp.drop(columns=['term', 'order']).sum().round(1).to_dict())
        fig_budget_bars(
            dfp, ylabel='% of final increment  <a_S,d_full>/||d_full||²',
            suptitle=('Full factorial budget, projection framing: signed share of the final '
                      'increment d_full = m_Vel - m_prior that each term accounts for\n'
                      '(beta_S = <a_S,d_full>/<d_full,d_full>; terms + residual sum to ~100%)'),
            fname='budget_bars_projection.png', ylim=(-12, 100))
        print('-> factorial_budget{,_projection}.csv, heatmap.png, '
              'budget_bars{,_projection}.png')

    if cmd == 'cache3d':
        # persist reduced EUC/TIW fields before diag_state is deleted. Default:
        # every run that currently has diag_state on disk.
        runs = sys.argv[2:] or [r for r in tiw.RUNS if _has_3d(r)]
        print(f'caching EUC/TIW for {runs}')
        cache_3d_fields(runs)
        print(f'-> {CACHE}/euctiw_*.nc')

    if cmd in ('euc', 'all'):
        df = euc_metric_anova()
        print(df.pivot(index='term', columns='metric', values='a').to_string())
        euc_metric_anova_bylon()
        print('-> euc_anova{,_bylon}.csv, euc_metrics{,_bylon}_raw.csv, euc_anova{,_bylon}.png')

    if cmd in ('tiw', 'all'):
        for ch in ('V', 'SST'):
            tiw_phase_amp(ch)
        print('-> tiw_phaseamp_{V,SST}.{png,csv}, tiw_phase_{V,SST}.csv, tiw_envelope_{V,SST}.png')

    if cmd in ('phaseamp', 'all'):
        # curated phase/amplitude-control folder: needs the euc + tiw figures to exist
        # first, so run after those (or via 'all').
        controlling_phase_amp()

    if cmd in ('maps', 'all'):
        # dominant higher-order terms + residual, SST (largest interaction) and SSH.
        fig_grid_map(['TU', 'TS', 'TSU', 'TSUW'], field='sst')
        fig_grid_map(['residual'], field='sst')
        fig_grid_map(['residual'], field='ssh')
        print('-> SST/SSH interaction maps')

    if cmd in ('fields', 'pairs', 'all'):
        # EUC/TIW field decomposition for any-order terms (default: all 6 pairs +
        # 4 triads + the 4-way). A term needs its run + ALL subset runs cached.
        valid = {''.join(sorted(c, key='TSUW'.index))
                 for k in range(2, len(FACTORS) + 1)
                 for c in itertools.combinations(FACTORS, k)}
        args = [''.join(sorted(a, key='TSUW'.index)) for a in sys.argv[2:]]
        terms = [a for a in args if a in valid] or sorted(valid, key=lambda s: (len(s), s))
        need = {'prior'}
        for t in terms:
            need |= {RUN[frozenset(c)] for k in range(1, len(t) + 1)
                     for c in itertools.combinations(t, k)}
        # a run is usable if its reduced EUC/TIW fields are cached or diag_state is on disk.
        missing = sorted({r for r in need
                          if not (os.path.exists(f'{CACHE}/euctiw_{r}.nc') or _has_3d(r))})
        if missing:
            print(f'EUC/TIW skipped: no cached fields or diag_state for {missing}. '
                  f'Cache them with `anova.py cache3d {" ".join(missing)}` while diag_state '
                  f'is on disk, or re-run those models.')
            sys.exit(0)
        euc, tiv = _load_3d(sorted(need))
        for t in terms:
            fig_decomp(t, euc, 'euc')
            fig_decomp(t, tiv, 'tiw')
        print(f'-> EUC/TIW decomposition for {terms}')

    if cmd == 'check':
        # fundamental identity (per obs point, one common mask across all runs):
        #   Σ_{nonempty S} a_S == m_top4 - m_prior   and   d_top4 + r == d_full.
        for key, name in PRODS:
            runs = {r: _load_obs(key, r).ravel() for r in
                    ['prior', 'Vel'] + [RUN[S] for S in subsets()]}
            v = np.all([np.isfinite(x) for x in runs.values()], axis=0)
            m = {r: runs[r][v] for r in runs}
            a_sum = sum(
                sum((-1) ** (len(S) - len(T)) * m[RUN[frozenset(T)]]
                    for k in range(len(S) + 1)
                    for T in itertools.combinations(S, k))
                for S in subsets())
            d_top4 = m[RUN[frozenset(FACTORS)]] - m['prior']
            r = m['Vel'] - m[RUN[frozenset(FACTORS)]]
            d_full = m['Vel'] - m['prior']
            e1 = np.max(np.abs(a_sum - d_top4))
            e2 = np.max(np.abs((d_top4 + r) - d_full))
            print(f'{name:7s} max|Σa_S - d_top4|={e1:.2e}  max|d_top4+r - d_full|={e2:.2e}')
        print('identity check done')
