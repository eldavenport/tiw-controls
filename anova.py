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
SYM = {'T': 'θ', 'S': 'S', 'U': 'u', 'W': 'sw'}
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


# ---- (2) controls x controls heatmap (main effects + pairwise) --------------
def fig_heatmap():
    n = len(FACTORS)
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (key, name) in zip(axs.flat, PRODS):
        M = np.full((n, n), np.nan)
        for i, fi in enumerate(FACTORS):
            for j, fj in enumerate(FACTORS):
                S = frozenset(fi) if i == j else frozenset(fi + fj)
                try:
                    a, _, dfull = anova_term(S, key)
                    M[i, j] = 100 * _rms(a) / _rms(dfull) if _rms(dfull) else np.nan
                except FileNotFoundError:
                    pass
        im = ax.imshow(M, cmap='viridis', vmin=0)
        ax.set_xticks(range(n)); ax.set_xticklabels([SYM[f] for f in FACTORS])
        ax.set_yticks(range(n)); ax.set_yticklabels([SYM[f] for f in FACTORS])
        for i in range(n):
            for j in range(n):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f'{M[i, j]:.0f}', ha='center', va='center',
                            color='w' if M[i, j] < 0.6 * np.nanmax(M) else 'k', fontsize=10)
        ax.set_title(f'{name}'); fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle('Interaction matrix  ||a_S|| / ||d_full||  (%)   —   diagonal = main effect, '
                 'off-diagonal = pairwise interaction\n(3-way / 4-way / residual in budget_bars.png)')
    fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=3)
    fig.savefig(f'{FIGS}/heatmap.png', dpi=140); plt.close(fig)


# ---- (2b) all-orders + residual budget bars ---------------------------------
def fig_budget_bars(df):
    cmap = {1: 'tab:green', 2: 'tab:orange', 3: 'tab:red', 4: 'tab:purple', 'resid': '0.4'}
    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    for ax, (_, name) in zip(axs.flat, PRODS):
        x = np.arange(len(df)); colors = [cmap[o] for o in df['order']]
        ax.bar(x, df[name], color=colors)
        ax.set_xticks(x); ax.set_xticklabels(df['term'], rotation=90, fontsize=7)
        ax.set_title(name); ax.set_ylabel('% of ||d_full||')
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[o]) for o in [1, 2, 3, 4, 'resid']]
    fig.legend(handles, ['main', 'pair', '3-way', '4-way', 'residual'],
               loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('Full factorial budget: every ANOVA term + excluded-controls residual, '
                 'as share of the full-solution increment ||d_full|| = ||m_Vel - m_prior||')
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(f'{FIGS}/budget_bars.png', dpi=140); plt.close(fig)


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


def _bars(ax, d, unit, title, ref=None):
    ax.bar(np.arange(len(d)), d['a'], color=[CMAP[o] for o in d['order']])
    ax.axhline(0, color='k', lw=.6)
    if ref is not None:                               # final Vel change = Σ all bars
        ax.axhline(ref, color='k', ls='--', lw=1.3, zorder=5)
    ax.set_xticks(np.arange(len(d))); ax.set_xticklabels(d['term'], rotation=90, fontsize=7)
    ax.set_title(title); ax.set_ylabel(f'signed contribution ({unit})')


def _legend(fig):
    h = [plt.Rectangle((0, 0), 1, 1, color=CMAP[o]) for o in [1, 2, 3, 4, 'resid']]
    h += [plt.Line2D([0], [0], color='k', ls='--', lw=1.3)]
    fig.legend(h, ['main', 'pair', '3-way', '4-way', 'residual', 'Vel−prior (total)'],
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
    return df


# ---- (2d) TIW phase + amplitude decomposition (vs Vel) ------------------------
# Band-pass each run's TIW Hovmöller to the westward TIW band, take the analytic
# (complex) signal z(x,t), project onto Vel's normalized TIW field: c = <ẑ_Vel, z>.
# Band-pass+Hilbert+projection are LINEAR, so c factor-separates EXACTLY:
#   Re(a_S) = amplitude contribution IN PHASE with Vel; Im(a_S) = quadrature (phase shift).
# Complemented by a nonlinear total-TIW-energy scalar (net wave activity incl. cancellation).
TIW_BAND = dict(per=(15, 40), wav=(1000, 1600))   # days, km (period, zonal wavelength)
TIW_REGION = (180, 265)                            # cold-tongue lon window (all-ocean)
TIW_SST_LAT = (0, 5)                               # SST TIW band (cusps on cold-tongue N flank)
TIW_MAP_TERMS = ['TU', 'TS', 'TSU']                # dominant terms for envelope Hovmöllers


def _tiw_hov(run, channel):
    """TIW Hovmöller (time, XC) over the cold-tongue window: V (cached tiv) or SST band-mean."""
    if channel == 'V':
        h = xr.open_dataset(f'{CACHE}/euctiw_{run}.nc')['tiv']
    else:
        s = xr.open_dataarray(f'{CACHE}/grid_{run}_sst.nc')
        h = s.sel(YC=slice(*TIW_SST_LAT)).mean('YC')
    return h.sel(XC=slice(*TIW_REGION))


def _analytic_tiw(h):
    """2D-FFT band-pass to the westward TIW band, single spectral lobe -> complex analytic field."""
    f = h.values.astype(float)
    f = np.nan_to_num(f - np.nanmean(f, axis=0, keepdims=True))   # demean per lon, fill NaN
    nt, nx = f.shape
    nu = np.fft.fftfreq(nt, 1.0)[:, None]                         # cpd
    ka = np.fft.fftfreq(nx, float(np.mean(np.diff(h.XC))) * 111.32)[None, :]   # cpkm
    (plo, phi), (wlo, whi) = TIW_BAND['per'], TIW_BAND['wav']
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
    for ax in axs:
        ax.set_xlabel('lon (°E)')
    fig.suptitle(f'TIW {channel}: injected-wave envelope |a_S^z| (where/when each term adds TIW amplitude)')
    fig.tight_layout(); fig.savefig(f'{FIGS}/tiw_envelope_{channel}.png', dpi=140); plt.close(fig)
    return df


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
    fig, axs = plt.subplots(1, len(terms) + 1, figsize=(4.6 * (len(terms) + 1), 4.2),
                            sharex=True, sharey=True)
    _, dfull = _grid_timemean(terms[0], field)
    lim0 = np.nanpercentile(np.abs(dfull.values), 99)
    pc = axs[0].pcolormesh(dfull.XC, dfull.YC, dfull.values, cmap='RdBu_r',
                           vmin=-lim0, vmax=lim0)
    axs[0].set_title(f'd_full  m_Vel-m_prior\n(RMS={_rms(dfull.values):.3f} {unit})')
    fig.colorbar(pc, ax=axs[0], shrink=.8)
    for ax, S in zip(axs[1:], terms):
        a, _ = _grid_timemean(S, field)
        lab = 'residual r' if S == 'residual' else f'a[{term_label(S)}]'
        lim = np.nanpercentile(np.abs(a.values), 99)
        pc = ax.pcolormesh(a.XC, a.YC, a.values, cmap='RdBu_r', vmin=-lim, vmax=lim)
        ax.set_title(f'{lab}\n(RMS={_rms(a.values):.3f} {unit})')
        fig.colorbar(pc, ax=ax, shrink=.8)
    for ax in axs:
        ax.set_xlabel('lon'); ax.axhline(0, color='k', lw=.5, ls=':')
    axs[0].set_ylabel('lat')
    fig.suptitle(f'{field.upper()} time-mean interaction fields (lat×lon)')
    name = '_'.join('resid' if S == 'residual' else term_label(S).replace('×', '') for S in terms)
    fig.tight_layout(); fig.savefig(f'{FIGS}/map_{name}_{field}.png', dpi=140); plt.close(fig)


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
    # shared-scale increment panels; aS gets its own (smaller) scale.
    inc = [(dS, 'd[combined]')] + [(mains[i], f'd[{SYM[i]}]') for i in S] + [(summ, 'Σ mains')]
    if len(S) >= 3:                                   # for pairs total interaction == a_ij
        inc += [(dS - summ, 'total interaction')]
    panels = inc + [(aS, f'{len(S)}-way a' if len(S) > 1 else 'main')]
    cols = [(_slice(F, cfg)[2], lab) for F, lab in panels]  # (2D array, label)
    xi, Y, _ = _slice(dS, cfg)
    lim = max(np.nanpercentile(np.abs(A), 99) for A, _ in cols[:-1])
    alim = np.nanpercentile(np.abs(cols[-1][0]), 99)
    n = len(panels)
    fig, axs = plt.subplots(1, n, figsize=(3.7 * n, 4.8), sharey=True)
    for k, (ax, (A, lab)) in enumerate(zip(axs, cols)):
        L = alim if k == n - 1 else lim
        pc = ax.pcolormesh(xi, Y, A, cmap='RdBu_r', vmin=-L, vmax=L, shading='auto')
        ax.set_title(f'{lab}\n(RMS={_rms(A):{cfg["fmt"][1:]}} m/s)'); ax.set_xlabel('lon (deg E)')
        fig.colorbar(pc, ax=ax, shrink=.8)
    if cfg['ylim']:
        axs[0].set_ylim(*cfg['ylim'])
    axs[0].set_ylabel(cfg['ylab'])
    fig.suptitle(f'{cfg["name"]}: {term_label(S)} fixed-increment decomposition  '
                 f'(rightmost = irreducible {len(S)}-way interaction)')
    fig.tight_layout(); fig.savefig(f'{FIGS}/{RUN[frozenset(S)]}_{kind}.png', dpi=140); plt.close(fig)


if __name__ == '__main__':
    import sys
    os.makedirs(FIGS, exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'budget'

    if cmd in ('budget', 'all'):
        df = budget_table()
        print(df.to_string(index=False))
        fig_heatmap(); fig_budget_bars(df)
        print(f'-> factorial_budget.csv, heatmap.png, budget_bars.png')

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
