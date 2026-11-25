"""
Model-data misfit (chi^2 per observation) for prior / noVel / Vel across all
four assimilation windows and all observation types.

chi^2/obs = sum_i w_i (o_i - m_i)^2 / N,  w = 1/sigma^2  (framing A, no mult).
  - 1.0  => solution fits that obs type to within its stated error bar.
  - profiles: w = prof_*weight (MITprof).  SST/SSH: w = 1/err^2 (ECCO grids).
The MITgcm mult_profiles / mult_gencost factors are NOT applied: they are an
assimilation weighting choice, not a data uncertainty, so they drop out of a
pure data-misfit statement. Same estimator for every solution x window x obs
type -> apples-to-apples.  GLD excluded (mult_profiles=0 everywhere).

Usage: misfit_cost.py            # compute (if no cache) + plot
       misfit_cost.py compute    # (re)compute the CSV
       misfit_cost.py plot       # plot from the CSV

Env: ~/miniforge3/envs/tpose/bin/python
"""
import os
# cap BLAS/OpenMP thread pools BEFORE numpy import: on the big multi-core box the
# default pools spawn ~250 threads that thrash on a futex during field loads.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import interpn
import dask
dask.config.set(scheduler='synchronous')   # xmitgcm field loads: avoid dask thread thrash
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import tiw

CSV = f'{tiw.CACHE}/misfit_cost.csv'
FIG = 'figs/comparison/misfit_cost_allwindows.png'
LAT_LIMIT = 17.0   # assimilation domain is 17S-17N (matches the ECCO error grids);
#                    score every obs type only within this band for consistency.
ROLES = ['prior', 'noVel', 'Vel']   # sampled/cached (prior still computed)
# prior is the *Vel* optimization background, not a baseline noVel shares, so it
# is not a fair cross-solution comparison here -> display noVel vs Vel only.
PLOT_ROLES = ['noVel', 'Vel']
NTY = {2012: 366, 2013: 365}            # days in the RADS/SST yearly files

_C = '/data/SO6/TPOSE/constraints/profiles'
_TAO = '/data/SO3/edavenport/tpose6/tao_profiles'
_SO3 = '/data/SO3/edavenport/tpose6/diags'
_SO6 = '/data/SO6/TPOSE_diags/tpose6'


def _tao(stem, suf2012):            # 2012 files carry _v2/_ED; 2013 are plain
    return lambda y: f'{_TAO}/TAO_WO_{y}_{stem}{suf2012 if y == 2012 else ""}.nc'


# one entry per obs *file*; each emits a bar per variable.
PROFILE_PRODUCTS = [
    dict(group='Argo', vars=['T', 'S'], path=lambda y: f'{_C}/USGO_TP_{y}_PFL_D_w.nc',
         names={'T': 'Argo T', 'S': 'Argo S'}),
    dict(group='TAO', vars=['T', 'S'],       # _ED exists for both 2012 and 2013
         path=lambda y: f'{_TAO}/TAO_WO_{y}_CTD_daily_ED.nc',
         names={'T': 'TAO T', 'S': 'TAO S'}),
    dict(group='CTD', vars=['T', 'S'], path=lambda y: f'{_C}/HDO_TP_{y}_CTD_w.nc',
         names={'T': 'CTD T', 'S': 'CTD S'}),
    dict(group='CTD', vars=['T', 'S'], path=lambda y: f'{_C}/WOD_TP_{y}_CTD_OSD_w.nc',
         names={'T': 'WOD CTD T', 'S': 'WOD CTD S'}),
    dict(group='NODC', vars=['T', 'S'], path=lambda y: f'{_C}/NODC_TP_{y}_MRB_w.nc',
         names={'T': 'NODC T', 'S': 'NODC S'}),
    dict(group='XBT', vars=['T'], path=lambda y: f'{_C}/WOD_TP_{y}_XBT_w.nc',
         names={'T': 'WOD XBT T'}),
    dict(group='XBT', vars=['T'], path=lambda y: f'{_C}/SIO_TP_{y}_XBT_w.nc',
         names={'T': 'SIO XBT T'}),
    dict(group='TAO', vars=['U', 'V'], path=_tao('ADCP', '_v2'),
         names={'U': 'TAO ADCP U', 'V': 'TAO ADCP V'}),
    dict(group='TAO', vars=['U', 'V'], path=_tao('CUR', '_v2'),
         names={'U': 'TAO CUR U', 'V': 'TAO CUR V'}),
]

WINDOWS = {
    'sep2012': dict(ref='2012-09-01', years=[2012], runs={
        'prior': (f'{_SO3}/sep2012/run_iter0', 72, 1200),
        # canonical noVel (matches vel-assim-manuscript/VelocityAssim), 48 it/file
        'noVel': (f'{_SO6}/sep2012/diags_iter7_daily', 48, 1800),
        'Vel':   (f'{_SO3}/sep2012/run_iter14', 72, 1200)}),
    'nov2012': dict(ref='2012-11-01', years=[2012, 2013], runs={
        'prior': (f'{_SO3}/nov2012/run_iter0', 72, 1200),
        'noVel': (f'{_SO6}/nov2012/diags', 72, 1200),
        'Vel':   (f'{_SO3}/nov2012/run_iter20', 72, 1200)}),
    'jan2013': dict(ref='2013-01-01', years=[2013], runs={
        'prior': (f'{_SO3}/jan2013/run_iter0', 72, 1200),
        # jan2013/diags is averdy 0600 (unreadable); diags_daily is the same run, readable.
        'noVel': (f'{_SO6}/jan2013/diags_daily', 72, 1200),
        'Vel':   (f'{_SO3}/jan2013/run_iter14', 72, 1200)}),
    'mar2013': dict(ref='2013-03-01', years=[2013], runs={
        'prior': (f'{_SO3}/mar2013/run_iter0', 72, 1200),
        # canonical noVel in averdy's SO3 tree (readable; SO6 .../mar2013/diags is 0600-locked)
        'noVel': ('/data/SO3/averdy/TPOSE6/mar2013/diags_daily', 72, 1200),
        'Vel':   (f'{_SO3}/mar2013/run_iter16', 72, 1200)}),
}

# noVel prior (iter0) per window — the noVel counterpart to the Vel prior (the 'prior'
# role above). Scattered across two trees; sep2012 noVel iter0 == Vel prior (same run).
# Used by the iter-0 bar plot (compute_noVel_prior) and profile_windows.
NOVEL_PRIOR = {
    'sep2012': WINDOWS['sep2012']['runs']['prior'],
    'nov2012': ('/data/SO3/averdy/TPOSE6/nov2012/diags_iter0', 72, 1200),
    'jan2013': (f'{_SO6}/jan2013/diags_iter0', 72, 1200),
    'mar2013': (f'{_SO6}/mar2013/diags_iter0', 72, 1200),
}


# ---- compute -----------------------------------------------------------------
def _load(win, prefix):
    """Load each run's output; skip (with a warning) any run whose files are
    unreadable (e.g. averdy 0600-locked noVel diags) rather than aborting."""
    w, out = WINDOWS[win], {}
    for r, s in w['runs'].items():
        try:
            out[r] = tiw.load_run(r, (prefix,), ref_date=w['ref'], spec=s)
        except (OSError, PermissionError) as e:
            print(f'  SKIP {win}/{r} {prefix}: unreadable ({type(e).__name__})', flush=True)
    return out


def _prep_field(ds, var):
    """Preload a run's full model field into memory with dry cells NaN'd and the
    Z axis ascending, ready for scipy.interpn. Loaded once per run per field."""
    fld = ds[tiw.FIELD[var][0]]
    xn = next(c for c in fld.dims if c in ('XC', 'XG'))
    yn = next(c for c in fld.dims if c in ('YC', 'YG'))
    arr = np.where(tiw.wet_mask(ds, fld).values[None], fld.values, np.nan)  # (t,Z,Y,X)
    z = ds['Z'].values
    o = np.argsort(z)                                 # Z is descending -> ascending
    return dict(arr=arr[:, o], Z=z[o], Y=ds[yn].values, X=ds[xn].values,
                days=pd.to_datetime(ds['time'].values).floor('D'))


def _sample(prep, obs):
    """Linear-interp the preloaded field to obs (iPROF, iDEPTH). Vectorized per day."""
    lon, lat = obs['lon'].values, obs['lat'].values
    zneg, oday = obs['depth'].values, pd.to_datetime(obs['time'].values).floor('D')
    D = zneg.size
    out = np.full((lon.size, D), np.nan)
    axes = (prep['Z'], prep['Y'], prep['X'])
    for di, day in enumerate(prep['days']):
        sel = np.where(oday == day)[0]
        if sel.size == 0:
            continue
        q = np.empty((sel.size * D, 3))
        q[:, 0] = np.tile(zneg, sel.size)
        q[:, 1] = np.repeat(lat[sel], D)
        q[:, 2] = np.repeat(lon[sel], D)
        v = interpn(axes, prep['arr'][di], q, method='linear',
                    bounds_error=False, fill_value=np.nan)
        out[sel] = v.reshape(sel.size, D)
    return out


S_FLOOR = 28.0   # psu; casts dipping below this are corrupted (see _good_profiles)


def _good_profiles(obs):
    """Per-cast QC mask (True = keep): drop whole profiles with non-physical
    salinity (any valid S < S_FLOOR). Removes a cluster of corrupted WOD-CTD
    warm-pool casts (12-15N, 140-143E, 2013) whose surface S collapses to ~20 psu
    and otherwise dominates the CTD-S cost. Model-independent (data QC, not a
    misfit clip); genuine tropical-Pacific Argo fresh lenses bottom out ~28.3 psu,
    so the floor touches no legitimate profile. Omits the whole cast (T and S)."""
    if 'prof_S' not in obs:
        return np.ones(obs['lat'].size, bool)
    S, w = obs['prof_S'].values, obs['prof_Sweight'].values
    return np.where((S != -9999.) & (w > 0), S, np.inf).min(1) >= S_FLOOR


def _profiles(rows, win, state):
    # open every obs file for this window once (obs are solution-independent)
    obs_sets = []        # (spec, obs, good-cast mask)
    for spec in PROFILE_PRODUCTS:
        for y in WINDOWS[win]['years']:
            p = spec['path'](y)
            if os.path.exists(p):
                obs = tiw.open_obs(p)
                obs_sets.append((spec, obs, _good_profiles(obs)))
            else:
                print(f'  {win} skip missing {os.path.basename(p)}', flush=True)
    acc = {}             # (role, name, group) -> [sum_wsq, n]
    for r in state:                         # only roles that loaded
        ds, prep = state[r], {}
        for spec, obs, good in obs_sets:
            for v in spec['vars']:
                if v not in prep:                      # load each field once per run
                    prep[v] = _prep_field(ds, v)
                man = _sample(prep[v], obs)
                keep = (np.abs(obs['lat'].values) <= LAT_LIMIT) & good   # domain + QC
                wt0 = np.where(keep[:, None], obs[f'prof_{v}weight'].values, 0.0)
                o, m, wt = tiw._mask(obs[f'prof_{v}'].values, man, wt0)
                a = acc.setdefault((r, spec['names'][v], spec['group']), [0.0, 0])
                a[0] += float(np.sum(wt * (o - m) ** 2))
                a[1] += o.size
        del prep                                        # free this run's fields
    for (r, name, group), (s, n) in acc.items():
        chi2 = s / n if n else np.nan
        rows.append((win, r, name, group, n, chi2))
        print(f'  {win} {r:5s} {name:12s} n={n:7d} chi2/n={chi2:.3f}', flush=True)


def _obs_at_days(model_days, byyear):
    """byyear: {year: (ny,nx,nt)} -> (nday,ny,nx) selected at each model day."""
    out = []
    for ts in model_days:
        doy = (ts - pd.Timestamp(f'{ts.year}-01-01')).days
        out.append(byyear[ts.year][:, :, doy])
    return np.stack(out, 0)


def _weighted(resid, w2d):
    """sum_i w_i resid_i^2 and N over finite cells. w2d is (ny,nx), resid (nt,ny,nx)."""
    W = np.broadcast_to(w2d[None], resid.shape)
    m = np.isfinite(resid) & np.isfinite(W)
    return float(np.sum(W[m] * resid[m] ** 2)), int(m.sum())


def _grids(rows, win, state, surf, ref):
    years = WINDOWS[win]['years']
    nx, ny = ref.sizes['XC'], ref.sizes['YC']
    wetC = (ref['hFacC'].isel(Z=0) > 0).values
    # SST: mw_fusion vs surface THETA, static error grid, weight 1/err^2
    sst = {y: tiw.load_sst(nx, ny, year=y, nt=NTY[y]) for y in years}
    wsst = 1.0 / sst[years[0]][1] ** 2
    for r in state:
        ds = state[r]
        md = pd.to_datetime(ds['time'].values)
        mod = np.where(wetC[None], ds['THETA'].isel(Z=0).values, np.nan)
        s, n = _weighted(mod - _obs_at_days(md, {y: sst[y][0] for y in years}), wsst)
        rows.append((win, r, 'SST', 'Satellite', n, s / n if n else np.nan))
        print(f'  {win} {r:5s} {"SST":12s} n={n:7d} chi2/n={s / n:.3f}', flush=True)
    # SSH: RADS ja(3cm)+c2(6cm) SLA vs ETAN anomaly, per-sat error grid
    err = {'ja': tiw.load_ssh_err(nx, ny, '3cm'), 'c2': tiw.load_ssh_err(nx, ny, '6cm')}
    rads = {y: tiw.load_rads(nx, ny, year=y, nt=NTY[y]) for y in years}
    for r in surf:
        ds = surf[r]
        md = pd.to_datetime(ds['time'].values)
        eta = np.where(wetC[None], ds['ETAN'].values, np.nan)
        eta = eta - np.nanmean(eta, axis=0, keepdims=True)
        s = n = 0
        for sat in ('ja', 'c2'):
            o = _obs_at_days(md, {y: rads[y][sat] for y in years})
            o = o - np.nanmean(o, axis=0, keepdims=True)
            si, ni = _weighted(eta - o, 1.0 / err[sat] ** 2)
            s += si
            n += ni
        rows.append((win, r, 'SSH', 'Satellite', n, s / n if n else np.nan))
        print(f'  {win} {r:5s} {"SSH":12s} n={n:7d} chi2/n={s / n:.3f}', flush=True)


COLS = ['window', 'solution', 'product', 'group', 'nobs', 'chi2']


def compute(windows=None):
    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    windows = list(windows or WINDOWS)
    rows = []
    if os.path.exists(CSV):                     # keep checkpointed rows for other windows
        old = pd.read_csv(CSV)
        rows += [r for r in old[~old['window'].isin(windows)].itertuples(index=False, name=None)]
    for win in windows:
        print('===', win, flush=True)
        state = _load(win, 'diag_state')
        _profiles(rows, win, state)
        surf = _load(win, 'diag_surf')
        _grids(rows, win, state, surf, state['Vel'])
        pd.DataFrame(rows, columns=COLS).to_csv(CSV, index=False)   # checkpoint per window
        print(f'  checkpoint -> {CSV} ({len(rows)} rows)', flush=True)
    return pd.DataFrame(rows, columns=COLS)


def compute_noVel_prior(windows=None):
    """Sample ONLY the noVel prior (iter0) and merge its rows into the CSV; the
    other roles are already cached, so this is ~1/4 the cost of a full recompute."""
    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    windows = list(windows or WINDOWS)
    old = pd.read_csv(CSV)
    rows = [r for r in old.itertuples(index=False, name=None) if r[1] != 'noVel_prior']
    for win in windows:
        print('===', win, flush=True)
        sp = NOVEL_PRIOR.get(win)
        if sp is None:
            continue
        ref = WINDOWS[win]['ref']
        try:
            st = {'noVel_prior': tiw.load_run('noVel_prior', ('diag_state',), ref_date=ref, spec=sp)}
            surf = {'noVel_prior': tiw.load_run('noVel_prior', ('diag_surf',), ref_date=ref, spec=sp)}
        except (OSError, PermissionError) as e:
            print(f'  SKIP {win}/noVel_prior: {type(e).__name__}', flush=True)
            continue
        _profiles(rows, win, st)
        _grids(rows, win, st, surf, st['noVel_prior'])
        pd.DataFrame(rows, columns=COLS).to_csv(CSV, index=False)
        print(f'  checkpoint ({len(rows)} rows)', flush=True)
    return pd.DataFrame(rows, columns=COLS)


# ---- plot --------------------------------------------------------------------
# Consolidated reporting cells. Redundant/renamed files fold together, and each
# solution is scored against the dataset IT assimilated (_keep_row):
#   Moored T/S  = TAO_WO moorings (Vel/prior) OR NODC_MRB (noVel)   [one category]
#   TAO U/V     = ADCP + CUR velocities (assimilated only by Vel)
#   CTD T/S     = HDO (Vel/prior) OR WOD-CTD-OSD (noVel, except nov2012=HDO)
#   XBT T       = WOD + SIO (Vel/prior) OR WOD only (noVel); redundant -> combine
#   gliders (GLD) dropped: neither Vel nor noVel assimilates them.
CELL_OF = {
    'Argo T': 'Argo T', 'Argo S': 'Argo S',
    'TAO T': 'Moored T', 'TAO S': 'Moored S', 'NODC T': 'Moored T', 'NODC S': 'Moored S',
    'TAO ADCP U': 'TAO U', 'TAO CUR U': 'TAO U', 'TAO ADCP V': 'TAO V', 'TAO CUR V': 'TAO V',
    'CTD T': 'CTD T', 'WOD CTD T': 'CTD T', 'CTD S': 'CTD S', 'WOD CTD S': 'CTD S',
    'WOD XBT T': 'XBT T', 'SIO XBT T': 'XBT T',
    'SST': 'SST', 'SSH': 'SSH',
}
CELL_ORDER = ['Argo T', 'Argo S', 'Moored T', 'Moored S', 'TAO U', 'TAO V',
              'CTD T', 'CTD S', 'XBT T', 'SST', 'SSH']
CELL_CAT = {'Argo T': 'Argo', 'Argo S': 'Argo', 'Moored T': 'Moored', 'Moored S': 'Moored',
            'TAO U': 'Velocity', 'TAO V': 'Velocity', 'CTD T': 'CTD', 'CTD S': 'CTD',
            'XBT T': 'XBT', 'SST': 'SST', 'SSH': 'SSH'}
GROUP_COLOR = {'Argo': '#1f77b4', 'Moored': '#17becf', 'Velocity': '#2ca02c',
               'CTD': '#9467bd', 'XBT': '#ff7f0e', 'SST': '#d62728', 'SSH': '#e377c2'}
HATCH = {'prior': '', 'noVel': '...', 'Vel': '//', 'noVel_prior': 'xx'}


def _keep_row(window, solution, product):
    """Pick the dataset each solution is scored against (what it assimilated).
    Per the updated noVel config: noVel assimilates Argo, WOD-CTD, WOD-XBT, SST,
    SSH only -- NOT moorings and NOT velocities (both withheld for noVel)."""
    if product in ('TAO T', 'TAO S'):          # moored T/S: all vs TAO_WO array
        return True                            #   (noVel didn't assimilate -> withheld)
    if product in ('NODC T', 'NODC S'):        # NODC no longer used by any solution
        return False
    if product in ('CTD T', 'CTD S'):          # HDO is subtropical (17-30N) -> out of the
        return False                           #   ±17 evaluation band; drop from CTD cell
    if product in ('WOD CTD T', 'WOD CTD S'):  # WOD-CTD-OSD = CTD for ALL solutions in-domain
        return True
    if product == 'SIO XBT T':                 # SIO assimilated only by Vel/prior (score each
        return solution not in ('noVel', 'noVel_prior')   # noVel lineage used WOD XBT only
    if product in ('WOD GLD T', 'WOD GLD S'):  # gliders: dropped entirely
        return False
    return True


def _grouped(df):
    """Aggregate per-product rows to consolidated cells (chi^2/obs), after
    selecting the dataset each solution assimilated (see _keep_row)."""
    df = df.copy()
    df = df[df['product'].isin(CELL_OF)]
    df = df[[_keep_row(w, s, p) for w, s, p in
             zip(df['window'], df['solution'], df['product'])]]
    df['cell'] = df['product'].map(CELL_OF)
    df['sw'] = df['chi2'] * df['nobs']                       # sum_i w(o-m)^2
    g = df.groupby(['window', 'solution', 'cell'], as_index=False).agg(
        sw=('sw', 'sum'), nobs=('nobs', 'sum'))
    g['chi2'] = g['sw'] / g['nobs']
    g['cat'] = g['cell'].map(CELL_CAT)
    return g


def table(df=None):
    """Readable pivot of consolidated chi^2/obs: rows=cell, cols=solution, per window."""
    if df is None:
        df = pd.read_csv(CSV)
    g = _grouped(df)
    g['cell'] = pd.Categorical(g['cell'], CELL_ORDER, ordered=True)
    for win in [w for w in WINDOWS if w in set(g['window'])]:
        piv = (g[g['window'] == win].pivot_table(index='cell', columns='solution',
               values='chi2', observed=True).reindex(columns=PLOT_ROLES))
        print(f'\n=== {win}  (normalized cost; 1 = fit to stated error) ===')
        print(piv.to_string(float_format=lambda v: f'{v:6.2f}'))
    out = f'{tiw.CACHE}/misfit_table.csv'
    _grouped(df).to_csv(out, index=False)
    print('\nwrote', out)


def _cell_bars(ax, cells, color, getval, roles):
    """Grouped bars (one group per role) for the consolidated cells on `ax`."""
    x, bw = np.arange(len(cells)), 0.38
    for j, role in enumerate(roles):
        off = (j - (len(roles) - 1) / 2) * bw
        ax.bar(x + off, [getval(role, c) for c in cells], bw,
               color=[color[c] for c in cells], hatch=HATCH[role],
               edgecolor='k', linewidth=0.4)
    ax.axhline(1.0, ls='--', c='0.4', lw=1)
    ax.set_ylabel('Normalized Cost', fontsize=14)
    ax.set_xticks(x); ax.set_xticklabels(cells, rotation=55, ha='right', fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.margins(x=0.01)


def plot(df=None, out=FIG, roles=None, labels=None):
    roles = roles or PLOT_ROLES
    labels = labels or {r: r for r in roles}
    if df is None:
        df = pd.read_csv(CSV)
    g = _grouped(df)
    color = {c: GROUP_COLOR[CELL_CAT[c]] for c in CELL_ORDER}
    gi = g.set_index(['window', 'solution', 'cell'])['chi2']
    cells = [c for c in CELL_ORDER if c in set(g['cell'])]
    wins = [w for w in WINDOWS if w in set(df['window'])]
    # all-windows aggregate: obs-count-weighted pooled normalized cost = Σ w(o-m)² / Σ N
    pooled = g.groupby(['solution', 'cell']).agg(sw=('sw', 'sum'), nobs=('nobs', 'sum'))
    pi = (pooled['sw'] / pooled['nobs'])
    # layout: pooled panel centered on top (row 0), the 4 windows in 2x2 below
    fig = plt.figure(figsize=(14, 11.5))
    gs = fig.add_gridspec(3, 4, hspace=0.62, wspace=0.3)
    ax_agg = fig.add_subplot(gs[0, 1:3])
    _cell_bars(ax_agg, cells, color, lambda role, c: pi.get((role, c), np.nan), roles)
    ax_agg.set_title('all windows pooled', fontweight='bold')
    for (r, c0, c1), win in zip([(1, 0, 2), (1, 2, 4), (2, 0, 2), (2, 2, 4)], wins):
        ax = fig.add_subplot(gs[r, c0:c1], sharey=ax_agg)
        _cell_bars(ax, cells, color, lambda role, c, win=win: gi.get((win, role, c), np.nan), roles)
        ax.set_title(win)
    cats = {CELL_CAT[c] for c in cells}        # only legend the categories actually shown
    grp = [Patch(facecolor=c, edgecolor='k', label=gname)
           for gname, c in GROUP_COLOR.items() if gname in cats]
    sol = [Patch(facecolor='0.85', edgecolor='k', hatch=HATCH[r], label=labels[r]) for r in roles]
    lax = fig.add_subplot(gs[0, 0]); lax.axis('off')       # legends fill the empty top corners
    lax.legend(handles=grp, loc='center', frameon=False, fontsize=13)
    rax = fig.add_subplot(gs[0, 3]); rax.axis('off')
    rax.legend(handles=sol, loc='center', frameon=False, fontsize=13)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print('wrote', out, flush=True)


def plot_iter0(out='figs/comparison/misfit_cost_allwindows_iter0.png'):
    """iter-0 version: Vel prior vs noVel prior (the no-DA background misfit)."""
    plot(roles=['prior', 'noVel_prior'],
         labels={'prior': 'Vel prior', 'noVel_prior': 'noVel prior'}, out=out)


# ---- equatorial / off-equatorial split (second version) ----------------------
# Every (solution x cell) is split by observation latitude into equatorial
# (|lat| <= 5 deg) and off-equatorial (5 < |lat| <= 17 deg) model-data misfit.
# Velocity obs (TAO ADCP/CUR) all sit at the equator, so their off-equatorial
# bars come out empty by construction -- that is intentional, it shows velocity
# is only measured on the equator. Separate cache (CSV_R) and figure (FIG_R) so
# the originals are untouched. Same sampling cost as compute() (one pass).
CSV_R = f'{tiw.CACHE}/misfit_cost_byregion.csv'
FIG_R = 'figs/comparison/misfit_cost_allwindows_byregion.png'
COLS_R = ['window', 'solution', 'product', 'group', 'region', 'nobs', 'chi2']
WIN_LABEL = {'sep2012': 'Sep - Dec 2012', 'nov2012': 'Nov 2012 - Feb 2013',
             'jan2013': 'Jan - Apr 2013', 'mar2013': 'Mar - Jun 2013'}
REGION_ORDER = ['eq', 'off']
REGION_LABEL = {'total': 'all obs (|lat| ≤ 17°)', 'eq': 'equatorial (≤ 5°)',
                'off': 'off-equator (5–17°)'}
# Two cues per region so the eq/off split bars read as distinct from the plain
# pooled-total bars: fill alpha + edge weight. (solution stays encoded by hatch.)
REGION_STYLE = {
    'total': dict(alpha=1.0, linewidth=0.4, edgecolor='black'),
    'eq':    dict(alpha=0.9, linewidth=1.7, edgecolor='black'),
    'off':   dict(alpha=0.4, linewidth=1.7, edgecolor='black'),
}


def _region_mask(abslat, region):
    """|lat| band mask. eq = [0, 5] deg (includes the equator), off = (5, 17]."""
    if region == 'eq':
        return abslat <= 5.0
    return (abslat > 5.0) & (abslat <= LAT_LIMIT)


def _profiles_region(rows, win, state):
    """_profiles, but the model field is sampled once then split into |lat| bands."""
    obs_sets = []
    for spec in PROFILE_PRODUCTS:
        for y in WINDOWS[win]['years']:
            p = spec['path'](y)
            if os.path.exists(p):
                obs = tiw.open_obs(p)
                obs_sets.append((spec, obs, _good_profiles(obs)))
            else:
                print(f'  {win} skip missing {os.path.basename(p)}', flush=True)
    acc = {}
    for r in state:
        ds, prep = state[r], {}
        for spec, obs, good in obs_sets:
            alat = np.abs(obs['lat'].values)
            for v in spec['vars']:
                if v not in prep:
                    prep[v] = _prep_field(ds, v)
                man = _sample(prep[v], obs)                 # sample once per run/field
                raw, wgt = obs[f'prof_{v}'].values, obs[f'prof_{v}weight'].values
                for region in REGION_ORDER:
                    keep = _region_mask(alat, region) & good
                    o, m, wt = tiw._mask(raw, man, np.where(keep[:, None], wgt, 0.0))
                    a = acc.setdefault((r, spec['names'][v], spec['group'], region), [0.0, 0])
                    a[0] += float(np.sum(wt * (o - m) ** 2))
                    a[1] += o.size
        del prep
    for (r, name, group, region), (s, n) in acc.items():
        chi2 = s / n if n else np.nan
        rows.append((win, r, name, group, region, n, chi2))
        print(f'  {win} {r:5s} {name:12s} {region:3s} n={n:7d} chi2/n={chi2:.3f}', flush=True)


def _grids_region(rows, win, state, surf, ref):
    """_grids, but each gridded term (SST, SSH) is split by the model grid |YC|."""
    years = WINDOWS[win]['years']
    nx, ny = ref.sizes['XC'], ref.sizes['YC']
    wetC = (ref['hFacC'].isel(Z=0) > 0).values
    alat2d = np.broadcast_to(np.abs(ref['YC'].values)[:, None], (ny, nx))

    def regw(w2d, region):          # NaN outside band -> excluded by _weighted
        return np.where(_region_mask(alat2d, region), w2d, np.nan)

    sst = {y: tiw.load_sst(nx, ny, year=y, nt=NTY[y]) for y in years}
    wsst = 1.0 / sst[years[0]][1] ** 2
    for r in state:
        ds = state[r]
        md = pd.to_datetime(ds['time'].values)
        mod = np.where(wetC[None], ds['THETA'].isel(Z=0).values, np.nan)
        resid = mod - _obs_at_days(md, {y: sst[y][0] for y in years})
        for region in REGION_ORDER:
            s, n = _weighted(resid, regw(wsst, region))
            rows.append((win, r, 'SST', 'Satellite', region, n, s / n if n else np.nan))
    err = {'ja': tiw.load_ssh_err(nx, ny, '3cm'), 'c2': tiw.load_ssh_err(nx, ny, '6cm')}
    rads = {y: tiw.load_rads(nx, ny, year=y, nt=NTY[y]) for y in years}
    for r in surf:
        ds = surf[r]
        md = pd.to_datetime(ds['time'].values)
        eta = np.where(wetC[None], ds['ETAN'].values, np.nan)
        eta = eta - np.nanmean(eta, axis=0, keepdims=True)
        for region in REGION_ORDER:
            s = n = 0
            for sat in ('ja', 'c2'):
                o = _obs_at_days(md, {y: rads[y][sat] for y in years})
                o = o - np.nanmean(o, axis=0, keepdims=True)
                si, ni = _weighted(eta - o, regw(1.0 / err[sat] ** 2, region))
                s += si
                n += ni
            rows.append((win, r, 'SSH', 'Satellite', region, n, s / n if n else np.nan))


def compute_region(windows=None):
    """Region-split recompute -> CSV_R (leaves the original CSV untouched)."""
    os.makedirs(os.path.dirname(CSV_R), exist_ok=True)
    windows = list(windows or WINDOWS)
    rows = []
    if os.path.exists(CSV_R):
        old = pd.read_csv(CSV_R)
        rows += [r for r in old[~old['window'].isin(windows)].itertuples(index=False, name=None)]
    for win in windows:
        print('===', win, flush=True)
        state = _load(win, 'diag_state')
        _profiles_region(rows, win, state)
        surf = _load(win, 'diag_surf')
        _grids_region(rows, win, state, surf, state['Vel'])
        pd.DataFrame(rows, columns=COLS_R).to_csv(CSV_R, index=False)
        print(f'  checkpoint -> {CSV_R} ({len(rows)} rows)', flush=True)
    return pd.DataFrame(rows, columns=COLS_R)


def _grouped_region(df):
    df = df.copy()
    df = df[df['product'].isin(CELL_OF)]
    df = df[[_keep_row(w, s, p) for w, s, p in zip(df['window'], df['solution'], df['product'])]]
    df['cell'] = df['product'].map(CELL_OF)
    df['sw'] = df['chi2'] * df['nobs']
    g = df.groupby(['window', 'solution', 'cell', 'region'], as_index=False).agg(
        sw=('sw', 'sum'), nobs=('nobs', 'sum'))
    g['chi2'] = g['sw'] / g['nobs']
    return g


def _cell_bars_region(ax, cells, color, getval, roles, regions):
    """Grouped bars for `regions` x `roles` per cell (contiguous, centered).
    region -> alpha+edge weight (REGION_STYLE), solution -> hatch, cell -> color."""
    x = np.arange(len(cells))
    series = [(reg, role) for reg in regions for role in roles]
    n = len(series)
    bw = 0.82 / n          # fill a fixed fraction of each cell -> wider bars when fewer
    for i, (reg, role) in enumerate(series):
        st = REGION_STYLE[reg]
        ax.bar(x + (i - (n - 1) / 2) * bw, [getval(role, reg, c) for c in cells], bw,
               color=[color[c] for c in cells], hatch=HATCH[role],
               alpha=st['alpha'], edgecolor=st['edgecolor'], linewidth=st['linewidth'])
    ax.axhline(1.0, ls='--', c='0.4', lw=1)
    ax.set_ylabel('Normalized Cost', fontsize=14)
    ax.set_xticks(x); ax.set_xticklabels(cells, rotation=55, ha='right', fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.margins(x=0.01)


def plot_byregion(df=None, out=FIG_R, roles=None, labels=None, drop=()):
    roles = roles or PLOT_ROLES
    labels = labels or {r: r for r in roles}
    if df is None:
        df = pd.read_csv(CSV_R)
    g = _grouped_region(df)
    color = {c: GROUP_COLOR[CELL_CAT[c]] for c in CELL_ORDER}
    cells = [c for c in CELL_ORDER if c in set(g['cell']) and c not in drop]
    wins = [w for w in WINDOWS if w in set(df['window'])]
    # top panel: pooled TOTAL (summed over both regions) -> 2 bars/cell
    tot = g.groupby(['solution', 'cell']).agg(sw=('sw', 'sum'), nobs=('nobs', 'sum'))
    ti = tot['sw'] / tot['nobs']
    # per-window panels: region-resolved -> 4 bars/cell
    gi = g.set_index(['window', 'solution', 'cell', 'region'])['chi2']
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.62, wspace=0.3)
    ax_agg = fig.add_subplot(gs[0, 0:3])                 # wide total panel
    _cell_bars_region(ax_agg, cells, color,
                      lambda role, reg, c: ti.get((role, c), np.nan), roles, ['total'])
    ax_agg.set_title('all windows and regions', fontweight='bold')
    for (r, c0, c1), win in zip([(1, 0, 2), (1, 2, 4), (2, 0, 2), (2, 2, 4)], wins):
        ax = fig.add_subplot(gs[r, c0:c1], sharey=ax_agg)
        _cell_bars_region(ax, cells, color,
                          lambda role, reg, c, win=win: gi.get((win, role, c, reg), np.nan),
                          roles, REGION_ORDER)
        ax.set_title(WIN_LABEL.get(win, win))
    cats = {CELL_CAT[c] for c in cells}
    grp = [Patch(facecolor=c, edgecolor='k', label=gname)
           for gname, c in GROUP_COLOR.items() if gname in cats]
    sol = [Patch(facecolor='0.85', edgecolor='k', hatch=HATCH[r], label=labels[r]) for r in roles]
    reg = [Patch(facecolor='0.6', label=REGION_LABEL[rg], **REGION_STYLE[rg])
           for rg in ('total', 'eq', 'off')]
    # one combined legend (category, then solution, then region) -> no overlap.
    # blank spacers separate the three groups.
    spacer = Patch(facecolor='none', edgecolor='none', label='')
    handles = grp + [spacer] + sol + [spacer] + reg
    lax = fig.add_subplot(gs[0, 3]); lax.axis('off')
    lax.legend(handles=handles, loc='center', frameon=False, fontsize=10, labelspacing=0.55)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print('wrote', out, flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'all'
    wins = [w for w in sys.argv[2:] if w in WINDOWS] or None   # subset merges with CSV
    if cmd in ('compute', 'all'):
        compute(wins)
    if cmd in ('plot', 'all'):
        plot()
    if cmd == 'table':
        table()
    if cmd == 'compute_prior':       # sample only the noVel prior (merges into CSV)
        compute_noVel_prior(wins)
    if cmd in ('plot_prior', 'compute_prior'):
        plot_iter0()
    if cmd == 'compute_region':      # region-split recompute -> CSV_R
        compute_region(wins)
    if cmd in ('plot_region', 'compute_region'):
        plot_byregion()
        plot_byregion(out=FIG_R.replace('.png', '_noCTD.png'), drop=('CTD T', 'CTD S'))
