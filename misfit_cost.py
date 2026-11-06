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


def _profiles(rows, win, state):
    # open every obs file for this window once (obs are solution-independent)
    obs_sets = []        # (spec, obs)
    for spec in PROFILE_PRODUCTS:
        for y in WINDOWS[win]['years']:
            p = spec['path'](y)
            if os.path.exists(p):
                obs_sets.append((spec, tiw.open_obs(p)))
            else:
                print(f'  {win} skip missing {os.path.basename(p)}', flush=True)
    acc = {}             # (role, name, group) -> [sum_wsq, n]
    for r in state:                         # only roles that loaded
        ds, prep = state[r], {}
        for spec, obs in obs_sets:
            for v in spec['vars']:
                if v not in prep:                      # load each field once per run
                    prep[v] = _prep_field(ds, v)
                man = _sample(prep[v], obs)
                wt0 = np.where(np.abs(obs['lat'].values)[:, None] <= LAT_LIMIT,
                               obs[f'prof_{v}weight'].values, 0.0)   # domain restriction
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
HATCH = {'prior': '', 'noVel': '...', 'Vel': '//'}


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
        return solution != 'noVel'             #   solution vs what it used; in-domain & minor)
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
        print(f'\n=== {win}  (chi^2 / obs; 1 = fit to stated error) ===')
        print(piv.to_string(float_format=lambda v: f'{v:6.2f}'))
    out = f'{tiw.CACHE}/misfit_table.csv'
    _grouped(df).to_csv(out, index=False)
    print('\nwrote', out)


def plot(df=None, out=FIG):
    if df is None:
        df = pd.read_csv(CSV)
    g = _grouped(df)
    color = {c: GROUP_COLOR[CELL_CAT[c]] for c in CELL_ORDER}
    gi = g.set_index(['window', 'solution', 'cell'])
    cells = [c for c in CELL_ORDER if c in set(g['cell'])]
    x = np.arange(len(cells))
    bw = 0.38
    wins = [w for w in WINDOWS if w in set(df['window'])]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    for ax, win in zip(axes.ravel(), wins):
        for j, role in enumerate(PLOT_ROLES):
            off = (j - (len(PLOT_ROLES) - 1) / 2) * bw
            vals = [gi['chi2'].get((win, role, c), np.nan) for c in cells]
            ax.bar(x + off, vals, bw, color=[color[c] for c in cells],
                   hatch=HATCH[role], edgecolor='k', linewidth=0.4)
        ax.axhline(1.0, ls='--', c='0.4', lw=1)
        ax.set_title(win)
        ax.set_ylabel(r'$\chi^2$ / obs')
        ax.set_xticks(x)
        ax.set_xticklabels(cells, rotation=55, ha='right', fontsize=8)
        ax.margins(x=0.01)
    for ax in axes.ravel()[len(wins):]:
        ax.set_visible(False)
    cats = {CELL_CAT[c] for c in cells}        # only legend the categories actually shown
    grp = [Patch(facecolor=c, edgecolor='k', label=gname)
           for gname, c in GROUP_COLOR.items() if gname in cats]
    sol = [Patch(facecolor='0.85', edgecolor='k', hatch=HATCH[r], label=r) for r in PLOT_ROLES]
    fig.legend(handles=grp + sol, ncol=10, loc='upper center', frameon=False,
               bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(r'Model$-$data misfit ($\chi^2$ per observation; dashed = fit to stated error)',
                 y=1.07)
    fig.tight_layout()
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
