"""
tiw-controls: sample TPOSE6 runs like the data and score them against obs.

Decomposes how much of the TPOSE6-Vel assimilation improvement is explained by
a single optimized control field (first experiment: initial temperature,
`run_initialT`). All four sep2012 runs share config; we sample each run's model
output at the observation coordinates and compute misfit metrics.

Env: run with ~/miniforge3/envs/tpose/bin/python
Weights: prof_*weight == 1/sigma^2 (verified). weight==0 => excluded.
"""
import glob
import os
import numpy as np
import pandas as pd
import xarray as xr
from xmitgcm import open_mdsdataset

GRID = '/data/SO6/TPOSE_diags/tpose6/grid_6/'
# Intermediate/cached data lives on /data, not in the repo or home dir.
CACHE = '/data/SO3/edavenport/tiw-controls/cache'
REF_DATE = '2012-09-01'
NT = 122                      # daily records, Sep 1 - Dec 31 2012
DAYS = pd.date_range(REF_DATE, periods=NT, freq='D')
_EXP = '/data/SO3/edavenport/tpose6/withheld_controls_exp/sep2012'
# noVel-anchored single-control experiments (same fixed-increment design, controls set to
# their TPOSE6-noVel optimized values). Dir names use the noVel convention (uwind, not
# zonalWind); run config matches the Vel experiments (72 iters/file, delta_t 1200).
_EXP_NOVEL = '/data/SO3/edavenport/tpose6/withheld_controls_exp/sep2012_noVel'
# TAO-TS-only family: same prior/fixed-increment design, controls set to the
# TAO-T/S-constrained (no velocity obs) optimized values. Dir names are Vel-style
# (zonalWind, not uwind). Config 72 iters/file, delta_t 1200 (same as Vel exps).
_EXP_TSONLY = '/data/SO3/edavenport/tpose6/withheld_controls_exp/sep2012_TSOnly'

# role -> (data_dir, itPerFile, delta_t). noVel writes at 48*1800s, others 72*1200s.
RUNS = {
    'prior':    ('/data/SO3/edavenport/tpose6/diags/sep2012/run_iter0',  72, 1200),
    'noVel':    ('/data/SO6/TPOSE_diags/tpose6/sep2012/diags',           48, 1800),
    'Vel':      ('/data/SO3/edavenport/tpose6/diags/sep2012/run_iter14', 72, 1200),
    # full TAO-T/S solution (assimilates TAO T&S, NOT U&V) — anchor for the TSOnly
    # family; its optimized increments are what the ts_* experiments hold fixed.
    'TSOnly':   ('/data/SO3/edavenport/tpose6/diags/sep2012/velocity_withhold/run_iter14', 72, 1200),
    # single-control experiments (hold all else at background)
    'initialT':       (f'{_EXP}/run_initialT',       72, 1200),
    'initialS':       (f'{_EXP}/run_initialS',       72, 1200),
    'zonalWind':      (f'{_EXP}/run_zonalWind',      72, 1200),
    'meridionalWind': (f'{_EXP}/run_meridionalWind', 72, 1200),
    'swdown':         (f'{_EXP}/run_swdown',         72, 1200),
    'atmosT':         (f'{_EXP}/run_atmosT',         72, 1200),  # xx_atemp
    'atmosQ':         (f'{_EXP}/run_atmosQ',         72, 1200),  # xx_aqh
    'lwdown':         (f'{_EXP}/run_lwdown',         72, 1200),
    'precip':         (f'{_EXP}/run_precip',         72, 1200),
    'uvel':           (f'{_EXP}/run_uvel',           72, 1200),
    'vvel':           (f'{_EXP}/run_vvel',           72, 1200),
    # combination runs (fixed-increment: named controls ON at full-solution values).
    'initialT_uwind': (f'{_EXP}/run_initialT_uwind', 72, 1200),  # theta x uwind
    'theta_salt':     (f'{_EXP}/run_theta_salt',     72, 1200),  # theta x salt
    'theta_swdown':   (f'{_EXP}/run_theta_swdown',   72, 1200),  # theta x swdown
    'uwind_swdown':   (f'{_EXP}/run_uwind_swdown',   72, 1200),  # uwind x swdown
    'salt_swdown':    (f'{_EXP}/run_salt_swdown',    72, 1200),  # salt x swdown
    'salt_uwind':     (f'{_EXP}/run_salt_uwind',     72, 1200),  # salt x uwind
    # reduced-full: subset {theta,uwind,salt,swdown} all ON (not a pair -> not in COMBOS).
    'top4':           (f'{_EXP}/run_top4',           72, 1200),
    # noVel-family single controls (anchor=noVel; see RUN_FAMILY['novel'] in anova.py).
    # key uses the Vel-family control name, dir uses the noVel name (uwind == zonalWind).
    'nv_initialT':    (f'{_EXP_NOVEL}/run_initialT', 72, 1200),  # xx_theta (noVel)
    'nv_initialS':    (f'{_EXP_NOVEL}/run_initialS', 72, 1200),  # xx_salt  (noVel)
    'nv_zonalWind':   (f'{_EXP_NOVEL}/run_uwind',    72, 1200),  # xx_uwind (noVel)
    'nv_swdown':      (f'{_EXP_NOVEL}/run_swdown',   72, 1200),  # xx_swdown(noVel)
    # noVel-family extra forcing singles (screen only; NOT in the TSUW factorial)
    'nv_atmosT':         (f'{_EXP_NOVEL}/run_atemp',  72, 1200),  # xx_atemp
    'nv_atmosQ':         (f'{_EXP_NOVEL}/run_aqh',    72, 1200),  # xx_aqh
    'nv_precip':         (f'{_EXP_NOVEL}/run_precip', 72, 1200),  # xx_precip
    'nv_meridionalWind': (f'{_EXP_NOVEL}/run_vwind',  72, 1200),  # xx_vwind
    'nv_lwdown':         (f'{_EXP_NOVEL}/run_lwdown', 72, 1200),  # xx_lwdown
    # noVel combos (IN the TSUW factorial; keys match RUN_FAMILY['novel'] in anova.py).
    # dir uses noVel naming; TU run has 121/122 records (one missing, handled by load_run).
    'nv_theta_salt':       (f'{_EXP_NOVEL}/run_theta_salt',              72, 1200),  # TS
    'nv_initialT_uwind':   (f'{_EXP_NOVEL}/run_theta_uwind',             72, 1200),  # TU
    'nv_theta_swdown':     (f'{_EXP_NOVEL}/run_theta_swdown',            72, 1200),  # TW
    'nv_salt_uwind':       (f'{_EXP_NOVEL}/run_salt_uwind',              72, 1200),  # SU
    'nv_salt_swdown':      (f'{_EXP_NOVEL}/run_salt_swdown',             72, 1200),  # SW
    'nv_uwind_swdown':     (f'{_EXP_NOVEL}/run_uwind_swdown',            72, 1200),  # UW
    'nv_theta_salt_uwind':   (f'{_EXP_NOVEL}/run_theta_uwind_salt',         72, 1200),  # TSU
    'nv_theta_salt_swdown':  (f'{_EXP_NOVEL}/run_theta_salt_swdown',        72, 1200),  # TSW
    'nv_theta_uwind_swdown': (f'{_EXP_NOVEL}/run_theta_uwind_swdown',       72, 1200),  # TUW
    'nv_salt_uwind_swdown':  (f'{_EXP_NOVEL}/run_salt_uwind_swdown',        72, 1200),  # SUW
    'nv_top4':             (f'{_EXP_NOVEL}/run_theta_uwind_salt_swdown', 72, 1200),  # TSUW
    # TSOnly-family single controls (anchor=TSOnly; dir names Vel-style). 6 with
    # output so far; more experiments copied in as diag_state is freed.
    'ts_initialT':        (f'{_EXP_TSONLY}/run_initialT',        72, 1200),  # xx_theta
    'ts_initialS':        (f'{_EXP_TSONLY}/run_initialS',        72, 1200),  # xx_salt
    'ts_zonalWind':       (f'{_EXP_TSONLY}/run_zonalWind',       72, 1200),  # xx_uwind
    'ts_meridionalWind':  (f'{_EXP_TSONLY}/run_meridionalWind',  72, 1200),  # xx_vwind
    'ts_swdown':          (f'{_EXP_TSONLY}/run_swdown',          72, 1200),  # xx_swdown
    'ts_precip':          (f'{_EXP_TSONLY}/run_precip',          72, 1200),  # xx_precip
    'ts_lwdown':          (f'{_EXP_TSONLY}/run_lwdown',          72, 1200),  # xx_lwdown
    # TSOnly combos (keys mirror RUN_FAMILY subsets; dir names vary by content)
    'ts_theta_salt':        (f'{_EXP_TSONLY}/run_initialT_initialS',  72, 1200),  # TS
    'ts_initialT_uwind':    (f'{_EXP_TSONLY}/run_initialT_uwind',     72, 1200),  # TU
    'ts_salt_uwind':        (f'{_EXP_TSONLY}/run_salt_uwind',         72, 1200),  # SU
    'ts_theta_salt_uwind':  (f'{_EXP_TSONLY}/run_theta_salt_uwind',   72, 1200),  # TSU
    'ts_top4':              (f'{_EXP_TSONLY}/run_top4',               72, 1200),  # TSUW
    'ts_atmosT':            (f'{_EXP_TSONLY}/run_atemp',              72, 1200),  # xx_atemp
    'ts_atmosQ':            (f'{_EXP_TSONLY}/run_aqh',                72, 1200),  # xx_aqh
    'ts_theta_swdown':        (f'{_EXP_TSONLY}/run_initialT_swdown',      72, 1200),  # TW
    'ts_salt_swdown':         (f'{_EXP_TSONLY}/run_salt_swdown',          72, 1200),  # SW
    'ts_uwind_swdown':        (f'{_EXP_TSONLY}/run_uwind_swdown',         72, 1200),  # UW
    'ts_theta_salt_swdown':   (f'{_EXP_TSONLY}/run_theta_salt_swdown',    72, 1200),  # TSW
    'ts_theta_uwind_swdown':  (f'{_EXP_TSONLY}/run_theta_uwind_swdown',   72, 1200),  # TUW
    'ts_salt_uwind_swdown':   (f'{_EXP_TSONLY}/run_salt_uwind_swdown',    72, 1200),  # SUW
    # triads (not pairs -> not in COMBOS; see TRIADS).
    'theta_salt_uwind':   (f'{_EXP}/run_theta_salt_uwind',   72, 1200),
    'theta_salt_swdown':  (f'{_EXP}/run_theta_salt_swdown',  72, 1200),  # pending run
    'theta_uwind_swdown': (f'{_EXP}/run_theta_uwind_swdown', 72, 1200),  # pending run
    'salt_uwind_swdown':  (f'{_EXP}/run_salt_uwind_swdown',  72, 1200),  # pending run
}
# triad run -> (i, j, k) singles. Total interaction = d_ijk - (d_i+d_j+d_k); the clean
# 3-way a_ijk also needs all three pair runs (salt_uwind pair still pending/broken).
TRIADS = {
    'theta_salt_uwind':   ('initialT', 'initialS',  'zonalWind'),
    'theta_salt_swdown':  ('initialT', 'initialS',  'swdown'),
    'theta_uwind_swdown': ('initialT', 'zonalWind', 'swdown'),
    'salt_uwind_swdown':  ('initialS', 'zonalWind', 'swdown'),
}
# combo run -> (single_i, single_j) it decomposes. a_ij = m_ij - m_i - m_j + m_prior.
# Verified fixed-increment: each ON field is byte-identical to its single-control run,
# each OFF field byte-identical to background (iter0).
COMBOS = {
    'initialT_uwind': ('initialT', 'zonalWind'),
    'theta_salt':     ('initialT', 'initialS'),
    'theta_swdown':   ('initialT', 'swdown'),
    'uwind_swdown':   ('zonalWind', 'swdown'),
    'salt_swdown':    ('initialS', 'swdown'),
    'salt_uwind':     ('initialS', 'zonalWind'),
}
# reduced-full run (subset all-ON); scored like any run, decomposed separately (not pairwise).
REDUCED_FULL = 'top4'
# single-control experiments, all verified applied + complete (122 daily records).
# forcing/IC controls kept for decomposition + combination runs.
CONTROLS = ['initialT', 'initialS', 'zonalWind', 'meridionalWind', 'swdown',
            'atmosT', 'atmosQ', 'lwdown', 'precip']
# velocity-IC solo null-checks: analyzed individually, EXCLUDED from combos (TODO).
CONTROLS_NULL = ['uvel', 'vvel']
# the control field each run perturbs (for labels / IC-increment plots)
CONTROL_XX = {'initialT': 'xx_theta', 'initialS': 'xx_salt', 'zonalWind': 'xx_uwind',
              'meridionalWind': 'xx_vwind', 'swdown': 'xx_swdown',
              'atmosT': 'xx_atemp', 'atmosQ': 'xx_aqh', 'lwdown': 'xx_lwdown',
              'precip': 'xx_precip', 'uvel': 'xx_uvel', 'vvel': 'xx_vvel',
              # multi-control combos (for per-experiment figures of combined runs)
              'initialT_uwind': 'θ+u wind', 'theta_salt': 'θ+S',
              'theta_salt_uwind': 'θ+S+u wind', 'top4': 'θ+S+u wind+sw',
              # noVel-family singles (same control fields, noVel-optimized increments)
              'nv_initialT': 'xx_theta', 'nv_initialS': 'xx_salt',
              'nv_zonalWind': 'xx_uwind', 'nv_swdown': 'xx_swdown',
              'nv_atmosT': 'xx_atemp', 'nv_atmosQ': 'xx_aqh',
              'nv_precip': 'xx_precip', 'nv_meridionalWind': 'xx_vwind',
              'nv_lwdown': 'xx_lwdown',
              # noVel combos (labels for per-experiment combo figures)
              'nv_theta_salt': 'θ+S', 'nv_initialT_uwind': 'θ+u wind',
              'nv_theta_salt_uwind': 'θ+S+u wind', 'nv_top4': 'θ+S+u wind+sw'}

# model field name + its horizontal grid coords, per obs variable
FIELD = {
    'T': ('THETA', ('XC', 'YC')),
    'S': ('SALT',  ('XC', 'YC')),
    'U': ('UVEL',  ('XG', 'YC')),
    'V': ('VVEL',  ('XC', 'YG')),
}

# wet-cell mask (hFac>0) by horizontal grid; dry cells are 0.0 in MDS output and
# must be dropped before interpolation or they corrupt the result.
_HFAC = {('XC', 'YC'): 'hFacC', ('XG', 'YC'): 'hFacW', ('XC', 'YG'): 'hFacS'}


def wet_mask(ds, field):
    xy = tuple(c for c in field.dims if c in ('XC', 'XG', 'YC', 'YG'))
    xy = (next(c for c in xy if c.startswith('X')), next(c for c in xy if c.startswith('Y')))
    return ds[_HFAC[xy]] > 0


def load_run(name, prefix=('diag_state',)):
    """Open a run's daily MDS output; relabel time to physical day dates.

    Iters are discovered from disk (robust to missing records, e.g. zonalWind
    has 121/122). A daily-average record at iter N covers physical day
    N/itPerFile - 1 after REF_DATE; we set `time` to that date so obs-day
    matching is exact and gaps leave a true hole rather than shifting days.
    """
    data_dir, itpf, dt = RUNS[name]
    # a .data with no matching .meta can't be read (xmitgcm needs dims/dtype
    # from the meta). Missing model output is a breaking error: fail loud with
    # the offending files so it gets fixed, don't silently drop records.
    datafiles = glob.glob(f'{data_dir}/{prefix[0]}.*.data')
    orphans = [f for f in datafiles if not os.path.exists(f[:-5] + '.meta')]
    if orphans:
        raise FileNotFoundError(
            f'{name}/{prefix[0]}: {len(orphans)} .data file(s) missing their .meta '
            f'(cannot be read): ' + ', '.join(sorted(os.path.basename(f) for f in orphans)))
    its = sorted(int(os.path.basename(f).split('.')[1]) for f in datafiles)
    expected = list(range(its[0], its[-1] + 1, itpf))
    if len(its) != len(expected):                       # never drop a day silently
        missing = sorted(set(expected) - set(its))
        print(f'WARNING {name}/{prefix[0]}: missing {len(missing)} record(s) -> '
              f'iters {missing}, days {[i // itpf - 1 for i in missing]}', flush=True)
    ds = open_mdsdataset(data_dir=data_dir, grid_dir=GRID, iters=its,
                         prefix=list(prefix), ref_date=REF_DATE, delta_t=dt)
    for c in ('XC', 'YC', 'Z', 'XG', 'YG'):
        if c in ds.coords:
            ds[c] = ds[c].astype(float)
    days = pd.to_datetime(REF_DATE) + pd.to_timedelta([i // itpf - 1 for i in its], 'D')
    ds = ds.assign_coords(time=days)
    return ds


def load_profile_obs(name, root):
    """Load a MITprof *_model.nc for a run and attach obs datetime/lon/lat/depth.

    Returns the dataset with added coords: time(iPROF), lon(iPROF), lat(iPROF),
    depth(iDEPTH, negative/down). Missing value is -9999.
    """
    data_dir = RUNS[name][0]
    ds = xr.open_dataset(f'{data_dir}/PROF_eq/{root}_model.nc')
    dates = ds['prof_YYYYMMDD'].values.astype(np.int64).astype(str)
    times = [t.zfill(6) for t in ds['prof_HHMMSS'].values.astype(np.int64).astype(str)]
    dt = pd.to_datetime([d + t for d, t in zip(dates, times)], format='%Y%m%d%H%M%S')
    # obs lon may be -180..180 (TAO); model grid is 0..360 -> normalize.
    lon = ds['prof_lon'].values.astype(float) % 360.0
    ds = ds.assign_coords(time=('iPROF', dt),
                          lon=('iPROF', lon),
                          lat=('iPROF', ds['prof_lat'].values.astype(float)),
                          depth=('iDEPTH', -ds['prof_depth'].values.astype(float)))
    return ds


def sample_profiles(field, obs, mask=None):
    """Sample a model DataArray at scattered profile obs (vectorized per day).

    field : xr.DataArray with dims (time, Z, <y>, <x>), time = physical days.
    obs   : dataset from load_profile_obs (coords time,lon,lat,depth).
    mask  : optional wet-cell mask (Z,<y>,<x>); dry cells set to NaN pre-interp.
    Returns ndarray (iPROF, iDEPTH) of model values interpolated to obs points.

    ponytail: nearest model day (daily cadence); linear in x,y,z. Upgrade to
    linear-in-time if sub-daily obs timing matters.
    """
    xname, yname = [c for c in field.dims if c in ('XC', 'XG')][0], \
                   [c for c in field.dims if c in ('YC', 'YG')][0]
    if mask is not None:
        field = field.where(mask)
    lon = obs['lon'].values
    lat = obs['lat'].values
    zneg = obs['depth'].values
    oday = pd.to_datetime(obs['time'].values).floor('D')
    ftimes = pd.to_datetime(field['time'].values)   # actual model days (may have gaps)
    out = np.full((lon.size, zneg.size), np.nan)
    zDA = xr.DataArray(zneg, dims='iDEPTH')
    for di, day in enumerate(ftimes):
        sel = np.where(oday == day)[0]
        if sel.size == 0:
            continue
        fld = field.isel(time=di).load()
        m = fld.interp({xname: xr.DataArray(lon[sel], dims='p'),
                        yname: xr.DataArray(lat[sel], dims='p'),
                        'Z': zDA}, method='linear')
        out[sel, :] = m.transpose('p', 'iDEPTH').values
    return out


# ---- metrics (profiles) ------------------------------------------------------
def _mask(obs, model, weight):
    m = (np.isfinite(obs) & np.isfinite(model) & (obs != -9999.)
         & np.isfinite(weight) & (weight > 0))
    return obs[m], model[m], weight[m]


def rmse(obs, model, weight):
    o, m, _ = _mask(obs, model, weight)
    return float(np.sqrt(np.mean((o - m) ** 2))) if o.size else np.nan


def bias(obs, model, weight):
    o, m, _ = _mask(obs, model, weight)
    return float(np.mean(m - o)) if o.size else np.nan


def wcost(obs, model, weight, normalize=True):
    """Cost-function contribution Sum w*(o-m)^2, w=1/sigma^2. normalize => per-point."""
    o, m, w = _mask(obs, model, weight)
    if o.size == 0:
        return np.nan
    j = float(np.sum(w * (o - m) ** 2))
    return j / o.size if normalize else j


def rmse_profile(obs, model, weight):
    """RMSE as a function of depth (over iPROF). Arrays shaped (iPROF, iDEPTH)."""
    o = np.where((obs != -9999.) & (weight > 0), obs, np.nan)
    m = np.where(np.isfinite(o), model, np.nan)
    return np.sqrt(np.nanmean((o - m) ** 2, axis=0))


def fraction_explained(m_prior, m_initialT, m_full):
    """Fraction of (prior->full) improvement captured by the single control."""
    denom = m_prior - m_full
    return np.nan if denom == 0 else (m_prior - m_initialT) / denom


# ---- gridded obs loaders (SST / SSH) ----------------------------------------
def _read_fortran(filename, nx, ny, nt):
    data = np.fromfile(filename, dtype='>f4').reshape((nx, ny, nt), order='F')
    return data.transpose(1, 0, 2).astype(float)   # -> (ny, nx, nt)


def load_rads(nx, ny, year=2012, nt=366):
    """RADS altimetry SSH (m) on the model XC/YC grid, daily. -9999 -> NaN."""
    out = {}
    for sat in ('ja', 'c2'):
        d = _read_fortran(f'/data/SO6/TPOSE/constraints/rads/rads_{sat}_tpose6_QC_{year}', nx, ny, nt)
        d[d == -9999.] = np.nan
        out[sat] = d / 100.0   # cm -> m
    return out


def load_sst(nx, ny, year=2012, nt=366):
    """mw_fusion SST obs (ny,nx,nt) and gridded SST error field (ny,nx)."""
    sst = _read_fortran(f'/data/SO6/TPOSE/constraints/sst/mw_fusion_tpose6_{year}', nx, ny, nt)
    sst[sst == -9999.] = np.nan
    err = np.fromfile('/data/SO6/TPOSE/model_setup/linked_files/TP6/SST_error_tpose6_17Sto17N.bin',
                      dtype='>f4').reshape((nx, ny), order='F').T
    return sst, err


if __name__ == '__main__':
    # Validation: manual sampling must reproduce the run's own *_model.nc estim.
    ARGO = 'USGO_TP_2012_PFL_D_w'
    for run in ('initialT', 'Vel'):
        ds = load_run(run, prefix=('diag_state',))
        obs = load_profile_obs(run, ARGO)
        for var in ('T', 'S'):
            fld = ds[FIELD[var][0]]
            man = sample_profiles(fld, obs, mask=wet_mask(ds, fld))
            est = obs[f'prof_{var}estim'].values
            o = obs[f'prof_{var}'].values
            w = obs[f'prof_{var}weight'].values
            valid = (est != -9999.) & np.isfinite(est) & (w > 0)
            d = (man - est)[valid]
            d = d[np.isfinite(d)]
            r_m3 = rmse(o, man, w)          # our misfit vs obs
            r_eq = rmse(o, est, w)          # equivalent misfit vs obs
            print(f'{run:9s} {var}: manual-vs-equiv RMSE={np.sqrt(np.mean(d**2)):.4f} '
                  f'(n={d.size})  |  RMSE_obs manual={r_m3:.4f} equiv={r_eq:.4f}')
            assert np.sqrt(np.mean(d ** 2)) < 0.6, 'manual sampling diverges from equivalent'
    print('OK: manual sampling reproduces profile equivalents within tolerance.')
