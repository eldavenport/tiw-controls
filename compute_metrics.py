"""
Sample runs at the obs and write caches + a metrics table.

Outputs (in tiw.CACHE, on /data not the repo):
  obs_<product>.npz          obs/weight/depth/lon/lat/day (shared across runs)
  model_<run>_<product>.npy  model sampled to obs, same shape as obs
  grid_<run>_<field>.nc      gridded model SST/SSH anomaly (for maps)
  metrics.csv                run,product,metric,value

Usage: compute_metrics.py [run ...]   (default: prior noVel Vel + tiw.CONTROLS)
  recomputes the named runs and merges with existing metrics.csv rows.
"""
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import tiw

CACHE = tiw.CACHE                      # on /data, not the repo
os.makedirs(CACHE, exist_ok=True)

# profile products: obs source file (a *_model.nc, used only for obs columns) + vars
PROFILE_PRODUCTS = {
    'argo': dict(root='USGO_TP_2012_PFL_D_w', vars=('T', 'S')),
    'tao':  dict(root='TAO_WO_2012_ADCP_v2',  vars=('U', 'V')),
}
# obs columns live in whichever run has the *_model.nc; obs are identical.
OBS_SOURCE = {'argo': 'initialT', 'tao': 'initialT'}
SEP1 = 244  # 0-based day-of-year index of 2012-09-01 (leap year)


def day_index(obs):
    return (pd.to_datetime(obs['time'].values).floor('D')
            - pd.Timestamp(tiw.REF_DATE)).days.values


def do_profiles(rows, runs):
    for prod, spec in PROFILE_PRODUCTS.items():
        obs = tiw.load_profile_obs(OBS_SOURCE[prod], spec['root'])
        shared = dict(depth=obs['depth'].values, lon=obs['lon'].values,
                      lat=obs['lat'].values, day=day_index(obs))
        for var in spec['vars']:
            shared[f'obs_{var}'] = obs[f'prof_{var}'].values
            shared[f'w_{var}'] = obs[f'prof_{var}weight'].values
        np.savez(f'{CACHE}/obs_{prod}.npz', **shared)

        for run in runs:
            ds = tiw.load_run(run)
            for var in spec['vars']:
                fld = ds[tiw.FIELD[var][0]]
                man = tiw.sample_profiles(fld, obs, mask=tiw.wet_mask(ds, fld))
                np.save(f'{CACHE}/model_{run}_{prod}_{var}.npy', man)
                o, w = shared[f'obs_{var}'], shared[f'w_{var}']
                rows += [
                    (run, f'{prod}_{var}', 'rmse', tiw.rmse(o, man, w)),
                    (run, f'{prod}_{var}', 'wcost', tiw.wcost(o, man, w)),
                    (run, f'{prod}_{var}', 'bias', tiw.bias(o, man, w)),
                ]
                print(f'  {run:9s} {prod}_{var}: rmse={rows[-3][3]:.4f}', flush=True)


def _surface_mask(ds):
    return (ds['hFacC'].isel(Z=0) > 0).values


def _obs_at(model_days, full366):
    """Select gridded daily obs (ny,nx,366 for 2012) at the model's actual days."""
    idx = (pd.to_datetime(model_days) - pd.Timestamp('2012-01-01')).days
    return np.moveaxis(full366[:, :, idx], 2, 0)      # (n, ny, nx)


def _has(run, prefix):
    import glob
    return bool(glob.glob(f'{tiw.RUNS[run][0]}/{prefix}.*.data'))


def do_grids(rows, runs):
    ref = tiw.load_run('Vel', prefix=('diag_state',))
    ny, nx = ref.sizes['YC'], ref.sizes['XC']
    wetC = _surface_mask(ref)
    sst_full, _ = tiw.load_sst(nx, ny)                # (ny,nx,366)
    rads_full = tiw.load_rads(nx, ny)                 # dict ja/c2 (ny,nx,366)

    for run in runs:
        ds = tiw.load_run(run, prefix=('diag_state',))
        md = ds['time'].values
        # ---- SST: mw_fusion vs surface THETA (every run has THETA) ----
        mod = np.where(wetC[None], ds['THETA'].isel(Z=0).values, np.nan)
        d = mod - _obs_at(md, sst_full)
        v = np.isfinite(d)
        rows += [(run, 'sst', 'rmse', float(np.sqrt(np.nanmean(d[v] ** 2)))),
                 (run, 'sst', 'bias', float(np.nanmean(d[v])))]
        xr.DataArray(mod, dims=['time', 'YC', 'XC'],
                     coords={'time': md, 'YC': ref.YC, 'XC': ref.XC}
                     ).to_netcdf(f'{CACHE}/grid_{run}_sst.nc')
        print(f'  {run:9s} sst: rmse={rows[-2][3]:.4f}', flush=True)

        # ---- SSH: RADS SLA vs ETAN anomaly (only runs with diag_surf) ----
        if not _has(run, 'diag_surf'):
            continue
        dss = tiw.load_run(run, prefix=('diag_surf',))
        eta = np.where(wetC[None], dss['ETAN'].values, np.nan)
        eta = eta - np.nanmean(eta, axis=0, keepdims=True)
        xr.DataArray(eta, dims=['time', 'YC', 'XC'],
                     coords={'time': dss['time'].values, 'YC': ref.YC, 'XC': ref.XC}
                     ).to_netcdf(f'{CACHE}/grid_{run}_ssh.nc')
        diffs = []
        for full in rads_full.values():
            o = _obs_at(dss['time'].values, full)
            o = o - np.nanmean(o, axis=0, keepdims=True)
            dd = eta - o
            diffs.append(dd[np.isfinite(dd)])
        dd = np.concatenate(diffs)
        rows += [(run, 'ssh', 'rmse', float(np.sqrt(np.mean(dd ** 2)))),
                 (run, 'ssh', 'bias', float(np.mean(dd)))]
        print(f'  {run:9s} ssh: rmse={rows[-2][3]:.4f}', flush=True)


# runs for the current (initialT) analysis; add CONTROLS as they're verified.
DEFAULT_RUNS = ['prior', 'noVel', 'Vel'] + tiw.CONTROLS


def main(runs):
    csv = f'{CACHE}/metrics.csv'
    rows = []
    if os.path.exists(csv):                           # keep rows for runs we aren't redoing
        old = pd.read_csv(csv)
        rows += [r for r in old.itertuples(index=False, name=None) if r[0] not in runs]
    do_profiles(rows, runs)
    do_grids(rows, runs)
    pd.DataFrame(rows, columns=['run', 'product', 'metric', 'value']).to_csv(csv, index=False)
    print('wrote', csv, flush=True)


if __name__ == '__main__':
    main(sys.argv[1:] or DEFAULT_RUNS)
