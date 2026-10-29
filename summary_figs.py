"""
Summary comparison figures across the three full solutions (noVel / TSOnly / Vel),
focused on SST, Argo T/S and TAO U/V (SSH deliberately excluded). Deliverables land
in figs/summary/. Reuses the cached obs/model samples, grids, and euctiw fields, plus
the TAO daily-series helpers in anova.py.

Env: ~/miniforge3/envs/tpose/bin/python
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.signal import hilbert, find_peaks
import tiw
import anova
import make_figures as mf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f'{HERE}/figs/summary'
CACHE = tiw.CACHE

# distinct, colorblind-friendly palette (Okabe-Ito) — used across the summary figs
PRIOR, NOVEL, TSONLY, VEL = '#7a7a7a', '#D55E00', '#009E73', '#0072B2'
SOLUTIONS = [('prior', PRIOR, 'prior (no DA)'),
             ('noVel', NOVEL, 'noVel (noTAO)'),
             ('TSOnly', TSONLY, 'TSOnly (TAO T/S)'),
             ('Vel', VEL, 'Vel (TAO T/S/U/V)')]
PROF = [('argo', 'T', 'Argo θ', '°C', -500),
        ('argo', 'S', 'Argo S', 'psu', -500),
        ('tao', 'U', 'TAO U', 'm/s', -315),
        ('tao', 'V', 'TAO V', 'm/s', -315)]
MOORINGS = [(190.0, '170°W'), (220.0, '140°W'), (250.0, '110°W')]
ZBAND = (-150.0, -20.0)          # subsurface depth band for mooring series
TIW_PER = (15.0, 40.0)           # TIW period band (days)


# ============================ (1) depth error profiles ============================
def fig_error_profiles():
    fig, axs = plt.subplots(1, 4, figsize=(15, 6.5))
    for ax, (prod, var, label, unit, zlim) in zip(axs, PROF):
        o = np.load(f'{CACHE}/obs_{prod}.npz')
        depth, obs, w = o['depth'], o[f'obs_{var}'], o[f'w_{var}']
        for run, color, rlabel in SOLUTIONS:
            m = np.load(f'{CACHE}/model_{run}_{prod}_{var}.npy')
            dashed = run == 'prior'
            ax.plot(tiw.rmse_profile(obs, m, w), depth, color=color,
                    lw=2.6 if run in ('Vel', 'TSOnly') else 1.9,
                    ls='--' if dashed else '-', label=rlabel,
                    marker='o' if dashed else None, ms=3, markevery=4, zorder=3)
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_xlabel(f'RMSE ({unit})'); ax.set_ylim(zlim, 0); ax.set_xlim(left=0)
        ax.grid(alpha=0.25, lw=0.6); ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=10)
    axs[0].set_ylabel('depth (m)', fontsize=12)
    axs[0].legend(fontsize=10, loc='lower right', frameon=True, framealpha=0.9)
    fig.suptitle('Error vs depth: prior → noVel → TSOnly → Vel\n'
                 'TAO T/S corrects θ/S everywhere; only the velocity constraint pulls '
                 'TAO U/V error down through the thermocline',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f'{OUT}/error_profiles_depth.png', dpi=150); plt.close(fig)
    print('wrote error_profiles_depth.png')


def fig_error_and_decomp_profiles():
    fig = plt.figure(figsize=(15, 11.5), constrained_layout=True)
    sfigs = fig.subfigures(2, 1, hspace=0.14)
    top, bot = sfigs[0].subplots(1, 4), sfigs[1].subplots(1, 4)
    # top row: prior -> noVel -> TSOnly -> Vel
    for ax, (prod, var, label, unit, zlim) in zip(top, PROF):
        o = np.load(f'{CACHE}/obs_{prod}.npz')
        depth, obs, w = o['depth'], o[f'obs_{var}'], o[f'w_{var}']
        for run, color, rlabel in SOLUTIONS:
            m = np.load(f'{CACHE}/model_{run}_{prod}_{var}.npy')
            dashed = run == 'prior'
            ax.plot(tiw.rmse_profile(obs, m, w), depth, color=color,
                    lw=1.9 if dashed else 2.6,
                    ls='--' if dashed else '-', label=rlabel,
                    marker='o' if dashed else None, ms=3, markevery=4, zorder=3)
        ax.set_title(label, fontsize=13)
        ax.set_xlabel(f'RMSE ({unit})'); ax.set_ylim(zlim, 0); ax.set_xlim(left=0)
        ax.grid(alpha=0.25, lw=0.6); ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=10)
    top[0].set_ylabel('depth (m)', fontsize=12)
    top[1].legend(fontsize=12, loc='lower right', frameon=True, framealpha=0.9)
    sfigs[0].suptitle('Full Solutions', fontsize=16, fontweight='bold')
    # bottom row: single-control & combo decomposition vs full solution
    for ax, (prod, var, label, unit, _) in zip(bot, PROF):
        obs = np.load(f'{CACHE}/obs_{prod}.npz')
        depth = obs['depth']; o = obs[f'obs_{var}']; w = obs[f'w_{var}']
        for run, color, lbl, kw in mf.DECOMP_PROFILE_RUNS:
            fp = f'{CACHE}/model_{run}_{prod}_{var}.npy'
            if not os.path.exists(fp):
                continue
            if run == 'prior':
                style = dict(lw=1.9, ls='--', marker='o', ms=3, markevery=4)
            else:
                style = dict(lw=1.3); style.update(kw)
            ax.plot(tiw.rmse_profile(o, np.load(fp), w), depth,
                    color=color, label=lbl or mf._dlabel(run), **style)
        ax.set_title(label, fontsize=13)
        ax.set_xlabel(f'RMSE ({unit})'); ax.set_ylim(-500 if prod == 'argo' else -315, 0)
        ax.set_xlim(left=0); ax.grid(alpha=0.25, lw=0.6)
        ax.spines[['top', 'right']].set_visible(False); ax.tick_params(labelsize=10)
    bot[0].set_ylabel('depth (m)', fontsize=12)
    bot[1].legend(fontsize=10, loc='lower right')
    sfigs[1].suptitle('TPOSE-Vel Withheld Controls Experiments', fontsize=16, fontweight='bold')
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f'{OUT}/error_and_decomp_profiles.png', dpi=150); plt.close(fig)
    print('wrote error_and_decomp_profiles.png')


# ============================ (2) SST skill decomposition map =====================
REGION = dict(lon=(170, 285), lat=(-10, 10))
TAO_LONS = [165, 180, 190, 220, 250, 265]
TAO_EQ = [(190, 0), (220, 0), (250, 0)]


def _obs_on_days(full366, days):
    idx = (pd.to_datetime(days) - pd.Timestamp('2012-01-01')).days
    return np.moveaxis(full366[:, :, idx], 2, 0)


def _sst_rmse_map(run, obs_days):
    m = xr.open_dataarray(f'{CACHE}/grid_{run}_sst.nc').values
    d = m - obs_days
    return np.sqrt(np.nanmean(d ** 2, axis=0))


def fig_skill_maps():
    ref = xr.open_dataarray(f'{CACHE}/grid_Vel_sst.nc')
    days, yc, xc = ref['time'].values, ref['YC'].values, ref['XC'].values
    ny, nx = yc.size, xc.size
    sst_full, _ = tiw.load_sst(nx, ny)
    obs = _obs_on_days(sst_full, days)
    mean_sst = np.nanmean(obs, axis=0)
    rmse = {r: _sst_rmse_map(r, obs) for r in ('noVel', 'TSOnly', 'Vel')}

    def crop_coarsen(a):
        da = xr.DataArray(a, coords={'YC': yc, 'XC': xc}, dims=['YC', 'XC'])
        da = da.sel(YC=slice(*REGION['lat']), XC=slice(*REGION['lon']))
        return da.coarsen(YC=3, XC=3, boundary='trim').mean()

    COLS = [('Vel', 'noVel', 'All TAO obs\nRMSE(Vel) − RMSE(noVel)'),
            ('TSOnly', 'noVel', 'via TAO T/S\nRMSE(TSOnly) − RMSE(noVel)'),
            ('Vel', 'TSOnly', 'via velocity obs\nRMSE(Vel) − RMSE(TSOnly)')]
    fig, axs = plt.subplots(1, 3, figsize=(22, 3.6), sharex=True, sharey=True,
                            constrained_layout=True)
    for ax, (a, b, title) in zip(axs, COLS):
        red = crop_coarsen(rmse[a] - rmse[b])
        pc = ax.pcolormesh(red.XC, red.YC, red.values, cmap='RdBu_r',
                           vmin=-0.4, vmax=0.4, shading='auto')
        ax.plot([m[0] for m in TAO_EQ], [m[1] for m in TAO_EQ], 'kv', ms=8,
                markerfacecolor='yellow', markeredgewidth=0.9)
        ax.set_title(title, fontsize=12); ax.set_xlabel('lon (°E)')
    axs[0].set_ylabel('lat (°N)')
    fig.colorbar(pc, ax=axs, shrink=0.85, label='SST RMSE difference (°C)')
    fig.savefig(f'{OUT}/skill_maps_sst_by_obs.png', dpi=150); plt.close(fig)
    print('wrote skill_maps_sst_by_obs.png')


# ============================ (3) SST TIW Hovmoller ===============================
HOV = dict(lat=(2, 6), lon=(180, 280))


def _sst_hov(arr_tyx, yc, xc):
    da = xr.DataArray(arr_tyx, dims=['time', 'YC', 'XC'], coords={'YC': yc, 'XC': xc})
    h = da.sel(YC=slice(*HOV['lat']), XC=slice(*HOV['lon'])).mean('YC')
    return h - h.mean('time')          # anomaly from time-mean (reveals propagating TIWs)


def fig_tiw_hovmoller():
    ref = xr.open_dataarray(f'{CACHE}/grid_Vel_sst.nc')
    days, yc, xc = ref['time'].values, ref['YC'].values, ref['XC'].values
    sst_full, _ = tiw.load_sst(yc.size * 0 + xc.size, yc.size)
    oh = _sst_hov(_obs_on_days(sst_full, days), yc, xc)
    panels = [('obs', oh, 'mw_fusion obs')] + [
        (r, _sst_hov(xr.open_dataarray(f'{CACHE}/grid_{r}_sst.nc').values, yc, xc), lab)
        for r, lab in [('noVel', 'noVel'), ('TSOnly', 'TSOnly'), ('Vel', 'Vel')]]
    fig, axs = plt.subplots(2, 4, figsize=(16, 12), sharex=True, sharey=True,
                            constrained_layout=True)
    for j, (r, h, lab) in enumerate(panels):
        pa = axs[0, j].pcolormesh(h.XC, days, h.values, cmap='RdBu_r', vmin=-1.2, vmax=1.2,
                                  shading='auto')
        if r != 'obs':                 # overlay obs TIW crests as contours
            axs[0, j].contour(oh.XC, days, oh.values, levels=[-0.6, 0.6], colors='k',
                              linewidths=0.6, alpha=0.7)
        axs[0, j].set_title(lab, fontsize=12)
        if r == 'obs':                 # row 2, col 1 is blank (obs has no error vs itself)
            axs[1, j].axis('off')
        else:
            pe = axs[1, j].pcolormesh(h.XC, days, (h - oh).values, cmap='PuOr_r',
                                      vmin=-1.0, vmax=1.0, shading='auto')
            axs[1, j].set_title(f'{lab} − obs', fontsize=12)
            axs[1, j].set_xlabel('lon (°E)')
    axs[0, 0].invert_yaxis()
    axs[0, 0].set_ylabel('time (2012)'); axs[1, 0].set_ylabel('time (2012)')
    fig.colorbar(pa, ax=axs[0, :], shrink=0.8, label='SST anomaly (°C)')
    fig.colorbar(pe, ax=axs[1, 1:], shrink=0.8, label='SST error, model − obs (°C)')
    fig.suptitle('SST TIW Hovmöller (2–6°N): anomaly (top) and model error (bottom). '
                 'When & where do the model TIWs match the observed ones?', fontsize=13)
    fig.savefig(f'{OUT}/tiw_hovmoller_sst.png', dpi=150); plt.close(fig)
    print('wrote tiw_hovmoller_sst.png')


# ============================ (4) EUC depth-lon section ===========================
def fig_euc_section():
    E = {r: xr.open_dataset(f'{CACHE}/euctiw_{r}.nc')['euc'] for r in ('noVel', 'TSOnly', 'Vel')}
    sl = dict(Z=slice(0, -300), XG=slice(180, 280))

    def cr(da):
        return da.sel(XG=slice(180, 280)).where(da.Z > -300, drop=True)
    vel, ts_imp, v_imp = cr(E['Vel']), cr(E['TSOnly'] - E['noVel']), cr(E['Vel'] - E['TSOnly'])
    fig, axs = plt.subplots(1, 3, figsize=(21, 4.2), sharey=True, constrained_layout=True)
    p0 = axs[0].pcolormesh(vel.XG, vel.Z, vel.values, cmap='RdBu_r', vmin=-1, vmax=1, shading='auto')
    axs[0].contour(vel.XG, vel.Z, vel.values, levels=[0.2, 0.5, 0.8], colors='k', linewidths=0.6)
    axs[0].set_title('TPOSE-Vel: mean zonal current')
    fig.colorbar(p0, ax=axs[0], shrink=0.9, label='U (m/s)')
    for ax, (da, title) in zip(axs[1:], [(ts_imp, 'Influence of TAO T/S (TSOnly − noVel)'),
                                         (v_imp, 'Influence of TAO U/V (Vel − TSOnly)')]):
        pc = ax.pcolormesh(da.XG, da.Z, da.values, cmap='PuOr_r', vmin=-0.3, vmax=0.3, shading='auto')
        ax.contour(vel.XG, vel.Z, vel.values, levels=[0.2, 0.5, 0.8], colors='0.4', linewidths=0.5)
        ax.set_title(title)
        fig.colorbar(pc, ax=ax, shrink=0.9, label='ΔU (m/s)')
    for ax in axs:
        ax.set_xlabel('lon (°E)')
    axs[0].set_ylabel('depth (m)')
    fig.savefig(f'{OUT}/euc_section_by_obs.png', dpi=150); plt.close(fig)
    print('wrote euc_section_by_obs.png')


# ====================== (5,6) TAO U/V frequency + filtered series =================
def _mooring_series(run, var, mlon, zband=ZBAND):
    o = np.load(f'{CACHE}/obs_tao.npz')
    lon, lat, day, depth = o['lon'], o['lat'], o['day'], o['depth']
    sel = (np.abs(lon - mlon) < 0.6) & (np.abs(lat) < 0.6)
    zmask = (depth <= zband[1]) & (depth >= zband[0])
    obs_arr = np.where(o[f'w_{var}'] > 0, o[f'obs_{var}'], -9999.)
    mod = np.load(f'{CACHE}/model_{run}_tao_{var}.npy')
    do = anova._tao_daily(obs_arr, day, sel, tiw.NT)[:, zmask]
    dm = anova._tao_daily(mod, day, sel, tiw.NT)[:, zmask]
    return np.nanmean(do, axis=1), np.nanmean(dm, axis=1)     # (nt,) obs, model


def _psd(x):
    x = pd.Series(x).interpolate(limit_direction='both').values
    x = x - np.nanmean(x)
    n = x.size; w = np.hanning(n)
    A = np.fft.rfft(x * w); f = np.fft.rfftfreq(n, 1.0)
    return f[1:], (np.abs(A) ** 2 / (w ** 2).sum())[1:]


def _bp(x, pmin, pmax):
    x = pd.Series(x).interpolate(limit_direction='both').values
    x = x - np.nanmean(x)
    n = x.size; A = np.fft.rfft(x); f = np.fft.rfftfreq(n, 1.0)
    per = np.where(f > 0, 1.0 / np.where(f > 0, f, np.nan), np.inf)
    return np.fft.irfft(A * ((per >= pmin) & (per <= pmax)), n)


def fig_error_spectrum():
    """Where in frequency does each solution's TAO U/V error live (moorings-averaged)."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    for j, var in enumerate(('U', 'V')):
        # moorings-averaged series
        series = {r: [] for r, *_ in SOLUTIONS}
        obs_series = []
        for mlon, _ in MOORINGS:
            vo, _ = _mooring_series('Vel', var, mlon)
            obs_series.append(vo)
            for r, *_ in SOLUTIONS:
                series[r].append(_mooring_series(r, var, mlon)[1])
        # field variance spectrum (top) + error spectrum (bottom)
        fo = [_psd(s) for s in obs_series]
        f = fo[0][0]; per = 1.0 / f
        Pobs = np.nanmean([p for _, p in fo], axis=0)
        axs[0, j].plot(per, Pobs, 'k', lw=2.4, label='TAO obs', zorder=5)
        for r, color, lab in SOLUTIONS:
            Pf = np.nanmean([_psd(s)[1] for s in series[r]], axis=0)
            Pe = np.nanmean([_psd(series[r][i] - obs_series[i])[1] for i in range(len(MOORINGS))], axis=0)
            axs[0, j].plot(per, Pf, color=color, lw=1.8, label=lab)
            axs[1, j].plot(per, Pe, color=color, lw=1.8, label=lab)
        for row in (0, 1):
            ax = axs[row, j]
            ax.axvspan(*TIW_PER, color='gold', alpha=0.18, label='TIW band' if row == 0 else None)
            ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(60, 4)
            ax.grid(alpha=0.25, which='both', lw=0.5)
            ax.set_xlabel('period (days)')
        axs[0, j].set_title(f'TAO {var}: variance spectrum (obs vs solutions)')
        axs[1, j].set_title(f'TAO {var}: ERROR spectrum (model − obs)')
    axs[0, 0].set_ylabel('PSD'); axs[1, 0].set_ylabel('error PSD')
    axs[0, 0].legend(fontsize=8)
    fig.suptitle('Where in frequency the TAO U/V signal and error live (3 eq. moorings, 20–150 m)\n'
                 'top: does each solution match obs variance? bottom: at which periods does error remain?',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{OUT}/velocity_error_spectrum.png', dpi=150); plt.close(fig)
    print('wrote velocity_error_spectrum.png')


def fig_filtered_timeseries():
    """TIW-band vs low-frequency TAO U/V at 140°W: obs vs each solution."""
    mlon = 220.0
    fig, axs = plt.subplots(2, 2, figsize=(15, 8), sharex=True)
    for j, var in enumerate(('U', 'V')):
        vo, _ = _mooring_series('Vel', var, mlon)
        for row, (pmin, pmax, bl) in enumerate([(TIW_PER[0], TIW_PER[1], 'TIW band 15–40 d'),
                                                (40, 1e6, 'low-freq >40 d')]):
            ax = axs[row, j]
            ax.plot(tiw.DAYS, _bp(vo, pmin, pmax), 'k', lw=2.4, label='TAO obs', zorder=5)
            for r, color, lab in SOLUTIONS:
                _, vm = _mooring_series(r, var, mlon)
                ax.plot(tiw.DAYS, _bp(vm, pmin, pmax), color=color, lw=1.6, label=lab)
            ax.set_title(f'TAO {var} @ 140°W, {bl}'); ax.grid(alpha=0.25)
            ax.set_ylabel(f'{var} (m/s)')
    axs[0, 0].legend(fontsize=8, ncol=2)
    for ax in axs[-1, :]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    fig.suptitle('Filtered TAO U/V at 140°W (20–150 m): which solution tracks obs in the TIW band vs at low freq',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{OUT}/velocity_filtered_timeseries.png', dpi=150); plt.close(fig)
    print('wrote velocity_filtered_timeseries.png')


# ===================== (7) TIW-band error envelope over time =====================
def _tiw_envelope(x):
    """|analytic signal| of the TIW-band-passed series — the slowly-varying amplitude."""
    return np.abs(hilbert(_bp(x, *TIW_PER)))


def fig_error_envelope():
    """When is each solution's TAO U/V error large at the TIW timescale? Rows=moorings,
    cols=U/V; each line = |TIW-band (model-obs)| envelope for one solution."""
    fig, axs = plt.subplots(len(MOORINGS), 2, figsize=(14, 9), sharex=True)
    for i, (mlon, mname) in enumerate(MOORINGS):
        for j, var in enumerate(('U', 'V')):
            ax = axs[i, j]
            # reference: obs TIW amplitude envelope (how big the real TIW signal is)
            vo, _ = _mooring_series('Vel', var, mlon)
            ax.fill_between(tiw.DAYS, 0, _tiw_envelope(vo), color='0.85',
                            label='obs TIW amplitude' if i == 0 and j == 1 else None, zorder=0)
            for run, color, lab in SOLUTIONS:
                vo2, vm = _mooring_series(run, var, mlon)
                ax.plot(tiw.DAYS, _tiw_envelope(vm - vo2), color=color, lw=1.8,
                        ls='--' if run == 'prior' else '-',
                        label=lab if i == 0 and j == 1 else None)
            ax.grid(alpha=0.25); ax.set_title(f'{mname}  TAO {var}', fontsize=11)
            if j == 0:
                ax.set_ylabel('|TIW-band error| (m/s)')
    axs[0, 1].legend(fontsize=8, loc='upper left')
    for ax in axs[-1, :]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    fig.suptitle('TIW-band error envelope over time (20–150 m): when does each solution '
                 'mis-fit the observed TIW velocity?\n(gray fill = observed TIW amplitude; '
                 'lower line = better TIW tracking)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f'{OUT}/velocity_error_envelope.png', dpi=150); plt.close(fig)
    print('wrote velocity_error_envelope.png')


# ============ (8) V @ 50 m Hovmoller with mooring-V wiggle overlays ===============
VHOV = dict(lat=(-1.0, 1.0), lon=(180, 280), z=-50.0)


def _vhov50(run):
    """Equatorial (±1°) meridional velocity at ~50 m as a (time, lon) Hovmoller.
    Cached (reads diag_state once) so the figure is cheap to re-render."""
    c = f'{CACHE}/vhov50_{run}.nc'
    if os.path.exists(c):
        return xr.open_dataarray(c)
    ds = tiw.load_run(run)
    v = (ds['VVEL'].sel(Z=VHOV['z'], method='nearest')
         .sel(YG=slice(*VHOV['lat'])).mean('YG').sel(XC=slice(*VHOV['lon'])))
    v = v.reindex(time=tiw.DAYS).load()
    v.to_netcdf(c)
    return v


DEG_M = 111320.0                 # m per degree longitude at the equator


def _phase_speed(a, dlon):
    """Dominant westward TIW phase speed in a (time, lon) box, via the 2D-FFT peak in the
    TIW period/wavelength band. Returns (c_m_per_s<0 west, slope deg/day)."""
    a = a - a.mean()
    nt, nx = a.shape
    P = np.abs(np.fft.fft2(a)) ** 2
    f = np.fft.fftfreq(nt, 1.0)              # cycles/day
    k = np.fft.fftfreq(nx, dlon)             # cycles/deg
    F, K = np.meshgrid(f, k, indexing='ij')
    with np.errstate(divide='ignore'):
        per = np.where(F != 0, 1 / np.abs(F), np.inf)       # days
        wav = np.where(K != 0, 1 / np.abs(K), np.inf)       # deg
    # westward: phase fronts x = -(f/k) t move west => f,k same sign; take the f>0,k>0 rep.
    mask = (F > 0) & (K > 0) & (per >= 12) & (per <= 45) & (wav >= 4) & (wav <= 25)
    if not mask.any():
        return np.nan, np.nan
    i = np.argmax(np.where(mask, P, -np.inf))
    f0, k0 = F.flat[i], K.flat[i]
    slope = -f0 / k0                          # deg/day (negative = westward)
    return slope * DEG_M / 86400.0, slope


def fig_vhov_mooring():
    """V@50m equatorial Hovmoller (noVel/TSOnly/Vel) with lines tracing the dominant westward
    TIW events, annotated with their phase speed — shows how timing & propagation adjust."""
    VMAX = 0.5
    hov = {r: _vhov50(r) for r in ('noVel', 'TSOnly', 'Vel')}
    ha = {r: (hov[r] - hov[r].mean('time')) for r in hov}
    lon = ha['Vel'].XC.values
    dlon = float(np.mean(np.diff(lon)))
    # detect the 2 dominant TIW events from the Vel longitude-RMS amplitude envelope
    amp = np.sqrt((ha['Vel'].values ** 2).mean(axis=1))
    amp_s = pd.Series(amp).rolling(7, center=True, min_periods=1).mean().values
    pk, _ = find_peaks(amp_s, distance=20)
    pk = np.sort(pk[np.argsort(amp_s[pk])[::-1][:2]])
    events = [(max(0, p - 18), min(len(tiw.DAYS), p + 18)) for p in pk]

    fig, axs = plt.subplots(1, 3, figsize=(16, 8), sharey=True, constrained_layout=True)
    for ax, r in zip(axs, ('noVel', 'TSOnly', 'Vel')):
        pc = ax.pcolormesh(lon, tiw.DAYS, ha[r].values, cmap='RdBu_r',
                           vmin=-VMAX, vmax=VMAX, shading='auto')
        for (t0, t1) in events:
            box = ha[r].isel(time=slice(t0, t1)).values
            c_ms, slope = _phase_speed(box, dlon)
            if not np.isfinite(slope):
                continue
            bi = np.unravel_index(np.argmax(np.abs(box)), box.shape)  # anchor on strongest crest
            ta, la = t0 + bi[0], lon[bi[1]]
            tt = np.array([t0, t1 - 1])
            xx = la + slope * (tt - ta)
            ax.plot(xx, [tiw.DAYS[t0], tiw.DAYS[t1 - 1]], 'k', lw=2.2, zorder=6,
                    path_effects=[pe.Stroke(linewidth=4, foreground='white'), pe.Normal()])
            tm = (t0 + t1) // 2
            ax.annotate(f'{abs(c_ms):.2f} m/s', (la + slope * (tm - ta), tiw.DAYS[tm]),
                        fontsize=10, fontweight='bold', ha='left', va='center',
                        xytext=(6, 0), textcoords='offset points',
                        bbox=dict(fc='white', alpha=0.85, ec='none', pad=1), zorder=7)
        ax.set_title(r, fontsize=13, fontweight='bold')
        ax.set_xlabel('lon (°E)'); ax.set_xlim(lon.min(), lon.max()); ax.invert_yaxis()
    axs[0].set_ylabel('time (2012)')
    fig.colorbar(pc, ax=axs, shrink=0.85, label='V @ 50 m anomaly (m/s)')
    fig.suptitle('Equatorial V at 50 m (±1°N): dominant westward TIW events and their phase speed '
                 '(black lines; label = westward phase speed)', fontsize=13)
    fig.savefig(f'{OUT}/vhov50_mooring_overlay.png', dpi=150); plt.close(fig)
    print('wrote vhov50_mooring_overlay.png')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    fig_error_profiles()
    fig_error_and_decomp_profiles()
    fig_skill_maps()
    fig_tiw_hovmoller()
    fig_euc_section()
    fig_error_spectrum()
    fig_filtered_timeseries()
    fig_error_envelope()
    fig_vhov_mooring()
