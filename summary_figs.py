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
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.signal import hilbert
import tiw
import anova

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

    COLS = [('noVel', 'Vel', 'All TAO obs\n(noVel→Vel)'),
            ('noVel', 'TSOnly', 'via TAO T/S\n(noVel→TSOnly)'),
            ('TSOnly', 'Vel', 'via velocity obs\n(TSOnly→Vel)')]
    ms = crop_coarsen(mean_sst)
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.6), sharex=True, sharey=True)
    for ax, (a, b, title) in zip(axs, COLS):
        red = crop_coarsen(rmse[a] - rmse[b])
        pc = ax.pcolormesh(red.XC, red.YC, red.values, cmap='RdBu_r',
                           vmin=-0.4, vmax=0.4, shading='auto')
        ys, xs = np.where(red.values > 0.08)
        ax.plot(red.XC.values[xs], red.YC.values[ys], '.', color='k', ms=1.1, alpha=0.3)
        ax.contour(ms.XC, ms.YC, ms.values, levels=[23, 25], colors='0.3',
                   linewidths=1.0, linestyles='--')
        for lon0 in TAO_LONS:
            ax.axvline(lon0, color='0.5', lw=0.5, alpha=0.4)
        ax.plot([m[0] for m in TAO_EQ], [m[1] for m in TAO_EQ], 'kv', ms=8,
                markerfacecolor='yellow', markeredgewidth=0.9)
        ax.set_title(title, fontsize=12); ax.set_xlabel('lon (°E)')
    axs[0].set_ylabel('lat (°N)')
    fig.colorbar(pc, ax=axs, shrink=0.85, pad=0.01, label='SST RMSE reduction (°C)')
    fig.suptitle('Where TAO reduces SST error, split by obs type  (red = error reduced; '
                 '▼ eq. moorings, dashed = array lons, gray = 23/25 °C front; stipple > 0.08 °C)',
                 fontsize=12)
    fig.savefig(f'{OUT}/skill_maps_sst_by_obs.png', dpi=150, bbox_inches='tight'); plt.close(fig)
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
    fig, axs = plt.subplots(1, 4, figsize=(16, 7), sharey=True)
    for ax, (r, h, lab) in zip(axs, panels):
        pc = ax.pcolormesh(h.XC, days, h.values, cmap='RdBu_r', vmin=-1.2, vmax=1.2,
                           shading='auto')
        if r != 'obs':                 # overlay obs TIW crests as contours
            ax.contour(oh.XC, days, oh.values, levels=[-0.6, 0.6], colors='k',
                       linewidths=0.6, alpha=0.7)
        ax.set_title(lab, fontsize=12); ax.set_xlabel('lon (°E)')
        ax.invert_yaxis()
    axs[0].set_ylabel('time (2012)')
    fig.colorbar(pc, ax=axs, shrink=0.85, pad=0.01, label='SST anomaly (°C), 2–6°N mean')
    fig.suptitle('SST TIW Hovmöller (2–6°N): when & where the TIWs appear; black contours = obs crests\n'
                 '(do the model crests line up with obs in longitude and time?)', fontsize=12)
    fig.savefig(f'{OUT}/tiw_hovmoller_sst.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    print('wrote tiw_hovmoller_sst.png')


# ============================ (4) EUC depth-lon section ===========================
def fig_euc_section():
    E = {r: xr.open_dataset(f'{CACHE}/euctiw_{r}.nc')['euc'] for r in ('noVel', 'TSOnly', 'Vel')}
    sl = dict(Z=slice(0, -300), XG=slice(180, 280))

    def cr(da):
        return da.sel(XG=slice(180, 280)).where(da.Z > -300, drop=True)
    vel, ts_imp, v_imp = cr(E['Vel']), cr(E['TSOnly'] - E['noVel']), cr(E['Vel'] - E['TSOnly'])
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.8), sharey=True)
    p0 = axs[0].pcolormesh(vel.XG, vel.Z, vel.values, cmap='RdBu_r', vmin=-1, vmax=1, shading='auto')
    axs[0].contour(vel.XG, vel.Z, vel.values, levels=[0.2, 0.5, 0.8], colors='k', linewidths=0.6)
    axs[0].set_title('Vel: mean zonal current (EUC)')
    fig.colorbar(p0, ax=axs[0], shrink=0.9, label='U (m/s)')
    for ax, (da, title) in zip(axs[1:], [(ts_imp, 'via TAO T/S  (TSOnly−noVel)'),
                                         (v_imp, 'via velocity obs  (Vel−TSOnly)')]):
        pc = ax.pcolormesh(da.XG, da.Z, da.values, cmap='PuOr_r', vmin=-0.3, vmax=0.3, shading='auto')
        ax.contour(vel.XG, vel.Z, vel.values, levels=[0.2, 0.5], colors='0.4', linewidths=0.5)
        ax.set_title(title)
        fig.colorbar(pc, ax=ax, shrink=0.9, label='ΔU (m/s)')
    for ax in axs:
        for lon0, _ in MOORINGS:
            ax.axvline(lon0, color='0.5', lw=0.6, ls=':')
        ax.set_xlabel('lon (°E)')
    axs[0].set_ylabel('depth (m)')
    fig.suptitle('Equatorial undercurrent (time-mean U): how TAO reshapes it, via T/S vs via velocity obs '
                 '(dotted = 170/140/110°W moorings)', fontsize=12)
    fig.savefig(f'{OUT}/euc_section_by_obs.png', dpi=150, bbox_inches='tight'); plt.close(fig)
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
                            label='obs TIW amplitude' if i == 0 and j == 0 else None, zorder=0)
            for run, color, lab in SOLUTIONS:
                vo2, vm = _mooring_series(run, var, mlon)
                ax.plot(tiw.DAYS, _tiw_envelope(vm - vo2), color=color, lw=1.8,
                        ls='--' if run == 'prior' else '-',
                        label=lab if i == 0 and j == 0 else None)
            ax.grid(alpha=0.25); ax.set_title(f'{mname}  TAO {var}', fontsize=11)
            if j == 0:
                ax.set_ylabel('|TIW-band error| (m/s)')
    axs[0, 0].legend(fontsize=8, loc='upper right')
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


# agreement colors: hot pink where the model matches the observed V sign, black where it
# doesn't. A thin white halo (below) keeps both legible over red, blue AND white shading.
AGREE, DISAGREE = '#ff1493', '#000000'


def _agree_column(ax, lon0, obs, mod, ydates, lw):
    """Vertical line at lon0, colored green where sign(mod)==sign(obs), magenta otherwise."""
    from matplotlib.colors import ListedColormap
    y = mdates.date2num(ydates)
    x = np.full_like(y, lon0, dtype=float)
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    agree = (np.sign(obs) == np.sign(mod)).astype(float)
    import matplotlib.patheffects as pe
    lc = LineCollection(segs, cmap=ListedColormap([DISAGREE, AGREE]),
                        norm=plt.Normalize(0, 1), zorder=6)
    lc.set_array(agree[:-1]); lc.set_linewidth(lw); lc.set_capstyle('butt')
    lc.set_path_effects([pe.Stroke(linewidth=lw + 1.8, foreground='white'), pe.Normal()])
    ax.add_collection(lc)


def fig_vhov_mooring():
    """V@50m Hovmoller (noVel/TSOnly/Vel); at each mooring lon a thin column colored by
    whether the modeled V agrees in SIGN with the observed mooring V (TIW phase correct?)."""
    VMAX = 0.3
    hov = {r: _vhov50(r) for r in ('noVel', 'TSOnly', 'Vel')}
    obsw = {}
    for mlon, _ in MOORINGS:
        vo, _ = _mooring_series('Vel', 'V', mlon, zband=(-60, -40))
        vo = pd.Series(vo).interpolate(limit_direction='both').values
        obsw[mlon] = vo - np.nanmean(vo)

    fig, axs = plt.subplots(1, 3, figsize=(16, 8), sharey=True)
    for ax, r in zip(axs, ('noVel', 'TSOnly', 'Vel')):
        h = hov[r]; ha = h - h.mean('time')
        pc = ax.pcolormesh(h.XC, tiw.DAYS, ha.values, cmap='RdBu_r',
                           vmin=-VMAX, vmax=VMAX, shading='auto')
        for mlon, mname in MOORINGS:
            mod = ha.sel(XC=mlon, method='nearest').values         # this run's V at the mooring
            _agree_column(ax, mlon, obsw[mlon], mod, tiw.DAYS, lw=3.5)
            ax.text(mlon, tiw.DAYS[-1], mname, fontsize=8, ha='center', va='top',
                    bbox=dict(fc='white', alpha=0.8, ec='none', pad=0.6))
        ax.set_title(r, fontsize=13, fontweight='bold')
        ax.set_xlabel('lon (°E)'); ax.set_xlim(*VHOV['lon']); ax.invert_yaxis()
    axs[0].set_ylabel('time (2012)')
    fig.colorbar(pc, ax=axs, shrink=0.8, pad=0.01, label='model V @ 50 m anomaly (m/s)')
    leg = [Line2D([0], [0], color=AGREE, lw=5, label='model agrees with obs V sign'),
           Line2D([0], [0], color=DISAGREE, lw=5, label='model disagrees (wrong TIW phase)')]
    fig.legend(handles=leg, fontsize=10, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.02), framealpha=1)
    fig.suptitle('Equatorial V at 50 m (±1°N): where each model agrees with the observed mooring V',
                 fontsize=14)
    fig.text(0.5, 0.925, 'columns at 170/140/110°W: pink = modeled V has the observed sign, '
             'black = wrong sign. More pink west→east and noVel→Vel = better TIW phase.',
             ha='center', fontsize=9, style='italic')
    fig.savefig(f'{OUT}/vhov50_mooring_overlay.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    print('wrote vhov50_mooring_overlay.png')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    fig_error_profiles()
    fig_skill_maps()
    fig_tiw_hovmoller()
    fig_euc_section()
    fig_error_spectrum()
    fig_filtered_timeseries()
    fig_error_envelope()
    fig_vhov_mooring()
