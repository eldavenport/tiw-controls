"""
TIW spectra for the control decomposition: frequency, zonal-wavenumber, and
wavenumber-frequency (omega-k) spectra of SST, SSH, and top-50 m V, for the
prior / theta / u-wind / theta+u-wind / Full set, in both the Vel and noVel families.

Quantities are reduced to a (time, lon) Hovmoller over the TIW window
(lon 180-265 E, lat 2-5 N): SST/SSH from the gridded caches (lat-mean), V from
the cached euctiw 'tiv' (already 2-5 N, top-50 m). omega-k panels carry the
equatorial-wave dispersion curves (Kelvin, Yanai/MRG, Rossby n=1,2) + the TIW band.

Env: ~/miniforge3/envs/tpose/bin/python.  Figures -> figs/comparison/spectra/.
"""
import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import tiw

CACHE = tiw.CACHE
OUT = os.path.join(os.path.dirname(__file__), 'figs', 'comparison', 'spectra')
os.makedirs(OUT, exist_ok=True)

LON = (180.0, 265.0)        # TIW cold-tongue window (deg E)
LAT = (2.0, 5.0)            # TIW meridional band (matches the cached V)
DEG_M = 111320.0            # m per degree longitude (equatorial)
BETA = 2.28e-11             # 1/(m s), df/dy at the equator
CBC = 2.6                   # m/s, first baroclinic gravity speed (eq. Pacific)

# (key, short label) per family row; prior repeated in both rows.
ROWS = {
    'Vel':   [('prior', 'prior'), ('initialT', 'θ'), ('zonalWind', 'u-wind'),
              ('initialT_uwind', 'θ+u'), ('Vel', 'Full Vel')],
    'noVel': [('prior', 'prior'), ('nv_initialT', 'θ'), ('nv_zonalWind', 'u-wind'),
              ('nv_initialT_uwind', 'θ+u'), ('noVel', 'Full noVel')],
}
COLORS = ['0.5', 'tab:green', 'tab:purple', 'tab:orange', 'k']
# (name, source, unit) — source 'sst'/'ssh' = gridded, 'V' = cached tiv Hovmoller.
QTYS = [('SST', 'sst', '°C'), ('SSH', 'ssh', 'm'), ('V (top-50 m, 2–5°N)', 'V', 'm/s')]
TIW_PER = (15.0, 40.0)      # days
TIW_WAV = (700.0, 1600.0)   # km


def hovmoller(run, src):
    """(time, lon) anomaly Hovmoller over the TIW window; time on the 122-day axis."""
    if src == 'V':
        da = xr.open_dataset(f'{CACHE}/euctiw_{run}.nc')['tiv']
        h = da.reindex(time=tiw.DAYS).interpolate_na('time').sel(XC=slice(*LON))
    else:
        da = xr.open_dataarray(f'{CACHE}/grid_{run}_{src}.nc')
        h = (da.reindex(time=tiw.DAYS).interpolate_na('time')
             .sel(YC=slice(*LAT)).mean('YC').sel(XC=slice(*LON)))
    x = h['XC'].values
    a = h.values.astype(float)                          # (nt, nx)
    # fill any residual gaps (coastal NaN) so the FFT stays finite
    a = np.where(np.isfinite(a), a, np.nan)
    col = np.nanmean(a, axis=0, keepdims=True)
    a = np.where(np.isfinite(a), a, np.broadcast_to(col, a.shape))
    a = np.nan_to_num(a - np.nanmean(a))
    return a, x


def _hann2(n):
    return np.hanning(n)


def freq_psd(a):
    """Power spectral density vs frequency (cpd): FFT in time, mean over lon."""
    nt = a.shape[0]
    d = a - a.mean(0, keepdims=True)
    d = d * _hann2(nt)[:, None]
    A = np.fft.rfft(d, axis=0)
    f = np.fft.rfftfreq(nt, d=1.0)                       # cycles/day
    P = (np.abs(A) ** 2).mean(1)
    return f[1:], P[1:]


def wavenum_psd(a, x):
    """PSD vs zonal wavenumber (cycles/deg): FFT in lon, mean over time."""
    nx = a.shape[1]
    dx = float(np.median(np.diff(x)))                   # deg
    d = a - a.mean(1, keepdims=True)
    d = d * _hann2(nx)[None, :]
    A = np.fft.rfft(d, axis=1)
    k = np.fft.rfftfreq(nx, d=dx)                        # cycles/deg
    P = (np.abs(A) ** 2).mean(0)
    return k[1:], P[1:]


def omega_k(a, x):
    """2-D (omega,k) power: full k, non-negative frequency half."""
    nt, nx = a.shape
    dx = float(np.median(np.diff(x)))
    d = a - a.mean(0, keepdims=True)
    d = d - d.mean(1, keepdims=True)
    d = d * _hann2(nt)[:, None] * _hann2(nx)[None, :]
    A = np.fft.fftshift(np.fft.fft2(d))
    f = np.fft.fftshift(np.fft.fftfreq(nt, d=1.0))       # cpd
    k = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))        # cycles/deg
    P = np.abs(A) ** 2
    pos = f >= 0                                          # fold to omega>=0
    # FFT puts a WESTWARD wave (c=w/k<0) at +k when w>0; negate k so westward sits at
    # k<0 (east→+), matching the dispersion-curve convention (verified w/ a synthetic).
    k = -k
    order = np.argsort(k)                                 # keep axis ascending
    return k[order], f[pos], P[pos][:, order]


# ---- equatorial-wave dispersion (plotted in cpd vs cycles/deg) ----------------
def _omega_roots(k_rad, n):
    """Positive real roots of  w^3 - (c^2 k^2 + (2n+1) beta c) w - beta c^2 k = 0."""
    c = CBC
    out = []
    for kk in k_rad:
        r = np.roots([1.0, 0.0, -(c ** 2 * kk ** 2 + (2 * n + 1) * BETA * c),
                      -BETA * c ** 2 * kk])
        r = r[np.abs(r.imag) < 1e-12].real
        out.append(r)
    return out


def dispersion_curves():
    """Return {label: (k_cyc_per_deg, f_cpd)} for Kelvin, Yanai, Rossby n=1,2."""
    kc = np.linspace(-0.25, 0.25, 401)                  # cycles/deg
    krad = 2 * np.pi * kc / DEG_M                        # rad/m
    to_cpd = 86400.0 / (2 * np.pi)
    curves = {}
    # Kelvin (eastward): w = c k, k>0
    ke = kc[kc > 0]
    curves['Kelvin'] = (ke, (CBC * (2 * np.pi * ke / DEG_M)) * to_cpd)
    # Yanai / mixed Rossby-gravity (n=0) and Rossby n=1,2: take lowest positive root
    for n, lab in [(0, 'Yanai (MRG)'), (1, 'Rossby n=1'), (2, 'Rossby n=2')]:
        roots = _omega_roots(krad, n)
        w = np.array([min(r[r > 0]) if np.any(r > 0) else np.nan for r in roots])
        curves[lab] = (kc, w * to_cpd)
    return curves


def _band_box(ax):
    """Shade the TIW band: period 15-40 d, wavelength 700-1600 km (westward, k<0)."""
    f0, f1 = 1.0 / TIW_PER[1], 1.0 / TIW_PER[0]         # cpd
    k_lo = (1.0 / (TIW_WAV[1] / (DEG_M / 1000.0)))       # cycles/deg (1600 km)
    k_hi = (1.0 / (TIW_WAV[0] / (DEG_M / 1000.0)))       # cycles/deg (700 km)
    ax.add_patch(plt.Rectangle((-k_hi, f0), k_hi - k_lo, f1 - f0,
                               fill=False, ec='cyan', lw=1.6, ls='--', zorder=6))


# ---- figures -----------------------------------------------------------------
def fig_freq():
    fig, axs = plt.subplots(2, len(QTYS), figsize=(5 * len(QTYS), 8), sharex=True)
    for i, fam in enumerate(('Vel', 'noVel')):
        for j, (qname, src, unit) in enumerate(QTYS):
            ax = axs[i, j]
            for (run, lbl), c in zip(ROWS[fam], COLORS):
                a, _ = hovmoller(run, src)
                f, P = freq_psd(a)
                ax.loglog(f, P, color=c, lw=1.6, label=lbl)
            for p in TIW_PER:
                ax.axvline(1.0 / p, color='cyan', lw=1, ls=':')
            if i == 0:
                ax.set_title(qname)
            if j == 0:
                ax.set_ylabel(f'{fam}\nPSD')
            if i == 1:
                ax.set_xlabel('frequency (cpd)')
            ax.grid(which='both', alpha=.25)
            if i == 0 and j == len(QTYS) - 1:
                ax.legend(fontsize=8, loc='lower left')
    fig.suptitle('TIW frequency spectra (TIW window, lat 2–5°N).  '
                 'cyan dotted = 15 & 40-day band edges', y=.995)
    fig.tight_layout()
    fig.savefig(f'{OUT}/spectra_frequency.png', dpi=140)
    plt.close(fig)


def fig_wavenumber():
    fig, axs = plt.subplots(2, len(QTYS), figsize=(5 * len(QTYS), 8), sharex=True)
    for i, fam in enumerate(('Vel', 'noVel')):
        for j, (qname, src, unit) in enumerate(QTYS):
            ax = axs[i, j]
            for (run, lbl), c in zip(ROWS[fam], COLORS):
                a, x = hovmoller(run, src)
                k, P = wavenum_psd(a, x)
                ax.loglog(k, P, color=c, lw=1.6, label=lbl)
            for wav in TIW_WAV:
                ax.axvline(1.0 / (wav / (DEG_M / 1000.0)), color='cyan', lw=1, ls=':')
            if i == 0:
                ax.set_title(qname)
            if j == 0:
                ax.set_ylabel(f'{fam}\nPSD')
            if i == 1:
                ax.set_xlabel('zonal wavenumber (cycles/deg)')
            ax.grid(which='both', alpha=.25)
            if i == 0 and j == len(QTYS) - 1:
                ax.legend(fontsize=8, loc='lower left')
    fig.suptitle('TIW zonal-wavenumber spectra (TIW window, lat 2–5°N).  '
                 'cyan dotted = 700 & 1600-km band edges', y=.995)
    fig.tight_layout()
    fig.savefig(f'{OUT}/spectra_wavenumber.png', dpi=140)
    plt.close(fig)


# dispersion-curve styles: Kelvin solid; Yanai dash-dot; Rossby dashed w/ lw = mode no.
CURVE_STYLE = {
    'Kelvin':      dict(ls='-',   lw=2.6),
    'Yanai (MRG)': dict(ls='-.',  lw=2.6),
    'Rossby n=1':  dict(ls='--',  lw=1.8),
    'Rossby n=2':  dict(ls='--',  lw=3.4),
}
_HALO = [pe.Stroke(linewidth=4.0, foreground='k', alpha=.6), pe.Normal()]


def _draw_curves(ax, curves):
    for lab, (kk, ww) in curves.items():
        ax.plot(kk, ww, color='w', path_effects=_HALO, **CURVE_STYLE[lab])


def fig_omega_k(qname, src):
    curves = dispersion_curves()
    fmax = 0.12
    fig, axs = plt.subplots(2, 5, figsize=(18, 7), sharex=True, sharey=True,
                            constrained_layout=True)
    pmax, cache = -np.inf, {}
    for fam in ('Vel', 'noVel'):
        for run, lbl in ROWS[fam]:
            a, x = hovmoller(run, src)
            k, f, P = omega_k(a, x)
            cache[(fam, run)] = (k, f, P)
            m = f <= fmax
            pmax = max(pmax, np.log10(P[m].max() + 1e-30))
    lo = pmax - 4
    flev = np.arange(np.floor(lo), np.ceil(pmax) + 0.5, 0.5)      # filled, 0.5 dex
    clev = np.arange(np.ceil(lo), np.floor(pmax) + 1)            # lines at integer log10
    cf = None
    for i, fam in enumerate(('Vel', 'noVel')):
        for j, (run, lbl) in enumerate(ROWS[fam]):
            ax = axs[i, j]
            k, f, P = cache[(fam, run)]
            m = f <= fmax
            Z = np.log10(P[m] + 1e-30)
            cf = ax.contourf(k, f[m], Z, levels=flev, cmap='turbo', extend='both')
            ax.contour(k, f[m], Z, levels=clev, colors='k', linewidths=0.5, alpha=.5)
            _draw_curves(ax, curves)
            _band_box(ax)
            ax.set_xlim(-0.2, 0.15); ax.set_ylim(0, fmax)
            ax.axvline(0, color='w', lw=.6)
            ax.set_title(f'{fam}: {lbl}', fontsize=10)
            if i == 1:
                ax.set_xlabel('k (cyc/deg)')
            if j == 0:
                ax.set_ylabel('frequency (cpd)')
    # legend (correct styles, no halo so dashes read cleanly) in the noVel-Full panel
    hs = [plt.Line2D([0], [0], color='k', lw=CURVE_STYLE[l]['lw'], ls=CURVE_STYLE[l]['ls'])
          for l in curves]
    axs[1, -1].legend(hs, list(curves.keys()), fontsize=8, loc='upper right',
                      handlelength=3.2, framealpha=.9)
    fig.colorbar(cf, ax=axs, shrink=.6, label='log₁₀ power (contour lines every 1 dex)')
    sub = ('westward = k<0 (left), eastward = k>0 (right);  curves = equatorial-wave '
           f'dispersion (c={CBC} m/s); cyan dashed box = TIW band')
    fig.suptitle(f'{qname}: wavenumber–frequency spectrum\n{sub}', fontsize=11)
    tag = qname.split()[0].lower()
    fig.savefig(f'{OUT}/omega_k_{tag}.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    fig_freq();        print('-> spectra_frequency.png')
    fig_wavenumber();  print('-> spectra_wavenumber.png')
    for qname, src, _ in QTYS:
        fig_omega_k(qname, src)
        print(f'-> omega_k_{qname.split()[0].lower()}.png')
    print(f'all spectra -> {OUT}')
