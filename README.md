# tiw-controls

Attribution analysis for the TPOSE6 **withheld-controls** experiments: which
assimilation control (adjustable initial conditions + surface forcing) is
responsible for the model's tropical-instability-wave (TIW) skill, and by what
mechanism.

Each experiment is a *fixed-increment* run: starting from the no-DA `prior`
(iter0), one control (or a named combination) is switched on at its
full-solution optimized value while everything else stays at background. We
sample every run's output at the observation coordinates, score it against the
obs, and decompose the skill back onto the individual controls.

Window: **Sep–Dec 2012** (122 daily records from 2012-09-01). Reference
full solutions: `noVel` (DA without velocity constraints), `TSOnly` (TAO T/S
assimilated, no U/V), `Vel` (full TPOSE6). `noVel`↔`Vel` differ only by whether
velocity obs are assimilated.

## Layout

Scripts read/write a cache on `/data` (`tiw.CACHE`), not the repo; only figures
and CSVs are committed under `figs/`.

| file | what it does |
|------|--------------|
| `tiw.py` | run registry (`RUNS`, `CONTROLS`, `CONTROL_XX`), obs loaders (Argo/TAO/SST/SSH/RADS), shared misfit metrics (`rmse`, `wcost`, `bias`, `rmse_profile`). |
| `compute_metrics.py` | samples each run to the obs → `cache/model_<run>_*`, gridded SST/SSH → `cache/grid_<run>_*`, and the scalar misfit table `cache/metrics.csv`. |
| `make_figures.py` | per-experiment figures (`figs/<family>/<control>/fig1–6`), cross-experiment intercomparison, increment decomposition, and the skill-ranking summary. `TIW_FAMILY=vel|novel`. |
| `compare_families.py` | three-solution comparison (noVel vs TSOnly vs Vel): full-solution skill + control attribution shifts as more obs are added. |
| `anova.py` | fixed-increment factorial ANOVA over the EUC/TIW regions + TIW phase/amplitude/envelope analysis and interaction-field maps. |
| `misfit_cost.py` | cross-window normalized model–data cost (χ²/obs) for prior/noVel/Vel, per obs type. |
| `profile_windows.py` | Argo/TAO RMSE profiles across the four assimilation windows. |
| `spectra.py` | ω–k / frequency / wavenumber spectra over the TIW window, with equatorial-wave dispersion curves. |

`TODO.md` is the detailed, running project log (per-figure notes, open threads).

## Special notes

- **Product comparisons are evaluated only within ±17° latitude (17°S–17°N),
  away from the domain boundaries, where every product was assimilated.** This
  is the band where the comparison between solutions is meaningful — the TPOSE
  assimilation error grids are themselves defined 17S–17N
  (`SST_error_tpose6_17Sto17N.bin`). The cut is applied in `compute_metrics.py`
  (`LAT_LIMIT`), `make_figures.py`, `summary_figs.py`, `misfit_cost.py`, and
  `profile_windows.py`. EUC diagnostics are the deliberate exception: they are
  equatorial (`|lat|<0.5` at the TAO moorings) and so already inside the band.
- **Regenerating `metrics.csv`:** the single-control run diag output is purged
  after caching, so a plain `compute_metrics.py <run>` can only redo the live
  runs (prior/noVel/Vel/TSOnly). To rebuild the whole table from the existing
  cache (e.g. after changing `LAT_LIMIT`), use
  `compute_metrics.py --from-cache`.
- Profile obs weights are `1/σ²` (`weight==0` ⇒ excluded). SST/SSH have no
  profile weights; they are scored against their own gridded error fields.
- Obs longitudes are −180..180 while the model grid is 0..360; dry cells read as
  0 and must be masked. See `tiw.py` loaders.

## Running

```bash
PY=~/miniforge3/envs/tpose/bin/python
$PY compute_metrics.py                 # sample live runs + write metrics.csv
$PY compute_metrics.py --from-cache    # rebuild metrics.csv from cache only
TIW_FAMILY=vel   $PY make_figures.py
TIW_FAMILY=novel $PY make_figures.py
$PY compare_families.py
```
