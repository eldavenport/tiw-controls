# TIW controls — project summary

Goal: identify which TPOSE6 assimilation controls (adjustable initial conditions +
forcing) are responsible for the model's tropical-instability-wave (TIW) skill, and
by what physical mechanism — separating overall misfit reduction from TIW-specific
improvement, and surface from subsurface.

Sep–Dec 2012 window. Runs compared: `prior` (iter0, no DA), `noVel` (DA without
velocity constraints), per-control and combo runs, `top4` (θ+salt+uwind+swdown), and
`Vel` (full TPOSE6). noVel↔Vel differ only by the **velocity data constraint** (whether
velocity obs are assimilated) — the controls are the same in both.

## Code
- `tiw.py` — run registry (`RUNS`, `COMBOS`, `CONTROL_XX`), obs loaders (Argo/TAO/SST/SSH), shared metrics.
- `compute_metrics.py <run>` — samples a run to the obs and writes `cache/metrics.csv` + `model_<run>_*`/`grid_<run>_*`.
- `make_figures.py` — per-experiment figures (figs/<control>/fig1–6), intercomparison, decomposition, summary.
- `anova.py {budget|euc|tiw|phaseamp|maps|fields|cache3d|check|all}` — factorial decomposition + TIW analysis.
  - Fixed-increment ANOVA: each control "on" = its full-solution increment (no re-optimization), so
    `a_S = Σ_{T⊆S}(−1)^{|S|−|T|} m_T` is exact; normalized by the shared full increment `d_full = m_Vel − m_prior`.
  - 3-D fields (EUC U, TIW V) are distilled to `cache/euctiw_<run>.nc` by `cache3d` before raw `diag_state` is purged.

## What we did & found

### Control screening and factorial decomposition
- All 11 single controls run + scored. Ranking by mean %RMSE-vs-prior: **θ ≫ uwind > salt > swdown**;
  vwind/atemp/aqh/lwdown/precip/uvel/vvel are ~null individually.
- `top4` (θ+salt+uwind+swdown) reproduces ~all of Vel's skill on every obs target → the other controls
  add little to the *total* misfit (but see SSH residual below).
- Full exact factorial of {θ,salt,uwind,swdown} (all singles, 6 pairs, 4 triads, 4-way) scored.
  - Two framings, both in `budget_bars{,_projection}.png`: (1) magnitude `‖a_S‖/‖d_full‖`; (2) **projection**
    `⟨a_S,d_full⟩/⟨d_full,d_full⟩`, which is signed and sums to ~100% of the final increment.
  - θ main dominates; **θ×uwind** and **θ×salt** are the leading interactions; triad/4-way are small but
    non-negligible (4-way 10–24% of d_full, larger than the weakest triads) → higher order can't be truncated.
  - Excluded-controls residual `r = m_Vel − m_top4`: small for most targets but **~34% of the SSH increment**
    (barotropic/mass controls) — top4≈Vel in RMSE does *not* mean the excluded controls are inert for SSH.
- EUC physical scalars (core speed/depth/transport) factor-separated: net prior→Vel EUC change is small
  because controls largely **cancel** (θ drives, u/sw oppose). By longitude the EUC response is a spatial
  redistribution (θ shifts transport eastward), hidden in the basin-mean.

### TIW phase & amplitude (the mechanism)
TIW field = band-passed (period 15–40 d, wavelength 700–1600 km, westward lobe) complex analytic field
`z(t,x)`; amplitude = √(band-passed variance), phase via the complex projection/coherence.
- **θ sets the TIW phase, uwind sets the amplitude/growth; the θ×uwind interaction amplifies a correctly-phased
  wave.** Temporally (SST): θ locks phase early (coherence ≈0.93 in the first third vs prior 0.58) with little
  amplitude; uwind grows amplitude over the window; θ+uwind gets both. θ×uwind is strongly phase-active.
- Mechanism: TIWs grow by instability of the equatorial current shear (uwind-controlled); their phase is set by
  projection onto the pre-existing density/wave structure (θ-controlled).

### Overall vs TIW-specific skill, and the role of velocity constraints
- Domain-wide RMSE reduction **understates** TIW improvement: SST reduction is ~8–15% domain-wide but 25–34%
  in the TIW band within the 2–6°N cold-tongue box (`tiw_region_rmse`, `tiw_box_rmse_3d`). The improvement is
  genuinely TIW-specific, not a mean-state artifact.
- **Scale/space split of the velocity constraint** (`tiw_band_sensitivity`, `tiw_lowfreq_rmse_3d`,
  `tiw_tao_depth_phase_amp`):
  - Long-wave (1100–1600 km) surface TIW phase is captured *without* velocity obs (noVel coherence ≈ Vel, even
    slightly higher); the **velocity constraint buys the short-wave (600–1100 km) phase** (SST coherence 0.27→0.65).
  - The velocity constraint adds the most skill in the **subsurface equatorial velocity** (TAO V TIW-band RMSE
    reduction: noVel 33% → Vel 57%; TAO U similar), invisible to SST/SSH.
  - noVel **degrades the large-scale (>40 d) equatorial zonal current** (TAO U >40 d: −14% vs prior) even while
    improving the TIW band — this is the source of noVel's worse *overall* U RMSE. Velocity obs hold the mean EUC/SEC
    in place.
- vs OBS (not vs Vel): band-passed skill vs mw_fusion SST, RADS SSH, and the TAO current meters
  (`tiw_*_skill_vs_obs`, `tiw_tao_skill_vs_obs`). Coherence + amplitude ratio are the informative metrics.

## Alternatives tried / considered
- **Normalization for interaction size:** per-run `‖a_S‖/‖d_comb‖` inflates terms whose own increment is small;
  standardized on the shared `d_full` baseline so orders are comparable. Dropped a "projection of a_ij onto d_full"
  pair diagnostic as misleading (d_full carries all controls).
- **Band:** widened TIW wavelength from 1000–1600 to 700–1600 km; use **disjoint** long/short sub-bands (not nested)
  for the scale sensitivity, since nested bands confound the regimes.
- **Phase lag** metric dropped from the skill figures — the net lag is only meaningful once coherence is high
  (kept in the CSVs).
- **Phase reference:** vs Vel (model–model) for the decomposition/time series; vs obs for the skill figures.

## Known / unknown
- Known: control ranking; the factorial budget and residual; the θ-phase / uwind-amplitude mechanism; that
  velocity constraints act mainly on short-wave + subsurface + large-scale-mean velocity.
- Unknown / caveats:
  - SSH amplitude vs RADS is unreliable (altimeter smoothing damps the observed TIW); SSH *coherence* is usable,
    SSH *amplitude/band-RMSE* is not. SST (mw_fusion) and TAO velocity are the trustworthy amplitude references.
  - `fig6_control_*.png` (applied-control fingerprint) can't be regenerated — needs raw `diag_state`, now purged.
  - vwind is a borderline control (<~2% everywhere); not run in any combo.
