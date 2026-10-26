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
    `a_S = Σ_{T⊆S}(−1)^{|S|−|T|} m_T` is exact; normalized by the shared full increment `d_full = m_anchor − m_prior`.
  - 3-D fields (EUC U, TIW V) are distilled to `cache/euctiw_<run>.nc` by `cache3d` before raw `diag_state` is purged.

## Spectra
`spectra.py` → `figs/comparison/spectra/` (frequency, zonal-wavenumber, ω–k with equatorial-wave dispersion
curves) for prior/θ/u-wind/θ+u/Full in both families, over the TIW window (180–265°E, 2–5°N). Quantities: SST,
SSH, and cached top-50 m V (U and true 50–100 m dropped — control/combo `diag_state` purged). 122-day record →
coarse ω frequency bins. To get true 50–100 m U/V, restore the 6 control/combo runs' `diag_state` + add a 50–100 m
extractor to `anova.cache3d`.
ω–k CONVENTION (verified w/ synthetic wave): westward (TIW/Rossby, c=ω/k<0) plotted at **k<0**; `omega_k` negates the
FFT wavenumber to enforce this (raw 2-D FFT puts westward at +k). Dispersion: equatorial β-plane cubic
`ω³−(c²k²+(2n+1)βc)ω−βc²k=0`, c=2.6 m/s (first baroclinic; Kelvin 2.6, Rossby-n1 0.40, Yanai 0.58 m/s at the TIW k —
bracket the obs TIW 0.45–0.53). Styles: Kelvin solid, Yanai dash-dot, Rossby dashed w/ lw=mode#; legend in the noVel-Full panel (no
path_effects halo on legend handles, else dash-dot reads as solid). Doppler overlay was tried and REMOVED
(2026-10-23) as premature clutter: a single ω→ω+kU shift is only a uniform-flow approx, but U is strongly
lat-sheared across 2–5°N (−0.55 @2N SEC → +0.14 @5N NECC, sign change) and the spectra are 2–5°N meridional
averages — doing it right needs U(y) weighted by the TIW meridional mode (+ per-run flow, which needs the
purged 3-D). Revisit only if that's built.

## Two solution families (anchor)
The factorial can be anchored on either full solution via `TIW_FAMILY=vel|novel` (env var, default `vel`),
read by both `anova.py` and `make_figures.py`:
- `vel` → anchor TPOSE6-**Vel** (velocity constraint). RUN subsets map to `run_<control>`/`top4`; `d_full = m_Vel − m_prior`.
- `novel` → anchor TPOSE6-**noVel** (same controls, no velocity data constraint). RUN subsets map to `nv_<control>`/`nv_top4`
  (added to `tiw.RUNS` when those runs land); `d_full = m_noVel − m_prior`. Lets us compare how the *same* controls
  attribute differently under the velocity constraint.
- Figures: `figs/vel/`, `figs/novel/` (per-family: `anova/`, per-control, summary/story/…); `figs/comparison/`
  holds the solution-agnostic cross figures (skill-vs-obs, band-sensitivity, tao-depth, phase/amp) that overlay prior/noVel/Vel.
  `controlling_phase_amp`/comparison figs are built once from the default `vel` run (they curate Vel-family anova figs).
- noVel scope: **singles first** (4 mains); obs-space budget/heatmap degrade gracefully (NaN) on the partial factorial,
  and `euc`/`tiw`/`maps`/`check` skip with a message until the full factorial (pairs/triads/`nv_top4`) is cached.
- **Full-parity prereq:** run `anova.py cache3d <nv_run> noVel` BEFORE each run's `diag_state` is purged, or the
  EUC/TIW/V-field decomposition is blocked.

### noVel singles — DONE (2026-10-20)
4 single-control runs in `/data/SO3/edavenport/tpose6/withheld_controls_exp/sep2012_noVel/run_{initialT,initialS,uwind,swdown}`
(72×1200, diag_state+diag_surf, no PROF_eq; md5-verified fixed-increment, noVel-optimized, shared iter0 prior).
Registered as `tiw.RUNS['nv_initialT','nv_initialS','nv_zonalWind','nv_swdown']` (key=Vel name, dir=noVel name `uwind==zonalWind`).
Done: `euctiw_{noVel,nv_*}.nc` cached (diag_state now purge-safe); `compute_metrics` (obs + grid caches); noVel main-effect
budget in `figs/novel/anova/`; per-experiment figs1–6 + intercomparison/decomposition/summary in `figs/novel/`.
**Main-effect attribution (projection % of own d_full) θ dominates both families; dropping the velocity constraint shifts
share toward u-wind on nearly every target (Argo T θ 83→64, u-wind 8→21; SSH θ 64→44); SST attribution ~unchanged.**

Screen extended to **8 noVel singles** (2026-10-20): added forcing controls `nv_atmosT,nv_atmosQ,nv_precip,nv_meridionalWind`
(dirs `run_{atemp,aqh,precip,vwind}`; `lwdown` dir still empty). md5-verified single-control (all 4 main fields at bg).
All ~null individually (like the Vel family): %RMSE-vs-prior θ>u-wind>swdown>salt, forcing controls ≤1.4% (precip SSH 1.4%).
NOTE single controls individually DEGRADE TAO U under noVel (θ −2.3, u-wind −5.1, salt −1.2%) — consistent with "noVel
weakens the large-scale equatorial zonal current." These forcing singles are SCREEN-ONLY, not in the TSUW factorial;
figs in `figs/novel/{nv_*, intercomparison, decomposition, summary}`. (Core-4 diag_state purged after caching → their fig6
is retained from first run; `fig_control` now skips with a clear message when diag_state is gone.)

### noVel combos — PARTIAL factorial DONE (2026-10-21)
Processed 4 combos + 9th single `nv_lwdown`: `nv_theta_salt`(TS, run_theta_salt), `nv_initialT_uwind`(TU, run_theta_uwind,
**121 recs** — one missing day), `nv_theta_salt_uwind`(TSU, run_theta_uwind_salt), `nv_top4`(TSUW, run_theta_uwind_salt_swdown).
Done: md5-verified fixed-increment, cache3d (euctiw_nv_*), compute_metrics, `anova.py budget` (now has a_TS, a_TU, top4+residual),
`anova.py fields TS TU` (EUC/TIW field decomp), SST/SSH interaction maps for TS/TU/residual, per-experiment combo figs.
**Findings:** θ×S small-negative both families. **noVel SSH excluded-control residual ~25% vs Vel 9%** (noVel leans harder
on controls outside {θ,S,u,sw} for SSH). CAUTION on TAO U: the projection budget is in MODEL-STATE space, not skill, and is
normalized by each family's own d_full. noVel is WORSE than prior on TAO U (RMSE prior 0.262 → noVel 0.282; Vel 0.159), so
noVel's d_full points toward a WORSE fit — a positive noVel share there means a term aligns with the degradation, NOT that
it helps. The apparent "θ×u sign flip" (Vel −10.9% → noVel +8.2%) is a reference-frame artifact (projection onto an
improving vector vs a worsening one). SKILL FACT: velocity obs carry TAO U; θ+u-wind together (nv_initialT_uwind 0.287) is
worse than either single (θ 0.268, u 0.275) and worse than prior — no beneficial θ×u synergy for the zonal current under
noVel. Projection shares are only skill-meaningful on targets where noVel improves over prior (TAO V, SST, SSH, Argo).
**121-day-run fix:** grid + TIW-Hovmöller loads now reindex to canonical `tiw.DAYS` (anova `_load_obs`/`_load_3d`/`_tiw_hov`;
make_figures `_grid_vals`) so a run missing a day stays shape-compatible for inclusion-exclusion (gap → NaN, dropped by mask).

### noVel FULL FACTORIAL COMPLETE (2026-10-22)
Final 7 runs filled + verified + cached (all diag_state now purge-safe): `θ×sw`,`S×u`,`S×sw`,`u×sw` pairs,
`θ×S×sw`,`θ×u×sw`,`S×u×sw` triads. Identity check holds to machine precision. Ran budget/euc/tiw/maps/fields
→ `figs/novel/anova/` (budget bars, heatmap, euc_anova, tiw_phaseamp, 11 higher-order EUC/TIW field decomps, maps).
Skill grounding (noVel vs prior): improves Argo T/S, TAO V, SST, SSH; **degrades TAO U −7.9%** (only target; its shares
are state-space, not skill).
**KEY RESULT: noVel is markedly MORE NONLINEAR than Vel** — summed interaction magnitude (Σ‖a_S‖/‖d_full‖, order≥2) is
~1.5× larger for noVel on every target (SST 400 vs 291%, Argo T 265 vs 146%; inter/main ≈1.6–2.0 vs 0.9–1.5). The velocity
constraint makes {θ,S,u,sw} combine more additively; without it they interact/cancel more. Leading interactions: θ×S×u
triad positive+prominent (TAO V +7.0, SST +5.7); S×u negative (TAO V −5.9, SST −4.4); 4-way stays ≤1.8% (both families).
SSH excluded-control residual noVel 25% vs Vel 9% (but noVel SSH gain only +3.2% total).
NOT run: `phaseamp` (that rebuilds the vel-sourced `figs/comparison/`). To fold noVel into the cross-solution comparison
figures, extend `VS_OBS_RUNS`/`PA_RUNS` and regen from the default (vel) family.

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
