# TODO

## Open
(none)

## Decomposition experiment plan (main-effect screening → subset → interactions)
Framework: multilinear/ANOVA decomposition of the full solution. Each control's
"on" value = the FULL solution's optimized increment for that field (do NOT
re-optimize in combos), so a_ij = M(i+j) − M(i) − M(j) + M(prior) is exact.
Decompose in increment/innovation space (βᵢ = ⟨d_i,d_full⟩/⟨d_full,d_full⟩),
keep RMSE as the fit/phase metric.

### Step 1 — finish main effects (singles)
Done: θ, salt, uwind, vwind, swdown. Remaining:
- [ ] **xx_atemp**, **xx_aqh**, **xx_lwdown**, **xx_precip** — PRIORITY (forcing; small
      increments still integrate over the window, so amplitude ≠ effect).
- [ ] xx_uvel, xx_vvel — run the solo experiments to confirm the null, then
      EXCLUDE from the subset / combination runs. Pre-screen: full-solution
      increment is ~0.0005 m/s RMS = 0.4–0.7% of field variability (vs θ 4%, S 8%),
      and IC-velocity memory decays over the window, so effect is expected ~0.
      (Don't add to tiw.CONTROLS beyond the solo check.)

### Step 2 — screen into a subset
Rank all singles by βᵢ (projection share) AND RMSE improvement, PER TARGET
(EUC U, TIW V, SST, Argo T/S, SSH). Include any control exceeding ~5% on any
target of interest, plus any dynamically-essential one.
Caveat: screening on main effects is a good INCLUDE filter, weak EXCLUDE filter
(synergy). Guards: (a) flag controls with large full increment but ~0 solo
effect; (b) after reduced-full, check Σ(singles) vs reduced-full; (c) leave-one-out
for borderline controls.

### Step 3 — the subset defines the combination runs
- [ ] reduced-full: all subset controls ON together → total interaction of the set.
- [ ] priority pairs within the subset: θ×uwind (EUC/TIW shear), θ×salt (density/
      barrier layer), θ×swdown, uwind×swdown (SST front↔wind). Add θ×uvel / uwind×uvel
      only if uvel is kept.
- [ ] optional triad θ×uwind×(uvel|swdown) for 3rd-order on the EUC/TIW core.

## ANOVA module (NOT YET BUILT — design only)
Build once combination runs (reduced-full + pairs) land. Add to the pipeline
(new `anova.py` + figures in figs/anova/). Reuses existing caches (model_<run>_*
and grid_<run>_*); no new sampling logic.
- Inputs: prior, each single control, each combo run (e.g. theta+uwind), Vel.
- Main effects:   a_i  = d_i = m_i − m_prior.
- Pairwise:       a_ij = m_ij − m_i − m_j + m_prior   (needs the i+j combo run).
- Higher order analogously (triads from 3-control runs).
- Per obs product + per field: report a_i, a_ij as (1) scalar projection shares
  onto d_full, and (2) maps/depth profiles of the interaction field a_ij.
- Budget checks: Σ(main effects of subset) vs reduced-full = total interaction;
  reduced-full vs Vel = untested-control contribution; leave-one-out marginals.
- Targeted metrics to decompose (not just global RMSE): EUC core speed/depth/
  transport; TIW-band distance metric (phase-sensitive RMSE of SSH/SST/V filtered
  to ~2–6°N, 15–40 day, ~1000–1600 km) so "TIWs in phase with data" is scored.
- Deliverable: interaction matrix heatmap (controls × controls) per target, plus
  a_ij maps for the dominant pairs (expected θ×uwind for EUC/TIW).

## Done
- [x] iter0 `diag_surf` (ETAN) now present → SSH baseline is prior (iter0) for all
      figures/decomposition (dropped the noVel fallback). prior SSH RMSE 0.0489 m
      > noVel 0.0473 > Vel 0.0468, so Vel improves SSH ~4% over prior (the noVel
      baseline was masking it). Reran compute_metrics.py prior + make_figures.py.
- [x] Verified + sampled the 5 single-control runs; metrics in cache/metrics.csv.
- [x] Per-experiment figures figs/<control>/ ; cross-experiment figs/intercomparison/ ;
      increment-decomposition figs/decomposition/ (work shares, depth, SST/SSH maps).
