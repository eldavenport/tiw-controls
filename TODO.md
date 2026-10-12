# TODO

## Open
(none)

## Decomposition experiment plan (main-effect screening → subset → interactions)
Framework: multilinear/ANOVA decomposition of the full solution. Each control's
"on" value = the FULL solution's optimized increment for that field (do NOT
re-optimize in combos), so a_ij = M(i+j) − M(i) − M(j) + M(prior) is exact.
Decompose in increment/innovation space (βᵢ = ⟨d_i,d_full⟩/⟨d_full,d_full⟩),
keep RMSE as the fit/phase metric.

### Step 1 — finish main effects (singles)  — DONE (all 11 singles run + sampled)
Done: θ, salt, uwind, vwind, swdown.
- [x] **xx_atemp**, **xx_aqh**, **xx_lwdown**, **xx_precip** — run + sampled. Result:
      individually ~null on every target (mean %RMSE-vs-prior ≤0.4%; precip only
      exceeds 1% anywhere, on SSH). Despite integrating over the window, their T/S
      response (p99 ~0.06°C / 0.03 psu at day121) barely clears the ~0.05°C null-drift
      floor. See figs/summary/summary_ranking.png + summary_pct_vs_prior.csv.
- [x] xx_uvel, xx_vvel — solo null CONFIRMED (≤0.8% on any target; day0 IC increment
      ~0.0005 m/s RMS as pre-screened, decays over window). EXCLUDED from combos:
      kept in tiw.CONTROLS_NULL, not tiw.CONTROLS. See summary_ranking.png bottom rows.
  Ranking (mean %RMSE vs prior): θ 11.7 > uwind 5.1 > salt 4.8 > swdown 1.8 >
  vwind 0.5 > precip 0.4 > vvel/aqh/uvel/lwdown/atemp ≤0.16.

### Step 2 — screen into a subset  — RESOLVED
Rule: include any control exceeding ~5% RMSE-vs-prior on any target, plus any
dynamically-essential one. Applied to summary_pct_vs_prior.csv (βᵢ in
figs/decomposition/decomp_work_shares.png agrees):
- SUBSET (kept for combos): **θ** (all targets), **uwind** (TAO U 10%, TAO V 14%),
  **salt** (Argo S 19%, TAO U 5%), **swdown** (SST 6%).
- EXCLUDED, <5% everywhere: vwind (peak ~2% TAO U) → BORDERLINE, hold as a
  leave-one-out candidate; atemp/aqh/lwdown/precip ~null; uvel/vvel null.
Caveat: main-effect screen is a good INCLUDE filter, weak EXCLUDE filter (synergy).
Guards carried into Step 3: (a) flag controls with large full increment but ~0 solo
effect; (b) after reduced-full, check Σ(singles) vs reduced-full; (c) leave-one-out
for borderline controls (vwind first).

### Step 3 — combination runs to do (FIXED-INCREMENT, NO RE-OPTIMIZATION)
Each combo = ONE forward run: the named controls set to their FULL-solution
optimized increments, every other control at background. Do NOT re-solve the
optimization. Then just sample + score with the existing pipeline (add each run
to tiw.RUNS and `compute_metrics.py <run>`); a_ij = m_ij − m_i − m_j + m_prior.
Subset = {θ, uwind, salt, swdown}.
- [ ] reduced-full: θ+uwind+salt+swdown ON together → total interaction of the set
      (and reduced-full vs Vel = contribution of the excluded/untested controls).
- [x] θ×uwind (EUC/TIW shear) — run_initialT_uwind, verified + scored. Combined beats
      BOTH singles on every target (e.g. TAO V 0.131 vs 0.165/0.170; TAO U 0.184 vs
      0.207/0.235). Interaction a_ij is 29–61% of the combined increment (largest for
      velocity + SST) → the pair is strongly nonlinear, combined ≠ sum of singles.
      TIW interaction carries organized westward-propagating wave structure (RMS 0.044 ≈
      sum-of-mains 0.052) → θ×uwind reshapes TIW phase, not just amplitude. Figs:
      figs/anova/initialT_uwind_{euc,tiw,budget}.png + _budget.csv. See combo.py.
- [ ] priority pairs remaining: θ×salt (density/barrier layer), θ×swdown (SST),
      uwind×swdown (SST front↔wind).
- [ ] secondary pairs (lower priority): salt×uwind, salt×swdown.
- [ ] triad: θ×uwind×swdown (EUC/TIW + SST core) for 3rd-order.
- [ ] leave-one-out (borderline guard): reduced-full minus vwind is already the
      reduced-full above; if vwind is later reconsidered, run θ+uwind+salt+swdown+vwind.
Dropped: all uvel/vvel combos (solo-null, excluded in Step 2).

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
