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
- [x] reduced-full (`run_top4` = θ+uwind+salt+swdown ON): scored, added to tiw.RUNS
      as `top4` (tiw.REDUCED_FULL). Result: the subset captures ~all of Vel's skill on
      every target — top4 vs Vel: Argo T 0.656/0.655, Argo S 0.093/0.092, TAO U 0.164/
      0.159, TAO V 0.122/0.117, SST 0.448/0.443, SSH 0.0477/0.0468. So the excluded/
      untested controls (vwind + atemp/aqh/lwdown/precip + uvel/vvel) add ~nothing beyond
      {θ,uwind,salt,swdown}. Set-interaction decomposition (Σsingles vs top4) = ANOVA-module
      work, still design-only below.
- [x] θ×uwind (EUC/TIW shear) — run_initialT_uwind, verified + scored. Combined beats
      BOTH singles on every target (e.g. TAO V 0.131 vs 0.165/0.170; TAO U 0.184 vs
      0.207/0.235). Interaction a_ij is 29–61% of the combined increment (largest for
      velocity + SST) → the pair is strongly nonlinear, combined ≠ sum of singles.
      TIW interaction carries organized westward-propagating wave structure (RMS 0.044 ≈
      sum-of-mains 0.052) → θ×uwind reshapes TIW phase, not just amplitude. Figs:
      figs/anova/initialT_uwind_{euc,tiw,budget}.png + _budget.csv. See combo.py.
- [x] priority pairs: θ×salt (`theta_salt`), θ×swdown (`theta_swdown`),
      uwind×swdown (`uwind_swdown`) — all verified fixed-increment, scored + ANOVA figs
      (figs/anova/*_{euc,tiw,budget}.png + _budget.csv). Interaction size ||a_ij||/
      ||d_combined|| (mean over targets): θ×salt ~39% (combined beats both singles on
      every target; SST 54%, SSH 46%), θ×swdown ~14% (near-additive; its SST gain comes
      mostly from swdown), uwind×swdown ~23%. θ×salt is the strongest new interaction.
- [x] secondary pair salt×swdown (`salt_swdown`) — scored + figs; interaction ~19–33%
      on velocity/SST but combined barely beats singles (both weak on T).
- [x] secondary pair salt×uwind (`salt_uwind`, re-run 2026-10-13 w/ theta off — verified
      fixed-increment, wired into tiw.RUNS+COMBOS, scored + ANOVA figs). STRONG interaction
      (25–51%: SST 51%, Argo T 41%, TAO U 40%, TAO V 36%); combined beats both singles
      everywhere. Completes all 6 subset pairs.
- [x] clean 3-way a_ijk for triad θ×salt×uwind (all 6 pairs now exist). NOT negligible:
      20–44% of the triad increment (SST 44%, velocity 30–32%, T/S/SSH 20–25%) — the
      irreducible triple is comparable to each pairwise term. The dynamical trio is strongly
      nonlinear at 3rd order → higher-order can NOT be dropped as a residual.
- [x] triad θ×salt×uwind (`run_theta_salt_uwind`, tiw.TRIADS): verified fixed-increment,
      scored. Reaches top4/Vel on T,S,V,SSH; the only gap vs top4 is SST (0.474 vs 0.448)
      and TAO U (0.174 vs 0.164) = swdown's additive SST contribution. TOTAL interaction
      (beyond Σsingles) is large (40–75% of the triad increment; SST 75%, velocity ~60%)
      but that lumps the known pairwise θ-coupling + the 3-way. Clean a_ijk BLOCKED on the
      salt×uwind pair rerun.
  TRIAD/4-way sizes as share of the FULL solution ||a_S||/||d_full||, d_full=m_Vel−m_prior
  (single shared baseline for all orders), all 4 triads done: θ×salt×uwind 20–40% (SST 40%, vel
  28–30%) = dominant, all-dynamical. θ×uwind×swdown 9–21% (SST 21%). salt×uwind×swdown 8–19%
  (SST 19%). θ×salt×swdown 8–16% (SST 16%). Rule: 3-way strength tracks "dynamical-ness"; the
  3 swdown-triads are all weak/comparable (swdown most additive). SST carries the largest share
  at every order (16–40%), velocity next, T/S/SSH smallest. NOTE earlier per-run normalization
  (||a||/||d_comb_of_that_run||) inflated salt×uwind×swdown to 12–29% because its own increment
  is small (no θ); the shared d_full baseline is the canonical one (triad_anova.py).
- [x] remaining triads DONE (2026-10-14): run_{theta_uwind_swdown, salt_uwind_swdown} verified
      md5 fixed-increment (ON fields byte-identical to singles, OFF=bg) + scored. The 4 triads
      ARE the 4 leave-one-out marginals of top4. Leave-one-out skill: drop θ catastrophic (Argo T
      .813 ≈ prior .851, worst U/V/SST); drop uwind → velocity (TAO V .158); drop swdown → SST
      only (.474); drop salt → Argo S only (.116). Importance θ≫uwind>swdown≈salt (matches mains).
- [x] 4-way a_ijkl for top4 CLOSED → full exact factorial of {θ,salt,uwind,swdown}. 4-way size
      (share of d_full) 10–24% (SST 24%, U 19%, V 17%, S 13%, SSH 11%, T 10%) = LARGER than all
      3 swdown-triads (8–21%), can't truncate higher order. anova.py (incl–excl, obs space,
      normalized by shared ||d_full||=||m_Vel−m_prior||; ||d_comb|| shown for context only).
- [ ] leave-one-out (borderline guard): reduced-full minus vwind is already the
      reduced-full above; if vwind is later reconsidered, run θ+uwind+salt+swdown+vwind.
Dropped: all uvel/vvel combos (solo-null, excluded in Step 2).

## ANOVA module — BUILT 2026-10-14 (`anova.py`, folds in old combo.py + triad_anova.py)
`anova.py` is the single decomposition module. Reuses existing caches (model_<run>_*
and grid_<run>_*); no new sampling. CLI: `anova.py {budget|maps|pairs|check|all}`.
DONE in the module:
- [x] obs-space incl–excl engine `anova_term(S, prod)` for ANY subset + residual, shared
      ||d_full|| normalization. `check` verifies the identity Σa_S=d_top4 & d_top4+r=d_full
      to machine precision (profiles ~1e-14; SST/SSH ~1e-5/1e-7 = float32 grid roundoff).
- [x] `factorial_budget.csv` — all 15 terms + residual × 6 targets (% of d_full).
- [x] `heatmap.png` — controls×controls per target (diag=main, off-diag=pair).
- [x] `budget_bars.png` — all orders + residual per target, colored by order.
- [x] gridded SST/SSH interaction field maps `map_<term>_<field>.png` (any order + residual).
- [x] residual r = m_Vel − m_top4 as its own term (SSH 34% — excluded controls drive ~1/3 SSH).
- [x] EUC/TIW field decomposition at ALL orders (`anova.py fields`): order-agnostic `fig_decomp`
      (replaces pairwise fig_euc/fig_tiw) → <run>_{euc,tiw}.png for every pair/triad/4-way from
      cached euctiw_<run>.nc. Field-space identity Σa_S=d_top4 verified ~5e-7 (float32). Headlines:
      θ×S×u 3-way carries organized westward TIW stripes (reshapes phase); 4-way a tiny in EUC-U
      (0.005 m/s; equatorial jet is θ+u driven).
3D-DATA WORKFLOW (resolved 2026-10-15): diag_state is purged from experiment dirs to save space, so
`anova.py cache3d <run>` distills each run's EUC/TIW to euctiw_<run>.nc (872K) BEFORE its diag_state
is deleted. All 16 factorial runs now cached → EUC/TIW decomposition fully reproducible; `_load_3d`
is cache-first. (Restored one missing meta via cp from a sibling same-iter — metas iter-identical.)
- [x] TARGETED EUC SCALAR decomposition (`anova.py euc`): factor-separate physical EUC scalars
      (not RMSE) from cached euc fields → euc_anova.{csv,png} + euc_metrics_raw.csv. Scalars:
      core_speed (peak eastward U), core_depth (eastward-U-weighted centroid), transport (+U over
      depth × ±0.5° band at 140°W). Signed a_S in physical units; budget identity Σa_S+r=d_full
      EXACT (float64). Needs euctiw_Vel.nc (cached). RESULT: net EUC change prior→Vel is SMALL
      (speed −0.019 m/s, depth +6.4 m shoaling, transport −0.20 Sv) because controls LARGELY
      CANCEL. θ is the dominant actor (speed −0.049, depth +5.4 m shoal, transport −0.90 Sv) and
      u/sw/θ×sw/θ×u push OPPOSITE on speed+transport → near-cancellation. core_depth cleanest
      (θ-dominated). CAVEAT: %-of-full explodes for speed/transport (d_full≈0); read the SIGNED
      physical a, not %. Different story than obs-RMSE (where θ looked purely beneficial).
- [x] EUC scalars ALSO decomposed AT the TAO obs longitudes 170°W/140°W/110°W (euc_scalars_at;
      `anova.py euc` now writes euc_anova_bylon.{csv,png}+euc_metrics_bylon_raw.csv; 3×3 signed bars
      rows=metric cols=lon; budget EXACT at all 9). RESULT: the EUC response is a SPATIAL
      REDISTRIBUTION the basin-window number hid — assimilation weakens transport in west/central
      (170°W −0.63, 140°W −0.20 Sv) but STRENGTHENS it in the far east (110°W +1.31 Sv). θ main on
      transport FLIPS SIGN with longitude: −0.85 (170°W), −0.90 (140°W), +1.73 (110°W) → θ shifts
      EUC transport eastward. Core shoals +14 m at 170/140°W, slightly deepens −4 m at 110°W.
      Cancellation strongest at 170°W (θ −0.85 offset by u +0.66). Basin euc_scalars analysis kept.
- [x] TIW PHASE + AMPLITUDE decomposition vs Vel (`anova.py tiw`, channels V + SST). Method:
      2D-FFT band-pass each run's TIW Hovmöller to the WESTWARD band (15–40 d, 1000–1600 km, cold
      tongue 180–265°E), single spectral lobe → complex analytic field z(x,t); project onto Vel's
      unit TIW field c=<ẑ_Vel,z>. Linear ⇒ c factor-separates EXACTLY (identity ~1e-15): Re(a_S)=
      amplitude IN PHASE w/ Vel, Im(a_S)=quadrature (phase shift). Plus nonlinear total-TIW-energy
      (RMS|z|) scalar. Outputs tiw_phaseamp_{V,SST}.{png,csv}, tiw_phase_{V,SST}.csv (per-run lag
      deg/days + coherence), tiw_envelope_{V,SST}.png (|a_S^z| Hovmöllers). Tdom=27.5d (V)/25.9d
      (SST). RESULTS: prior TIWs LAG Vel (−2.2 d in V, −6.8 d in SST) & are weaker; controls
      amplify + advance them into phase (top4 → −0.4/−0.5 d, coh 0.99/0.98). V amplitude is uwind-
      driven (u 21%, θ×u 11%, θ 9%); SST amplitude θ+u co-dominant (θ 40%, u 37%, θ×u 19%). θ×u is
      strongly PHASE-active — in SST its quadrature is +51% of Vel amp (rotates SST TIW phase far
      more than it adds in-phase amp), quantifying the earlier field-map "θ×u reshapes TIW phase".
      θ is the biggest single SST phase-corrector (−94°→−53°), uwind the biggest V phase-corrector.
FUTURE TASK (deferred by user 2026-10-15): TIW phase/amplitude vs OBSERVATIONS (skill, not vs Vel).
Band-pass RADS SSH + mw_fusion SST obs (loaders exist: tiw.load_rads/load_sst) to the same TIW band,
same complex-projection pipeline but project onto/compare with filtered obs. Scores whether controls
put TIWs in phase+amplitude with the REAL ocean. Reuses anova.py tiw machinery (swap z_Vel→z_obs).
Original design notes (for reference):
- Inputs: prior, each single control, each combo run (e.g. theta+uwind), Vel.
- Main effects:   a_i  = d_i = m_i − m_prior.
- Pairwise:       a_ij = m_ij − m_i − m_j + m_prior   (needs the i+j combo run).
- Higher order analogously (triads from 3-control runs).
- Per obs product + per field: report a_i, a_ij as (1) scalar projection shares
  onto d_full, and (2) maps/depth profiles of the interaction field a_ij.
- Budget checks: Σ(main effects of subset) vs reduced-full = total interaction;
  reduced-full vs Vel = untested-control contribution; leave-one-out marginals.
- EXCLUDED-CONTROLS RESIDUAL TERM (decided 2026-10-13, INCLUDE in the module): define
  r = m_Vel − m_top4 = contribution of the excluded controls (vwind, atemp/aqh/lwdown/
  precip, uvel/vvel) + all their interactions with the subset. Closes the budget exactly:
  m_Vel − m_prior = Σ(all subset ANOVA terms) + r. Report r as its own term per target.
  NOTE r is LARGER than the top4↔Vel RMSE gap implies — ||r||/||d_full|| (increment space)
  = 6% (T), 7% (S), 11% (TAO U), 12% (TAO V), 15% (SST), 34% (SSH). top4≈Vel in misfit
  ≠ excluded controls inert; they drive ~1/3 of the SSH increment (mass/barotropic).
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
