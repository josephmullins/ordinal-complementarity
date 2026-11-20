# TODO

Current paper/slide numbers come from the 24-start regularized fit
(`output/estimation_v3_eps_reg.jld2`, seed 1009, LL −1,109,721.3, λ_logw = 0.01)
and its B=200 child bootstrap (`output/bootstrap_reg.jld2`).

## NEXT: choose the technology specification methodically (added 2026-11-20)

The new pipeline (below) changed the results substantially, so the technology
form has to be chosen by a documented comparison, not by default. On the same data
(`R_I=4 NO_PIAT3=1 GROUP_INV=1 BINS=5 AGEBIN=1 TIDY=1`), several cold-start seeds each,
fit and compare by LL / BIC, seed agreement, and the shape of Π̄:

- per-skill ordered robit (current q5a spec, `OrderedRobitPerThetaV3`);
- per-skill + a quadratic in investment for some skill levels only
  (`OrderedRobitPerThetaQuadV3` currently puts δ_θ·I² on every θ — needs a variant with
  δ_θ = 0 for the other levels; `MODE=quad` in `run_v3_eps_ordrobit.jl`);
- robit NN with K = 2 and K = 3 (`MODE=nn K=2|3`).

Then check whether the OC/OS, t± and counterfactual conclusions move across the
specifications that the data cannot tell apart.

## Simple ordered-robit pipeline (2026-11-17 → 20; exploratory, not in the paper)

Cold-start, reproducible across seeds: per-skill ordered robit technology, HOME items
grouped by content in advance (`src/investment_groups.jl`), no PIAT at ages 4–5,
r_I = 4, grouped technology TG = [[1,2,3],[4,5,6]], B = 500 child bootstrap.
Scripts: `scripts/_ordrobit_data.jl`, `run_v3_eps_ordrobit.jl`, `bootstrap_ordrobit.jl`,
`run_test_battery.sh`, `_postboot_grouped.sh`; tables in `output/tests_ordrobit_q5a/`,
Hong–Li t± in both directions in `output/ndm_flip_q5a.txt` (`scripts/_ndm_flip.jl`).

- Ages 0–5: supermodularity (OC) not rejected, OS rejected; "no Q gives OC" rejected.
- Ages 6–13: OC and OS both rejected at the raw labels; some Q gives OC and another gives OS
  (substitution at the bottom thresholds, complementarity at the top).
- Counterfactuals (`scripts/_cf_ordrobit.jl`, exploratory): optimal investment is V-shaped,
  not anti-comonotone; the income gain is almost entirely equalization and the y(θ₁) profile
  tilts toward high θ₁ — the opposite of the paper. The static income counterpart is new.
  Caveat: the data-rule baseline chain does not reproduce the empirical F_θT.

## 0. URGENT: empty latent investment level in the paper fit (found 2026-11-17)

In both the reg fit and the MLE, latent investment I=4 has posterior share 0.000 in
every period (CCP cutpoints c3 = c4). Π̄ at I=4 is pure network interpolation, and
the I-steps 3→4 and 4→5 — half the cells in every OC/OS/t±/triangle test — pass
through an unpopulated level. Decide: tests on the populated lattice {1,2,3,5},
or refit with r_I = 4. Cold-start ordered-robit fits (`scripts/run_v3_eps_ordrobit.jl`)
populate all five levels.

## 1. Full estimation pipeline + thorough multi-start

- Establish one end-to-end estimation pipeline (data → fit → bootstrap → tests →
  counterfactuals) and test it; compare its output to the current results
  (`output/paper_tables_boot/`).
- Run a more thorough multi-start routine:
  - The 96-start run (`output/reg_multistart96.log`, 2026-06-08) died at 40/96.
    Seed 1062 reached LL@λ0.01 = −1,109,709.3 vs −1,109,725.0 for seed 1009 at
    the same stage (~16 better); never converged. Seeds 1001–1024 replicate exactly.
  - `reg_multistart.jl` randomizes only the transition layer; emissions, CCPs,
    income model, π0 and regime probabilities warm-start from the old MLE
    (`estimation_v3_eps.jld2`). Add fully cold starts.
  - **Change `reg_multistart.jl`'s output path first** — it writes to
    `estimation_v3_eps_reg.jld2` and would overwrite the paper fit.
- Check flatness in investment across optima: in the reg fit (seed 1009) the
  ages 0–5 technology is flat for I ≥ 3 (max |ΔΠ̄| ≤ 6e-4; `scripts/_pibar_flat_in_I.jl`),
  which pins 6 of 12 cross-differences at 0 and forces t̂⁺ ≤ 0 ≤ t̂⁻. The MLE is not
  flat (max |ΔΠ̄| 0.07–0.17; `scripts/_pibar_flat_mle.jl`). Compare across the top optima.
- If the fit changes: redo the bootstrap (`scripts/bootstrap_reg.jl`) and everything downstream.

## 2. Numerical bootstrap for the rationalizability tests

- Replace the bootstrap in `scripts/comonotonicity_test.jl` (currently a
  parametric bootstrap from the asymptotic distribution of Π̄, `bootstrap_tstar`)
  with the numerical bootstrap of Hong & Li (2018), which handles the
  non-differentiable t* statistic.

### 2b. Refining beyond OC/OS — explored 2026-11-10 / 11-17 (come back to)

All exploratory, on the reg fit + B=200 bootstrap; results in
`output/paper_tables_boot/*_exploratory.csv`, figures in `figures/` (gitignored).

| Test | Script | Finding |
|---|---|---|
| Hong–Li NDM on t± (unrestricted, convex, concave Q) | `comonotonicity_test.jl` (ndm modes) | no t± null rejected; percentile CIs in draft invalid (toy MC `_ndm_toy_mc.jl`) |
| NDM vs standard bootstrap distributions | `_ndm_compare.jl` | NDM shifts CIs where cells near-tie (t⁺ up, t⁻ down) |
| Closest-to-OC/OS valuation Q̂ + E[Q̂] cross-diffs | `_rationalize_Q_school.jl` (GROUP, SENSE, SHAPE) | 6–13: best Q convex, returns fall with θ; best concave Q = linear, all cells < 0 |
| Flatness of Π̄ in I | `_pibar_flat_in_I.jl`, `_pibar_flat_mle.jl` | reg fit flat for I≥3 at 0–5 (MLE not) — see item 1 |
| Studentized t± (cell s.e. weights) + NDM | `_ndm_studentized.jl` | rejects induced 6–13 non-complementary, induced 0–5 no convex-Q substitutability |
| Dual-certificate IUT (equal / precision / in-sample dual p) | `_certificate_test.jl`, `_concave_Q_cells.jl` | rejects "some concave Q gives complementarity" at 6–13 (both valid weightings) |
| Fixed-Q "some Q works" IUT (6 pre-specified Q, Bonferroni) | `_fixedQ_rationalize_test.jl` | nothing confirmed; closest 6–13 sub at (0,1,1,1), weakest z 1.38 |
| Q-triangle map, pointwise RSW + Fang–Seo per Q | `_Q_triangle_map.jl` | 6–13 RSW rejects supermodularity for all but top-weighted Q; FS rejects nothing (noisy cells); induced 0–5 splits cleanly by Q |
| Q-triangle confirmation map (IUT, points + regions) | `_Q_triangle_confirm.jl` | nothing confirmed; 6–13 all cells < 0 at estimate over concave/bottom-weighted region |

Open: uniform (region-level) version of the triangle map; anchors (HS, college,
earnings, employment) as points in the triangle; studentized t± / certificate
test would change the paper's procedure — co-author decision.

## 3. One figure pipeline

- Output routing is spread over `PAPER_FIGS`, `REG_RUN`, `BOOT_COV`,
  `SLIDES_FIG_DIR`. Make one build script that sets the fit + bootstrap
  explicitly and writes to one canonical figure folder; include
  `counterfactual_sequential.jl` (currently in neither build script).
- Diff the output against the Iowa deck's figures (14 of 23 cannot currently be
  traced to a run).

## 4. Project.toml + Manifest.toml

- Add both so the environment is reproducible.

## Dropped

- Piecewise income transfers (`counterfactual_income_piecewise_10k.jl`,
  `counterfactual_income_transfers_piecewise.jl`) — gitignored, not deleted;
  remove `cf_income_piecewise_10k.pdf` and the transfer table from slides/draft.
  β5/δ now comes from `scripts/ccp_income_semielasticity.jl`.
