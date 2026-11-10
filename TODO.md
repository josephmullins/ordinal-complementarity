# TODO

## Complete re-run of the paper pipeline (priority)

Current paper/slide numbers come from the 24-start regularized fit
(`output/estimation_v3_eps_reg.jld2`, seed 1009, LL −1,109,721.3, λ_logw = 0.01)
and its B=200 child bootstrap (`output/bootstrap_reg.jld2`). Known gaps:

1. **Multi-start not finished.** The 96-start run (`output/reg_multistart96.log`,
   2026-06-08) died at 40/96. Seed 1062 reached LL@λ0.01 = −1,109,709.3 vs
   −1,109,725.0 for seed 1009 at the same stage (~16 better); never converged.
   - Change `reg_multistart.jl`'s output path first — it writes to
     `estimation_v3_eps_reg.jld2` and would overwrite the paper fit.
   - Finish 96 starts (seeds 1001–1024 already replicate exactly), converge the top few.
2. **Fully cold starts.** `reg_multistart.jl` randomizes only the transition
   layer; emissions, CCPs, income model, π0 and regime probabilities start at
   the old MLE (`estimation_v3_eps.jld2`). Run a few all-random starts to show the
   fit does not depend on that warm start.
3. **If the fit changes:** redo the bootstrap (`scripts/bootstrap_reg.jl`), then
   all tests, counterfactuals and figures.
4. **Single figure pipeline.** Output routing is spread over `PAPER_FIGS`,
   `REG_RUN`, `BOOT_COV`, `SLIDES_FIG_DIR`. Make one build script that sets the
   fit + bootstrap explicitly and writes to one canonical figure folder;
   include `counterfactual_sequential.jl` (currently in neither build script).
   Then diff against the Iowa deck's figures (14 of 23 cannot currently be traced
   to a run).
5. **Reproducibility.** Add `Project.toml` + `Manifest.toml`.

## Dropped
- Piecewise income transfers (`counterfactual_income_piecewise_10k.jl`,
  `counterfactual_income_transfers_piecewise.jl`) — no longer committed; remove
  `cf_income_piecewise_10k.pdf` and the transfer table from slides/draft.
  β5/δ now comes from `scripts/ccp_income_semielasticity.jl`.
