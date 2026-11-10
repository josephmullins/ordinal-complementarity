# TODO

Current paper/slide numbers come from the 24-start regularized fit
(`output/estimation_v3_eps_reg.jld2`, seed 1009, LL −1,109,721.3, λ_logw = 0.01)
and its B=200 child bootstrap (`output/bootstrap_reg.jld2`).

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
- If the fit changes: redo the bootstrap (`scripts/bootstrap_reg.jl`) and everything downstream.

## 2. Numerical bootstrap for the rationalizability tests

- Replace the bootstrap in `scripts/comonotonicity_test.jl` (currently a
  parametric bootstrap from the asymptotic distribution of Π̄, `bootstrap_tstar`)
  with the numerical bootstrap of Hong & Li (2018), which handles the
  non-differentiable t* statistic.

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
