# Nonparametric child bootstrap for the simple ordered-robit pipeline (2026-11-18).
# Resample children with replacement, warm-start EM from the fit, run to TOL; record the
# marginal technology Π̄ (both age groups), β₅/δ, latent investment shares, and the
# period-specific technology re-fitted at each resample's converged posteriors
# (so the period-specific estimates get bootstrap SEs too).
# Data ENV as in run_v3_eps_ordrobit.jl (R_I, NO_PIAT3, GROUP_INV, BINS, TIDY) — must match FIT.
# Also records the induced income transition Π_ind(θ'|θ,y) on the point estimate's income
# grid, every resample's params_v3_eps (for later calculations), and writes the draws in
# the bootstrap_reg.jld2 layout (…_testfmt.jld2) so every test script can consume them.
# Run: julia -t 12 scripts/bootstrap_ordrobit.jl
# ENV FIT, TGFIT (grouped | period), B (200), SEED, MAXITER (500), TOL (1e-4)
using LinearAlgebra, Random, Statistics, JLD2, Optim, Printf
using Base.Threads
const REPO = joinpath(@__DIR__, "..")
for f in ["parametric_models.jl", "parametric_models_v3.jl", "parametric_models_v3_eps.jl", "prepare_data.jl",
          "estimation.jl", "estimation_multiperiod.jl", "estimation_multiperiod_v3.jl",
          "estimation_multiperiod_v3_eps.jl", "parametric_models_v3_robit.jl", "investment_groups.jl",
          "induced_transition.jl"]
    include(joinpath(REPO, "src", f))
end
BLAS.set_num_threads(1)
include(joinpath(REPO, "scripts", "_ordrobit_data.jl"))
include(joinpath(REPO, "scripts", "_period_tech.jl"))
const TGFIT = get(ENV, "TGFIT", "grouped")
const TG = TGFIT == "period" ? TG_PERIOD : [[1, 2, 3], [4, 5, 6]]
const FIT = get(ENV, "FIT", "")
const B = parse(Int, get(ENV, "B", "200"))
const BSEED = parse(Int, get(ENV, "SEED", "20261118"))
const MAXITER = parse(Int, get(ENV, "MAXITER", "500"))
const BTOL = parse(Float64, get(ENV, "TOL", "1e-4"))
F = load(joinpath(REPO, "output", FIT)); const PV = F["params_v3_eps"]; const PSK = F["params_skill"]
(; r_θ, r_I) = PSK; const N = data.N
@assert F["data_dims"].N == N && F["data_dims"].r_I == r_I "data ENV does not match FIT"

function resample_data(d, idx)
    merge(d, (; N = length(idx), X_θ = [[m[idx] for m in d.X_θ[t]] for t in 1:d.T],
              X_I = [[m[idx] for m in d.X_I[t]] for t in 1:d.T], X_P = [m[idx] for m in d.X_P],
              Y_income = d.Y_income[idx, :], X_adult = isempty(d.X_adult) ? d.X_adult : [m[idx] for m in d.X_adult],
              kids = d.kids[idx]))
end
ratio(pv) = pv.ccp_spec.beta5 / pv.income_model.delta
pibar_g(pv) = (Pb = marginalize_Pi_v3(pv.trans_specs, TG, pv.π_Pω, r_θ, r_I, pv.r_P, pv.r_ω); [Pb[g[1]] for g in TG])
ishares(γ) = (s = vec(sum(γ, dims = (1, 2, 4))); s ./ sum(s))

const N_Y = 5; const Y_LEVELS = model_income_grid(PV; n_y = N_Y)
pind_g(pv) = induced_transitions_by_group(pv, TG, Y_LEVELS; r_θ, r_I)
refit_period(pv, ξ) = TGFIT == "period" ? [pibar_g(pv)[t] for t in 1:6] : period_pibar(pv, period_specific_specs(pv, ξ, r_θ, r_I), r_θ, r_I)
pib0 = pibar_g(PV); per0 = refit_period(PV, F["xi_sum_z"]); pind0 = pind_g(PV)
@printf("FIT %s (%s)  LL %.1f  β₅/δ %.3f  N %d  B %d  threads %d\n", FIT, TGFIT, F["best_ll"], ratio(PV), N, B, nthreads()); flush(stdout)

rng = MersenneTwister(BSEED); idx_draws = [rand(rng, 1:N, N) for _ in 1:B]
G = length(TG)
boot_pibar = [fill(NaN, B, r_θ, r_θ, r_I) for _ in 1:G]
boot_pind = [fill(NaN, B, r_θ, r_θ, N_Y) for _ in 1:G]
boot_period = [fill(NaN, B, r_θ, r_θ, r_I) for _ in 1:6]
boot_ratio = fill(NaN, B); boot_ll = fill(NaN, B); boot_ish = fill(NaN, B, r_I); ok = falses(B)
pv_draws = Vector{Any}(nothing, B)
sfx = tag * (TGFIT == "period" ? "_period" : "")
OUT = joinpath(REPO, "output", "bootstrap_ordrobit$(sfx).jld2")
OUT_T = joinpath(REPO, "output", "bootstrap_ordrobit$(sfx)_testfmt.jld2")
pibar_free_n(Π, nc) = Float64[Π[j, kθ, kI] for kI in 1:nc for kθ in 1:r_θ for j in 1:(r_θ - 1)]
surv_free_n(Π, nc) = Float64[sum(Π[j, kθ, kI] for j in k+1:r_θ) for kI in 1:nc for kθ in 1:r_θ for k in 1:(r_θ - 1)]
flat(D, f, nc) = [vcat([f(D[g][b, :, :, :], nc)' for b in 1:B]...) for g in 1:G]
lk = ReentrantLock()
function save_state()
    jldsave(OUT; fit = FIT, tgfit = TGFIT, B, seed = BSEED, ok, idx_draws, boot_pibar, boot_pind, boot_period, boot_ratio,
            boot_ll, boot_ish, pv_draws, pibar0 = pib0, pind0, period0 = per0, ratio0 = ratio(PV),
            ishares0 = ishares(F["gamma_agg"]), transition_groups = TG, r_θ, r_I, n_y = N_Y, y_levels = Y_LEVELS)
    # bootstrap_reg.jld2 layout (free vectors per group) for the test scripts
    jldsave(OUT_T; fit = FIT, B, lambda = 0.0, seed = BSEED, n_ok = count(ok), ok, transition_groups = TG,
            boot_pibar = flat(boot_pibar, pibar_free_n, r_I), boot_surv = flat(boot_pibar, surv_free_n, r_I),
            boot_ratio, boot_ll, idx_draws,
            pibar0 = [pibar_free_n(pib0[g], r_I) for g in 1:G], surv0 = [surv_free_n(pib0[g], r_I) for g in 1:G],
            ratio0 = ratio(PV), r_θ, r_I, n_y = N_Y, y_levels = Y_LEVELS,
            boot_pibar_ind = flat(boot_pind, pibar_free_n, N_Y), boot_surv_ind = flat(boot_pind, surv_free_n, N_Y),
            pibar_ind0 = [pibar_free_n(pind0[g], N_Y) for g in 1:G], surv_ind0 = [surv_free_n(pind0[g], N_Y) for g in 1:G])
end
done = Atomic{Int}(0); t0 = time()
@threads for b in 1:B
    try
        db = resample_data(data, idx_draws[b])
        pv, _, γ, ξ, _, ll = expectation_maximization_v3_eps(db, PV, PSK; transition_groups = TG,
                                                             maxiter = MAXITER, tol = BTOL, verbose = false)
        Pg = pibar_g(pv); Pi = pind_g(pv); per = refit_period(pv, ξ)
        lock(lk) do
            for g in 1:G; boot_pibar[g][b, :, :, :] = Pg[g]; boot_pind[g][b, :, :, :] = Pi[g]; end
            for t in 1:6; boot_period[t][b, :, :, :] = per[t]; end
            boot_ratio[b] = ratio(pv); boot_ll[b] = ll; boot_ish[b, :] = ishares(γ); pv_draws[b] = pv; ok[b] = true
        end
    catch e
        @warn "resample $b failed" exception = (e, catch_backtrace())
    end
    d = atomic_add!(done, 1) + 1
    if d % 10 == 0 || d == B
        lock(lk) do; save_state(); end
        @printf("  %3d/%d done (%.0f min)\n", d, B, (time() - t0) / 60); flush(stdout)
    end
end
save_state()
@printf("\n%d/%d ok.  β₅/δ = %.3f (boot SE %.3f)\nSaved → %s\n", count(ok), B, ratio(PV), std(boot_ratio[ok]), OUT)
