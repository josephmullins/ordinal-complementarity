#!/usr/bin/env julia
# ─────────────────────────────────────────────────────────────────
# Patch estimation_multiperiod_mixture.jld2:
#   - Recompute complementarity test stats (with degenerate-V fix)
#   - Add parametric Var_free per transition (for survivor SEs)
#
# Loads the saved EM output, recomputes ψ_all from the saved params,
# then overwrites the .jld2 with the corrected test_results and Var_free.
# ─────────────────────────────────────────────────────────────────

using LinearAlgebra, Random, Statistics, JLD2, ForwardDiff

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod.jl"))
include(joinpath(@__DIR__, "..", "src", "inference.jl"))

# ── Load existing results ──
outdir = joinpath(dirname(@__DIR__), "output")
inpath = joinpath(outdir, "estimation_multiperiod_mixture.jld2")
println("Loading: $inpath")

d = load(inpath)
best_params = d["best_params"]
best_γ = haskey(d, "best_gamma") ? d["best_gamma"] : d["best_γ"]
data = d["data"]
best_ll = d["best_ll"]
best_specs = d["best_specs"]
parametric_config = d["parametric_config"]

(; r_θ, r_I, T, N) = data
println("T = $T, N = $N, r_θ = $r_θ, r_I = $r_I")

# ── Recompute pairwise posteriors ──
println("\nComputing per-observation pairwise posteriors...")
ψ_all = compute_psi_all(best_params, data)

# ── Recompute test stats (with degenerate-V fix) ──
println("Running nonparametric complementarity tests...")
test_results = get_test_stat_mp(ψ_all, best_params, data; α=0.05)

for res in test_results
    status = isnan(res.T_n) ? "SKIPPED (degenerate)" :
             res.reject ? "REJECT" : "FAIL TO REJECT"
    println("  Transition $(res.t): T_n = $(round(res.T_n; digits=4)),  " *
            "c_α = $(round(res.c_α; digits=4))  [$status]")
end

# ── Compute parametric Var(Π̂_free) per transition ──
println("\nComputing parametric Var(Π̂_free) per transition...")
Var_free_list = Vector{Matrix{Float64}}(undef, T - 1)

for t in 1:T - 1
    spec_t = best_specs[:transition][t]::MixtureProbitTransition
    Σ_score_np = test_results[t].Σ_score

    φ = pack_mixture_probit(spec_t)
    K = spec_t.K

    # Jacobian: φ → Π_free
    f = φ_vec -> let
        Π_full = _build_mixture_probit_pi(φ_vec, r_θ, r_I, K)
        v = similar(φ_vec, (r_θ - 1) * r_θ * r_I)
        idx = 0
        for kI in 1:r_I, kθ in 1:r_θ, j in 1:(r_θ - 1)
            idx += 1
            v[idx] = Π_full[j, kθ, kI]
        end
        v
    end
    J = ForwardDiff.jacobian(f, φ)

    # Parametric variance via chain rule
    Σ_φ = J' * Σ_score_np * J
    Var_φ = pinv(Σ_φ) / N
    Var_free_list[t] = J * Var_φ * J'

    # Quick sanity check
    SE_t = survivor_se_all(Var_free_list[t], best_params.Π[t])
    max_se = maximum(SE_t)
    println("  Transition $t: n_φ = $(length(φ)), max survivor SE = $(round(max_se; digits=4))")
end

# ── Save patched results ──
outpath = joinpath(outdir, "estimation_multiperiod_mixture.jld2")
println("\nSaving patched results to: $outpath")
jldsave(outpath; best_params, best_gamma=best_γ, data, best_ll, test_results,
        best_specs, parametric_config, Var_free=Var_free_list)

println("Done. Re-run notebooks/analyze_results_mixture.ipynb to see corrected SEs.")
