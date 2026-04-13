# ─────────────────────────────────────────────────────────────────
# compare_shared_vs_unrestricted.jl
#
# Estimates the age group (4,5)→(6,7) under two models:
#   1. Unrestricted: separate emission matrices P_θ and P_Y
#   2. Shared:       P_θ = P_Y (only using measures present at both periods)
#
# Then compares:
#   - Estimated parameters (Π, π0, emission matrices)
#   - Log-likelihoods on the shared-measure data
#   - Likelihood ratio test
#   - Diagnoses Π for zero/near-zero entries (score singularity source)
# ─────────────────────────────────────────────────────────────────

using LinearAlgebra, Random, Statistics, JLD2

include(joinpath(@__DIR__, "..", "src", "prepare_data.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation.jl"))

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────

const CFG = (age_t1=(4,5), age_t2=(6,7), invest_items=[
    :HowOftenRead, :HowManyBooks, :HowManyZines, :HowOftenOuting,
    :HowOftenMuseum, :LearnNumbers, :LearnAlphabet, :LearnColors,
    :LearnShapes, :RecordTapeCD, :FoodChoice])

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

"""
    make_shared_data(data) -> NamedTuple

Filter X_θ and Y to measures present at BOTH periods (matched by name AND κ).
"""
function make_shared_data(data)
    (; N, r_θ, r_I,
       X_θ, κ_θ, measure_names_θ,
       X_I, κ_I, measure_names_I,
       Y, κ_Y, measure_names_Y,
       kids) = data

    t1_lookup = Dict{String, Tuple{Int, Int}}()
    for (m, name) in enumerate(measure_names_θ)
        t1_lookup[name] = (m, κ_θ[m])
    end

    t2_lookup = Dict{String, Tuple{Int, Int}}()
    for (m, name) in enumerate(measure_names_Y)
        t2_lookup[name] = (m, κ_Y[m])
    end

    shared_names = String[]
    for name in measure_names_θ
        if haskey(t2_lookup, name)
            if t1_lookup[name][2] == t2_lookup[name][2]
                push!(shared_names, name)
            end
        end
    end
    shared_names = unique(shared_names)

    t1_only = setdiff(measure_names_θ, shared_names)
    t2_only = setdiff(measure_names_Y, shared_names)

    println("\n  Shared skill emission filtering:")
    println("    Period 1 measures: $(length(measure_names_θ))")
    println("    Period 2 measures: $(length(measure_names_Y))")
    println("    Shared measures:   $(length(shared_names))")
    if !isempty(t1_only)
        println("    Dropped (t1 only): $(length(t1_only))")
        for name in t1_only; println("      - $name"); end
    end
    if !isempty(t2_only)
        println("    Dropped (t2 only): $(length(t2_only))")
        for name in t2_only; println("      - $name"); end
    end
    for name in shared_names
        println("      + $name (κ=$(t1_lookup[name][2]))")
    end

    isempty(shared_names) && error("No shared skill measures between periods.")

    X_θ_shared = Vector{Vector{Int}}()
    Y_shared   = Vector{Vector{Int}}()
    κ_shared   = Int[]
    names_shared = String[]

    for name in shared_names
        m1 = t1_lookup[name][1]
        m2 = t2_lookup[name][1]
        push!(X_θ_shared, X_θ[m1])
        push!(Y_shared, Y[m2])
        push!(κ_shared, t1_lookup[name][2])
        push!(names_shared, name)
    end

    return (;
        N, r_θ, r_I,
        X_θ = X_θ_shared, κ_θ = κ_shared, measure_names_θ = names_shared,
        X_I, κ_I, measure_names_I,
        Y = Y_shared, κ_Y = κ_shared, measure_names_Y = names_shared,
        kids,
        shared_skill_emission = true,
    )
end

"""
    count_free_params(data; shared::Bool) -> Int

Count the number of free parameters in the model.
"""
function count_free_params(data; shared::Bool)
    r_θ = data.r_θ
    r_I = data.r_I

    # Π: (r_θ-1) * r_θ * r_I free params (last row determined by sum-to-1)
    n_Π = (r_θ - 1) * r_θ * r_I

    # π0: r_θ * r_I - 1 free params (sum to 1)
    n_π0 = r_θ * r_I - 1

    # Investment emissions: each P_I[m] has (κ-1)*r_I free params
    n_inv = sum((κ - 1) * r_I for κ in data.κ_I)

    if shared
        # Shared: one set of skill emission matrices
        n_skill = sum((κ - 1) * r_θ for κ in data.κ_θ)
    else
        # Unrestricted: separate P_θ and P_Y
        n_skill_t1 = sum((κ - 1) * r_θ for κ in data.κ_θ)
        n_skill_t2 = sum((κ - 1) * r_θ for κ in data.κ_Y)
        n_skill = n_skill_t1 + n_skill_t2
    end

    return n_Π + n_π0 + n_inv + n_skill
end

"""
    diagnose_transition(Π; threshold=1e-6)

Report entries of Π that are zero or near-zero (these cause score singularities).
"""
function diagnose_transition(Π; threshold=1e-6)
    r_θ, _, r_I = size(Π)
    n_zero = 0
    n_near_zero = 0
    println("\n  Π diagnostics (entries < $threshold):")
    for kI in 1:r_I, kθ in 1:r_θ, kθ1 in 1:r_θ
        v = Π[kθ1, kθ, kI]
        if v == 0.0
            println("    Π[$kθ1, $kθ, $kI] = 0.0  (EXACT ZERO)")
            n_zero += 1
        elseif v < threshold
            println("    Π[$kθ1, $kθ, $kI] = $(round(v; sigdigits=4))  (near zero)")
            n_near_zero += 1
        end
    end
    if n_zero == 0 && n_near_zero == 0
        println("    All entries > $threshold")
    else
        println("    Summary: $n_zero exact zeros, $n_near_zero near-zeros")
    end
end

"""
    diagnose_score(Q, Π)

Check the score matrix for NaN/Inf entries and report the score covariance condition.
"""
function diagnose_score(Q, Π)
    dL = get_score(Q, Π)
    n_nan = count(isnan, dL)
    n_inf = count(isinf, dL)
    println("\n  Score matrix diagnostics:")
    println("    Size: $(size(dL))")
    println("    NaN entries: $n_nan")
    println("    Inf entries: $n_inf")
    println("    Range: [$(minimum(dL)), $(maximum(dL))]")

    if n_nan == 0 && n_inf == 0
        Σ = cov(dL')
        ev = eigvals(Hermitian(Σ))
        println("    Σ_score eigenvalues: min=$(round(minimum(ev); sigdigits=4)), max=$(round(maximum(ev); sigdigits=4))")
        println("    Condition number: $(round(maximum(ev) / max(minimum(ev), 1e-16); sigdigits=4))")
        n_neg = count(x -> x < 0, ev)
        n_tiny = count(x -> x < 1e-10, ev)
        println("    Negative eigenvalues: $n_neg")
        println("    Eigenvalues < 1e-10: $n_tiny")
    else
        println("    Cannot compute Σ_score due to NaN/Inf in scores")
        # Identify which parameters have the problem
        for j in 1:size(dL, 1)
            row = dL[j, :]
            n_bad = count(x -> !isfinite(x), row)
            if n_bad > 0
                println("      Parameter $j: $n_bad non-finite entries")
            end
        end
    end
end

# ─────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────

println("Loading datasets...")
datasets = load_all_datasets()

println("\n" * "="^70)
println("  AGE GROUP: $(CFG.age_t1) → $(CFG.age_t2)")
println("="^70)

# ─────────────────────────────────────────────────────────────────
# Prepare full (unrestricted) data
# ─────────────────────────────────────────────────────────────────

println("\n" * "-"^70)
println("  STEP 1: Prepare data")
println("-"^70)

data_full = prepare_age_group_data(datasets;
    age_range_t1=CFG.age_t1,
    age_range_t2=CFG.age_t2,
    r_θ=3, r_I=3,
    invest_items=CFG.invest_items)

println("  N = $(data_full.N)")
println("  Period 1 skill measures: $(length(data_full.X_θ)) (names: $(join(data_full.measure_names_θ[1:min(5,end)], ", "))...)")
println("  Period 2 skill measures: $(length(data_full.Y)) (names: $(join(data_full.measure_names_Y[1:min(5,end)], ", "))...)")
println("  Investment measures: $(length(data_full.X_I))")

# ─────────────────────────────────────────────────────────────────
# Prepare shared data (subset of measures)
# ─────────────────────────────────────────────────────────────────

data_shared = make_shared_data(data_full)

# Also prepare an unrestricted dataset using the SAME shared measures
# so log-likelihoods are directly comparable
data_unrestricted_on_shared = (;
    data_shared.N, data_shared.r_θ, data_shared.r_I,
    data_shared.X_θ, data_shared.κ_θ, data_shared.measure_names_θ,
    data_shared.X_I, data_shared.κ_I, data_shared.measure_names_I,
    data_shared.Y, data_shared.κ_Y, data_shared.measure_names_Y,
    data_shared.kids,
    shared_skill_emission = false,
)

println("\n  Comparable datasets prepared:")
println("    Shared measures: $(length(data_shared.X_θ))")
println("    Same observations: N=$(data_shared.N)")

# ─────────────────────────────────────────────────────────────────
# STEP 2: Estimate unrestricted model (on shared measures)
# ─────────────────────────────────────────────────────────────────

println("\n" * "-"^70)
println("  STEP 2: Estimate UNRESTRICTED model (P_θ ≠ P_Y, shared measures)")
println("-"^70)

params_unr, Q_unr, ll_unr = run_em_restarts(data_unrestricted_on_shared;
    shared_skill_emission=false)
println("\n  Best log-likelihood (unrestricted): $(round(ll_unr; digits=2))")

# Normalize
params_unr, Q_unr = normalize_params(params_unr, Q_unr, data_unrestricted_on_shared)

# ─────────────────────────────────────────────────────────────────
# STEP 3: Estimate shared model (P_θ = P_Y)
# ─────────────────────────────────────────────────────────────────

println("\n" * "-"^70)
println("  STEP 3: Estimate SHARED model (P_θ = P_Y)")
println("-"^70)

params_sh, Q_sh, ll_sh = run_em_restarts(data_shared;
    shared_skill_emission=true)
println("\n  Best log-likelihood (shared): $(round(ll_sh; digits=2))")

# Normalize
params_sh, Q_sh = normalize_params(params_sh, Q_sh, data_shared)

# ─────────────────────────────────────────────────────────────────
# STEP 4: Compare parameters
# ─────────────────────────────────────────────────────────────────

println("\n" * "-"^70)
println("  STEP 4: Parameter comparison")
println("-"^70)

println("\n  ── Unrestricted model ──")
display_results(params_unr, data_unrestricted_on_shared)

println("\n  ── Shared model ──")
display_results(params_sh, data_shared)

# Transition matrix comparison
println("\n" * "="^60)
println("  Π COMPARISON (unrestricted vs shared)")
println("="^60)
r_θ = data_shared.r_θ
r_I = data_shared.r_I
max_abs_diff = 0.0
for kI in 1:r_I
    println("  I = $kI:")
    for kθ1 in 1:r_θ
        print("    θ'=$kθ1: ")
        for kθ in 1:r_θ
            u = params_unr.Π[kθ1, kθ, kI]
            s = params_sh.Π[kθ1, kθ, kI]
            d = s - u
            max_abs_diff = max(max_abs_diff, abs(d))
            print("  $(round(u; digits=4))/$(round(s; digits=4)) (Δ=$(round(d; digits=4)))")
        end
        println()
    end
end
println("  Max |Δ| in Π: $(round(max_abs_diff; digits=6))")

# Emission matrix comparison (shared measures only)
println("\n" * "="^60)
println("  SKILL EMISSION COMPARISON (P_θ)")
println("  Format: unrestricted / shared")
println("="^60)
for m in eachindex(data_shared.measure_names_θ)
    name = data_shared.measure_names_θ[m]
    κ_m = data_shared.κ_θ[m]
    println("  $name (κ=$κ_m):")
    for j in 1:κ_m
        print("    cat=$j: ")
        for kθ in 1:r_θ
            u = params_unr.P_θ[m][j, kθ]
            s = params_sh.P_θ[m][j, kθ]
            print("  θ=$kθ: $(round(u; digits=4))/$(round(s; digits=4))")
        end
        println()
    end
end

println("\n" * "="^60)
println("  SKILL EMISSION COMPARISON (P_Y)")
println("  Format: unrestricted / shared")
println("="^60)
for m in eachindex(data_shared.measure_names_Y)
    name = data_shared.measure_names_Y[m]
    κ_m = data_shared.κ_Y[m]
    println("  $name (κ=$κ_m):")
    for j in 1:κ_m
        print("    cat=$j: ")
        for kθ in 1:r_θ
            u = params_unr.P_Y[m][j, kθ]
            s = params_sh.P_Y[m][j, kθ]
            print("  θ=$kθ: $(round(u; digits=4))/$(round(s; digits=4))")
        end
        println()
    end
end

# ─────────────────────────────────────────────────────────────────
# STEP 5: Likelihood ratio test
# ─────────────────────────────────────────────────────────────────

println("\n" * "-"^70)
println("  STEP 5: Likelihood ratio test")
println("-"^70)

n_free_unr = count_free_params(data_unrestricted_on_shared; shared=false)
n_free_sh  = count_free_params(data_shared; shared=true)
df = n_free_unr - n_free_sh

println("  Free parameters (unrestricted): $n_free_unr")
println("  Free parameters (shared):       $n_free_sh")
println("  Degrees of freedom:             $df")
println()
println("  Log-likelihood (unrestricted):  $(round(ll_unr; digits=4))")
println("  Log-likelihood (shared):        $(round(ll_sh; digits=4))")
println("  Δ log-likelihood:               $(round(ll_unr - ll_sh; digits=4))")
println("  LR statistic:                   $(round(2 * (ll_unr - ll_sh); digits=4))")

using Distributions
lr_stat = 2 * (ll_unr - ll_sh)
if df > 0 && lr_stat >= 0
    p_value = 1.0 - cdf(Chisq(df), lr_stat)
    println("  p-value (χ² with $df df):       $(round(p_value; sigdigits=4))")
    println("  Reject shared at 5%:            $(p_value < 0.05 ? "YES" : "no")")
else
    println("  WARNING: df=$df, LR=$lr_stat — cannot compute p-value")
end

# ─────────────────────────────────────────────────────────────────
# STEP 6: Diagnose singularity issues
# ─────────────────────────────────────────────────────────────────

println("\n" * "-"^70)
println("  STEP 6: Singularity diagnostics")
println("-"^70)

println("\n  ── Unrestricted model ──")
diagnose_transition(params_unr.Π)
diagnose_score(Q_unr, params_unr.Π)

println("\n  ── Shared model ──")
diagnose_transition(params_sh.Π)
diagnose_score(Q_sh, params_sh.Π)

# ─────────────────────────────────────────────────────────────────
# STEP 7: Also estimate the full unrestricted model (all measures)
#         and compare log-likelihoods
# ─────────────────────────────────────────────────────────────────

println("\n" * "-"^70)
println("  STEP 7: Full unrestricted model (all measures) for reference")
println("-"^70)

params_full, Q_full, ll_full = run_em_restarts(data_full;
    shared_skill_emission=false)
println("\n  Best log-likelihood (full, all measures): $(round(ll_full; digits=2))")

params_full, Q_full = normalize_params(params_full, Q_full, data_full)
n_free_full = count_free_params(data_full; shared=false)

println("\n  Free parameters (full):          $n_free_full")
println("  Free parameters (shared subset): $n_free_unr")
println("  Free parameters (shared model):  $n_free_sh")
println()
println("  Log-likelihoods:")
println("    Full model (all measures):     $(round(ll_full; digits=4))")
println("    Unrestricted (shared measures): $(round(ll_unr; digits=4))")
println("    Shared (shared measures):       $(round(ll_sh; digits=4))")

println("\n  ── Full model Π diagnostics ──")
diagnose_transition(params_full.Π)
diagnose_score(Q_full, params_full.Π)

# ─────────────────────────────────────────────────────────────────
# Save all results
# ─────────────────────────────────────────────────────────────────

outdir = joinpath(dirname(@__DIR__), "output")
mkpath(outdir)
outpath = joinpath(outdir, "comparison_4_5__6_7.jld2")
@save outpath params_unr Q_unr data_unrestricted_on_shared ll_unr params_sh Q_sh data_shared ll_sh params_full Q_full data_full ll_full
println("\n  All results saved to $outpath")

# ─────────────────────────────────────────────────────────────────
# STEP 8: Plot production functions E[θ'|θ, I] for both models
# ─────────────────────────────────────────────────────────────────

println("\n" * "-"^70)
println("  STEP 8: Production function plots")
println("-"^70)

using Plots

# Compute E[θ'|θ, I] for a given Π
function expected_theta_prime(Π)
    r_θ, _, r_I = size(Π)
    E = zeros(r_θ, r_I)
    for kθ in 1:r_θ, kI in 1:r_I
        E[kθ, kI] = sum(kθ1 * Π[kθ1, kθ, kI] for kθ1 in 1:r_θ)
    end
    return E
end

E_unr  = expected_theta_prime(params_unr.Π)
E_sh   = expected_theta_prime(params_sh.Π)
E_full = expected_theta_prime(params_full.Π)

theta_colors = [:steelblue, :darkorange, :forestgreen]

prod_plots = []
for (label, E_mat) in [("Unrestricted (shared measures)", E_unr),
                        ("Shared (P_θ = P_Y)", E_sh),
                        ("Unrestricted (all measures)", E_full)]
    p = plot(
        title = label,
        xlabel = "Investment (I)",
        ylabel = "E[θ' | θ, I]",
        ylim = (0.8, r_θ + 0.2),
        xticks = 1:r_I,
        legend = :topleft,
        grid = true, gridalpha = 0.3,
    )
    for kθ in 1:r_θ
        plot!(p, 1:r_I, E_mat[kθ, :];
            label = "θ=$kθ",
            linewidth = 2.5,
            marker = :circle, markersize = 6,
            color = theta_colors[kθ],
        )
    end
    push!(prod_plots, p)
end

fig_prod = plot(prod_plots...,
    layout = (1, 3),
    size = (1400, 450),
    margin = 6Plots.mm,
    plot_title = "Production Function: E[θ' | θ, I]  —  Age (4-5) → (6-7)",
    plot_titlefontsize = 12,
)

save_path = joinpath(outdir, "production_function_comparison_4_5__6_7.png")
savefig(fig_prod, save_path)
println("  Production function plot saved to $save_path")

# Also plot survivor functions side by side
surv_plots = []
for (label, Π_mat) in [("Unrestricted (shared measures)", params_unr.Π),
                        ("Shared (P_θ = P_Y)", params_sh.Π),
                        ("Unrestricted (all measures)", params_full.Π)]
    for k in 1:r_θ-1
        p = plot(
            title = "$label\nP(θ' > $k | θ, I)",
            xlabel = "Investment (I)",
            ylabel = "Survival probability",
            ylim = (0, 1.05),
            xticks = 1:r_I,
            legend = (k == 1) ? :bottomright : false,
            titlefontsize = 8,
            grid = true, gridalpha = 0.3,
        )
        for kθ in 1:r_θ
            surv = [1.0 - sum(Π_mat[1:k, kθ, kI]) for kI in 1:r_I]
            plot!(p, 1:r_I, surv;
                label = "θ=$kθ",
                linewidth = 2.5,
                marker = :circle, markersize = 5,
                color = theta_colors[kθ],
            )
        end
        push!(surv_plots, p)
    end
end

fig_surv = plot(surv_plots...,
    layout = (r_θ - 1, 3),
    size = (1400, 350 * (r_θ - 1)),
    margin = 5Plots.mm,
    plot_title = "Survivor Functions: Age (4-5) → (6-7)",
    plot_titlefontsize = 12,
)

save_path_surv = joinpath(outdir, "survivor_comparison_4_5__6_7.png")
savefig(fig_surv, save_path_surv)
println("  Survivor function plot saved to $save_path_surv")

println("\n" * "="^70)
println("  DONE")
println("="^70)
