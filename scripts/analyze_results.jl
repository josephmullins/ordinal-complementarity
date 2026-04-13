using LinearAlgebra, Random, Statistics, JLD2

include(joinpath(@__DIR__, "..", "src", "estimation.jl"))
include(joinpath(@__DIR__, "..", "src", "inference.jl"))

# ─────────────────────────────────────────────────────────────────
# Find all estimation result files
# ─────────────────────────────────────────────────────────────────

outdir = joinpath(dirname(@__DIR__), "output")
result_files = sort(filter(f -> startswith(f, "estimation_") && endswith(f, ".jld2"), readdir(outdir)))

if isempty(result_files)
    error("No estimation result files found in $outdir. Run run_estimation.jl first.")
end

println("Found $(length(result_files)) estimation result file(s):")
for f in result_files
    println("  $f")
end

# ─────────────────────────────────────────────────────────────────
# Analyze each age group
# ─────────────────────────────────────────────────────────────────

using Plots

summary_rows = []

for result_file in result_files
    inpath = joinpath(outdir, result_file)
    @load inpath best_params best_Q data best_ll

    # Extract age group label from filename
    label = replace(replace(result_file, "estimation_" => ""), ".jld2" => "")
    age_parts = split(label, "__")
    t1 = replace(age_parts[1], "_" => "-")
    t2 = replace(age_parts[2], "_" => "-")
    age_label = "($t1) → ($t2)"

    println("\n" * "="^60)
    println("  Age group: $age_label")
    println("="^60)
    println("  Log-likelihood: $(round(best_ll; digits=2))")
    println("  N = $(data.N), r_θ = $(data.r_θ), r_I = $(data.r_I)")

    # ── Normalize latent factor ordering ──
    println("\n  Normalize Latent Factors:")
    best_params, best_Q = normalize_params(best_params, best_Q, data)

    # ── Display results ──
    display_results(best_params, data)

    # ── Test for ordinal complementarity ──
    println("\n  Test for Ordinal Complementarity:")

    r_θ = data.r_θ
    r_I = data.r_I
    N = data.N

    result = get_test_stat(best_Q, best_params, data)
    T_n = result.T_n
    Ω = result.Ω
    Σ_score = result.Σ_score

    eigvals_Ω = eigvals(Hermitian(Ω))
    println("  Ω eigenvalues: min=$(round(minimum(eigvals_Ω); sigdigits=4)), max=$(round(maximum(eigvals_Ω); sigdigits=4))")
    println("  Test statistic T_n = $(round(T_n; digits=4))")

    rng_cv = MersenneTwister(2025)
    c_α = crit_value(Hermitian(Ω), 0.05; rng=rng_cv)
    println("  Critical value (5%): $(round(c_α; digits=4))")
    reject = T_n > c_α
    println("  Reject H0 (supermodularity): $reject")
    if reject
        println("  → Evidence AGAINST ordinal complementarity")
    else
        println("  → FAIL to reject ordinal complementarity (consistent with H0)")
    end

    push!(summary_rows, (;
        age_group = age_label, N, r_θ, r_I,
        n_skill_t1 = length(data.X_θ), n_skill_t2 = length(data.Y),
        n_invest = length(data.X_I),
        ll = best_ll, T_n, c_α, reject))

    # ── CDF diagnostics ──
    println("\n  CDF difference-in-differences (supermodularity conditions):")
    println("  (All should be ≤ 0 under H0)")

    Π = best_params.Π
    for kθ in 1:r_θ, kI in 1:r_I
        for kθ2 in kθ+1:r_θ, kI2 in kI+1:r_I
            println("    (θ=$kθ,I=$kI) vs (θ=$kθ2,I=$kI2):")
            for j in 1:r_θ-1
                F(θ,I) = sum(Π[1:j, θ, I])
                did = F(kθ, kI) + F(kθ2, kI2) - F(kθ, kI2) - F(kθ2, kI)
                println("      CDF at θ'≤$j: $(round(did; digits=4))  $(did ≤ 1e-10 ? "✓" : "✗")")
            end
        end
    end

    # ── Survivor function plots ──
    println("\n  Survivor Function Plots:")
    SE = survivor_se_all(Σ_score, best_params.Π, N)

    z = 1.96
    plots_arr = []
    for k in 1:r_θ-1
        p = plot(
            title="P(θ' > $k | θ, I)",
            xlabel="Investment (I)",
            ylabel="Survival probability",
            legend=:best,
            ylim=(0, 1),
            xticks=1:r_I,
        )
        for kθ in 1:r_θ
            surv = [1.0 - sum(Π[1:k, kθ, kI]) for kI in 1:r_I]
            se = [SE[k, kθ, kI] for kI in 1:r_I]
            lo = clamp.(surv .- z .* se, 0, 1)
            hi = clamp.(surv .+ z .* se, 0, 1)

            plot!(p, 1:r_I, surv;
                ribbon=(surv .- lo, hi .- surv),
                fillalpha=0.15,
                label="θ=$kθ", linewidth=2, marker=:circle, markersize=5)
        end
        push!(plots_arr, p)
    end

    n_plots = r_θ - 1
    fig = plot(plots_arr..., layout=(1, n_plots), size=(450 * n_plots, 400),
               plot_title="Age group: $age_label")
    save_path = joinpath(outdir, "survivor_$(label).png")
    savefig(fig, save_path)
    println("  Plot saved to $save_path")
end

# ─────────────────────────────────────────────────────────────────
# Summary table across age groups
# ─────────────────────────────────────────────────────────────────
println("\n" * "="^60)
println("  SUMMARY ACROSS AGE GROUPS")
println("="^60)
println()

# Header
println(rpad("Age group", 20),
        rpad("N", 7),
        rpad("Sk(t1)", 8), rpad("Sk(t2)", 8), rpad("Inv", 6),
        rpad("T_n", 10), rpad("c_α", 10), "Reject")
println("-"^75)

for row in summary_rows
    println(rpad(row.age_group, 20),
            rpad(row.N, 7),
            rpad(row.n_skill_t1, 8), rpad(row.n_skill_t2, 8), rpad(row.n_invest, 6),
            rpad(round(row.T_n; digits=4), 10),
            rpad(round(row.c_α; digits=4), 10),
            row.reject ? "YES" : "no")
end

println("\nDone.")
