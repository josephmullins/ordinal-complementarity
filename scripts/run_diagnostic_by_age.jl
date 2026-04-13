#!/usr/bin/env julia
# ─────────────────────────────────────────────────────────────────
# Diagnostic: All-Parametric Two-Period Estimation by Single Age
#
# Runs a fully parametric (ordered probit everything) T=2 model
# for each single-age transition: age a → age a+2, for a ∈ 0..11.
# Each transition has its own production function Π.
#
# r_θ = 3, r_I = 3, fully parametric (ordered probit emissions,
# ordered probit transition, ordered probit CCP, ordered probit
# initial I distribution).
#
# Purpose: diagnose which age transitions are problematic.
# ─────────────────────────────────────────────────────────────────

using LinearAlgebra, Random, Statistics, JLD2

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "prepare_data.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod.jl"))
include(joinpath(@__DIR__, "..", "src", "inference.jl"))

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────

const R_THETA = 3
const R_I = 3
const N_RESTARTS = 3
const MAXITER = 2000
const TOL = 1e-5

const CONFIG = (
    emission_θ = :ordered_probit,
    emission_I = :ordered_probit,
    transition = :ordered_probit,
    ccp = :ordered_probit,
    initial = :ordered_probit,
)

# HOME investment items by Part (determined by child age)
const INVEST_PART_A = [  # ages 0–2
    :HowOftenRead, :HowManyBooks, :HowManyToysA, :HowManyToysB,
    :OutOfHouse, :HowOftenTalks, :HowOftenGrocery,
]

const INVEST_PART_B = [  # ages 3–5
    :HowOftenRead, :HowManyBooks, :HowManyZines, :HowOftenOuting,
    :FoodChoice, :LearnNumbers, :LearnAlphabet, :LearnColors,
    :LearnShapes, :RecordTapeCD,
]

const INVEST_PART_C = [  # ages 6–9
    :HowOftenRead, :HowManyBooks, :HowOftenMuseum, :HowOftenTheater,
    :HowOftenFamily, :HowOftenOutdoors, :ChildMusic, :SpecialActivities,
    :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
    :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy,
]

const INVEST_PART_D = [  # ages 10–14
    :HowManyBooks, :HowOftenMuseum, :HowOftenTheater, :HowOftenFamily,
    :HowOftenOutdoors, :HowOftenFather, :ChildMusic, :SpecialActivities,
    :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
    :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy,
    :ToldPositive, :HowOftenClean, :HowOftenChores, :HowOftenTime,
]

"""Return HOME investment items appropriate for a given age."""
function invest_items_for_age(age::Int)
    if age <= 2
        return INVEST_PART_A
    elseif age <= 5
        return INVEST_PART_B
    elseif age <= 9
        return INVEST_PART_C
    else
        return INVEST_PART_D
    end
end

# ─────────────────────────────────────────────────────────────────
# Single transition estimation
# ─────────────────────────────────────────────────────────────────

function run_single_transition(datasets, age_t1::Int, age_t2::Int;
                               n_restarts=N_RESTARTS, maxiter=MAXITER, tol=TOL)
    age_ranges = [(age_t1, age_t1), (age_t2, age_t2)]
    invest_items = [invest_items_for_age(age_t1), invest_items_for_age(age_t2)]

    data = prepare_panel_data(datasets;
        age_ranges=age_ranges,
        invest_items=invest_items,
        r_θ=R_THETA,
        r_I=R_I,
    )

    println("  N = $(data.N) children")
    for t in 1:data.T
        n_skill = length(data.X_θ[t])
        n_inv = length(data.X_I[t])
        println("    Period $t (age $(age_ranges[t])): $n_skill skill measures, $n_inv investment measures")
    end

    if data.N < 50
        println("  WARNING: Only $(data.N) observations — skipping.")
        return nothing
    end

    best_ll = -Inf
    best_params = nothing
    best_γ = nothing
    best_ξ_sum = nothing
    best_specs = nothing

    for restart in 1:n_restarts
        rng = MersenneTwister(42 + restart)

        p_init, specs_init = initialize_params_mp_parametric(data, CONFIG; rng)
        params_est, γ, ξ_sum, specs = expectation_maximization_mp_parametric(
            data, p_init, specs_init, CONFIG;
            verbose=false, maxiter=maxiter, tol=tol)

        ll = expectation_step_mp!(γ, ξ_sum, params_est, data)

        if ll > best_ll
            best_ll = ll
            best_params = params_est
            best_γ = copy(γ)
            best_ξ_sum = [copy(x) for x in ξ_sum]
            best_specs = specs
        end
    end

    println("  Best log-likelihood: $(round(best_ll; digits=2))")

    # Print transition parameters
    trans_spec = best_specs[:transition][1]
    println("  Transition parameters:")
    println("    c (cutpoints) = $(round.(trans_spec.c; digits=4))")
    println("    β₁ (θ)   = $(round(trans_spec.beta1; digits=4))")
    println("    β₂ (I)   = $(round(trans_spec.beta2; digits=4))")
    println("    β₃ (θ·I) = $(round(trans_spec.beta3; digits=4))")

    # Print CCP parameters
    ccp_spec = best_specs[:ccp][1]
    println("  CCP parameters:")
    println("    c (cutpoints) = $(round.(ccp_spec.c; digits=4))")
    println("    β₁ (θ')  = $(round(ccp_spec.beta1; digits=4))")
    println("    β₂ (I)   = $(round(ccp_spec.beta2; digits=4))")
    println("    β₃ (θ'·I)= $(round(ccp_spec.beta3; digits=4))")

    # Print initial I parameters
    if haskey(best_specs, :initial_I)
        init_spec = best_specs[:initial_I]
        println("  Initial I parameters:")
        println("    c (cutpoints) = $(round.(init_spec.c; digits=4))")
        println("    β₁ (θ)  = $(round(init_spec.beta1; digits=4))")
    end

    # Print Π
    Π = best_params.Π[1]
    println("  Production function Π[θ'|θ,I]:")
    for kI in 1:R_I
        println("    I=$kI:")
        for kθ in 1:R_THETA
            probs = round.(Π[:, kθ, kI]; digits=4)
            println("      θ=$kθ → $probs")
        end
    end

    # Complementarity test
    println("  Computing test statistic...")
    ψ_all = compute_psi_all(best_params, data)
    test_np = get_test_stat_mp(ψ_all, best_params, data; α=0.05)
    res = test_np[1]
    println("  NP Test: T_n=$(round(res.T_n; digits=4)), c_α=$(round(res.c_α; digits=4)) → $(res.reject ? "REJECT" : "fail to reject")")

    return (data=data, params=best_params, γ=best_γ, ξ_sum=best_ξ_sum,
            ll=best_ll, specs=best_specs, test_np=test_np)
end

# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

println("Loading datasets...")
datasets = load_all_datasets()

outdir = joinpath(dirname(@__DIR__), "output")
mkpath(outdir)

# All single-age transitions: age a → age a+2
transitions = [(a, a+2) for a in 0:11]

all_results = Dict{Tuple{Int,Int}, Any}()

for (age_t1, age_t2) in transitions
    println("\n" * "="^60)
    println("  TRANSITION: age $age_t1 → age $age_t2")
    println("="^60)

    result = run_single_transition(datasets, age_t1, age_t2)
    all_results[(age_t1, age_t2)] = result
end

# ─────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────

println("\n" * "="^80)
println("  SUMMARY")
println("="^80)
println()
println(rpad("Transition", 15) *
        rpad("N", 8) *
        rpad("LL", 12) *
        rpad("β₁(θ)", 10) *
        rpad("β₂(I)", 10) *
        rpad("β₃(θ·I)", 10) *
        rpad("T_n", 10) *
        rpad("c_α", 10) *
        "Reject")
println("-"^95)

for (age_t1, age_t2) in transitions
    result = all_results[(age_t1, age_t2)]
    label = "$age_t1 → $age_t2"

    if result === nothing
        println(rpad(label, 15) * "SKIPPED (too few observations)")
        continue
    end

    trans_spec = result.specs[:transition][1]
    test = result.test_np[1]

    println(rpad(label, 15) *
            rpad(result.data.N, 8) *
            rpad(round(result.ll; digits=1), 12) *
            rpad(round(trans_spec.beta1; digits=4), 10) *
            rpad(round(trans_spec.beta2; digits=4), 10) *
            rpad(round(trans_spec.beta3; digits=4), 10) *
            rpad(round(test.T_n; digits=3), 10) *
            rpad(round(test.c_α; digits=3), 10) *
            (test.reject ? "YES" : "no"))
end

# Save all results
save_path = joinpath(outdir, "diagnostic_by_age.jld2")
jldsave(save_path;
    results=all_results,
    transitions=transitions,
    config=CONFIG,
    r_theta=R_THETA, r_I=R_I)
println("\n  Results saved to $save_path")
println("Done.")
