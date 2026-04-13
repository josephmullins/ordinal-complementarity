#!/usr/bin/env julia
# ─────────────────────────────────────────────────────────────────
# Baseline Specification Estimation
#
# Two-stage multi-period estimation:
#   Stage 1: Ordered probit emissions + ordered probit Π + ordered probit CCP
#            (multiple restarts)
#   Stage 2: Nonparametric Π, initialized from best Stage 1 estimates
#            (single run with emissions and CCP carried over from Stage 1)
#
# Each transition has its own production function Π_t.
#
# Run for r_I ∈ {3, 5} (set via command-line or SPEC_LIST below).
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

const PANEL_AGE_RANGES = [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13)]

const PANEL_INVEST_ITEMS = [
    # Period 1: ages 0-1 (HOME Part A)
    [:HowOftenRead, :HowManyBooks, :HowManyToysA, :HowManyToysB,
     :OutOfHouse, :HowOftenTalks, :HowOftenGrocery],
    # Period 2: ages 2-3 (HOME Part A/B)
    [:HowOftenRead, :HowManyBooks, :HowManyToysA, :HowManyToysB,
     :OutOfHouse, :HowOftenTalks, :HowOftenGrocery,
     :HowManyZines, :HowOftenOuting, :FoodChoice,
     :LearnNumbers, :LearnAlphabet, :LearnColors, :LearnShapes, :RecordTapeCD],
    # Period 3: ages 4-5 (HOME Part B/C)
    [:HowOftenRead, :HowManyBooks, :HowManyZines, :HowOftenOuting,
     :HowOftenMuseum, :LearnNumbers, :LearnAlphabet, :LearnColors,
     :LearnShapes, :RecordTapeCD, :FoodChoice],
    # Period 4: ages 6-7 (HOME Part C)
    [:HowOftenRead, :HowManyBooks, :HowOftenMuseum, :HowOftenTheater,
     :HowOftenFamily, :HowOftenOutdoors, :ChildMusic, :SpecialActivities,
     :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
     :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy],
    # Period 5: ages 8-9 (HOME Part C/D)
    [:HowOftenRead, :HowManyBooks, :HowOftenMuseum, :HowOftenTheater,
     :HowOftenFamily, :HowOftenOutdoors, :HowOftenFather, :ChildMusic,
     :SpecialActivities, :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
     :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy],
    # Period 6: ages 10-11 (HOME Part D)
    [:HowManyBooks, :HowOftenMuseum, :HowOftenTheater, :HowOftenFamily,
     :HowOftenOutdoors, :HowOftenFather, :ChildMusic, :SpecialActivities,
     :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
     :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy,
     :ToldPositive, :HowOftenClean, :HowOftenChores, :HowOftenTime],
    # Period 7: ages 12-13 (HOME Part D)
    [:HowManyBooks, :HowOftenMuseum, :HowOftenTheater, :HowOftenFamily,
     :HowOftenOutdoors, :HowOftenFather, :ChildMusic, :SpecialActivities,
     :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
     :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy,
     :ToldPositive, :HowOftenClean, :HowOftenChores, :HowOftenTime],
]

# Specifications to run (modify this to run a subset)
const SPEC_LIST = [
    (label="r3", r_θ=3, r_I=3),
    (label="r5", r_θ=3, r_I=5),
]

const N_RESTARTS_STAGE1 = 5
const MAXITER = 2000
const TOL = 1e-5

# Production function groupings: transitions within each group share a single Π.
# Stage A: transitions 1-2 (ages 1-2→3-4, 3-4→5-6)
# Stage B: transitions 3-6 (ages 5-6→7-8, 7-8→9-10, 9-10→11-12, 11-12→13-14)
#const TRANSITION_GROUPS = Vector{Int}[[1,2], [3,4,5,6]]
const TRANSITION_GROUPS = nothing
# ─────────────────────────────────────────────────────────────────
# Stage 1: Ordered probit emissions + parametric (ordered probit) Π
# ─────────────────────────────────────────────────────────────────

const CONFIG_STAGE1 = (
    emission_θ = :ordered_probit,
    emission_I = :ordered_probit,
    transition = :ordered_probit,
    ccp = :ordered_probit_simple,          # :ordered_probit, :ordered_probit_simple, or :nonparametric
    initial = :ordered_probit,       # :ordered_probit (factored π0 = p(θ)·P(I|θ)) or :nonparametric
    monotone_I = true,               # constrain β₂ > 0 so Π is monotonically increasing in I
)

function run_stage1(data; n_restarts=N_RESTARTS_STAGE1, maxiter=MAXITER, tol=TOL,
                    transition_groups=TRANSITION_GROUPS)
    best_ll = -Inf
    best_params = nothing
    best_γ = nothing
    best_ξ_sum = nothing
    best_specs = nothing

    for restart in 1:n_restarts
        println("\n  Stage 1 — Restart $restart / $n_restarts:")
        rng = MersenneTwister(42 + restart)

        p_init, specs_init = initialize_params_mp_parametric(data, CONFIG_STAGE1; rng)
        params_est, γ, ξ_sum, specs = expectation_maximization_mp_parametric(
            data, p_init, specs_init, CONFIG_STAGE1;
            verbose=true, maxiter=maxiter, tol=tol,
            transition_groups=transition_groups, accelerate=false)

        ll = expectation_step_mp!(γ, ξ_sum, params_est, data)
        println("  Final log-likelihood: $(round(ll; digits=2))")

        if ll > best_ll
            best_ll = ll
            best_params = params_est
            best_γ = copy(γ)
            best_ξ_sum = [copy(x) for x in ξ_sum]
            best_specs = specs
        end
    end

    return best_params, best_γ, best_ξ_sum, best_ll, best_specs
end

# ─────────────────────────────────────────────────────────────────
# Stage 2: Nonparametric Π from Stage 1 posteriors
#
# Fix emission parameters from Stage 1. Run EM with nonparametric
# transitions, initialized from the Stage 1 parametric Π.
# ─────────────────────────────────────────────────────────────────

const CONFIG_STAGE2 = (
    emission_θ = :ordered_probit,
    emission_I = :ordered_probit,
    transition = :nonparametric,
    ccp = :ordered_probit_simple,          # carried over from Stage 1
    initial = :ordered_probit,       # carried over from Stage 1
)

function run_stage2(data, stage1_params, stage1_specs;
                    maxiter=MAXITER, tol=TOL,
                    transition_groups=TRANSITION_GROUPS)
    println("\n  Stage 2 — Nonparametric transitions (from Stage 1 parametric):")

    # Build specs dict carrying over Stage 1 emission and CCP specs (no transition specs for NP)
    specs = Dict{Symbol, Any}()
    specs[:probit_θ] = stage1_specs[:probit_θ]
    specs[:probit_I] = stage1_specs[:probit_I]
    if haskey(stage1_specs, :ccp)
        specs[:ccp] = stage1_specs[:ccp]
    end
    if haskey(stage1_specs, :initial_I)
        specs[:initial_I] = stage1_specs[:initial_I]
    end

    # Build initial params: use Stage 1 emissions + CCPs + Stage 1 Π as starting point
    params_init = stage1_params

    # Run EM with nonparametric transitions
    params_est, γ, ξ_sum, specs_est = expectation_maximization_mp_parametric(
        data, params_init, specs, CONFIG_STAGE2;
        verbose=true, maxiter=maxiter, tol=tol,
        transition_groups=transition_groups, accelerate=false)

    ll = expectation_step_mp!(γ, ξ_sum, params_est, data)
    println("  Stage 2 final log-likelihood: $(round(ll; digits=2))")

    return params_est, γ, ξ_sum, ll, specs_est
end

# ─────────────────────────────────────────────────────────────────
# Complementarity tests (both NP and parametric variance)
# ─────────────────────────────────────────────────────────────────

function run_tests(params, data, specs, config_label)
    (; r_θ, r_I, T, N) = data
    println("\n  Computing pairwise posteriors ($config_label)...")
    ψ_all = compute_psi_all(params, data)

    # Nonparametric test (per transition)
    println("  Running nonparametric test (per transition)...")
    test_np = get_test_stat_mp(ψ_all, params, data; α=0.05)

    return test_np, ψ_all
end

function run_parametric_tests(params, data, specs, ψ_all)
    (; r_θ, r_I, T, N) = data
    config = CONFIG_STAGE1
    n_transitions = T - 1

    # Compute parametric Var(Π̂_free) via delta method per transition
    Var_free_list = Vector{Matrix{Float64}}(undef, n_transitions)
    for t in 1:n_transitions
        Q_t = ψ_all[t, :, :, :, :]
        Π_t = copy(params.Π[t])
        Π_t .= max.(Π_t, 1e-15)
        for kI in 1:r_I, kθ in 1:r_θ
            Π_t[:, kθ, kI] ./= sum(Π_t[:, kθ, kI])
        end

        trans_spec = specs[:transition][t]
        specs_t = Dict{Symbol,Any}(:transition => trans_spec)
        config_t = (transition = config.transition,)
        dL_param = get_score_parametric(Q_t, Π_t, specs_t, config_t)
        Σ_score_param = cov(dL_param')
        Var_free_list[t] = pi_free_var_parametric(trans_spec, config.transition,
                                                   Σ_score_param, r_θ, r_I, N)
    end

    # Parametric test (per transition)
    println("  Running parametric test (per transition)...")
    test_param = get_test_stat_mp_parametric(Var_free_list, params, data; α=0.05)

    return test_param, Var_free_list
end

# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

println("Loading datasets...")
datasets = load_all_datasets()

outdir = joinpath(dirname(@__DIR__), "output")
mkpath(outdir)

for spec in SPEC_LIST
    println("\n" * "="^70)
    println("  BASELINE ESTIMATION: r_θ=$(spec.r_θ), r_I=$(spec.r_I) ($(spec.label))")
    println("="^70)

    # ── Print transition groupings ──
    if TRANSITION_GROUPS !== nothing
        println("\n  Production function groups (shared Π within group):")
        for (g, group) in enumerate(TRANSITION_GROUPS)
            ages_str = join(["$(PANEL_AGE_RANGES[t])→$(PANEL_AGE_RANGES[t+1])" for t in group], ", ")
            println("    Group $g: transitions $group ($ages_str)")
        end
    else
        println("\n  No transition grouping — each transition has its own Π")
    end

    # ── Data preparation ──
    println("\n  Data Preparation:")
    data = prepare_panel_data(datasets;
        age_ranges=PANEL_AGE_RANGES,
        invest_items=PANEL_INVEST_ITEMS,
        r_θ=spec.r_θ,
        r_I=spec.r_I,
    )

    println("  N = $(data.N) children, T = $(data.T) periods")
    for t in 1:data.T
        n_skill = length(data.X_θ[t])
        n_inv = length(data.X_I[t])
        println("    Period $t (ages $(PANEL_AGE_RANGES[t])): $n_skill skill measures, $n_inv investment measures")
    end

    if data.N < 50
        error("Only $(data.N) observations — too few for estimation.")
    end

    # ── Stage 1: Parametric Π (ordered probit) ──
    println("\n" * "-"^50)
    println("  STAGE 1: Ordered probit emissions + ordered probit Π")
    println("-"^50)

    s1_params, s1_γ, s1_ξ_sum, s1_ll, s1_specs = run_stage1(data)
    println("\n  Stage 1 best log-likelihood: $(round(s1_ll; digits=2))")

    display_results_mp(s1_params, data)

    # Print parametric transition parameters (one per transition)
    println("\n  Ordered Probit Transition Parameters (Π):")
    for t in 1:data.T-1
        ages_str = "$(PANEL_AGE_RANGES[t])→$(PANEL_AGE_RANGES[t+1])"
        println("    Transition $t ($ages_str):")
        trans_spec = s1_specs[:transition][t]
        println("      c (cutpoints) = $(round.(trans_spec.c; digits=4))")
        println("      β₁ (θ)   = $(round(trans_spec.beta1; digits=4))")
        println("      β₂ (I)   = $(round(trans_spec.beta2; digits=4))")
        println("      β₃ (θ·I) = $(round(trans_spec.beta3; digits=4))")
    end

    # Print parametric CCP parameters (one per transition)
    if haskey(s1_specs, :ccp)
        ccp_label = s1_specs[:ccp][1] isa OrderedProbitCCPSimple ?
            "Simple Ordered Probit CCP Parameters (I'|θ')" :
            "Ordered Probit CCP Parameters (I'|θ',I)"
        println("\n  $ccp_label:")
        for t in 1:data.T-1
            ages_str = "$(PANEL_AGE_RANGES[t])→$(PANEL_AGE_RANGES[t+1])"
            println("    Transition $t ($ages_str):")
            ccp_spec = s1_specs[:ccp][t]
            println("      c (cutpoints) = $(round.(ccp_spec.c; digits=4))")
            println("      β₁ (θ')  = $(round(ccp_spec.beta1; digits=4))")
            if ccp_spec isa OrderedProbitCCP
                println("      β₂ (I)   = $(round(ccp_spec.beta2; digits=4))")
                println("      β₃ (θ'·I)= $(round(ccp_spec.beta3; digits=4))")
            end
        end
    end

    # Print initial I parameters if parametric
    if haskey(s1_specs, :initial_I)
        println("\n  Ordered Probit Initial I Parameters (I|θ):")
        init_spec = s1_specs[:initial_I]
        println("    c (cutpoints) = $(round.(init_spec.c; digits=4))")
        println("    β₁ (θ)  = $(round(init_spec.beta1; digits=4))")
    end

    # Stage 1 tests (parametric)
    s1_test_np, s1_ψ_all = run_tests(s1_params, data, s1_specs, "Stage 1 Parametric")
    s1_test_param, s1_Var_free = run_parametric_tests(s1_params, data, s1_specs, s1_ψ_all)

    println("\n  Stage 1 Complementarity Tests:")
    println("  " * rpad("Transition", 30) * rpad("NP T_n", 12) * rpad("NP c_α", 12) *
            rpad("Param T_n", 12) * rpad("Param c_α", 12) * "Reject (param)")
    println("  " * "-"^80)
    for t in eachindex(s1_test_np)
        np = s1_test_np[t]
        p = s1_test_param[t]
        ages_str = "$(PANEL_AGE_RANGES[t])→$(PANEL_AGE_RANGES[t+1])"
        status = p.reject ? "YES" : "no"
        println("  " * rpad("$t ($ages_str)", 30) *
                rpad(round(np.T_n; digits=2), 12) *
                rpad(round(np.c_α; digits=2), 12) *
                rpad(round(p.T_n; digits=2), 12) *
                rpad(round(p.c_α; digits=2), 12) *
                status)
    end

    # Save Stage 1
    s1_path = joinpath(outdir, "baseline_stage1_$(spec.label).jld2")
    jldsave(s1_path; params=s1_params, gamma=s1_γ, data, ll=s1_ll,
            test_np=s1_test_np, test_param=s1_test_param,
            Var_free=s1_Var_free,
            specs=s1_specs, config=CONFIG_STAGE1,
            r_theta=spec.r_θ, r_I=spec.r_I,
            transition_groups=TRANSITION_GROUPS)
    println("\n  Stage 1 saved to $s1_path")

    # ── Stage 2: Nonparametric Π (from Stage 1) ──
    println("\n" * "-"^50)
    println("  STAGE 2: Nonparametric transitions (from Stage 1)")
    println("-"^50)

    s2_params, s2_γ, s2_ξ_sum, s2_ll, s2_specs = run_stage2(data, s1_params, s1_specs)
    println("\n  Stage 2 log-likelihood: $(round(s2_ll; digits=2))")

    display_results_mp(s2_params, data)

    # Stage 2 tests (NP only — no parametric transition specs)
    s2_test_np, s2_ψ_all = run_tests(s2_params, data, s2_specs, "Stage 2 NP")

    println("\n  Stage 2 Complementarity Test (nonparametric Π):")
    for res in s2_test_np
        t = res.t
        ages_str = "$(PANEL_AGE_RANGES[t])→$(PANEL_AGE_RANGES[t+1])"
        status = res.reject ? "REJECT" : "fail to reject"
        println("    Transition $t ($ages_str): T_n=$(round(res.T_n; digits=4)), c_α=$(round(res.c_α; digits=4)) → $status")
    end

    # Save Stage 2
    s2_path = joinpath(outdir, "baseline_stage2_$(spec.label).jld2")
    jldsave(s2_path; params=s2_params, gamma=s2_γ, data, ll=s2_ll,
            test_np=s2_test_np,
            specs=s2_specs, config=CONFIG_STAGE2,
            r_theta=spec.r_θ, r_I=spec.r_I,
            stage1_ll=s1_ll, stage1_test_np=s1_test_np,
            stage1_test_param=s1_test_param,
            transition_groups=TRANSITION_GROUPS)
    println("\n  Stage 2 saved to $s2_path")
end

println("\n" * "="^70)
println("  ALL DONE")
println("="^70)
