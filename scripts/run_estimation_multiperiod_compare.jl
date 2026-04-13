using LinearAlgebra, Random, Statistics, JLD2

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "prepare_data.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod.jl"))
include(joinpath(@__DIR__, "..", "src", "inference.jl"))

# ─────────────────────────────────────────────────────────────────
# Parametric configuration: ordered probit emissions + transitions
# ─────────────────────────────────────────────────────────────────

const PARAMETRIC_CONFIG = (
    emission_θ  = :ordered_probit,
    emission_I  = :ordered_probit,
    transition  = :ordered_probit,
)

# ─────────────────────────────────────────────────────────────────
# Panel configuration: 7 age points, 6 transitions
# ─────────────────────────────────────────────────────────────────

const PANEL_AGE_RANGES = [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13)]

const PANEL_INVEST_ITEMS = [
    # Period 1: ages 0-1
    [:HowOftenRead, :HowManyBooks, :HowManyToysA, :HowManyToysB,
     :OutOfHouse, :HowOftenTalks, :HowOftenGrocery],
    # Period 2: ages 2-3
    [:HowOftenRead, :HowManyBooks, :HowManyToysA, :HowManyToysB,
     :OutOfHouse, :HowOftenTalks, :HowOftenGrocery,
     :HowManyZines, :HowOftenOuting, :FoodChoice,
     :LearnNumbers, :LearnAlphabet, :LearnColors, :LearnShapes, :RecordTapeCD],
    # Period 3: ages 4-5
    [:HowOftenRead, :HowManyBooks, :HowManyZines, :HowOftenOuting,
     :HowOftenMuseum, :LearnNumbers, :LearnAlphabet, :LearnColors,
     :LearnShapes, :RecordTapeCD, :FoodChoice],
    # Period 4: ages 6-7
    [:HowOftenRead, :HowManyBooks, :HowOftenMuseum, :HowOftenTheater,
     :HowOftenFamily, :HowOftenOutdoors, :ChildMusic, :SpecialActivities,
     :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
     :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy],
    # Period 5: ages 8-9
    [:HowOftenRead, :HowManyBooks, :HowOftenMuseum, :HowOftenTheater,
     :HowOftenFamily, :HowOftenOutdoors, :HowOftenFather, :ChildMusic,
     :SpecialActivities, :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
     :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy],
    # Period 6: ages 10-11
    [:HowManyBooks, :HowOftenMuseum, :HowOftenTheater, :HowOftenFamily,
     :HowOftenOutdoors, :HowOftenFather, :ChildMusic, :SpecialActivities,
     :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
     :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy,
     :ToldPositive, :HowOftenClean, :HowOftenChores, :HowOftenTime],
    # Period 7: ages 12-13
    [:HowManyBooks, :HowOftenMuseum, :HowOftenTheater, :HowOftenFamily,
     :HowOftenOutdoors, :HowOftenFather, :ChildMusic, :SpecialActivities,
     :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
     :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy,
     :ToldPositive, :HowOftenClean, :HowOftenChores, :HowOftenTime],
]

# ─────────────────────────────────────────────────────────────────
# Two specifications to compare
# ─────────────────────────────────────────────────────────────────

const SPECS = [
    (label="r3", r_θ=3, r_I=3),
    (label="r5", r_θ=5, r_I=5),
]

# ─────────────────────────────────────────────────────────────────
# Load datasets
# ─────────────────────────────────────────────────────────────────

println("Loading datasets...")
datasets = load_all_datasets()

outdir = joinpath(dirname(@__DIR__), "output")
mkpath(outdir)

for spec in SPECS
    println("\n" * "="^60)
    println("  MULTI-PERIOD ESTIMATION: r_θ=$(spec.r_θ), r_I=$(spec.r_I)")
    println("="^60)

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

    # ── EM estimation ──
    println("\n  Parametric EM Estimation:")
    println("    emission_θ = $(PARAMETRIC_CONFIG.emission_θ)")
    println("    emission_I = $(PARAMETRIC_CONFIG.emission_I)")
    println("    transition = $(PARAMETRIC_CONFIG.transition)")
    best_params, best_γ, best_ξ_sum, best_ll, best_specs = run_em_restarts_mp(data;
        config=PARAMETRIC_CONFIG)
    println("\n  Best log-likelihood: $(round(best_ll; digits=2))")

    println("\n  Skipping normalization (parametric emissions enforce monotonicity)")

    display_results_mp(best_params, data)

    # ── Parametric parameter estimates ──
    if best_specs !== nothing
        println("\n" * "="^60)
        println("  PARAMETRIC PARAMETER ESTIMATES ($(spec.label))")
        println("="^60)

        if haskey(best_specs, :transition)
            for (t, trans_spec) in enumerate(best_specs[:transition])
                age_from = PANEL_AGE_RANGES[t]
                age_to = PANEL_AGE_RANGES[t + 1]
                println("\n  Transition $t: ages $age_from → $age_to")
                if trans_spec isa OrderedProbitTransition
                    println("    c (cutpoints) = $(round.(trans_spec.c; digits=4))")
                    println("    β₁ (θ) = $(round(trans_spec.beta1; digits=4))")
                    println("    β₂ (I) = $(round(trans_spec.beta2; digits=4))")
                    println("    β₃ (θ·I) = $(round(trans_spec.beta3; digits=4))")
                elseif trans_spec isa ParsimoniousLPMTransition
                    println("    c = $(round.(trans_spec.c; digits=4))")
                    println("    γ₁ (I) = $(round(trans_spec.gamma1; digits=4))")
                    println("    γ₂ (θ) = $(round(trans_spec.gamma2; digits=4))")
                    println("    γ₃ (θ·I) = $(round(trans_spec.gamma3; digits=4))")
                elseif trans_spec isa FlexibleLPMTransition
                    println("    c = $(round.(trans_spec.c; digits=4))")
                    println("    γ₀(θ) = $(round.(trans_spec.gamma0; digits=4))")
                    println("    γ₁(θ) = $(round.(trans_spec.gamma1; digits=4))")
                end
            end
        end
    end

    # ── Complementarity tests ──
    println("\n" * "="^60)
    println("  PER-PERIOD COMPLEMENTARITY TESTS ($(spec.label))")
    println("="^60)

    println("\n  Computing per-observation pairwise posteriors...")
    ψ_all = compute_psi_all(best_params, data)

    println("  Running tests...")
    test_results = get_test_stat_mp(ψ_all, best_params, data; α=0.05)

    for res in test_results
        age_from = PANEL_AGE_RANGES[res.t]
        age_to = PANEL_AGE_RANGES[res.t + 1]
        status = res.reject ? "REJECT" : "FAIL TO REJECT"
        println("\n  Transition $(res.t): ages $age_from → $age_to")
        println("    T_n = $(round(res.T_n; digits=4)),  c_α = $(round(res.c_α; digits=4))")
        println("    Result: $status ordinal complementarity (H₀)")
    end

    # ── Save results ──
    outpath = joinpath(outdir, "estimation_multiperiod_compare_$(spec.label).jld2")
    jldsave(outpath; best_params, best_gamma=best_γ, data, best_ll, test_results,
            best_specs, parametric_config=PARAMETRIC_CONFIG,
            r_theta=spec.r_θ, r_I=spec.r_I)
    println("\n  Results saved to $outpath")
end

println("\nDone.")
