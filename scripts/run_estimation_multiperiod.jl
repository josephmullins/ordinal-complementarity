using LinearAlgebra, Random, Statistics, JLD2

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "prepare_data.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation.jl"))          # for skill_index, investment_index, get_score
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod.jl"))
include(joinpath(@__DIR__, "..", "src", "inference.jl"))

# ─────────────────────────────────────────────────────────────────
# Parametric configuration
#
# Set to `nothing` for fully nonparametric estimation (default).
# Each component can be :nonparametric or its parametric alternative:
#   emission_θ / emission_I: :ordered_probit
#   transition: :flexible_lpm, :parsimonious_lpm, or :ordered_probit
# ─────────────────────────────────────────────────────────────────

# const PARAMETRIC_CONFIG = nothing
const PARAMETRIC_CONFIG = (
    emission_θ  = :ordered_probit,
    emission_I  = :ordered_probit,
    transition  = :ordered_probit,
    ccp = :ordered_probit_simple
)

# ─────────────────────────────────────────────────────────────────
# Panel configuration: 7 age points, 6 transitions
# ─────────────────────────────────────────────────────────────────

const PANEL_AGE_RANGES = [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13)]

# Investment items per period (same items as AGE_GROUP_CONFIGS in run_estimation.jl)
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
     :HowOftenMuseum, 
     #:LearnNumbers, :LearnAlphabet, :LearnColors, :LearnShapes,
     :RecordTapeCD, :FoodChoice],
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
    # Period 7: ages 12-13 (same HOME Part D items as period 6)
    [:HowManyBooks, :HowOftenMuseum, :HowOftenTheater, :HowOftenFamily,
     :HowOftenOutdoors, :HowOftenFather, :ChildMusic, :SpecialActivities,
     :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
     :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy,
     :ToldPositive, :HowOftenClean, :HowOftenChores, :HowOftenTime],
]

# ─────────────────────────────────────────────────────────────────
# Load datasets
# ─────────────────────────────────────────────────────────────────

println("Loading datasets...")
datasets = load_all_datasets()

# ─────────────────────────────────────────────────────────────────
# Prepare panel data
# ─────────────────────────────────────────────────────────────────

println("\n" * "="^60)
println("  MULTI-PERIOD ESTIMATION (T=7)")
println("="^60)

println("\n  Data Preparation:")
data = prepare_panel_data(datasets;
    age_ranges=PANEL_AGE_RANGES,
    invest_items=PANEL_INVEST_ITEMS,
    r_θ=4,
    r_I=5,
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

# ─────────────────────────────────────────────────────────────────
# EM estimation with restarts
# ─────────────────────────────────────────────────────────────────

use_parametric = PARAMETRIC_CONFIG !== nothing

if use_parametric
    println("\n  Parametric EM Estimation:")
    println("    emission_θ = $(PARAMETRIC_CONFIG.emission_θ)")
    println("    emission_I = $(PARAMETRIC_CONFIG.emission_I)")
    println("    transition = $(PARAMETRIC_CONFIG.transition)")
else
    println("\n  Nonparametric EM Estimation:")
end

best_params, best_γ, best_ξ_sum, best_ll, best_specs = run_em_restarts_mp(data;
    config=PARAMETRIC_CONFIG)
println("\n  Best log-likelihood: $(round(best_ll; digits=2))")

# ─────────────────────────────────────────────────────────────────
# Normalize ordering
# ─────────────────────────────────────────────────────────────────

if use_parametric
    println("\n  Skipping normalization (parametric emissions enforce monotonicity)")
else
    println("\n  Normalizing latent factor ordering:")
    best_params, best_γ = normalize_params_mp(best_params, best_γ, data)
end

# Display results
display_results_mp(best_params, data)

# Display parametric parameters if applicable
if use_parametric && best_specs !== nothing
    println("\n" * "="^60)
    println("  PARAMETRIC PARAMETER ESTIMATES")
    println("="^60)

    if haskey(best_specs, :transition)
        for (t, spec) in enumerate(best_specs[:transition])
            age_from = PANEL_AGE_RANGES[t]
            age_to = PANEL_AGE_RANGES[t + 1]
            println("\n  Transition $t: ages $age_from → $age_to")
            if spec isa ParsimoniousLPMTransition
                println("    c = $(round.(spec.c; digits=4))")
                println("    γ₁ (I) = $(round(spec.gamma1; digits=4))")
                println("    γ₂ (θ) = $(round(spec.gamma2; digits=4))")
                println("    γ₃ (θ·I, complementarity) = $(round(spec.gamma3; digits=4))")
            elseif spec isa FlexibleLPMTransition
                println("    c = $(round.(spec.c; digits=4))")
                println("    γ₀(θ) = $(round.(spec.gamma0; digits=4))")
                println("    γ₁(θ) = $(round.(spec.gamma1; digits=4))")
            elseif spec isa OrderedProbitTransition
                println("    c (cutpoints) = $(round.(spec.c; digits=4))")
                println("    β₁ (θ) = $(round(spec.beta1; digits=4))")
                println("    β₂ (I) = $(round(spec.beta2; digits=4))")
                println("    β₃ (θ·I) = $(round(spec.beta3; digits=4))")
            end
        end
    end

    if haskey(best_specs, :probit_θ)
        println("\n  Skill emission loadings (α) by period:")
        for t in 1:data.T
            alphas = round.(best_specs[:probit_θ][t].alpha; digits=4)
            println("    Period $t: α = $alphas")
        end
    end

    if haskey(best_specs, :probit_I)
        println("\n  Investment emission loadings (α) by period:")
        for t in 1:data.T
            alphas = round.(best_specs[:probit_I][t].alpha; digits=4)
            println("    Period $t: α = $alphas")
        end
    end
end

# ─────────────────────────────────────────────────────────────────
# Per-period complementarity tests
# ─────────────────────────────────────────────────────────────────

println("\n" * "="^60)
println("  PER-PERIOD COMPLEMENTARITY TESTS")
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

# ─────────────────────────────────────────────────────────────────
# Parametric Var(Π̂_free) per transition (for survivor SEs)
# ─────────────────────────────────────────────────────────────────

Var_free_list = nothing
if use_parametric && best_specs !== nothing && haskey(best_specs, :transition) &&
        PARAMETRIC_CONFIG.transition == :ordered_probit

    println("\n  Computing parametric Var(Π̂_free) per transition...")
    Var_free_list = Vector{Matrix{Float64}}(undef, data.T - 1)
    for t in 1:data.T - 1
        spec_t = best_specs[:transition][t]::OrderedProbitTransition
        Σ_score_np = test_results[t].Σ_score

        # Jacobian of φ → Π_free
        φ = pack_ordered_probit_transition(spec_t)
        function _pi_free_vec(φ_vec)
            Π_full = _build_ordered_probit_pi(φ_vec, data.r_θ, data.r_I)
            v = similar(φ_vec, (data.r_θ - 1) * data.r_θ * data.r_I)
            idx = 0
            for kI in 1:data.r_I, kθ in 1:data.r_θ, j in 1:(data.r_θ - 1)
                idx += 1
                v[idx] = Π_full[j, kθ, kI]
            end
            return v
        end
        J = ForwardDiff.jacobian(_pi_free_vec, φ)

        Σ_φ = J' * Σ_score_np * J
        Var_φ = pinv(Σ_φ) / data.N
        Var_free_list[t] = J * Var_φ * J'
    end
end

# ─────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────

outdir = joinpath(dirname(@__DIR__), "output")
mkpath(outdir)
suffix = use_parametric ? "_parametric" : ""
outpath = joinpath(outdir, "estimation_multiperiod$(suffix).jld2")
if use_parametric && Var_free_list !== nothing
    jldsave(outpath; best_params, best_gamma=best_γ, data, best_ll, test_results,
            best_specs, parametric_config=PARAMETRIC_CONFIG, Var_free=Var_free_list)
elseif use_parametric
    jldsave(outpath; best_params, best_gamma=best_γ, data, best_ll, test_results,
            best_specs, parametric_config=PARAMETRIC_CONFIG)
else
    jldsave(outpath; best_params, best_gamma=best_γ, data, best_ll, test_results)
end
println("\n  Results saved to $outpath")

println("\nDone.")
