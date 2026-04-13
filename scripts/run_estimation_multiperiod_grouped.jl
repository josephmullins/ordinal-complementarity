using LinearAlgebra, Random, Statistics, JLD2

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "prepare_data.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation.jl"))          # for skill_index, investment_index, get_score
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod.jl"))
include(joinpath(@__DIR__, "..", "src", "inference.jl"))

# ─────────────────────────────────────────────────────────────────
# Parametric configuration
# ─────────────────────────────────────────────────────────────────

const PARAMETRIC_CONFIG = (
    emission_θ  = :ordered_probit,
    emission_I  = :ordered_probit,
    transition  = :ordered_probit,
    ccp = :ordered_probit_simple
)

# ─────────────────────────────────────────────────────────────────
# Transition grouping: shared Π within each group
#
#   Group 1 (early childhood): transitions 1,2,3
#     (0,1)→(2,3), (2,3)→(4,5), (4,5)→(6,7)
#   Group 2: transition 4  — (6,7)→(8,9)
#   Group 3: transition 5  — (8,9)→(10,11)
#   Group 4: transition 6  — (10,11)→(12,13)
# ─────────────────────────────────────────────────────────────────

const TRANSITION_GROUPS = [[1, 2, 3], [4, 5, 6]]

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
    # Period 7: ages 12-13
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
println("  MULTI-PERIOD ESTIMATION (T=7, GROUPED TRANSITIONS)")
println("="^60)

println("\n  Transition groups:")
for (g, group) in enumerate(TRANSITION_GROUPS)
    labels = [string(PANEL_AGE_RANGES[t], "→", PANEL_AGE_RANGES[t+1]) for t in group]
    println("    Group $g: $(join(labels, ", "))")
end

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
# EM estimation with restarts (grouped transitions)
# ─────────────────────────────────────────────────────────────────

println("\n  Parametric EM Estimation:")
println("    emission_θ = $(PARAMETRIC_CONFIG.emission_θ)")
println("    emission_I = $(PARAMETRIC_CONFIG.emission_I)")
println("    transition = $(PARAMETRIC_CONFIG.transition)")
println("    transition_groups = $TRANSITION_GROUPS")

best_params, best_γ, best_ξ_sum, best_ll, best_specs = run_em_restarts_mp(data;
    config=PARAMETRIC_CONFIG,
    transition_groups=TRANSITION_GROUPS)
println("\n  Best log-likelihood: $(round(best_ll; digits=2))")

# ─────────────────────────────────────────────────────────────────
# Normalize ordering
# ─────────────────────────────────────────────────────────────────

println("\n  Skipping normalization (parametric emissions enforce monotonicity)")

# Display results
display_results_mp(best_params, data)

# Display parametric parameters
if best_specs !== nothing
    println("\n" * "="^60)
    println("  PARAMETRIC PARAMETER ESTIMATES")
    println("="^60)

    if haskey(best_specs, :transition)
        for (t, spec) in enumerate(best_specs[:transition])
            age_from = PANEL_AGE_RANGES[t]
            age_to = PANEL_AGE_RANGES[t + 1]
            # Find which group this transition belongs to
            g = findfirst(grp -> t ∈ grp, TRANSITION_GROUPS)
            shared = length(TRANSITION_GROUPS[g]) > 1 ? " (shared with group $g)" : ""
            println("\n  Transition $t: ages $age_from → $age_to$shared")
            if spec isa OrderedProbitTransition
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
# Per-group complementarity tests
# ─────────────────────────────────────────────────────────────────

println("\n" * "="^60)
println("  GROUPED COMPLEMENTARITY TESTS")
println("="^60)

println("\n  Computing per-observation pairwise posteriors...")
ψ_all = compute_psi_all(best_params, data)

println("  Running grouped tests...")
test_results = get_test_stat_mp_grouped(ψ_all, best_params, data, TRANSITION_GROUPS; α=0.05)

for (g, res) in enumerate(test_results)
    labels = [string(PANEL_AGE_RANGES[t], "→", PANEL_AGE_RANGES[t+1]) for t in res.group]
    status = res.reject ? "REJECT" : "FAIL TO REJECT"
    println("\n  Group $g: $(join(labels, ", "))")
    println("    T_n = $(round(res.T_n; digits=4)),  c_α = $(round(res.c_α; digits=4))")
    println("    Result: $status ordinal complementarity (H₀)")
end

# ─────────────────────────────────────────────────────────────────
# Parametric Var(Π̂_free) per group (for survivor SEs)
# ─────────────────────────────────────────────────────────────────

Var_free_list = nothing
if best_specs !== nothing && haskey(best_specs, :transition) &&
        PARAMETRIC_CONFIG.transition == :ordered_probit

    println("\n  Computing parametric Var(Π̂_free) per group...")
    Var_free_list = Vector{Matrix{Float64}}(undef, data.T - 1)
    for (g, group) in enumerate(TRANSITION_GROUPS)
        t_rep = group[1]
        spec_g = best_specs[:transition][t_rep]::OrderedProbitTransition

        # Pool nonparametric score covariance across transitions in group
        Σ_score_pooled = sum(test_results[g].Σ_score for _ in 1:1)  # already pooled in grouped test

        # Jacobian of φ → Π_free
        φ = pack_ordered_probit_transition(spec_g)
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

        Σ_φ = J' * test_results[g].Σ_score * J
        Var_φ = pinv(Σ_φ) / data.N
        Var_free_g = J * Var_φ * J'

        # Assign same Var_free to all transitions in the group
        for t in group
            Var_free_list[t] = Var_free_g
        end
    end
end

# ─────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────

outdir = joinpath(dirname(@__DIR__), "output")
mkpath(outdir)
outpath = joinpath(outdir, "estimation_multiperiod_grouped.jld2")
if Var_free_list !== nothing
    jldsave(outpath; best_params, best_gamma=best_γ, data, best_ll,
            test_results, best_specs, parametric_config=PARAMETRIC_CONFIG,
            transition_groups=TRANSITION_GROUPS, Var_free=Var_free_list)
else
    jldsave(outpath; best_params, best_gamma=best_γ, data, best_ll,
            test_results, best_specs, parametric_config=PARAMETRIC_CONFIG,
            transition_groups=TRANSITION_GROUPS)
end
println("\n  Results saved to $outpath")

println("\nDone.")
