using LinearAlgebra, Random, Statistics, JLD2

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "prepare_data.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation.jl"))          # for skill_index, investment_index, get_score
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod.jl"))
include(joinpath(@__DIR__, "..", "src", "inference.jl"))

# ─────────────────────────────────────────────────────────────────
# Parametric configuration: mixture probit transition, ordered probit emissions
# ─────────────────────────────────────────────────────────────────

const PARAMETRIC_CONFIG = (
    emission_θ  = :ordered_probit,
    emission_I  = :ordered_probit,
    transition  = :mixture_probit,
    K           = 4,                   # number of mixture components
)

const R_THETA = 4
const R_I = 5

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
# Load datasets
# ─────────────────────────────────────────────────────────────────

println("Loading datasets...")
datasets = load_all_datasets()

# ─────────────────────────────────────────────────────────────────
# Prepare panel data
# ─────────────────────────────────────────────────────────────────

println("\n" * "="^60)
println("  MULTI-PERIOD MIXTURE PROBIT ESTIMATION")
println("  r_θ=$R_THETA, r_I=$R_I, K=$(PARAMETRIC_CONFIG.K)")
println("="^60)

println("\n  Data Preparation:")
data = prepare_panel_data(datasets;
    age_ranges=PANEL_AGE_RANGES,
    invest_items=PANEL_INVEST_ITEMS,
    r_θ=R_THETA,
    r_I=R_I,
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

println("\n  Parametric EM Estimation:")
println("    emission_θ = $(PARAMETRIC_CONFIG.emission_θ)")
println("    emission_I = $(PARAMETRIC_CONFIG.emission_I)")
println("    transition = $(PARAMETRIC_CONFIG.transition), K = $(PARAMETRIC_CONFIG.K)")

best_params, best_γ, best_ξ_sum, best_ll, best_specs = run_em_restarts_mp(data;
    config=PARAMETRIC_CONFIG)
println("\n  Best log-likelihood: $(round(best_ll; digits=2))")

# Skip normalization (parametric emissions enforce monotonicity)
println("\n  Skipping normalization (parametric emissions enforce monotonicity)")

# Display results
display_results_mp(best_params, data)

# Display mixture probit parameters
if best_specs !== nothing && haskey(best_specs, :transition)
    println("\n" * "="^60)
    println("  MIXTURE PROBIT TRANSITION PARAMETERS")
    println("="^60)

    for (t, spec) in enumerate(best_specs[:transition])
        if spec isa MixtureProbitTransition
            age_from = PANEL_AGE_RANGES[t]
            age_to = PANEL_AGE_RANGES[t + 1]
            println("\n  Transition $t: ages $age_from → $age_to")

            c = _unpack_cutpoints(spec.c_raw)
            a = exp.(spec.a_raw)
            println("    Cutpoints c = $(round.(c; digits=4))")
            println("    Slopes a = $(round.(a; digits=4))")

            K = spec.K
            for kθ in 1:R_THETA
                logits = vcat(0.0, spec.lambda[:, kθ])
                exp_l = exp.(logits .- maximum(logits))
                w = exp_l ./ sum(exp_l)
                println("    θ=$kθ: weights = $(round.(w; digits=4))")
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

println("  Running nonparametric test...")
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

println("\n  Computing parametric Var(Π̂_free) per transition...")
Var_free_list = Vector{Matrix{Float64}}(undef, data.T - 1)
for t in 1:data.T - 1
    spec_t = best_specs[:transition][t]::MixtureProbitTransition
    Σ_score_np = test_results[t].Σ_score  # nonparametric: n_free × n_free

    # Jacobian of φ → Π_free
    φ = pack_mixture_probit(spec_t)
    K = spec_t.K
    function _pi_free_vec(φ_vec)
        Π_full = _build_mixture_probit_pi(φ_vec, data.r_θ, data.r_I, K)
        v = similar(φ_vec, (data.r_θ - 1) * data.r_θ * data.r_I)
        idx = 0
        for kI in 1:data.r_I, kθ in 1:data.r_θ, j in 1:(data.r_θ - 1)
            idx += 1
            v[idx] = Π_full[j, kθ, kI]
        end
        return v
    end
    J = ForwardDiff.jacobian(_pi_free_vec, φ)

    # Parametric score covariance via chain rule: Σ_φ = J' * Σ_score_np * J
    Σ_φ = J' * Σ_score_np * J
    Var_φ = pinv(Σ_φ) / data.N
    Var_free_list[t] = J * Var_φ * J'
end

# ─────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────

outdir = joinpath(dirname(@__DIR__), "output")
mkpath(outdir)
outpath = joinpath(outdir, "estimation_multiperiod_mixture.jld2")
jldsave(outpath; best_params, best_gamma=best_γ, data, best_ll, test_results,
        best_specs, parametric_config=PARAMETRIC_CONFIG, Var_free=Var_free_list)
println("\n  Results saved to $outpath")

println("\nDone. Run notebooks/analyze_results_mixture.ipynb to inspect estimates.")
