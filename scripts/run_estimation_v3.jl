using LinearAlgebra, Random, Statistics, JLD2

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3.jl"))
include(joinpath(@__DIR__, "..", "src", "prepare_data.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod_v3.jl"))

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────

const R_P  = 3     # θ^P cardinality
const R_ω  = 3     # ω cardinality
const K_NN = 4     # robit NN hidden units
const NU   = 7.0   # t-CDF degrees of freedom

const V3_CONFIG = (K = K_NN, ν = NU, n_groups = 2)
const TRANSITION_GROUPS = [[1, 2, 3], [4, 5, 6]]

const PANEL_AGE_RANGES = [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13)]

const PANEL_INVEST_ITEMS = [
    [:HowOftenRead, :HowManyBooks, :HowManyToysA, :HowManyToysB,
     :OutOfHouse, :HowOftenTalks, :HowOftenGrocery],
    [:HowOftenRead, :HowManyBooks, :HowManyToysA, :HowManyToysB,
     :OutOfHouse, :HowOftenTalks, :HowOftenGrocery,
     :HowManyZines, :HowOftenOuting, :FoodChoice,
     :LearnNumbers, :LearnAlphabet, :LearnColors, :LearnShapes, :RecordTapeCD],
    [:HowOftenRead, :HowManyBooks, :HowManyZines, :HowOftenOuting,
     :HowOftenMuseum, :RecordTapeCD, :FoodChoice],
    [:HowOftenRead, :HowManyBooks, :HowOftenMuseum, :HowOftenTheater,
     :HowOftenFamily, :HowOftenOutdoors, :ChildMusic, :SpecialActivities,
     :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
     :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy],
    [:HowOftenRead, :HowManyBooks, :HowOftenMuseum, :HowOftenTheater,
     :HowOftenFamily, :HowOftenOutdoors, :HowOftenFather, :ChildMusic,
     :SpecialActivities, :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
     :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy],
    [:HowManyBooks, :HowOftenMuseum, :HowOftenTheater, :HowOftenFamily,
     :HowOftenOutdoors, :HowOftenFather, :ChildMusic, :SpecialActivities,
     :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
     :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy,
     :ToldPositive, :HowOftenClean, :HowOftenChores, :HowOftenTime],
    [:HowManyBooks, :HowOftenMuseum, :HowOftenTheater, :HowOftenFamily,
     :HowOftenOutdoors, :HowOftenFather, :ChildMusic, :SpecialActivities,
     :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
     :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy,
     :ToldPositive, :HowOftenClean, :HowOftenChores, :HowOftenTime],
]

const OUTPUT_PATH = joinpath(dirname(@__DIR__), "output", "estimation_v3.jld2")

# ─────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────

println("Loading datasets...")
datasets = load_all_datasets()

println("\n" * "="^60)
println("  VERSION 3 ESTIMATION")
println("  r_θ=4, r_I=5, r_P=$R_P, r_ω=$R_ω, K=$K_NN, ν=$NU")
println("="^60)

panel_data = prepare_panel_data(datasets;
    age_ranges   = PANEL_AGE_RANGES,
    invest_items = PANEL_INVEST_ITEMS,
    r_θ = 4, r_I = 5,
)
data_v3 = merge(panel_data, (; r_P = R_P, r_ω = R_ω))

println("\n  N = $(data_v3.N), T = $(data_v3.T), r_Z = $(R_P * R_ω)")
for t in 1:data_v3.T
    println("    Period $t: $(length(data_v3.X_θ[t])) skill, $(length(data_v3.X_I[t])) invest measures")
end
println("  ASVAB measures: $(length(data_v3.X_P))")
println("  Income obs: $(sum(!isnan, data_v3.Y_income)) / $(data_v3.N * data_v3.T)")

# ─────────────────────────────────────────────────────────────────
# Run EM
# ─────────────────────────────────────────────────────────────────

println("\n" * "="^60)
println("  Running V3 EM with funnel restarts")
println("="^60)

best_pv3, best_psk, best_γ, best_ξz, best_wz, best_ll =
    run_em_restarts_v3(data_v3;
        n_initial    = 10,
        n_semifinal  = 3,
        phase1_iters = 15,
        phase2_iters = 75,
        maxiter      = 500,
        tol          = 1e-4,
        verbose      = true,
        config       = V3_CONFIG,
        transition_groups = TRANSITION_GROUPS,
    )

println("\n  Best log-likelihood: $(round(best_ll; digits=2))")

# ─────────────────────────────────────────────────────────────────
# Display and save
# ─────────────────────────────────────────────────────────────────

display_results_v3(best_pv3, best_psk, data_v3)

println("\nSaving to $OUTPUT_PATH ...")
save(OUTPUT_PATH,
    "params_v3",    best_pv3,
    "params_skill", best_psk,
    "gamma_agg",    best_γ,
    "w_z",          best_wz,
    "best_ll",      best_ll,
    "data_dims",    (N=data_v3.N, T=data_v3.T, r_θ=data_v3.r_θ, r_I=data_v3.r_I,
                     r_P=R_P, r_ω=R_ω, K=K_NN),
    "age_ranges",   PANEL_AGE_RANGES,
)
println("Done.")
