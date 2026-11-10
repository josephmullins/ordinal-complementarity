using LinearAlgebra, Random, Statistics, JLD2, Optim

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3.jl"))
include(joinpath(@__DIR__, "..", "src", "prepare_data.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod_v3.jl"))

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────

const R_P  = 3
const R_ω  = 3
const NU   = 7.0
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

const OUTPUT_DIR = joinpath(dirname(@__DIR__), "output")
mkpath(OUTPUT_DIR)

# ─────────────────────────────────────────────────────────────────
# Load V3 data (includes parental skills + income)
# ─────────────────────────────────────────────────────────────────

println("Loading datasets...")
datasets = load_all_datasets()

panel_data = prepare_panel_data(datasets;
    age_ranges   = PANEL_AGE_RANGES,
    invest_items = PANEL_INVEST_ITEMS,
    r_θ = 4, r_I = 5,
)
# Drop income: set all observations to NaN
panel_data_no_inc = merge(panel_data, (; Y_income = fill(NaN, size(panel_data.Y_income))))
data_v3 = merge(panel_data_no_inc, (; r_P = R_P, r_ω = R_ω))

(; r_θ, r_I, T, N, κ_θ, κ_I, κ_P) = panel_data
println("  Income: ALL observations set to NaN (dropped from likelihood)")
r_Z = R_P * R_ω

println("\n  N = $N, T = $T, r_θ = $r_θ, r_I = $r_I, r_P = $R_P, r_ω = $R_ω")

# ─────────────────────────────────────────────────────────────────
# Load V2 results
# ─────────────────────────────────────────────────────────────────

v2_path = joinpath(OUTPUT_DIR, "estimation_multiperiod_robit_grouped.jld2")
println("\nLoading V2 results from $v2_path ...")
v2 = load(v2_path)
v2_params = v2["best_params"]
v2_specs  = v2["best_specs"]
v2_ll     = v2["best_ll"]

println("  V2 log-likelihood: $(round(v2_ll; digits=2))")

# ─────────────────────────────────────────────────────────────────
# Convert V2 -> V3
# ─────────────────────────────────────────────────────────────────

println("\nConverting V2 parameters to V3 warm-start...")

rng = MersenneTwister(42)

# --- 1. Transitions: add alpha3=0, alpha4=0 to each V2 RobitNNTransition ---

v2_trans = v2_specs[:transition]  # Vector of RobitNNTransition (one per group representative)

# V2 has grouped transitions: for each group, take group[1] as the representative
trans_specs_v3 = Vector{RobitNNTransitionV3}(undef, length(TRANSITION_GROUPS))
for (g, group) in enumerate(TRANSITION_GROUPS)
    v2_spec = v2_trans[group[1]]
    K = v2_spec.K
    trans_specs_v3[g] = RobitNNTransitionV3(
        copy(v2_spec.c_raw),
        copy(v2_spec.alpha1),
        copy(v2_spec.alpha2_raw),
        zeros(K),              # alpha3 (theta^P) = 0
        zeros(K),              # alpha4 (omega) = 0
        copy(v2_spec.b),
        copy(v2_spec.log_w),
        K,
        v2_spec.ν,
    )
end

K_v2 = trans_specs_v3[1].K
println("  Transitions: V2 K=$K_v2, added alpha3=0, alpha4=0")

# --- 2. Emissions: transfer V2 probit_θ and probit_I directly ---

probit_θ = v2_specs[:probit_θ]
probit_I = v2_specs[:probit_I]
P_θ = [emission_matrices_from_probit(probit_θ[t], r_θ) for t in 1:T]
P_I = [emission_matrices_from_probit(probit_I[t], r_I) for t in 1:T]

println("  Skill emissions: $(length(probit_θ)) periods transferred")
println("  Investment emissions: $(length(probit_I)) periods transferred")

# --- 3. CCP: convert V2 OrderedProbitCCPSimple -> V3 OrderedProbitCCPV3 ---

v2_ccp_vec = v2_specs[:ccp]
v2_ccp = v2_ccp_vec[1]  # all transitions share the same CCP; take representative
ccp_spec = OrderedProbitCCPV3(
    copy(v2_ccp.c),    # cutpoints
    v2_ccp.beta1,      # beta1 (theta')
    0.0,               # beta2 (I_old) -- V2 CCPSimple has no beta2
    0.0,               # beta3 (theta^P)
    0.0,               # beta4 (omega)
    0.0,               # beta5 (log y)
)

println("  CCP: transferred beta1=$(round(v2_ccp.beta1; digits=3)), set beta2-5=0")

# --- 4. Parental skill emissions Q_P: initialize from scratch ---

probit_P = initialize_ordered_probit(collect(κ_P), R_P; rng)
Q_P = emission_matrices_from_probit(probit_P, R_P)

println("  Q_P: initialized $(length(κ_P)) ASVAB emission matrices (random)")

# --- 5. Income model: initialize from sample moments ---

income_model = initialize_income_model(data_v3.Y_income)

println("  Income model: beta0=$(round(income_model.beta0; digits=3)), log_sigma=$(round(income_model.log_sigma; digits=3))")

# --- 6. Initial distribution: uniform over (r_θ, r_P, r_ω) ---

π0  = ones(r_θ, R_P, R_ω) / (r_θ * R_P * R_ω)
π_Z = ones(R_P, R_ω) / (R_P * R_ω)

println("  pi0: uniform $(size(π0)), pi_Z: uniform $(size(π_Z))")

# ─────────────────────────────────────────────────────────────────
# Build V3 parameter tuples
# ─────────────────────────────────────────────────────────────────

params_v3 = (;
    r_P = R_P, r_ω = R_ω,
    π0, π_Z,
    trans_specs = trans_specs_v3,
    ccp_spec,
    probit_P,
    Q_P,
    income_model,
)

params_skill = (; r_θ, r_I, T, P_θ, P_I, probit_θ, probit_I)

# ─────────────────────────────────────────────────────────────────
# Run V3 EM
# ─────────────────────────────────────────────────────────────────

println("\n" * "="^60)
println("  Running V3 EM — NO INCOME (warm-started from V2 K=$K_v2)")
println("="^60)

pv3, psk, γ_agg, ξ_sum_z, w_z, ll_v3 = expectation_maximization_v3(
    data_v3, params_v3, params_skill;
    transition_groups = TRANSITION_GROUPS,
    maxiter = 2000,
    tol     = 1e-4,
    verbose = true,
)

println("\n  V3 final log-likelihood: $(round(ll_v3; digits=2))")
println("  V2 log-likelihood:       $(round(v2_ll; digits=2))")
println("  Improvement:             $(round(ll_v3 - v2_ll; digits=2))")

# ─────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────

outpath = joinpath(OUTPUT_DIR, "estimation_v3_no_income.jld2")
jldsave(outpath;
    params_v3  = pv3,
    params_skill = psk,
    gamma_agg  = γ_agg,
    xi_sum_z   = ξ_sum_z,
    w_z        = w_z,
    best_ll    = ll_v3,
    data_dims  = (; N, T, r_θ, r_I, r_P=R_P, r_ω=R_ω, K=K_v2),
    age_ranges = PANEL_AGE_RANGES,
    transition_groups = TRANSITION_GROUPS,
    v2_ll      = v2_ll,
)
println("\n  Results saved to $outpath")

# ─────────────────────────────────────────────────────────────────
# Display results
# ─────────────────────────────────────────────────────────────────

display_results_v3(pv3, psk, data_v3)

println("\nDone.")
