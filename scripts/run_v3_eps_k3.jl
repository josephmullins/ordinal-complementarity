using LinearAlgebra, Random, Statistics, JLD2, Optim, Printf

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3_eps.jl"))
include(joinpath(@__DIR__, "..", "src", "prepare_data.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod_v3.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod_v3_eps.jl"))

# ─────────────────────────────────────────────────────────────────
# V3+ε_Y K=3 — warm-started from V2 K=3 robit NN grouped
#   Same pattern as run_v3_eps.jl but using V2 K=3 source
# ─────────────────────────────────────────────────────────────────

const R_P  = 3
const R_ω  = 3
const R_Y  = 3
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
const V2_K3_FIT = joinpath(OUTPUT_DIR, "estimation_multiperiod_robit_grouped.jld2")
const V2_K4_FIT = joinpath(OUTPUT_DIR, "estimation_multiperiod_robit_k4_grouped.jld2")
const K4_FIT    = joinpath(OUTPUT_DIR, "estimation_v3_eps.jld2")
const OUT_FILE  = joinpath(OUTPUT_DIR, "estimation_v3_eps_k3.jld2")

# ── Load data ──
println("Loading datasets...")
datasets = load_all_datasets()
panel_data = prepare_panel_data(datasets;
    age_ranges = PANEL_AGE_RANGES, invest_items = PANEL_INVEST_ITEMS,
    r_θ = 4, r_I = 5)
data = merge(panel_data, (; r_P = R_P, r_ω = R_ω))
(; r_θ, r_I, T, N, κ_θ, κ_I, κ_P) = panel_data

println("  N = $N, T = $T, r_θ = $r_θ, r_I = $r_I")

# ── Load V2 K=3 ──
println("\nLoading V2 K=3 from $V2_K3_FIT...")
v2 = load(V2_K3_FIT)
v2_specs = v2["best_specs"]
v2_ll = v2["best_ll"]
println("  V2 K=3 LL: $(round(v2_ll; digits=2))")

# ── Convert V2 K=3 → V3 transitions (add alpha3=alpha4=0) ──
rng = MersenneTwister(42)
v2_trans = v2_specs[:transition]
K_v2 = v2_trans[1].K
println("  V2 K = $K_v2")

trans_specs_v3 = Vector{RobitNNTransitionV3}(undef, length(TRANSITION_GROUPS))
for (g, group) in enumerate(TRANSITION_GROUPS)
    v2_spec = v2_trans[group[1]]
    K = v2_spec.K
    trans_specs_v3[g] = RobitNNTransitionV3(
        copy(v2_spec.c_raw),
        copy(v2_spec.alpha1),
        copy(v2_spec.alpha2_raw),
        zeros(K),              # alpha3 (θ^P)
        zeros(K),              # alpha4 (ω)
        copy(v2_spec.b),
        copy(v2_spec.log_w),
        K, v2_spec.ν)
end
println("  Transitions: K=$K_v2, alpha3=alpha4=0")

# ── Skill/investment emissions from V2 ──
probit_θ = v2_specs[:probit_θ]
probit_I = v2_specs[:probit_I]
P_θ = [emission_matrices_from_probit(probit_θ[t], r_θ) for t in 1:T]
P_I = [emission_matrices_from_probit(probit_I[t], r_I) for t in 1:T]

# ── CCP from V2 → V3+ε_Y ──
v2_ccp = v2_specs[:ccp][1]
ccp_spec = OrderedProbitCCPV3Eps(
    copy(v2_ccp.c), v2_ccp.beta1,
    0.0, 0.0, 0.0, 0.3)   # β₅(ε_Y) = 0.3 positive anchor
println("  CCP: β₁=$(round(v2_ccp.beta1;digits=3)), β₅(ε_Y) init = 0.3")

# ── Q_P, income, priors, adult — fresh init ──
probit_P = initialize_ordered_probit(collect(κ_P), R_P; rng)
Q_P = emission_matrices_from_probit(probit_P, R_P)
income_model = initialize_income_model_eps(data.Y_income; delta_init=0.15)
π0   = ones(r_θ, R_P, R_ω) / (r_θ * R_P * R_ω)
π_Pω = ones(R_P, R_ω) / (R_P * R_ω)
π_ε  = fill(1.0 / R_Y, R_Y)

κ_adult = data.κ_adult
adult_emission = initialize_adult_outcome_emission(κ_adult, r_θ, R_ω; rng)
adult_matrices = build_adult_matrices(adult_emission, κ_adult, r_θ, R_ω)

params_v3_eps = (;
    r_P = R_P, r_ω = R_ω, r_Y = R_Y,
    π0, π_Pω, π_ε,
    trans_specs = trans_specs_v3,
    ccp_spec, probit_P, Q_P, income_model,
    adult_emission, adult_matrices)
params_skill = (; r_θ, r_I, T, P_θ, P_I, probit_θ, probit_I)

# ── Run EM ──
println("\n" * "="^60)
println("  Running V3+ε_Y K=$K_v2 EM (warm-started from V2 K=$K_v2)")
println("="^60)

pv3, psk, γ_agg, ξ_sum_z, w_z, ll = expectation_maximization_v3_eps(
    data, params_v3_eps, params_skill;
    transition_groups = TRANSITION_GROUPS,
    maxiter = 2000, tol = 1e-4, verbose = true)

println("\n  V3+ε_Y K=$K_v2 final LL: $(round(ll; digits=2))")

# ── BIC comparison with K=4 ──
ll_k4 = load(K4_FIT)["best_ll"]
n_trans_k3 = 2 * (r_θ - 1 + 6 * K_v2)
n_trans_k4 = 2 * (r_θ - 1 + 6 * 4)
bic_k3 = -2 * ll + n_trans_k3 * log(N)
bic_k4 = -2 * ll_k4 + n_trans_k4 * log(N)
println("\n  K=3: LL=$(round(ll;digits=2)),  BIC_trans=$(round(bic_k3;digits=2))")
println("  K=4: LL=$(round(ll_k4;digits=2)),  BIC_trans=$(round(bic_k4;digits=2))")
println("  ΔBIC (K3 - K4) = $(round(bic_k3 - bic_k4;digits=2))  (negative → K3 preferred)")

# Check degeneracy
for (g, spec) in enumerate(pv3.trans_specs)
    max_lw = maximum(abs.(spec.log_w))
    println("  Group $g: max|log_w| = $(round(max_lw;digits=1)),  log_w = $(round.(spec.log_w;digits=2))")
end

# ── Save ──
jldsave(OUT_FILE;
    params_v3_eps = pv3, params_skill = psk,
    gamma_agg = γ_agg, xi_sum_z = ξ_sum_z, w_z,
    best_ll = ll,
    data_dims = (; N, T, r_θ, r_I, r_P=R_P, r_ω=R_ω, r_Y=R_Y, K=K_v2),
    age_ranges = PANEL_AGE_RANGES,
    transition_groups = TRANSITION_GROUPS,
    ll_k4, bic_k3, bic_k4)
println("\nSaved to $OUT_FILE")

display_results_v3_eps(pv3, psk, data)
println("\nDone.")
