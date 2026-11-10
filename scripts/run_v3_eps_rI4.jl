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
# V3 + ε_Y at r_I=4 — K=4 and K=5, warm-started from V2 r_I=4 chain
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
const V2_K3_RI4 = joinpath(OUTPUT_DIR, "estimation_multiperiod_robit_k3_grouped_rI4.jld2")
const V2_K4_RI4 = joinpath(OUTPUT_DIR, "estimation_multiperiod_robit_k4_grouped_rI4.jld2")
const V2_K5_RI4 = joinpath(OUTPUT_DIR, "estimation_multiperiod_robit_k5_grouped_rI4.jld2")
const OUT_K3    = joinpath(OUTPUT_DIR, "estimation_v3_eps_k3_rI4.jld2")
const OUT_K4    = joinpath(OUTPUT_DIR, "estimation_v3_eps_k4_rI4.jld2")
const OUT_K5    = joinpath(OUTPUT_DIR, "estimation_v3_eps_k5_rI4.jld2")

# ─────────────────────────────────────────────────────────────────
# Load data at r_I=4
# ─────────────────────────────────────────────────────────────────

println("Loading datasets...")
datasets = load_all_datasets()

panel_data = prepare_panel_data(datasets;
    age_ranges   = PANEL_AGE_RANGES,
    invest_items = PANEL_INVEST_ITEMS,
    r_θ = 4, r_I = 4,
)
data_v3_eps = merge(panel_data, (; r_P = R_P, r_ω = R_ω))
(; r_θ, r_I, T, N, κ_θ, κ_I, κ_P) = panel_data
r_Z = R_P * R_ω * R_Y
println("  N = $N, T = $T, r_θ = $r_θ, r_I = $r_I, r_Z = $r_Z")

# ─────────────────────────────────────────────────────────────────
# Helper: build V3+ε_Y params warm-started from a V2 r_I=4 fit
# ─────────────────────────────────────────────────────────────────

function build_v3_eps_warmstart(v2_fit_path::String, data_v3_eps, panel_data;
                                 rng_seed::Int = 42)
    v2 = load(v2_fit_path)
    v2_params = v2["best_params"]
    v2_specs  = v2["best_specs"]
    v2_ll     = v2["best_ll"]
    println("  Loaded $(basename(v2_fit_path)),  V2 LL = $(round(v2_ll; digits=2))")

    (; r_θ, r_I, T, κ_P) = panel_data
    rng = MersenneTwister(rng_seed)

    # Transitions: V2 RobitNNTransition → V3 RobitNNTransitionV3 (pad alpha3,alpha4=0)
    v2_trans = v2_specs[:transition]
    trans_specs_v3 = Vector{RobitNNTransitionV3}(undef, length(TRANSITION_GROUPS))
    for (g, group) in enumerate(TRANSITION_GROUPS)
        v2_spec = v2_trans[group[1]]
        K = v2_spec.K
        trans_specs_v3[g] = RobitNNTransitionV3(
            copy(v2_spec.c_raw),
            copy(v2_spec.alpha1),
            copy(v2_spec.alpha2_raw),
            zeros(K),                 # alpha3 (θ^P)
            zeros(K),                 # alpha4 (ω)
            copy(v2_spec.b),
            copy(v2_spec.log_w),
            K, v2_spec.ν,
        )
    end
    K_v2 = trans_specs_v3[1].K
    println("  Transitions: V2 K=$K_v2, alpha3=alpha4=0 init")

    # Skill/investment emissions: transfer
    probit_θ = v2_specs[:probit_θ]
    probit_I = v2_specs[:probit_I]
    P_θ = [emission_matrices_from_probit(probit_θ[t], r_θ) for t in 1:T]
    P_I = [emission_matrices_from_probit(probit_I[t], r_I) for t in 1:T]

    # CCP: V2 OrderedProbitCCPSimple → V3+ε_Y with β₅=0.3 positive anchor
    v2_ccp = v2_specs[:ccp][1]
    ccp_spec = OrderedProbitCCPV3Eps(
        copy(v2_ccp.c),   # r_I-1 cutpoints (already 3 for r_I=4)
        v2_ccp.beta1,
        0.0, 0.0, 0.0,    # β₂ fixed, β₃,β₄ init 0
        0.3,              # β₅(ε_Y) positive anchor
    )

    # Q_P: random init
    probit_P = initialize_ordered_probit(collect(κ_P), R_P; rng)
    Q_P = emission_matrices_from_probit(probit_P, R_P)

    # Income model: β₀ from sample mean; δ=0.15 positive anchor
    income_model = initialize_income_model_eps(data_v3_eps.Y_income; delta_init=0.15)

    # Priors: uniform
    π0   = ones(r_θ, R_P, R_ω) / (r_θ * R_P * R_ω)
    π_Pω = ones(R_P, R_ω) / (R_P * R_ω)
    π_ε  = fill(1.0 / R_Y, R_Y)

    # Adult outcomes
    κ_adult = data_v3_eps.κ_adult
    adult_emission = initialize_adult_outcome_emission(κ_adult, r_θ, R_ω; rng)
    adult_matrices = build_adult_matrices(adult_emission, κ_adult, r_θ, R_ω)

    params_v3_eps = (;
        r_P = R_P, r_ω = R_ω, r_Y = R_Y,
        π0, π_Pω, π_ε,
        trans_specs = trans_specs_v3,
        ccp_spec, probit_P, Q_P, income_model,
        adult_emission, adult_matrices,
    )
    params_skill = (; r_θ, r_I, T, P_θ, P_I, probit_θ, probit_I)
    return params_v3_eps, params_skill, v2_ll, K_v2
end

# ─────────────────────────────────────────────────────────────────
# Helper: run one V3+ε_Y fit from a V2 r_I=4 warm start
# ─────────────────────────────────────────────────────────────────

function run_v3_eps_fit(v2_path::String, out_path::String, tag::String; rng_seed::Int)
    println("\n" * "="^60)
    println("  $tag  (warm from $(basename(v2_path)))")
    println("="^60)

    pv_init, psk_init, v2_ll, K_val = build_v3_eps_warmstart(v2_path, data_v3_eps, panel_data; rng_seed)

    println("\n  Running V3+ε_Y EM (K=$K_val, r_I=4)...")
    pv, psk, γ, ξz, wz, ll = expectation_maximization_v3_eps(
        data_v3_eps, pv_init, psk_init;
        transition_groups = TRANSITION_GROUPS,
        maxiter = 2000, tol = 1e-4, verbose = true,
    )
    println("  V3+ε_Y K=$K_val r_I=4 LL: $(round(ll; digits=2))")

    jldsave(out_path;
        params_v3_eps = pv, params_skill = psk,
        gamma_agg = γ, xi_sum_z = ξz, w_z = wz,
        best_ll = ll,
        data_dims = (; N, T, r_θ, r_I, r_P=R_P, r_ω=R_ω, r_Y=R_Y, K=K_val),
        age_ranges = PANEL_AGE_RANGES,
        transition_groups = TRANSITION_GROUPS,
        v2_ll = v2_ll,
    )
    println("  Saved to $out_path")

    return pv, psk, ll, K_val
end

# ─────────────────────────────────────────────────────────────────
# Stages C/D/E: V3+ε_Y at K=3, K=4, K=5 (r_I=4)
# ─────────────────────────────────────────────────────────────────

pv3_k3, psk_k3, ll_k3, K_k3 = run_v3_eps_fit(V2_K3_RI4, OUT_K3, "STAGE C: V3+ε_Y K=3, r_I=4"; rng_seed=41)
pv3_k4, psk_k4, ll_k4, K_k4 = run_v3_eps_fit(V2_K4_RI4, OUT_K4, "STAGE D: V3+ε_Y K=4, r_I=4"; rng_seed=42)
pv3_k5, psk_k5, ll_k5, K_k5 = run_v3_eps_fit(V2_K5_RI4, OUT_K5, "STAGE E: V3+ε_Y K=5, r_I=4"; rng_seed=43)

# ─────────────────────────────────────────────────────────────────
# BIC comparison K=3 vs K=4 vs K=5 at r_I=4
# ─────────────────────────────────────────────────────────────────

println("\n" * "="^60)
println("  BIC COMPARISON: V3+ε_Y K=3, K=4, K=5 at r_I=4")
println("="^60)

# Transition params per group: (r_θ-1) cutpoints + 6K (α₁,α₂,α₃,α₄,b,w each length K)
# Identifying restriction: one log_w fixed (handled by convention) — using full K parameterization for comparison
function n_trans_params(K, r_θ; n_groups=length(TRANSITION_GROUPS))
    n_groups * ((r_θ - 1) + 6 * K)
end

results = [
    (; K=K_k3, ll=ll_k3, pv=pv3_k3),
    (; K=K_k4, ll=ll_k4, pv=pv3_k4),
    (; K=K_k5, ll=ll_k5, pv=pv3_k5),
]

println(@sprintf("  %-5s  %12s  %12s  %12s", "K", "LL", "trans_df", "BIC_trans"))
for r in results
    n_trans = n_trans_params(r.K, r_θ)
    bic = -2 * r.ll + n_trans * log(N)
    println(@sprintf("  K=%-3d  %12.2f  %12d  %12.2f", r.K, r.ll, n_trans, bic))
end

bics = [(r.K, -2 * r.ll + n_trans_params(r.K, r_θ) * log(N)) for r in results]
best_K, best_bic = first(sort(bics; by = x -> x[2]))
println(@sprintf("\n  BIC-preferred: K=%d  (BIC_trans = %.2f)", best_K, best_bic))

# Degeneracy check
println("\n  Degeneracy check (max|log_w|):")
for r in results
    for (g, spec) in enumerate(r.pv.trans_specs)
        max_lw = maximum(abs.(spec.log_w))
        println(@sprintf("    K=%d group %d: max|log_w|=%.1f,  log_w=%s",
            r.K, g, max_lw, string(round.(spec.log_w; digits=2))))
    end
end

println("\n  BIC-preferred V3+ε_Y spec at r_I=4:  K=$best_K")
println("  Settled output file:  $(best_K == K_k3 ? OUT_K3 : best_K == K_k4 ? OUT_K4 : OUT_K5)")

println("\nDone.")
