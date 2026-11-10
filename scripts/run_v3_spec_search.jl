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
# Load data
# ─────────────────────────────────────────────────────────────────

println("Loading datasets...")
datasets = load_all_datasets()

panel_data = prepare_panel_data(datasets;
    age_ranges   = PANEL_AGE_RANGES,
    invest_items = PANEL_INVEST_ITEMS,
    r_θ = 4, r_I = 5,
)
data_v3 = merge(panel_data, (; r_P = R_P, r_ω = R_ω))

(; r_θ, r_I, T, N) = panel_data
r_Z = R_P * R_ω

println("\n  N = $N, T = $T, r_θ = $r_θ, r_I = $r_I, r_P = $R_P, r_ω = $R_ω")

# ─────────────────────────────────────────────────────────────────
# Parameter counting
# ─────────────────────────────────────────────────────────────────

function count_v3_params(data, n_trans_params_per_group, n_groups, r_P, r_ω)
    (; r_θ, r_I, T) = data
    n_trans = n_groups * n_trans_params_per_group
    n_emission = 0
    for t in 1:T
        for κ in data.κ_θ[t]; n_emission += κ; end
        for κ in data.κ_I[t]; n_emission += κ; end
    end
    n_ccp = (r_I - 1) + 5
    n_pi0 = r_θ * r_P * r_ω - 1
    n_qp = sum((data.κ_P[m]) for m in eachindex(data.κ_P))  # ordered probit: κ_m-1 cuts + 1 alpha per measure
    n_income = 4
    return n_trans + n_emission + n_ccp + n_pi0 + n_qp + n_income
end

# ─────────────────────────────────────────────────────────────────
# Warm-start helper: fit robit NN K transitions to existing ξ_sum_z
# ─────────────────────────────────────────────────────────────────

function warmstart_robit_transitions(xi_sum_z, K_new, r_θ, r_I, r_P, r_ω;
                                       ν=NU, n_restarts=15)
    r_Z = r_P * r_ω
    θ_grid = collect(1.0:r_θ); I_grid = collect(1.0:r_I)
    P_grid = collect(1.0:r_P); ω_grid = collect(1.0:r_ω)

    best_specs = nothing
    best_obj = Inf

    for i in 1:n_restarts
        rng = MersenneTwister(300 + i)
        init_specs = [initialize_robit_nn_v3(r_θ, K_new; rng, ν) for _ in 1:length(TRANSITION_GROUPS)]

        try
            new_specs = mstep_transition_v3_grouped(
                init_specs, xi_sum_z, TRANSITION_GROUPS,
                r_θ, r_I, r_P, r_ω; θ_grid, I_grid, P_grid, ω_grid)

            obj = 0.0
            for (g, group) in enumerate(TRANSITION_GROUPS)
                Q_sum_z = Vector{Array{Float64,3}}(undef, r_Z)
                for z in 1:r_Z
                    Q_sum_z[z] = _qsum_from_xi(xi_sum_z[z][group[1]])
                    for t in group[2:end]
                        Q_sum_z[z] = Q_sum_z[z] .+ _qsum_from_xi(xi_sum_z[z][t])
                    end
                end
                φ = pack_robit_nn_v3(new_specs[g])
                obj += transition_negll_robit_nn_v3(φ, Q_sum_z, r_θ, r_I, K_new, ν,
                                                      r_P, r_ω; θ_grid, I_grid, P_grid, ω_grid)
            end

            println("      Init $i/$n_restarts: negll = $(round(obj; digits=2))")
            if obj < best_obj
                best_obj = obj
                best_specs = new_specs
            end
        catch e
            println("      Init $i/$n_restarts: failed ($(typeof(e)))")
        end
    end

    return best_specs
end

# ═══════════════════════════════════════════════════════════════
# STEP 1: Ordered probit transitions (baseline, many restarts)
# ═══════════════════════════════════════════════════════════════

op_path = joinpath(OUTPUT_DIR, "estimation_v3_op.jld2")
if isfile(op_path)
    println("\n" * "="^60)
    println("  STEP 1: Loading saved ordered probit results")
    println("="^60)
    op_data = load(op_path)
    pv3_op = op_data["params_v3"]
    psk_op = op_data["params_skill"]
    γ_op = op_data["gamma_agg"]
    ξz_op = op_data["xi_sum_z"]
    wz_op = op_data["w_z"]
    ll_op = op_data["best_ll"]
    n_params_op = op_data["n_params"]
    aic_op = op_data["aic"]
    bic_op = op_data["bic"]
    println("  ll=$(round(ll_op;digits=2)), params=$n_params_op")
    println("  AIC=$(round(aic_op;digits=2)), BIC=$(round(bic_op;digits=2))")
    display_results_v3(pv3_op, psk_op, data_v3)
else
    println("\n" * "="^60)
    println("  STEP 1: Ordered probit transition (baseline)")
    println("="^60)

    config_op = (transition = :ordered_probit, n_groups = 2)
    pv3_op, psk_op, γ_op, ξz_op, wz_op, ll_op = run_em_restarts_v3(data_v3;
        n_initial    = 20,
        n_semifinal  = 5,
        phase1_iters = 20,
        phase2_iters = 100,
        maxiter      = 2000,
        tol          = 1e-5,
        verbose      = true,
        config       = config_op,
        transition_groups = TRANSITION_GROUPS)

    n_trans_per_group_op = (r_θ - 1) + 5  # cutpoints + 5 betas
    n_params_op = count_v3_params(data_v3, n_trans_per_group_op, 2, R_P, R_ω)
    aic_op = -2 * ll_op + 2 * n_params_op
    bic_op = -2 * ll_op + log(N) * n_params_op

    println("\n  Ordered probit: ll=$(round(ll_op;digits=2)), params=$n_params_op")
    println("    AIC=$(round(aic_op;digits=2)), BIC=$(round(bic_op;digits=2))")

    jldsave(op_path; params_v3=pv3_op, params_skill=psk_op,
            gamma_agg=γ_op, xi_sum_z=ξz_op, w_z=wz_op,
            best_ll=ll_op, data_dims=(; N, T, r_θ, r_I, r_P=R_P, r_ω=R_ω, K=0),
            age_ranges=PANEL_AGE_RANGES, aic=aic_op, bic=bic_op, n_params=n_params_op)
    println("  Saved to $op_path")
    display_results_v3(pv3_op, psk_op, data_v3)
end

# ═══════════════════════════════════════════════════════════════
# STEP 2: K=2 robit NN, warm-started from ordered probit
# ═══════════════════════════════════════════════════════════════

K_curr = 2
println("\n" * "="^60)
println("  STEP 2: K=$K_curr robit NN (warm start from ordered probit)")
println("="^60)

println("  Fitting K=$K_curr transitions to ordered probit posteriors...")
new_trans = warmstart_robit_transitions(ξz_op, K_curr, r_θ, r_I, R_P, R_ω; n_restarts=15)
if new_trans === nothing
    error("All K=$K_curr warm-start initializations failed.")
end

# Build warm-started params: keep emissions, CCP, Q_P, income from ordered probit
pv3_init = (; pv3_op..., trans_specs = new_trans)
psk_init = psk_op

pv3_k2, psk_k2, γ_k2, ξz_k2, wz_k2, ll_k2 = expectation_maximization_v3(
    data_v3, pv3_init, psk_init;
    transition_groups = TRANSITION_GROUPS,
    maxiter = 2000, tol = 1e-4, verbose = true)

n_trans_per_group_k2 = (r_θ - 1) + 6 * K_curr  # cutpoints + 6K
n_params_k2 = count_v3_params(data_v3, n_trans_per_group_k2, 2, R_P, R_ω)
aic_k2 = -2 * ll_k2 + 2 * n_params_k2
bic_k2 = -2 * ll_k2 + log(N) * n_params_k2

println("\n  K=$K_curr: ll=$(round(ll_k2;digits=2)), params=$n_params_k2")
println("    AIC=$(round(aic_k2;digits=2)), BIC=$(round(bic_k2;digits=2))")

outpath = joinpath(OUTPUT_DIR, "estimation_v3_k$(K_curr).jld2")
jldsave(outpath; params_v3=pv3_k2, params_skill=psk_k2,
        gamma_agg=γ_k2, w_z=wz_k2,
        best_ll=ll_k2, data_dims=(; N, T, r_θ, r_I, r_P=R_P, r_ω=R_ω, K=K_curr),
        age_ranges=PANEL_AGE_RANGES, aic=aic_k2, bic=bic_k2, n_params=n_params_k2)
println("  Saved to $outpath")
display_results_v3(pv3_k2, psk_k2, data_v3)

# ═══════════════════════════════════════════════════════════════
# STEP 3: K=3 robit NN, warm-started from K=2 (if BIC improves)
# ═══════════════════════════════════════════════════════════════

if bic_k2 < bic_op
    K_prev = K_curr; K_curr = 3
    println("\n" * "="^60)
    println("  STEP 3: K=$K_curr robit NN (warm start from K=$K_prev)")
    println("="^60)

    println("  Fitting K=$K_curr transitions to K=$K_prev posteriors...")
    new_trans = warmstart_robit_transitions(ξz_k2, K_curr, r_θ, r_I, R_P, R_ω; n_restarts=15)
    if new_trans === nothing
        error("All K=$K_curr warm-start initializations failed.")
    end

    pv3_init = (; pv3_k2..., trans_specs = new_trans)
    psk_init = psk_k2

    pv3_k3, psk_k3, γ_k3, ξz_k3, wz_k3, ll_k3 = expectation_maximization_v3(
        data_v3, pv3_init, psk_init;
        transition_groups = TRANSITION_GROUPS,
        maxiter = 2000, tol = 1e-4, verbose = true)

    n_trans_per_group_k3 = (r_θ - 1) + 6 * K_curr
    n_params_k3 = count_v3_params(data_v3, n_trans_per_group_k3, 2, R_P, R_ω)
    aic_k3 = -2 * ll_k3 + 2 * n_params_k3
    bic_k3 = -2 * ll_k3 + log(N) * n_params_k3

    println("\n  K=$K_curr: ll=$(round(ll_k3;digits=2)), params=$n_params_k3")
    println("    AIC=$(round(aic_k3;digits=2)), BIC=$(round(bic_k3;digits=2))")

    outpath = joinpath(OUTPUT_DIR, "estimation_v3_k$(K_curr).jld2")
    jldsave(outpath; params_v3=pv3_k3, params_skill=psk_k3,
            gamma_agg=γ_k3, w_z=wz_k3,
            best_ll=ll_k3, data_dims=(; N, T, r_θ, r_I, r_P=R_P, r_ω=R_ω, K=K_curr),
            age_ranges=PANEL_AGE_RANGES, aic=aic_k3, bic=bic_k3, n_params=n_params_k3)
    println("  Saved to $outpath")
    display_results_v3(pv3_k3, psk_k3, data_v3)
else
    println("\n  Skipping K=3: BIC(K=2) = $(round(bic_k2;digits=2)) ≥ BIC(OP) = $(round(bic_op;digits=2))")
end

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════

println("\n" * "="^60)
println("  MODEL COMPARISON SUMMARY")
println("="^60)
println("\n  Model          |  log-lik        |  params  |  AIC            |  BIC")
println("  ---------------|-----------------|----------|-----------------|----------------")
println("  Ord. Probit    |  $(round(ll_op;digits=2))  |  $n_params_op  |  $(round(aic_op;digits=2))  |  $(round(bic_op;digits=2))")
println("  Robit K=2      |  $(round(ll_k2;digits=2))  |  $n_params_k2  |  $(round(aic_k2;digits=2))  |  $(round(bic_k2;digits=2))")
if @isdefined(ll_k3)
    println("  Robit K=3      |  $(round(ll_k3;digits=2))  |  $n_params_k3  |  $(round(aic_k3;digits=2))  |  $(round(bic_k3;digits=2))")
end

# Save comparison
summary = [(model="OrderedProbit", ll=ll_op, n_params=n_params_op, aic=aic_op, bic=bic_op),
           (model="RobitK2", ll=ll_k2, n_params=n_params_k2, aic=aic_k2, bic=bic_k2)]
if @isdefined(ll_k3)
    push!(summary, (model="RobitK3", ll=ll_k3, n_params=n_params_k3, aic=aic_k3, bic=bic_k3))
end
jldsave(joinpath(OUTPUT_DIR, "v3_spec_search.jld2"); summary)
println("\nDone.")
