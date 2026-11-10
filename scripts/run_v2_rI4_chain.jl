using LinearAlgebra, Random, Statistics, JLD2, Optim, Printf

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "prepare_data.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod.jl"))
include(joinpath(@__DIR__, "..", "src", "inference.jl"))

# ─────────────────────────────────────────────────────────────────
# V2 chain at r_I=4 with relaxed tolerance
#   Stage A: surgery V2 K=4 (r_I=5) → V2 K=4 (r_I=4) + short EM
#   Stage B: warm K=5 from K=4 (r_I=4) + short EM
# ─────────────────────────────────────────────────────────────────

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
const V2_K4_FIT_RI5 = joinpath(OUTPUT_DIR, "estimation_multiperiod_robit_k4_grouped.jld2")
const V2_K3_OUT_RI4 = joinpath(OUTPUT_DIR, "estimation_multiperiod_robit_k3_grouped_rI4.jld2")
const V2_K4_OUT_RI4 = joinpath(OUTPUT_DIR, "estimation_multiperiod_robit_k4_grouped_rI4.jld2")
const V2_K5_OUT_RI4 = joinpath(OUTPUT_DIR, "estimation_multiperiod_robit_k5_grouped_rI4.jld2")

# Relaxed tolerances for the mini-chain
const MINI_TOL = 5e-4
const MINI_MAXITER = 800
const N_RESTARTS_TRANSITION = 5

# ─────────────────────────────────────────────────────────────────
# Load data at r_I=4
# ─────────────────────────────────────────────────────────────────

println("Loading datasets...")
datasets = load_all_datasets()

println("\n  Data Preparation (r_I=4):")
data = prepare_panel_data(datasets;
    age_ranges = PANEL_AGE_RANGES,
    invest_items = PANEL_INVEST_ITEMS,
    r_θ = 4, r_I = 4,
)

(; r_θ, r_I, T, N) = data
println("  N = $N, T = $T, r_θ = $r_θ, r_I = $r_I")

# ─────────────────────────────────────────────────────────────────
# Load V2 K=4 r_I=5 fit for surgery
# ─────────────────────────────────────────────────────────────────

println("\nLoading V2 K=4 r_I=5 fit from $V2_K4_FIT_RI5 ...")
v2_ri5 = load(V2_K4_FIT_RI5)
v2_params_ri5 = v2_ri5["best_params"]
v2_specs_ri5  = v2_ri5["best_specs"]
v2_ll_ri5     = v2_ri5["best_ll"]
println("  V2 K=4 r_I=5 LL: $(round(v2_ll_ri5; digits=2))")

# ─────────────────────────────────────────────────────────────────
# Stage A: surgery V2 K=4 r_I=5 → V2 K=4 r_I=4
# ─────────────────────────────────────────────────────────────────

println("\n" * "="^60)
println("  STAGE A: V2 K=4 surgery r_I=5 → r_I=4")
println("="^60)

# Transitions: keep robit NN parameters as-is (α₂·I still well-defined for I∈{1..4})
trans_specs_ri4 = [deepcopy(spec) for spec in v2_specs_ri5[:transition]]

# Skill emissions: r_θ unchanged, reuse
probit_θ_ri4 = v2_specs_ri5[:probit_θ]
P_θ_ri4 = [emission_matrices_from_probit(probit_θ_ri4[t], r_θ) for t in 1:T]

# Investment emissions: keep OrderedProbitEmission params (tau,alpha), rebuild with r_I=4
probit_I_ri4 = v2_specs_ri5[:probit_I]
P_I_ri4 = [emission_matrices_from_probit(probit_I_ri4[t], r_I) for t in 1:T]

# CCP: drop c_4 (top cutpoint was collapsing with c_3), keep β₁
ccp_specs_ri4 = Vector{OrderedProbitCCPSimple}(undef, T-1)
for t in 1:T-1
    ccp_old = v2_specs_ri5[:ccp][t]
    c_new = ccp_old.c[1:r_I-1]  # drop top cutpoint
    ccp_specs_ri4[t] = OrderedProbitCCPSimple(c_new, ccp_old.beta1)
end

# π0: drop last I column from V2's r_θ × r_I=5 matrix, renormalize
π0_old = v2_params_ri5.π0
π0_ri4 = π0_old[:, 1:r_I]
π0_ri4 ./= sum(π0_ri4)

# Build Π and CCP arrays
Π_ri4 = Vector{Array{Float64,3}}(undef, T-1)
for t in 1:T-1
    spec = trans_specs_ri4[t]
    Π_ri4[t] = _build_robit_nn_pi(pack_robit_nn(spec), r_θ, r_I, spec.K, spec.ν)
end
CCP_ri4 = Vector{Array{Float64,3}}(undef, T-1)
for t in 1:T-1
    CCP_ri4[t] = ccp_from_ordered_probit_simple(ccp_specs_ri4[t], r_θ, r_I)
end

params_init = (; Π=Π_ri4, CCP=CCP_ri4, π0=π0_ri4, P_θ=P_θ_ri4, P_I=P_I_ri4,
                 r_θ, r_I, T)
specs_init = Dict{Symbol,Any}(
    :probit_θ => probit_θ_ri4,
    :probit_I => probit_I_ri4,
    :ccp      => ccp_specs_ri4,
    :transition => trans_specs_ri4,
)

# Verify warm-start LL
γ_tmp = zeros(T, r_θ, r_I, N)
ξ_tmp = [zeros(r_θ, r_I, r_θ, r_I) for _ in 1:T-1]
ll_surgery = expectation_step_mp!(γ_tmp, ξ_tmp, params_init, data)
println("  Surgery warm-start LL: $(round(ll_surgery; digits=2))")

config_k4 = (
    emission_θ  = :ordered_probit,
    emission_I  = :ordered_probit,
    transition  = :robit_nn,
    K           = 4,
    ν           = 7.0,
    ccp         = :ordered_probit_simple,
)

println("\n  Running EM (K=4, r_I=4, tol=$MINI_TOL, maxiter=$MINI_MAXITER)...")
params_k4, γ_k4, ξ_sum_k4, specs_k4 = expectation_maximization_mp_parametric(
    data, params_init, specs_init, config_k4;
    maxiter = MINI_MAXITER, tol = MINI_TOL, verbose = true,
    transition_groups = TRANSITION_GROUPS,
)
ll_k4 = expectation_step_mp!(γ_k4, ξ_sum_k4, params_k4, data)
println("  V2 K=4 r_I=4 LL: $(round(ll_k4; digits=2))")

jldsave(V2_K4_OUT_RI4;
    best_params = params_k4, best_specs = specs_k4, best_ll = ll_k4,
    best_gamma = γ_k4, best_xi_sum = ξ_sum_k4,
    data, transition_groups = TRANSITION_GROUPS,
    parametric_config = config_k4)
println("  Saved to $V2_K4_OUT_RI4")

# ─────────────────────────────────────────────────────────────────
# Helper: warm from K=4 posteriors to a new K via random transition init
# ─────────────────────────────────────────────────────────────────

function warm_to_K(K_target::Int, base_params, base_specs, ξ_base, γ_tmp, ξ_tmp;
                   rng_seed::Int, tag::String, outpath::String)
    println("\n" * "="^60)
    println("  $tag: V2 K=$K_target warm from K=4 posteriors (r_I=4)")
    println("="^60)

    # Initialize K_target transitions with random restarts, fit to base posteriors
    println("  Initializing K=$K_target robit NN with $N_RESTARTS_TRANSITION restarts...")
    best_init_specs = nothing
    best_init_Π = nothing
    best_init_obj = Inf

    for i in 1:N_RESTARTS_TRANSITION
        rng_i = MersenneTwister(rng_seed + i)
        init_trans = [initialize_robit_nn(r_θ, K_target; rng=rng_i, r_I, ν=7.0) for _ in 1:T-1]

        try
            new_trans, new_Π = parametric_mstep_transition_mp_grouped(
                init_trans, ξ_base, r_θ, r_I, TRANSITION_GROUPS)

            obj = 0.0
            for (g, group) in enumerate(TRANSITION_GROUPS)
                Q_sum_pooled = _qsum_from_xi(ξ_base[group[1]])
                for t in group[2:end]
                    Q_sum_pooled = Q_sum_pooled .+ _qsum_from_xi(ξ_base[t])
                end
                φ = pack_robit_nn(new_trans[group[1]])
                ν = new_trans[group[1]].ν
                obj += transition_negll_robit_nn_fast(φ, Q_sum_pooled, r_θ, r_I, K_target, ν)
            end

            println("    Init $i/$N_RESTARTS_TRANSITION: transition negll = $(round(obj; digits=2))")
            if obj < best_init_obj
                best_init_obj = obj
                best_init_specs = new_trans
                best_init_Π = new_Π
            end
        catch e
            println("    Init $i/$N_RESTARTS_TRANSITION: failed ($(typeof(e)))")
        end
    end

    if best_init_specs === nothing
        error("All K=$K_target robit NN initializations failed.")
    end

    specs_new_init = Dict{Symbol,Any}(
        :probit_θ => base_specs[:probit_θ],
        :probit_I => base_specs[:probit_I],
        :ccp      => base_specs[:ccp],
        :transition => best_init_specs,
    )
    params_new_init = (;
        Π = best_init_Π, CCP = base_params.CCP, π0 = base_params.π0,
        P_θ = base_params.P_θ, P_I = base_params.P_I, r_θ, r_I, T
    )
    ll_init = expectation_step_mp!(γ_tmp, ξ_tmp, params_new_init, data)
    println("  K=$K_target init LL: $(round(ll_init; digits=2))")

    config_new = (
        emission_θ  = :ordered_probit,
        emission_I  = :ordered_probit,
        transition  = :robit_nn,
        K           = K_target,
        ν           = 7.0,
        ccp         = :ordered_probit_simple,
    )

    println("\n  Running EM (K=$K_target, r_I=4, tol=$MINI_TOL, maxiter=$MINI_MAXITER)...")
    params_new, γ_new, ξ_new, specs_new = expectation_maximization_mp_parametric(
        data, params_new_init, specs_new_init, config_new;
        maxiter = MINI_MAXITER, tol = MINI_TOL, verbose = true,
        transition_groups = TRANSITION_GROUPS,
    )
    ll_new = expectation_step_mp!(γ_new, ξ_new, params_new, data)
    println("  V2 K=$K_target r_I=4 LL: $(round(ll_new; digits=2))")

    jldsave(outpath;
        best_params = params_new, best_specs = specs_new, best_ll = ll_new,
        best_gamma = γ_new, best_xi_sum = ξ_new,
        data, transition_groups = TRANSITION_GROUPS,
        parametric_config = config_new)
    println("  Saved to $outpath")

    return params_new, specs_new, ll_new
end

# ─────────────────────────────────────────────────────────────────
# Stage B: V2 K=3 warm from K=4 posteriors (r_I=4)
# ─────────────────────────────────────────────────────────────────

params_k3, specs_k3, ll_k3 = warm_to_K(3, params_k4, specs_k4, ξ_sum_k4,
    γ_tmp, ξ_tmp;
    rng_seed = 400, tag = "STAGE B", outpath = V2_K3_OUT_RI4)

# ─────────────────────────────────────────────────────────────────
# Stage C: V2 K=5 warm from K=4 posteriors (r_I=4)
# ─────────────────────────────────────────────────────────────────

params_k5, specs_k5, ll_k5 = warm_to_K(5, params_k4, specs_k4, ξ_sum_k4,
    γ_tmp, ξ_tmp;
    rng_seed = 500, tag = "STAGE C", outpath = V2_K5_OUT_RI4)

# ─────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────

println("\n" * "="^60)
println("  V2 CHAIN SUMMARY (r_I=4)")
println("="^60)
println(@sprintf("  Surgery (V2 K=4 r_I=5→r_I=4 init):  LL = %.2f", ll_surgery))
println(@sprintf("  Stage A  V2 K=4 r_I=4 (converged):  LL = %.2f", ll_k4))
println(@sprintf("  Stage B  V2 K=3 r_I=4 (converged):  LL = %.2f", ll_k3))
println(@sprintf("  Stage C  V2 K=5 r_I=4 (converged):  LL = %.2f", ll_k5))

for (stage, spec_list) in [("K=3", specs_k3[:transition]), ("K=4", specs_k4[:transition]), ("K=5", specs_k5[:transition])]
    for (g, group) in enumerate(TRANSITION_GROUPS)
        spec = spec_list[group[1]]
        max_lw = maximum(abs.(spec.log_w))
        println("  $stage group $g: K=$(spec.K), max|log_w|=$(round(max_lw;digits=1))")
    end
end

println("\nDone.")
