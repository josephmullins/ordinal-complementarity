using LinearAlgebra, Random, Statistics, JLD2, Optim

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3.jl"))
include(joinpath(@__DIR__, "..", "src", "prepare_data.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod_v3.jl"))

const R_P  = 3; const R_ω  = 3; const NU = 7.0
const TRANSITION_GROUPS = [[1, 2, 3], [4, 5, 6]]
const OUTPUT_DIR = joinpath(dirname(@__DIR__), "output")

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

# Load data
println("Loading datasets...")
datasets = load_all_datasets()
panel_data = prepare_panel_data(datasets;
    age_ranges=PANEL_AGE_RANGES, invest_items=PANEL_INVEST_ITEMS, r_θ=4, r_I=5)
data_v3 = merge(panel_data, (; r_P=R_P, r_ω=R_ω))
(; r_θ, r_I, T, N) = panel_data
r_Z = R_P * R_ω

# Load K=3 results
println("Loading K=3 results...")
k3 = load(joinpath(OUTPUT_DIR, "estimation_v3_k3.jld2"))
pv3_k3 = k3["params_v3"]
psk_k3 = k3["params_skill"]
ll_k3 = k3["best_ll"]

# Need ξ_sum_z — rerun E-step from K=3 params
println("Running E-step to get K=3 posteriors...")
θ_grid = collect(1.0:r_θ); I_grid = collect(1.0:r_I)
P_grid = collect(1.0:R_P); ω_grid = collect(1.0:R_ω)
bin_map, bin_values = bin_income(data_v3.Y_income; n_bins=20)
n_total_bins = length(bin_values)

γ_agg      = zeros(T, r_θ, r_I, N)
xi_sum_z   = [[zeros(r_θ, r_I, r_θ, r_I) for _ in 1:T-1] for _ in 1:r_Z]
w_z        = zeros(r_Z, N)
gamma_z_t1 = zeros(r_θ, r_I, r_Z, N)
ccp_init_ss  = zeros(r_θ, r_I, r_Z, n_total_bins)
ccp_trans_ss = zeros(r_I, r_I, r_θ, r_Z, T-1, n_total_bins)
pi0_ss     = zeros(r_θ, R_P, R_ω)

Pi_z_pre = precompute_Pi_z(pv3_k3.trans_specs, TRANSITION_GROUPS,
                             r_θ, r_I, R_P, R_ω; θ_grid, I_grid, P_grid, ω_grid)
CCP_init_pre, CCP_trans_pre = precompute_CCP_z(
    pv3_k3.ccp_spec, r_Z, R_P, R_ω, bin_values, r_θ, r_I;
    θ_grid, I_grid, P_grid, ω_grid)
ll_check = expectation_step_v3!(
    γ_agg, xi_sum_z, w_z, gamma_z_t1, ccp_init_ss, ccp_trans_ss, pi0_ss,
    Pi_z_pre, CCP_init_pre, CCP_trans_pre, bin_map,
    pv3_k3, psk_k3, data_v3; θ_grid, I_grid, P_grid, ω_grid)
println("K=3 ll check: $(round(ll_check; digits=2))")

# Warm-start K=4 transitions
K_new = 4
println("\n" * "="^60)
println("  K=$K_new: warm start from K=3")
println("="^60)

println("  Fitting K=$K_new transitions to K=3 posteriors...")
best_specs = nothing
best_obj = Inf
for i in 1:15
    rng = MersenneTwister(400 + i)
    init_specs = [initialize_robit_nn_v3(r_θ, K_new; rng, ν=NU) for _ in 1:2]
    try
        new_specs = mstep_transition_v3_grouped(
            init_specs, xi_sum_z, TRANSITION_GROUPS,
            r_θ, r_I, R_P, R_ω; θ_grid, I_grid, P_grid, ω_grid)
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
            obj += transition_negll_robit_nn_v3(φ, Q_sum_z, r_θ, r_I, K_new, NU,
                                                  R_P, R_ω; θ_grid, I_grid, P_grid, ω_grid)
        end
        println("    Init $i/15: negll = $(round(obj; digits=2))")
        if obj < best_obj
            global best_obj = obj
            global best_specs = new_specs
        end
    catch e
        println("    Init $i/15: failed ($(typeof(e)))")
    end
end

if best_specs === nothing
    error("All K=$K_new initializations failed.")
end

# Run EM
pv3_init = (; pv3_k3..., trans_specs = best_specs)
pv3_k4, psk_k4, γ_k4, ξz_k4, wz_k4, ll_k4 = expectation_maximization_v3(
    data_v3, pv3_init, psk_k3;
    transition_groups = TRANSITION_GROUPS,
    maxiter = 2000, tol = 1e-4, verbose = true)

n_trans_per_group = (r_θ - 1) + 6 * K_new
# Simplified param count
n_params_k4 = 1285 + 12  # K=3 params + 12 more (6 per group)
aic_k4 = -2 * ll_k4 + 2 * n_params_k4
bic_k4 = -2 * ll_k4 + log(N) * n_params_k4

println("\n  K=$K_new: ll=$(round(ll_k4;digits=2)), params=$n_params_k4")
println("    AIC=$(round(aic_k4;digits=2)), BIC=$(round(bic_k4;digits=2))")

outpath = joinpath(OUTPUT_DIR, "estimation_v3_k4.jld2")
jldsave(outpath; params_v3=pv3_k4, params_skill=psk_k4,
        gamma_agg=γ_k4, w_z=wz_k4,
        best_ll=ll_k4, data_dims=(; N, T, r_θ, r_I, r_P=R_P, r_ω=R_ω, K=K_new),
        age_ranges=PANEL_AGE_RANGES, aic=aic_k4, bic=bic_k4, n_params=n_params_k4)
println("  Saved to $outpath")
display_results_v3(pv3_k4, psk_k4, data_v3)

# Compare
println("\n" * "="^60)
println("  K=3 vs K=4")
println("="^60)
println("  K=3: ll=$(round(ll_k3;digits=2)), BIC=$(round(-2*ll_k3 + log(N)*1285;digits=2))")
println("  K=4: ll=$(round(ll_k4;digits=2)), BIC=$(round(bic_k4;digits=2))")
bic_k3_val = -2*ll_k3 + log(N)*1285
preferred = bic_k4 < bic_k3_val ? "K=4" : "K=3"
println("  BIC prefers: $preferred")
println("\nDone.")
