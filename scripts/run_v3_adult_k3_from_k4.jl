using LinearAlgebra, Random, Statistics, JLD2, Optim

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3.jl"))
include(joinpath(@__DIR__, "..", "src", "prepare_data.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod_v3.jl"))

const R_P = 3; const R_ω = 3; const NU = 7.0
const TRANSITION_GROUPS = [[1, 2, 3], [4, 5, 6]]
const OUTPUT_DIR = joinpath(dirname(@__DIR__), "output")

const PANEL_AGE_RANGES = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13)]
const PANEL_INVEST_ITEMS = [
    [:HowOftenRead,:HowManyBooks,:HowManyToysA,:HowManyToysB,:OutOfHouse,:HowOftenTalks,:HowOftenGrocery],
    [:HowOftenRead,:HowManyBooks,:HowManyToysA,:HowManyToysB,:OutOfHouse,:HowOftenTalks,:HowOftenGrocery,:HowManyZines,:HowOftenOuting,:FoodChoice,:LearnNumbers,:LearnAlphabet,:LearnColors,:LearnShapes,:RecordTapeCD],
    [:HowOftenRead,:HowManyBooks,:HowManyZines,:HowOftenOuting,:HowOftenMuseum,:RecordTapeCD,:FoodChoice],
    [:HowOftenRead,:HowManyBooks,:HowOftenMuseum,:HowOftenTheater,:HowOftenFamily,:HowOftenOutdoors,:ChildMusic,:SpecialActivities,:Hobbies,:EnjoyRead,:DailyNewspaper,:DiscussTV,:GradesTeacher,:GradesTalk,:GradesHelp,:GradesStudy],
    [:HowOftenRead,:HowManyBooks,:HowOftenMuseum,:HowOftenTheater,:HowOftenFamily,:HowOftenOutdoors,:HowOftenFather,:ChildMusic,:SpecialActivities,:Hobbies,:EnjoyRead,:DailyNewspaper,:DiscussTV,:GradesTeacher,:GradesTalk,:GradesHelp,:GradesStudy],
    [:HowManyBooks,:HowOftenMuseum,:HowOftenTheater,:HowOftenFamily,:HowOftenOutdoors,:HowOftenFather,:ChildMusic,:SpecialActivities,:Hobbies,:EnjoyRead,:DailyNewspaper,:DiscussTV,:GradesTeacher,:GradesTalk,:GradesHelp,:GradesStudy,:ToldPositive,:HowOftenClean,:HowOftenChores,:HowOftenTime],
    [:HowManyBooks,:HowOftenMuseum,:HowOftenTheater,:HowOftenFamily,:HowOftenOutdoors,:HowOftenFather,:ChildMusic,:SpecialActivities,:Hobbies,:EnjoyRead,:DailyNewspaper,:DiscussTV,:GradesTeacher,:GradesTalk,:GradesHelp,:GradesStudy,:ToldPositive,:HowOftenClean,:HowOftenChores,:HowOftenTime],
]

# Load data
println("Loading datasets...")
datasets = load_all_datasets()
panel_data = prepare_panel_data(datasets;
    age_ranges=PANEL_AGE_RANGES, invest_items=PANEL_INVEST_ITEMS, r_θ=4, r_I=5)
data_v3 = merge(panel_data, (; r_P=R_P, r_ω=R_ω))
(; r_θ, r_I, T, N) = panel_data
r_Z = R_P * R_ω

# Load K=4 results
println("\nLoading K=4 results...")
k4 = load(joinpath(OUTPUT_DIR, "estimation_v3_adult.jld2"))
pv3_k4 = k4["params_v3"]
psk_k4 = k4["params_skill"]
ll_k4 = k4["best_ll"]
println("  K=4 ll: $(round(ll_k4; digits=2))")

# Run E-step to get K=4 posteriors
println("Running E-step for K=4 posteriors...")
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

Pi_z_pre = precompute_Pi_z(pv3_k4.trans_specs, TRANSITION_GROUPS,
                             r_θ, r_I, R_P, R_ω; θ_grid, I_grid, P_grid, ω_grid)
CCP_init_pre, CCP_trans_pre = precompute_CCP_z(
    pv3_k4.ccp_spec, r_Z, R_P, R_ω, bin_values, r_θ, r_I;
    θ_grid, I_grid, P_grid, ω_grid)
ll_check = expectation_step_v3!(
    γ_agg, xi_sum_z, w_z, gamma_z_t1, ccp_init_ss, ccp_trans_ss, pi0_ss,
    Pi_z_pre, CCP_init_pre, CCP_trans_pre, bin_map,
    pv3_k4, psk_k4, data_v3; θ_grid, I_grid, P_grid, ω_grid)
println("  K=4 ll check: $(round(ll_check; digits=2))")

# Warm-start K=3 transitions from K=4 posteriors
K_new = 3
println("\n" * "="^60)
println("  Fitting K=$K_new transitions to K=4 posteriors")
println("="^60)

best_specs = nothing
best_obj = Inf
for i in 1:15
    rng = MersenneTwister(500 + i)
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

# Build K=3 params: keep everything from K=4 except transitions
pv3_init = (; pv3_k4..., trans_specs = best_specs)

println("\n" * "="^60)
println("  Running V3 EM with K=$K_new (warm-started from K=4)")
println("="^60)

pv3_k3, psk_k3, γ_k3, ξz_k3, wz_k3, ll_k3 = expectation_maximization_v3(
    data_v3, pv3_init, psk_k4;
    transition_groups = TRANSITION_GROUPS,
    maxiter = 2000, tol = 1e-4, verbose = true)

println("\n  K=3 ll: $(round(ll_k3; digits=2))")
println("  K=4 ll: $(round(ll_k4; digits=2))")
println("  Δll: $(round(ll_k3 - ll_k4; digits=2))")

# BIC comparison
n_params_k3 = 0  # placeholder
n_params_k4 = 0
delta_params = 12  # K=4 has 12 more transition params
bic_penalty = delta_params * log(N)
delta_bic = -2*(ll_k4 - ll_k3) + bic_penalty
println("\n  ΔBIC (K=4 - K=3) = $(round(delta_bic; digits=2)) (negative favors K=4)")
println("  BIC prefers: $(delta_bic < 0 ? "K=4" : "K=3")")

outpath = joinpath(OUTPUT_DIR, "estimation_v3_adult_k3.jld2")
jldsave(outpath; params_v3=pv3_k3, params_skill=psk_k3,
        gamma_agg=γ_k3, w_z=wz_k3,
        best_ll=ll_k3, data_dims=(; N, T, r_θ, r_I, r_P=R_P, r_ω=R_ω, K=K_new),
        age_ranges=PANEL_AGE_RANGES)
println("  Saved to $outpath")

display_results_v3(pv3_k3, psk_k3, data_v3)
println("\nDone.")
