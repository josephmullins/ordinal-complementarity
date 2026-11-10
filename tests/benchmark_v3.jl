"""
V3 performance benchmarks:
  1. V2 E-step vs V3 E-step (r_P=3, r_ω=3) — expect ~9x slowdown
  2. V3 M-step component breakdown
  3. Full V3 EM iteration timing
  4. Scaling: V3 E-step with r_Z = 1, 4, 9
"""

using LinearAlgebra, Random, Statistics, Dates

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3.jl"))
include(joinpath(@__DIR__, "..", "src", "prepare_data.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod_v3.jl"))

# ─────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────

println("Loading datasets...")
datasets = load_all_datasets()

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

const TRANSITION_GROUPS = [[1,2,3],[4,5,6]]

panel_data = prepare_panel_data(datasets;
    age_ranges   = PANEL_AGE_RANGES,
    invest_items = PANEL_INVEST_ITEMS,
    r_θ = 4, r_I = 5,
)

(; r_θ, r_I, T, N) = panel_data

println("\n" * "="^60)
println("  V3 PERFORMANCE BENCHMARKS")
println("  $(Dates.now())")
println("  N=$N, T=$T, r_θ=$r_θ, r_I=$r_I")
println("="^60)

# Helper for timing
function bench(f, n_runs::Int=3; warmup::Bool=true)
    warmup && f()  # compile
    times = [(@elapsed f()) for _ in 1:n_runs]
    return (median=Statistics.median(times), min=minimum(times), max=maximum(times))
end

# ═════════════════════════════════════════════════════════════════
# BENCHMARK 1: V2 E-step baseline
# ═════════════════════════════════════════════════════════════════

println("\n" * "-"^60)
println("  BENCHMARK 1: V2 E-step baseline")
println("-"^60)

# Initialize V2 params
rng = MersenneTwister(42)
config_v2 = (emission_θ=:ordered_probit, emission_I=:ordered_probit,
             transition=:robit_nn, K=4, ν=7.0, ccp=:ordered_probit_simple)

probit_θ = [initialize_ordered_probit(panel_data.κ_θ[t], r_θ; rng) for t in 1:T]
P_θ      = [emission_matrices_from_probit(probit_θ[t], r_θ) for t in 1:T]
probit_I = [initialize_ordered_probit(panel_data.κ_I[t], r_I; rng) for t in 1:T]
P_I      = [emission_matrices_from_probit(probit_I[t], r_I) for t in 1:T]

trans_v2 = [initialize_robit_nn(r_θ, 4; rng, r_I, ν=7.0) for _ in 1:T-1]
for group in TRANSITION_GROUPS; for t in group[2:end]; trans_v2[t] = trans_v2[group[1]]; end; end
Π_v2 = [_build_robit_nn_pi(pack_robit_nn(trans_v2[t]), r_θ, r_I, 4, 7.0) for t in 1:T-1]
CCP_v2 = [let c = zeros(r_I, r_I, r_θ); for kθ in 1:r_θ, kI in 1:r_I; c[:, kI, kθ] .= 1.0/r_I; end; c; end for _ in 1:T-1]
π0_v2 = ones(r_θ, r_I) / (r_θ * r_I)
params_v2 = (; Π=Π_v2, CCP=CCP_v2, π0=π0_v2, P_θ, P_I, r_θ, r_I, T)

γ_v2     = zeros(T, r_θ, r_I, N)
ξ_sum_v2 = [zeros(r_θ, r_I, r_θ, r_I) for _ in 1:T-1]

t_v2 = bench(() -> expectation_step_mp!(γ_v2, ξ_sum_v2, params_v2, panel_data), 3)
println("  V2 E-step: median=$(round(t_v2.median; digits=3))s, " *
        "min=$(round(t_v2.min; digits=3))s, max=$(round(t_v2.max; digits=3))s")

# ═════════════════════════════════════════════════════════════════
# BENCHMARK 2: V3 E-step scaling with r_Z
# ═════════════════════════════════════════════════════════════════

println("\n" * "-"^60)
println("  BENCHMARK 2: V3 E-step scaling with r_Z")
println("-"^60)

for (r_P, r_ω) in [(1,1), (2,2), (3,3)]
    r_Z = r_P * r_ω
    data_v3 = merge(panel_data, (; r_P, r_ω))
    config = (K=4, ν=7.0, n_groups=2)
    rng_v3 = MersenneTwister(42)

    pv3, psk = initialize_params_v3(data_v3, config; rng=rng_v3)

    # Bin income and allocate
    bm, bv = bin_income(data_v3.Y_income; n_bins=20)
    nb = length(bv)
    γ_agg      = zeros(T, r_θ, r_I, N)
    xi_sum_z   = [[zeros(r_θ, r_I, r_θ, r_I) for _ in 1:T-1] for _ in 1:r_Z]
    w_z        = zeros(r_Z, N)
    gamma_z_t1 = zeros(r_θ, r_I, r_Z, N)
    ccp_init_ss  = zeros(r_θ, r_I, r_Z, nb)
    ccp_trans_ss = zeros(r_I, r_I, r_θ, r_Z, T-1, nb)
    pi0_ss     = zeros(r_θ, r_P, r_ω)

    θ_grid = collect(1.0:r_θ); I_grid = collect(1.0:r_I)
    P_grid = collect(1.0:r_P); ω_grid = collect(1.0:r_ω)

    Pi_z_pre = precompute_Pi_z(pv3.trans_specs, TRANSITION_GROUPS,
                                 r_θ, r_I, r_P, r_ω; θ_grid, I_grid, P_grid, ω_grid)
    CCP_i, CCP_t = precompute_CCP_z(pv3.ccp_spec, r_Z, r_P, r_ω, bv, r_θ, r_I;
                                       θ_grid, I_grid, P_grid, ω_grid)

    function run_estep()
        expectation_step_v3!(γ_agg, xi_sum_z, w_z, gamma_z_t1,
                              ccp_init_ss, ccp_trans_ss, pi0_ss,
                              Pi_z_pre, CCP_i, CCP_t, bm,
                              pv3, psk, data_v3;
                              θ_grid, I_grid, P_grid, ω_grid)
    end

    t_v3 = bench(run_estep, 3)
    ratio = t_v3.median / t_v2.median
    println("  r_Z=$r_Z (r_P=$r_P, r_ω=$r_ω): median=$(round(t_v3.median; digits=3))s, " *
            "ratio vs V2=$(round(ratio; digits=2))x")
end

# ═════════════════════════════════════════════════════════════════
# BENCHMARK 3: V3 M-step component breakdown (r_P=3, r_ω=3)
# ═════════════════════════════════════════════════════════════════

println("\n" * "-"^60)
println("  BENCHMARK 3: V3 M-step components (r_P=3, r_ω=3)")
println("-"^60)

r_P = 3; r_ω = 3; r_Z = r_P * r_ω
data_v3 = merge(panel_data, (; r_P, r_ω))
config = (K=4, ν=7.0, n_groups=2)
rng_v3 = MersenneTwister(42)
pv3, psk = initialize_params_v3(data_v3, config; rng=rng_v3)

# Bin income and allocate
bm3, bv3 = bin_income(data_v3.Y_income; n_bins=20)
nb3 = length(bv3)
γ_agg      = zeros(T, r_θ, r_I, N)
xi_sum_z   = [[zeros(r_θ, r_I, r_θ, r_I) for _ in 1:T-1] for _ in 1:r_Z]
w_z        = zeros(r_Z, N)
gamma_z_t1 = zeros(r_θ, r_I, r_Z, N)
ccp_init_ss  = zeros(r_θ, r_I, r_Z, nb3)
ccp_trans_ss = zeros(r_I, r_I, r_θ, r_Z, T-1, nb3)
pi0_ss     = zeros(r_θ, r_P, r_ω)

θ_grid = collect(1.0:r_θ); I_grid = collect(1.0:r_I)
P_grid = collect(1.0:r_P); ω_grid = collect(1.0:r_ω)

Pi_z_pre = precompute_Pi_z(pv3.trans_specs, TRANSITION_GROUPS,
                             r_θ, r_I, r_P, r_ω; θ_grid, I_grid, P_grid, ω_grid)
CCP_i3, CCP_t3 = precompute_CCP_z(pv3.ccp_spec, r_Z, r_P, r_ω, bv3, r_θ, r_I;
                                     θ_grid, I_grid, P_grid, ω_grid)
expectation_step_v3!(γ_agg, xi_sum_z, w_z, gamma_z_t1,
                      ccp_init_ss, ccp_trans_ss, pi0_ss,
                      Pi_z_pre, CCP_i3, CCP_t3, bm3,
                      pv3, psk, data_v3;
                      θ_grid, I_grid, P_grid, ω_grid)

# Benchmark individual M-step components
mstep_opts = DEFAULT_OPTIM_OPTIONS

t_emission_θ = bench(() -> parametric_mstep_emission_mp(
    psk.probit_θ, γ_agg, data_v3, :theta, r_θ; options=mstep_opts), 3)
println("  Emission θ:    $(round(t_emission_θ.median; digits=3))s")

t_emission_I = bench(() -> parametric_mstep_emission_mp(
    psk.probit_I, γ_agg, data_v3, :I, r_I; options=mstep_opts), 3)
println("  Emission I:    $(round(t_emission_I.median; digits=3))s")

t_transition = bench(() -> mstep_transition_v3_grouped(
    pv3.trans_specs, xi_sum_z, TRANSITION_GROUPS,
    r_θ, r_I, r_P, r_ω; θ_grid, I_grid, P_grid, ω_grid, options=mstep_opts), 3)
println("  Transition:    $(round(t_transition.median; digits=3))s")

t_ccp = bench(() -> begin
    x0 = pack_ccp_v3(pv3.ccp_spec)
    obj = x -> ccp_negll_v3_binned(x, ccp_init_ss, ccp_trans_ss, bv3,
                                     r_θ, r_I, r_P, r_ω, T; θ_grid, I_grid, P_grid, ω_grid)
    optimize(obj, x0, LBFGS(), mstep_opts; autodiff=:forward)
end, 3)
println("  CCP:           $(round(t_ccp.median; digits=3))s")

t_pi0 = bench(() -> mstep_pi0(pi0_ss), 3)
println("  π0:            $(round(t_pi0.median; digits=6))s")

t_qp = bench(() -> mstep_Q_P(w_z, data_v3, r_P, r_ω), 3)
println("  Q_P:           $(round(t_qp.median; digits=3))s")

t_income = bench(() -> update_income_model(w_z, data_v3.Y_income, r_P, r_ω; P_grid, ω_grid), 3)
println("  Income model:  $(round(t_income.median; digits=3))s")

t_mstep_total = t_emission_θ.median + t_emission_I.median + t_transition.median +
                t_ccp.median + t_pi0.median + t_qp.median + t_income.median
println("  M-step total:  $(round(t_mstep_total; digits=3))s")

# ═════════════════════════════════════════════════════════════════
# BENCHMARK 4: Full EM iteration (r_P=3, r_ω=3)
# ═════════════════════════════════════════════════════════════════

println("\n" * "-"^60)
println("  BENCHMARK 4: Full EM iteration (r_P=3, r_ω=3)")
println("-"^60)

t_em = bench(() -> expectation_maximization_v3(
    data_v3, pv3, psk;
    transition_groups=TRANSITION_GROUPS,
    maxiter=1, tol=0.0, verbose=false), 3)
println("  1 EM iteration: median=$(round(t_em.median; digits=3))s")

# ═════════════════════════════════════════════════════════════════
# BENCHMARK 5: Memory estimate
# ═════════════════════════════════════════════════════════════════

println("\n" * "-"^60)
println("  BENCHMARK 5: Memory estimates (r_P=3, r_ω=3)")
println("-"^60)

bytes_γ = sizeof(γ_agg)
bytes_ξ = sum(sum(sizeof(x) for x in xi_sum_z[z]) for z in 1:r_Z)
bytes_wz = sizeof(w_z)
bytes_gamma_t1 = sizeof(gamma_z_t1)
bytes_ccp_init = sizeof(ccp_init_ss)
bytes_ccp_trans = sizeof(ccp_trans_ss)
bytes_total = bytes_γ + bytes_ξ + bytes_wz + bytes_gamma_t1 + bytes_ccp_init + bytes_ccp_trans

println("  γ_agg:         $(round(bytes_γ / 1e6; digits=1)) MB")
println("  ξ_sum_z:       $(round(bytes_ξ / 1e6; digits=1)) MB")
println("  w_z:           $(round(bytes_wz / 1e6; digits=1)) MB")
println("  gamma_z_t1:    $(round(bytes_gamma_t1 / 1e6; digits=1)) MB")
println("  ccp_init_ss:   $(round(bytes_ccp_init / 1e6; digits=3)) MB")
println("  ccp_trans_ss:  $(round(bytes_ccp_trans / 1e6; digits=3)) MB")
println("  Total buffers: $(round(bytes_total / 1e6; digits=1)) MB")

println("\n" * "="^60)
println("  BENCHMARKS COMPLETE")
println("="^60)
