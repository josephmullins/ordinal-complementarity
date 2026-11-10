using LinearAlgebra, Random, Statistics, JLD2, Optim, ForwardDiff, Printf

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3.jl"))
include(joinpath(@__DIR__, "..", "src", "prepare_data.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod_v3.jl"))

# ─────────────────────────────────────────────────────────────────
# Complete-cases CCP diagnostic
#   Restrict the sample to children with observed income in ALL 7
#   periods.  Rerun the per-period CCP fit (contemporaneous log y,
#   β₂ = 0) and see how β₅ moves vs the full-sample diagnostic.
#   Purpose: gauge the attenuation from missing-is-zero coding.
# ─────────────────────────────────────────────────────────────────

const R_P  = 3
const R_ω  = 3
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
const FIT_FILE = joinpath(OUTPUT_DIR, "estimation_v3_adult.jld2")
const OUT_FILE = joinpath(OUTPUT_DIR, "v3_ccp_per_period_complete_cases.jld2")

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
data_full = merge(panel_data, (; r_P = R_P, r_ω = R_ω))

(; r_θ, r_I, T, N, κ_θ, κ_I, κ_P) = panel_data
r_Z = R_P * R_ω

# ─────────────────────────────────────────────────────────────────
# Restrict to complete-income children
# ─────────────────────────────────────────────────────────────────

complete_mask = [all(!isnan, data_full.Y_income[n, :]) for n in 1:N]
idx = findall(complete_mask)
N_c = length(idx)
println("\nComplete-cases filter: $N_c / $N children with income in all $T periods")

function filter_panel_data(d, idx)
    N_new = length(idx)
    X_θ_f = [[x[idx] for x in d.X_θ[t]] for t in 1:d.T]
    X_I_f = [[x[idx] for x in d.X_I[t]] for t in 1:d.T]
    X_P_f = [x[idx] for x in d.X_P]
    Y_income_f = d.Y_income[idx, :]
    X_adult_f = [x[idx] for x in d.X_adult]
    kids_f = d.kids[idx]
    merge(d, (; N = N_new,
               X_θ = X_θ_f, X_I = X_I_f, X_P = X_P_f,
               Y_income = Y_income_f, X_adult = X_adult_f,
               kids = kids_f))
end

data_v3 = filter_panel_data(data_full, idx)
N = data_v3.N
println("  Filtered N = $N")

# ─────────────────────────────────────────────────────────────────
# Load saved fit
# ─────────────────────────────────────────────────────────────────

println("\nLoading fit from $FIT_FILE...")
saved = load(FIT_FILE)
params_v3 = saved["params_v3"]
params_skill = saved["params_skill"]
println("  LL at save (full sample): ", round(saved["best_ll"]; digits=2))

# ─────────────────────────────────────────────────────────────────
# E-step on filtered data
# ─────────────────────────────────────────────────────────────────

println("\nRunning E-step on complete-cases subset...")

bin_map, bin_values = bin_income(data_v3.Y_income; n_bins=20)
n_bins = length(bin_values)

θ_grid = collect(1.0:r_θ); I_grid = collect(1.0:r_I)
P_grid = collect(1.0:R_P); ω_grid = collect(1.0:R_ω)

γ_agg        = zeros(T, r_θ, r_I, N)
xi_sum_z     = [[zeros(r_θ, r_I, r_θ, r_I) for _ in 1:T-1] for _ in 1:r_Z]
w_z          = zeros(r_Z, N)
gamma_z_t1   = zeros(r_θ, r_I, r_Z, N)
ccp_init_ss  = zeros(r_θ, r_I, r_Z, n_bins)
ccp_trans_ss = zeros(r_I, r_I, r_θ, r_Z, T-1, n_bins)
pi0_ss       = zeros(r_θ, R_P, R_ω)

Pi_z_pre = precompute_Pi_z(params_v3.trans_specs, TRANSITION_GROUPS,
                            r_θ, r_I, R_P, R_ω;
                            θ_grid, I_grid, P_grid, ω_grid)

CCP_init_pre, CCP_trans_pre = precompute_CCP_z(
    params_v3.ccp_spec, r_Z, R_P, R_ω, bin_values, r_θ, r_I;
    θ_grid, I_grid, P_grid, ω_grid)

ll = expectation_step_v3!(
    γ_agg, xi_sum_z, w_z, gamma_z_t1, ccp_init_ss, ccp_trans_ss, pi0_ss,
    Pi_z_pre, CCP_init_pre, CCP_trans_pre, bin_map,
    params_v3, params_skill, data_v3;
    θ_grid, I_grid, P_grid, ω_grid)

println("  LL on complete cases: ", round(ll; digits=2))

# ─────────────────────────────────────────────────────────────────
# Per-period fits
# ─────────────────────────────────────────────────────────────────

function ccp_negll_period_init(vec::AbstractVector{Tv},
                                init_ss::Array{Float64,4},
                                bin_values::Vector{Float64},
                                r_θ::Int, r_I::Int, r_P::Int, r_ω::Int;
                                θ_grid, P_grid, ω_grid) where Tv
    nc = r_I - 1
    c  = _unpack_cutpoints(vec[1:nc])
    β1, β3, β4, β5 = vec[nc+1], vec[nc+2], vec[nc+3], vec[nc+4]
    nll = zero(Tv)
    n_tb = length(bin_values)
    for kω in 1:r_ω, kP in 1:r_P
        z = (kP - 1) * r_ω + kω
        rp = β3 * P_grid[kP] + β4 * ω_grid[kω]
        for b in 1:n_tb
            ip = β5 * bin_values[b]
            for kθ in 1:r_θ
                η = β1 * θ_grid[kθ] + rp + ip
                for kI in 1:r_I
                    g = init_ss[kθ, kI, z, b]
                    g < 1e-15 && continue
                    nll -= g * _ordered_probit_logprob(c, η, kI, nc)
                end
            end
        end
    end
    return nll
end

function marginalize_trans_ss(trans_ss_t::Array{Float64,5})
    m = dropdims(sum(trans_ss_t; dims=2); dims=2)
    return permutedims(m, (2, 1, 3, 4))
end

ccp0 = params_v3.ccp_spec
nc = r_I - 1
x0 = vcat(_pack_cutpoints(ccp0.c), ccp0.beta1, ccp0.beta3, ccp0.beta4, ccp0.beta5)
opt_opts = Optim.Options(show_trace=false, g_tol=1e-8, iterations=500)

results = Vector{NamedTuple}(undef, T)

for t in 1:T
    println("Fitting period $t (complete cases, β₂=0)...")
    ss_t = t == 1 ? ccp_init_ss :
                    marginalize_trans_ss(ccp_trans_ss[:, :, :, :, t-1, :])
    objt = x -> ccp_negll_period_init(x, ss_t, bin_values,
                                        r_θ, r_I, R_P, R_ω;
                                        θ_grid, P_grid, ω_grid)
    rest = optimize(objt, x0, LBFGS(), opt_opts; autodiff=:forward)
    v = Optim.minimizer(rest)

    # Inverse-Hessian SEs (treats posterior weights as fixed; ignores EM uncertainty)
    H = ForwardDiff.hessian(objt, v)
    Hinv = try
        inv(H)
    catch
        pinv(H)
    end
    se = sqrt.(max.(diag(Hinv), 0.0))

    results[t] = (
        t = t,
        c = _unpack_cutpoints(v[1:nc]),
        β1 = v[nc+1],  se_β1 = se[nc+1],
        β3 = v[nc+2],  se_β3 = se[nc+2],
        β4 = v[nc+3],  se_β4 = se[nc+3],
        β5 = v[nc+4],  se_β5 = se[nc+4],
        ll = -Optim.minimum(rest),
    )
end

mean_logy = [mean(filter(!isnan, data_v3.Y_income[:, t])) for t in 1:T]

# ─────────────────────────────────────────────────────────────────
# Pooled CCP on the SAME complete-cases subset (apples-to-apples)
# Aggregate SS across periods:
#   init_ss  (t=1) already in init-shape (θ, I, z, b)
#   trans_ss[t] (t=2..T) marginalized over I_{t-1} → init-shape
# ─────────────────────────────────────────────────────────────────

ss_pooled = copy(ccp_init_ss)
for t in 2:T
    ss_pooled .+= marginalize_trans_ss(ccp_trans_ss[:, :, :, :, t-1, :])
end

println("Fitting pooled CCP on complete-cases subset...")
obj_pool = x -> ccp_negll_period_init(x, ss_pooled, bin_values,
                                        r_θ, r_I, R_P, R_ω;
                                        θ_grid, P_grid, ω_grid)
res_pool = optimize(obj_pool, x0, LBFGS(), opt_opts; autodiff=:forward)
v_pool = Optim.minimizer(res_pool)
H_pool = ForwardDiff.hessian(obj_pool, v_pool)
Hinv_pool = try inv(H_pool) catch; pinv(H_pool) end
se_pool = sqrt.(max.(diag(Hinv_pool), 0.0))

pooled_cc = (
    c  = _unpack_cutpoints(v_pool[1:nc]),
    β1 = v_pool[nc+1],  se_β1 = se_pool[nc+1],
    β3 = v_pool[nc+2],  se_β3 = se_pool[nc+2],
    β4 = v_pool[nc+3],  se_β4 = se_pool[nc+3],
    β5 = v_pool[nc+4],  se_β5 = se_pool[nc+4],
)

# ─────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────

ages_label = ["(0,1)", "(2,3)", "(4,5)", "(6,7)", "(8,9)", "(10,11)", "(12,13)"]

println("\n" * "="^105)
println("COMPLETE-CASES CCP — N_c = $N  (children with all 7 periods observed)")
println("  Survivor: P(I_t > k) = Φ(β₁·θ_t + β₃·θ^P + β₄·ω + β₅·log y_t − c_k), β₂=0")
println("  SEs: inverse-Hessian (posterior weights treated as fixed; does not account for EM latent uncertainty)")
println("="^105)
@printf("%-4s %-8s %9s %9s %9s %9s %10s %8s %7s\n",
        "t", "ages", "mean logy", "β₁(θ)", "β₃(θ^P)", "β₄(ω)", "β₅(logy)", "se(β₅)", "t-stat")
println("-"^105)
for r in results
    tstat = r.β5 / max(r.se_β5, 1e-12)
    @printf("%-4d %-8s %9.3f %9.3f %9.3f %9.3f %10.4f %8.4f %7.2f\n",
            r.t, ages_label[r.t], mean_logy[r.t],
            r.β1, r.β3, r.β4, r.β5, r.se_β5, tstat)
end
println("-"^105)
tstat_p = pooled_cc.β5 / max(pooled_cc.se_β5, 1e-12)
@printf("%-4s %-8s %9s %9.3f %9.3f %9.3f %10.4f %8.4f %7.2f\n",
        "pool", "cc", "—",
        pooled_cc.β1, pooled_cc.β3, pooled_cc.β4,
        pooled_cc.β5, pooled_cc.se_β5, tstat_p)
println("="^105)
println("\nPooled V3 CCP (full sample, saved fit, for reference):")
@printf("    β₁ = %.3f,  β₃ = %.3f,  β₄ = %.3f,  β₅ = %.4f\n",
        ccp0.beta1, ccp0.beta3, ccp0.beta4, ccp0.beta5)

# ─────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────

jldsave(OUT_FILE;
    results = results,
    pooled_cc = pooled_cc,
    ccp0_beta = (ccp0.beta1, ccp0.beta2, ccp0.beta3, ccp0.beta4, ccp0.beta5),
    ccp0_c    = ccp0.c,
    ll_complete = ll,
    N_complete = N,
    mean_logy = mean_logy,
)
println("\nSaved to $OUT_FILE")
