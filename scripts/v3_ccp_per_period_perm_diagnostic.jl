using LinearAlgebra, Random, Statistics, JLD2, Optim, ForwardDiff, Printf

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3.jl"))
include(joinpath(@__DIR__, "..", "src", "prepare_data.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod_v3.jl"))

# ─────────────────────────────────────────────────────────────────
# Per-period CCP diagnostic using PERMANENT income
#   permanent log income per child := log( mean_{t obs} exp(Y_income[n,t]) )
#   — same value applied at every period for that child.
# The regime posterior (E-step) still uses contemporaneous log y;
# only the CCP binning switches to the permanent version.
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
const OUT_FILE = joinpath(OUTPUT_DIR, "v3_ccp_per_period_perm_diagnostic.jld2")

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

(; r_θ, r_I, T, N, κ_θ, κ_I, κ_P) = panel_data
r_Z = R_P * R_ω

println("  N = $N, T = $T, r_θ = $r_θ, r_I = $r_I, r_P = $R_P, r_ω = $R_ω")

# ─────────────────────────────────────────────────────────────────
# Build permanent log income per child
# ─────────────────────────────────────────────────────────────────

# Y_perm[n, t] = log( mean_{t: observed} y_{n,t} ), constant across t per child
function build_permanent_log_income(Y_log::Matrix{Float64})
    N, Tp = size(Y_log)
    Y_perm = Matrix{Float64}(undef, N, Tp)
    n_obs_per_child = zeros(Int, N)
    for n in 1:N
        y_levels = Float64[]
        for t in 1:Tp
            v = Y_log[n, t]
            isnan(v) || push!(y_levels, exp(v))
        end
        n_obs_per_child[n] = length(y_levels)
        if isempty(y_levels)
            Y_perm[n, :] .= NaN
        else
            Y_perm[n, :] .= log(mean(y_levels))
        end
    end
    return Y_perm, n_obs_per_child
end

Y_perm, n_obs_child = build_permanent_log_income(data_v3.Y_income)

n_any = sum(n_obs_child .> 0)
mean_perm = mean(filter(!isnan, Y_perm[:, 1]))
println("\nPermanent income:")
println("  Children with ≥1 income obs: $n_any / $N")
@printf("  Mean periods/child (cond. on observed): %.2f\n",
        mean(n_obs_child[n_obs_child .> 0]))
@printf("  Mean log(mean y) across children: %.3f   (level %.0f)\n",
        mean_perm, exp(mean_perm))

# ─────────────────────────────────────────────────────────────────
# Load saved fit
# ─────────────────────────────────────────────────────────────────

println("\nLoading fit from $FIT_FILE...")
saved = load(FIT_FILE)
params_v3 = saved["params_v3"]
params_skill = saved["params_skill"]
println("  LL at save: ", round(saved["best_ll"]; digits=2))

# ─────────────────────────────────────────────────────────────────
# Bin the permanent income; run E-step with the permanent bin_map
#   — data.Y_income (contemporaneous) is still used by regime_loglik
#     for the normal income likelihood; only the CCP binning changes.
# ─────────────────────────────────────────────────────────────────

println("\nBinning permanent income (20 quantile bins)...")
bin_map, bin_values = bin_income(Y_perm; n_bins=20)
n_bins = length(bin_values)
println("  bin_values (perm): ",
        round.(bin_values[1:min(end,n_bins)]; digits=3))

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

# Build CCP arrays indexed by the new (permanent) bin values
CCP_init_pre, CCP_trans_pre = precompute_CCP_z(
    params_v3.ccp_spec, r_Z, R_P, R_ω, bin_values, r_θ, r_I;
    θ_grid, I_grid, P_grid, ω_grid)

println("Running E-step with permanent-income bin_map...")
ll = expectation_step_v3!(
    γ_agg, xi_sum_z, w_z, gamma_z_t1, ccp_init_ss, ccp_trans_ss, pi0_ss,
    Pi_z_pre, CCP_init_pre, CCP_trans_pre, bin_map,
    params_v3, params_skill, data_v3;
    θ_grid, I_grid, P_grid, ω_grid)
println("  LL: $(round(ll; digits=2))   (reference: $(round(saved["best_ll"]; digits=2)))")

# ─────────────────────────────────────────────────────────────────
# Per-period CCP negll (β₂ = 0, matching V3 spec)
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

# ─────────────────────────────────────────────────────────────────
# Fit each period
# ─────────────────────────────────────────────────────────────────

ccp0 = params_v3.ccp_spec
nc = r_I - 1
x0 = vcat(_pack_cutpoints(ccp0.c), ccp0.beta1, ccp0.beta3, ccp0.beta4, ccp0.beta5)
opt_opts = Optim.Options(show_trace=false, g_tol=1e-8, iterations=500)

results = Vector{NamedTuple}(undef, T)

for t in 1:T
    println("Fitting period $t (permanent y, β₂=0)...")
    ss_t = t == 1 ? ccp_init_ss :
                    marginalize_trans_ss(ccp_trans_ss[:, :, :, :, t-1, :])
    objt = x -> ccp_negll_period_init(x, ss_t, bin_values,
                                        r_θ, r_I, R_P, R_ω;
                                        θ_grid, P_grid, ω_grid)
    rest = optimize(objt, x0, LBFGS(), opt_opts; autodiff=:forward)
    v = Optim.minimizer(rest)
    results[t] = (
        t = t,
        c = _unpack_cutpoints(v[1:nc]),
        β1 = v[nc+1],
        β3 = v[nc+2],
        β4 = v[nc+3],
        β5 = v[nc+4],
        ll = -Optim.minimum(rest),
    )
end

# ─────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────

ages_label = ["(0,1)", "(2,3)", "(4,5)", "(6,7)", "(8,9)", "(10,11)", "(12,13)"]

println("\n" * "="^90)
println("PERMANENT income CCP — period-by-period coefficients  (β₂ = 0)")
println("  Survivor: P(I_t > k) = Φ(β₁·θ_t + β₃·θ^P + β₄·ω + β₅·log ȳ_n − c_k)")
println("="^90)
@printf("%-4s %-8s %9s %9s %9s %10s\n",
        "t", "ages", "β₁(θ)", "β₃(θ^P)", "β₄(ω)", "β₅(logȳ)")
println("-"^90)
for r in results
    @printf("%-4d %-8s %9.3f %9.3f %9.3f %10.4f\n",
            r.t, ages_label[r.t], r.β1, r.β3, r.β4, r.β5)
end
println("="^90)

# Pooled reference (contemporaneous pooled from saved fit)
println("\nPooled V3 CCP (contemporaneous, for reference):")
@printf("    β₁ = %.3f,  β₃ = %.3f,  β₄ = %.3f,  β₅ = %.4f\n",
        ccp0.beta1, ccp0.beta3, ccp0.beta4, ccp0.beta5)

# Cutpoints
println("\nCutpoints c_k by period:")
@printf("%-4s %-8s %9s %9s %9s %9s\n",
        "t", "ages", "c_1", "c_2", "c_3", "c_4")
println("-"^60)
for r in results
    @printf("%-4d %-8s", r.t, ages_label[r.t])
    for ci in r.c
        @printf(" %9.3f", ci)
    end
    println()
end

# ─────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────

jldsave(OUT_FILE;
    results = results,
    ccp0_beta = (ccp0.beta1, ccp0.beta2, ccp0.beta3, ccp0.beta4, ccp0.beta5),
    ccp0_c    = ccp0.c,
    ll_perm   = ll,
    bin_values = bin_values,
    Y_perm_summary = (
        n_any = n_any,
        mean_logperm = mean_perm,
    ),
)
println("\nSaved to $OUT_FILE")
