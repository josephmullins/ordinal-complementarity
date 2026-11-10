# ═══════════════════════════════════════════════════════════════
# Bootstrap validation of the asymptotic (OPG / Fisher-identity) SEs
# for the V3+ε_Y settled paper/slides spec: r_θ=4, r_I=5, K=4,
# r_P=r_ω=r_Y=3  (output/estimation_v3_eps.jld2).
#
# Each bootstrap replication:
#   1. resamples children (rows) with replacement,
#   2. re-runs EM *warm-started from the full-sample MLE*,
#   3. records the regime-marginalized transition Π̄ (per group),
#      the survivor S̄, and the scalar β₅/δ.
#
# We then compare the bootstrap SD of each quantity against the
# asymptotic SE from `compute_var_phi_v3_eps` + delta method.
#
# The asymptotic SEs are known to be *conditional* on the nuisance
# blocks (π_ε, π0, Q_P, probit_θ/I[t], adult) which are stubbed out of
# the score, and they zero-out "inactive"/rank-truncated directions.
# The bootstrap re-estimates everything, so it is the full-information
# yardstick.  Expectation: SE_boot ≥ SE_asymp.
#
# Config via ENV:
#   BOOT_B        number of bootstrap replications (default 200)
#   BOOT_MAXITER  EM maxiter per replication      (default 400)
#   BOOT_TOL      EM convergence tol on Π         (default 1e-4)
#   BOOT_SEED     RNG seed                         (default 20260604)
#   BOOT_FIT      fit file (default estimation_v3_eps_k3_rI4.jld2)
#
# Threads: run with `julia -t N`.  Replications are parallelized.
# ═══════════════════════════════════════════════════════════════

using LinearAlgebra, Random, Statistics, JLD2, Optim, Printf, ForwardDiff
using Base.Threads

const REPO = joinpath(@__DIR__, "..")
include(joinpath(REPO, "src", "parametric_models.jl"))
include(joinpath(REPO, "src", "parametric_models_v3.jl"))
include(joinpath(REPO, "src", "parametric_models_v3_eps.jl"))
include(joinpath(REPO, "src", "prepare_data.jl"))
include(joinpath(REPO, "src", "estimation.jl"))
include(joinpath(REPO, "src", "estimation_multiperiod.jl"))
include(joinpath(REPO, "src", "estimation_multiperiod_v3.jl"))
include(joinpath(REPO, "src", "estimation_multiperiod_v3_eps.jl"))
include(joinpath(REPO, "src", "inference_common.jl"))
include(joinpath(REPO, "src", "inference_v3_eps.jl"))

# Avoid BLAS oversubscription when threading over replications.
BLAS.set_num_threads(1)
const T_START = time()

const OUTPUT_DIR = joinpath(REPO, "output")
const TRANSITION_GROUPS = [[1, 2, 3], [4, 5, 6]]
const R_P, R_ω, R_Y = 3, 3, 3

const PANEL_AGE_RANGES = [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13)]
# Identical to scripts/run_v3_eps_rI4.jl (the fit being validated).
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

# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────
# NOTE: warm-started EM does not formally converge — it wanders on a flat
# likelihood ridge (robit-NN weak identification).  max|ΔΠ| oscillates ~1e-3
# and never crosses tol, so BOOT_MAXITER is effectively a *fixed* iteration
# budget applied identically to every rep (incl. the reference fit).  This
# makes the bootstrapped estimator a well-defined deterministic map of the
# resample.  LL is flat after ~50 iters; 300 is a generous, representative
# ridge depth.  The reference fit below centers the bias diagnostic so the
# stopping-depth difference vs the 1459-iter MLE doesn't masquerade as bias.
const BOOT_B       = parse(Int, get(ENV, "BOOT_B", "100"))
const BOOT_MAXITER = parse(Int, get(ENV, "BOOT_MAXITER", "300"))
const BOOT_TOL     = parse(Float64, get(ENV, "BOOT_TOL", "1e-4"))
const BOOT_SEED    = parse(Int, get(ENV, "BOOT_SEED", "20260604"))
# Paper / slides settled spec: r_θ=4, r_I=5, K=4, r_P=r_ω=r_Y=3.
const FIT_FILE     = get(ENV, "BOOT_FIT", "estimation_v3_eps.jld2")
const OUT_FILE     = joinpath(OUTPUT_DIR,
    "bootstrap_" * replace(FIT_FILE, ".jld2" => "") * ".jld2")

# ─────────────────────────────────────────────────────────────────
# Child-level resampling: rebuild the data NamedTuple on row index `idx`.
# Only the per-child arrays are reindexed; κ's, T, r_*, names are shared.
# ─────────────────────────────────────────────────────────────────
function resample_data(data, idx::Vector{Int})
    X_θ = [[m[idx] for m in data.X_θ[t]] for t in 1:data.T]
    X_I = [[m[idx] for m in data.X_I[t]] for t in 1:data.T]
    X_P = [m[idx] for m in data.X_P]
    Y_income = data.Y_income[idx, :]
    X_adult = isempty(data.X_adult) ? data.X_adult : [m[idx] for m in data.X_adult]
    return merge(data, (; N = length(idx),
                          X_θ, X_I, X_P, Y_income, X_adult,
                          kids = data.kids[idx]))
end

# ─────────────────────────────────────────────────────────────────
# Quantities of interest from a fitted (pv, psk)
# ─────────────────────────────────────────────────────────────────

"Flatten the free (first r_θ-1 rows) entries of Π̄ for a given group's period.
 Ordering matches `delta_cov_Pi_bar_free`'s `pi_bar_free_map`:
   for kI in 1:r_I, kθ in 1:r_θ, j in 1:(r_θ-1)."
function pi_bar_free_vec(Π_bar::Array{Float64,3}, r_θ::Int, r_I::Int)
    v = Float64[]
    for kI in 1:r_I, kθ in 1:r_θ, j in 1:(r_θ - 1)
        push!(v, Π_bar[j, kθ, kI])
    end
    return v
end

"Survivor free entries S̄(k|θ,I)=Σ_{j>k} Π̄(j,θ,I), k=1..r_θ-1.
 Ordering matches `delta_cov_survivor_bar`: for kI, kθ, k."
function survivor_bar_free_vec(Π_bar::Array{Float64,3}, r_θ::Int, r_I::Int)
    n_cut = r_θ - 1
    v = Float64[]
    for kI in 1:r_I, kθ in 1:r_θ, k in 1:n_cut
        s = 0.0
        for j in (k + 1):r_θ
            s += Π_bar[j, kθ, kI]
        end
        push!(v, s)
    end
    return v
end

"From a fitted (pv, psk): per-group Π̄_free, S̄_free, and β₅/δ."
function quantities_from_fit(pv, psk)
    (; r_θ, r_I) = psk
    Pi_bar = marginalize_Pi_v3(pv.trans_specs, TRANSITION_GROUPS, pv.π_Pω,
                               r_θ, r_I, pv.r_P, pv.r_ω)
    pibar_free = [pi_bar_free_vec(Pi_bar[g[1]], r_θ, r_I) for g in TRANSITION_GROUPS]
    surv_free  = [survivor_bar_free_vec(Pi_bar[g[1]], r_θ, r_I) for g in TRANSITION_GROUPS]
    ratio = pv.ccp_spec.beta5 / pv.income_model.delta
    return (; pibar_free, surv_free, ratio)
end

# ─────────────────────────────────────────────────────────────────
# Load data + MLE fit
# ─────────────────────────────────────────────────────────────────
println("="^64)
println("  Bootstrap SE validation — V3+ε_Y  ($FIT_FILE)")
println("  B=$BOOT_B  maxiter=$BOOT_MAXITER  tol=$BOOT_TOL  seed=$BOOT_SEED  threads=$(nthreads())")
println("="^64)

println("Loading MLE fit ...")
saved = load(joinpath(OUTPUT_DIR, FIT_FILE))
const PV_MLE  = saved["params_v3_eps"]
const PSK_MLE = saved["params_skill"]
(; r_θ, r_I, T) = PSK_MLE
const FIT_DIMS = haskey(saved, "data_dims") ? saved["data_dims"] : nothing
const FIT_K = FIT_DIMS === nothing ? PV_MLE.trans_specs[1].K : FIT_DIMS.K
println("  fit dims: r_θ=$r_θ  r_I=$r_I  K=$FIT_K  r_P=$R_P  r_ω=$R_ω  r_Y=$R_Y  T=$T")

# Prepare data at the *fit's* r_I (read from the saved fit, not hardcoded).
println("Loading data (r_I=$r_I to match fit) ...")
datasets = load_all_datasets()
panel = prepare_panel_data(datasets;
    age_ranges = PANEL_AGE_RANGES, invest_items = PANEL_INVEST_ITEMS,
    r_θ = r_θ, r_I = r_I)
const DATA = merge(panel, (; r_P = R_P, r_ω = R_ω))
const N = DATA.N
@assert panel.r_I == r_I "data r_I ($(panel.r_I)) != fit r_I ($r_I)"
println("  N=$N")

# ─────────────────────────────────────────────────────────────────
# Asymptotic SEs at the MLE (the thing we are validating)
# ─────────────────────────────────────────────────────────────────
println("\nComputing asymptotic (OPG/Fisher) SEs ...")
inf = compute_var_phi_v3_eps(PV_MLE, PSK_MLE, DATA;
    transition_groups = TRANSITION_GROUPS, verbose = true)

Var_pibar_list, Pi_bar_mle = delta_cov_Pi_bar_free(inf, PV_MLE, PSK_MLE;
    transition_groups = TRANSITION_GROUPS)
sb_results = delta_cov_survivor_bar(inf, PV_MLE, PSK_MLE;
    transition_groups = TRANSITION_GROUPS)
ratio_res = delta_se_β5_over_δ(inf, PV_MLE, PSK_MLE)

se_pibar_asymp = [sqrt.(max.(diag(Var_pibar_list[g[1]]), 0.0)) for g in TRANSITION_GROUPS]
se_surv_asymp  = [sb_results[gi].ses for gi in 1:length(TRANSITION_GROUPS)]
se_ratio_asymp = ratio_res.se

q_mle = quantities_from_fit(PV_MLE, PSK_MLE)
println("  β₅/δ = $(round(ratio_res.point; digits=4))  asymp SE = $(round(se_ratio_asymp; digits=4))")

# ─────────────────────────────────────────────────────────────────
# Iteration-budget sweep.
#
# EM does NOT converge here — it wanders on a flat likelihood ridge (robit-NN
# weak identification), so max|ΔΠ| never crosses tol and `maxiter` is a *fixed
# iteration budget*.  The bootstrap SD depends on that budget:
#   • too few iters  → reps haven't separated from the shared MLE warm start
#                      (under-dispersed, SD biased DOWN);
#   • enough iters   → each rep reaches its own resample ridge-position
#                      (genuine sampling SD — the object we want);
#   • too many iters → extra wander along the flat ridge inflates SD.
# We therefore sweep several budgets over the SAME resamples and report SD at
# each, so the reader sees where (and whether) the bootstrap SD stabilizes.
# ─────────────────────────────────────────────────────────────────
const MAXITER_LIST = [parse(Int, s) for s in split(get(ENV, "BOOT_MAXITER_LIST",
                       string(BOOT_MAXITER)), ',')]

n_groups = length(TRANSITION_GROUPS)
n_pibar  = (r_θ - 1) * r_θ * r_I
n_surv   = (r_θ - 1) * r_θ * r_I

# Same resamples across every budget (seeded once).
rng_master = MersenneTwister(BOOT_SEED)
idx_draws = [rand(rng_master, 1:N, N) for _ in 1:BOOT_B]

robust_sd(x) = 1.4826 * median(abs.(x .- median(x)))
colwise_robust(M) = [robust_sd(M[:, j]) for j in 1:size(M, 2)]

"Run reference fit + B bootstrap reps at one iteration budget."
function run_at_budget(maxiter::Int)
    # Reference fit: EM-from-MLE on the *original* data at this budget.
    pv_ref, psk_ref, _, _, _, ll_ref = expectation_maximization_v3_eps(
        DATA, PV_MLE, PSK_MLE; transition_groups = TRANSITION_GROUPS,
        maxiter = maxiter, tol = BOOT_TOL, verbose = false)
    q_ref = quantities_from_fit(pv_ref, psk_ref)

    boot_pibar = [fill(NaN, BOOT_B, n_pibar) for _ in 1:n_groups]
    boot_surv  = [fill(NaN, BOOT_B, n_surv)  for _ in 1:n_groups]
    boot_ratio = fill(NaN, BOOT_B)
    boot_ll    = fill(NaN, BOOT_B)
    boot_ok    = fill(false, BOOT_B)

    t0 = time(); done = Threads.Atomic{Int}(0)
    Threads.@threads for b in 1:BOOT_B
        try
            data_b = resample_data(DATA, idx_draws[b])
            pv_b, psk_b, _, _, _, ll_b = expectation_maximization_v3_eps(
                data_b, PV_MLE, PSK_MLE; transition_groups = TRANSITION_GROUPS,
                maxiter = maxiter, tol = BOOT_TOL, verbose = false)
            q = quantities_from_fit(pv_b, psk_b)
            for g in 1:n_groups
                boot_pibar[g][b, :] = q.pibar_free[g]
                boot_surv[g][b, :]  = q.surv_free[g]
            end
            boot_ratio[b] = q.ratio; boot_ll[b] = ll_b; boot_ok[b] = true
        catch err
            @warn "Replication $b (maxiter=$maxiter) failed" exception=(err, catch_backtrace())
        end
        d = Threads.atomic_add!(done, 1) + 1
        if d % 5 == 0 || d == BOOT_B
            el = time() - t0
            @printf("    [maxiter=%d] %3d/%d done (%.0fs elapsed)\n", maxiter, d, BOOT_B, el)
            flush(stdout)
        end
    end

    ok = findall(boot_ok)
    se_pibar = [vec(std(boot_pibar[g][ok, :]; dims=1)) for g in 1:n_groups]
    se_surv  = [vec(std(boot_surv[g][ok, :];  dims=1)) for g in 1:n_groups]
    se_ratio = std(boot_ratio[ok])
    se_pibar_rob = [colwise_robust(boot_pibar[g][ok, :]) for g in 1:n_groups]
    se_surv_rob  = [colwise_robust(boot_surv[g][ok, :])  for g in 1:n_groups]
    se_ratio_rob = robust_sd(boot_ratio[ok])
    bias_pibar  = [vec(mean(boot_pibar[g][ok, :]; dims=1)) .- q_ref.pibar_free[g] for g in 1:n_groups]
    depth_pibar = [q_ref.pibar_free[g] .- q_mle.pibar_free[g] for g in 1:n_groups]

    ll_ok = boot_ll[ok]
    return (; maxiter, n_ok = length(ok), ok,
            boot_pibar, boot_surv, boot_ratio, boot_ll,
            q_ref, ll_ref,
            se_pibar, se_surv, se_ratio, se_pibar_rob, se_surv_rob, se_ratio_rob,
            bias_pibar, depth_pibar,
            ll_med = median(ll_ok), ll_rng = (minimum(ll_ok), maximum(ll_ok)))
end

println("\nIteration-budget sweep: $(MAXITER_LIST)   (B=$BOOT_B each, same resamples)")
flush(stdout)
runs = NamedTuple[]
for mi in MAXITER_LIST
    println("\n▶ budget maxiter=$mi  ($(round((time()-T_START)/60; digits=1)) min in) ...")
    flush(stdout)
    r = run_at_budget(mi)
    println("  budget $mi done: n_ok=$(r.n_ok), β₅/δ bootSD=$(round(r.se_ratio; digits=4)) (asy $(round(se_ratio_asymp; digits=4)), ratio $(round(r.se_ratio/se_ratio_asymp; digits=2)))")
    flush(stdout)
    push!(runs, r)
end

# ─────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────
println("\n" * "="^72)
println("  RESULTS:  bootstrap SD vs asymptotic SE,  across iteration budgets")
println("="^72)

# β₅/δ row across budgets
println("\nβ₅/δ  (point=$(round(q_mle.ratio; digits=4)), asymp SE=$(round(se_ratio_asymp; digits=4))):")
@printf("  %-10s %10s %10s %8s %8s %10s\n", "maxiter", "bootSD", "robSD", "SD/asy", "rob/asy", "ref β₅/δ")
for r in runs
    @printf("  %-10d %10.4f %10.4f %8.2f %8.2f %10.4f\n",
            r.maxiter, r.se_ratio, r.se_ratio_rob,
            r.se_ratio / se_ratio_asymp, r.se_ratio_rob / se_ratio_asymp, r.q_ref.ratio)
end

for g in 1:n_groups
    println("\n── Transition group $g (periods $(TRANSITION_GROUPS[g])),  Π̄_free  (asymp SE mean=$(round(mean(se_pibar_asymp[g]); digits=4))) ──")
    @printf("  %-10s %8s %8s %8s %8s %10s %10s\n",
            "maxiter", "boot µ", "rob µ", "r µ", "r med", "r max", "max|bias|")
    for r in runs
        rr  = filter(isfinite, r.se_pibar[g]     ./ max.(se_pibar_asymp[g], 1e-12))
        rrr = filter(isfinite, r.se_pibar_rob[g] ./ max.(se_pibar_asymp[g], 1e-12))
        @printf("  %-10d %8.4f %8.4f %8.2f %8.2f %10.2f %10.4f\n",
                r.maxiter, mean(r.se_pibar[g]), mean(r.se_pibar_rob[g]),
                mean(rr), median(rr), maximum(rr), maximum(abs.(r.bias_pibar[g])))
    end
    @printf("  (stopping-depth max|ref − MLE| at largest budget = %.4f)\n",
            maximum(abs.(runs[end].depth_pibar[g])))
end

println("\nBootstrap LL by budget (median [min,max]):")
for r in runs
    @printf("  maxiter=%-5d  %.1f  [%.1f, %.1f]\n", r.maxiter, r.ll_med, r.ll_rng[1], r.ll_rng[2])
end

# ─────────────────────────────────────────────────────────────────
# Save (all budgets)
# ─────────────────────────────────────────────────────────────────
jldsave(OUT_FILE;
    fit_file = FIT_FILE, B = BOOT_B, tol = BOOT_TOL, seed = BOOT_SEED,
    maxiter_list = MAXITER_LIST, transition_groups = TRANSITION_GROUPS,
    idx_draws = idx_draws,
    se_pibar_asymp = se_pibar_asymp, se_surv_asymp = se_surv_asymp, se_ratio_asymp = se_ratio_asymp,
    pibar_free_mle = q_mle.pibar_free, surv_free_mle = q_mle.surv_free, ratio_mle = q_mle.ratio,
    # per-budget results
    runs_maxiter = [r.maxiter for r in runs],
    runs_n_ok    = [r.n_ok for r in runs],
    runs_boot_pibar = [r.boot_pibar for r in runs],
    runs_boot_surv  = [r.boot_surv  for r in runs],
    runs_boot_ratio = [r.boot_ratio for r in runs],
    runs_boot_ll    = [r.boot_ll    for r in runs],
    runs_se_pibar   = [r.se_pibar   for r in runs],
    runs_se_surv    = [r.se_surv    for r in runs],
    runs_se_ratio   = [r.se_ratio   for r in runs],
    runs_se_pibar_rob = [r.se_pibar_rob for r in runs],
    runs_se_ratio_rob = [r.se_ratio_rob for r in runs],
    runs_ratio_ref  = [r.q_ref.ratio for r in runs],
    runs_pibar_ref  = [r.q_ref.pibar_free for r in runs],
)
println("\nSaved → $OUT_FILE")
println("Done.")
