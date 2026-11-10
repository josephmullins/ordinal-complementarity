#####
# compare_se_blocks.jl
#
# (1) Finite-difference validation of the 6 newly-implemented nuisance
#     score blocks in src/inference_v3_eps.jl (§4.5–4.9), against the
#     observed-data log-likelihood returned by expectation_step_v3_eps!.
# (2) 4-block vs 10-block asymptotic SE comparison for β₅/δ and Π̄_free,
#     alongside the bootstrap SD (if the bootstrap file exists).
#
# Uses the PAPER/SLIDES settled spec: output/estimation_v3_eps.jld2
# (r_θ=4, r_I=5, K=4, r_P=r_ω=r_Y=3, T=7, N=9686).  Data prepared at
# r_I=5 with the exact items from scripts/run_v3_eps.jl.
#
# Run:  julia -t 2 scripts/compare_se_blocks.jl
#####

using JLD2, LinearAlgebra, Statistics, Printf, Random

const REPO = joinpath(@__DIR__, "..")
include(joinpath(REPO, "src", "parametric_models.jl"))
include(joinpath(REPO, "src", "parametric_models_v3.jl"))
include(joinpath(REPO, "src", "parametric_models_v3_eps.jl"))
include(joinpath(REPO, "src", "prepare_data.jl"))
include(joinpath(REPO, "src", "estimation.jl"))
include(joinpath(REPO, "src", "estimation_multiperiod.jl"))
include(joinpath(REPO, "src", "estimation_multiperiod_v3.jl"))
include(joinpath(REPO, "src", "estimation_multiperiod_v3_eps.jl"))
include(joinpath(REPO, "src", "inference.jl"))
include(joinpath(REPO, "src", "inference_common.jl"))
include(joinpath(REPO, "src", "inference_v3_eps.jl"))

const TRANSITION_GROUPS = [[1, 2, 3], [4, 5, 6]]

# Items VERBATIM from scripts/run_v3_eps.jl (PANEL_INVEST_ITEMS).
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

# ─────────────────────────────────────────────────────────────────────────
# Load fit + data
# ─────────────────────────────────────────────────────────────────────────
println("Loading estimation_v3_eps.jld2 ...")
fit = load(joinpath(REPO, "output", "estimation_v3_eps.jld2"))
pv3  = fit["params_v3_eps"]
psk  = fit["params_skill"]
dims = fit["data_dims"]
age_ranges = fit["age_ranges"]
(; r_θ, r_I, r_P, r_ω, r_Y, T, N) = dims

println("Loading raw data (r_I=$r_I) ...")
datasets = load_all_datasets()
panel = prepare_panel_data(datasets;
    age_ranges = age_ranges, invest_items = PANEL_INVEST_ITEMS,
    r_θ = r_θ, r_I = r_I)
data_nb = merge(panel, (; r_P = r_P, r_ω = r_ω))
println("  N=$(data_nb.N), T=$(data_nb.T)")

θ_grid = collect(1.0:r_θ); I_grid = collect(1.0:r_I)
P_grid = collect(1.0:r_P); ω_grid = collect(1.0:r_ω); Y_grid = collect(1.0:r_Y)

# ─────────────────────────────────────────────────────────────────────────
# Helper: run one E-step at given params, return total observed-data LL
# ─────────────────────────────────────────────────────────────────────────
function total_ll_at(pv, ps, data)
    Nn = data.N; r_Zf = r_P * r_ω * r_Y; Tm1 = data.T - 1
    γ_agg         = zeros(data.T, r_θ, r_I, Nn)
    xi_sum_z_full = [[zeros(r_θ, r_I, r_θ, r_I) for _ in 1:Tm1] for _ in 1:r_Zf]
    w_z           = zeros(r_Zf, Nn)
    gamma_z_t1    = zeros(r_θ, r_I, r_Zf, Nn)
    ccp_init_ss   = zeros(r_θ, r_I, r_Zf)
    ccp_trans_ss  = zeros(r_I, r_I, r_θ, r_Zf, Tm1)
    pi0_ss        = zeros(r_θ, r_P, r_ω)
    pi_eps_ss     = zeros(r_Y)
    Pi_z_pre = precompute_Pi_z(pv.trans_specs, TRANSITION_GROUPS, r_θ, r_I, r_P, r_ω;
        θ_grid, I_grid, P_grid, ω_grid)
    CCP_init_pre, CCP_trans_pre = precompute_CCP_z_eps(pv.ccp_spec,
        r_P, r_ω, r_Y, r_θ, r_I; θ_grid, I_grid, P_grid, ω_grid, Y_grid)
    return expectation_step_v3_eps!(γ_agg, xi_sum_z_full, w_z, gamma_z_t1,
        ccp_init_ss, ccp_trans_ss, pi0_ss, pi_eps_ss,
        Pi_z_pre, CCP_init_pre, CCP_trans_pre, pv, ps, data;
        θ_grid, I_grid, P_grid, ω_grid, Y_grid)
end

# Rebuild params NamedTuples from a perturbed packed vector for a given block.
# Returns (pv', ps').  Block ∈ :π_ε, :π0, :Q_P, :probit_θ_t, :probit_I_t, :adult.

# π_ε from softmax α (length r_Y-1)
function set_pi_ε(pv, α)
    π = softmax_to_probs(α)
    merge(pv, (; π_ε = collect(π)))
end
# π0 conditional: per-cell softmax; combine with current π_Pω marginal → joint π0
function set_pi0(pv, α_all)  # α_all length r_P*r_ω*(r_θ-1)
    nc = r_θ - 1
    π0_new = zeros(r_θ, r_P, r_ω)
    for kω in 1:r_ω, kP in 1:r_P
        cell = (kP - 1) * r_ω + kω
        αc = α_all[(cell-1)*nc+1 : cell*nc]
        pc = softmax_to_probs(αc)            # P(θ_1 | cell)
        w  = pv.π_Pω[kP, kω]                  # marginal
        for kθ in 1:r_θ; π0_new[kθ, kP, kω] = w * pc[kθ]; end
    end
    merge(pv, (; π0 = π0_new))
end
function set_Q_P(pv, φ)
    M = length(pv.Q_P); idx = 1
    tau = Vector{Vector{Float64}}(undef, M); alpha = Vector{Float64}(undef, M)
    Qnew = Vector{Matrix{Float64}}(undef, M)
    for m in 1:M
        κ = data_nb.κ_P[m]; nc = κ - 1
        tau[m] = _unpack_cutpoints(φ[idx:idx+nc-1]); idx += nc
        alpha[m] = φ[idx]; idx += 1
        Qnew[m] = ordered_probit_matrix(tau[m], alpha[m], r_P)
    end
    merge(pv, (; probit_P = OrderedProbitEmission(tau, alpha), Q_P = Qnew))
end
function set_probit_θ_t(ps, t, φ)
    M = length(ps.probit_θ[t].alpha); idx = 1
    tau = Vector{Vector{Float64}}(undef, M); alpha = Vector{Float64}(undef, M)
    for m in 1:M
        κ = data_nb.κ_θ[t][m]; nc = κ - 1
        tau[m] = _unpack_cutpoints(φ[idx:idx+nc-1]); idx += nc
        alpha[m] = φ[idx]; idx += 1
    end
    pθ = copy(ps.probit_θ); pθ[t] = OrderedProbitEmission(tau, alpha)
    Pθ = copy(ps.P_θ); Pθ[t] = emission_matrices_from_probit(pθ[t], r_θ)
    merge(ps, (; probit_θ = pθ, P_θ = Pθ))
end
function set_probit_I_t(ps, t, φ)
    M = length(ps.probit_I[t].alpha); idx = 1
    tau = Vector{Vector{Float64}}(undef, M); alpha = Vector{Float64}(undef, M)
    for m in 1:M
        κ = data_nb.κ_I[t][m]; nc = κ - 1
        tau[m] = _unpack_cutpoints(φ[idx:idx+nc-1]); idx += nc
        alpha[m] = φ[idx]; idx += 1
    end
    pI = copy(ps.probit_I); pI[t] = OrderedProbitEmission(tau, alpha)
    PI = copy(ps.P_I); PI[t] = emission_matrices_from_probit(pI[t], r_I)
    merge(ps, (; probit_I = pI, P_I = PI))
end
function set_adult(pv, φ)
    ae = unpack_adult_emission(φ, collect(data_nb.κ_adult))
    am = build_adult_matrices(ae, collect(data_nb.κ_adult), r_θ, r_ω)
    merge(pv, (; adult_emission = ae, adult_matrices = am))
end

# Pack current values per block
α_pi_ε0  = probs_to_softmax(collect(pv3.π_ε); ref=1)
function pi0_cond_pack(pv)
    nc = r_θ - 1; out = Float64[]
    for kω in 1:r_ω, kP in 1:r_P
        s = sum(max(pv.π0[kθ,kP,kω],0.0) for kθ in 1:r_θ)
        pc = [max(pv.π0[kθ,kP,kω],0.0)/s for kθ in 1:r_θ]
        append!(out, probs_to_softmax(pc; ref=1))
    end
    out
end
α_pi00 = pi0_cond_pack(pv3)
function emis_pack(probit, κ_list)
    v = Float64[]
    for m in eachindex(probit.alpha)
        append!(v, _pack_cutpoints(probit.tau[m])); push!(v, probit.alpha[m])
    end
    v
end
φ_QP0 = emis_pack(pv3.probit_P, collect(data_nb.κ_P))
φ_adult0 = pack_adult_emission(pv3.adult_emission)

# ─────────────────────────────────────────────────────────────────────────
# (1) FD VALIDATION
#   At a perturbed point (so gradient ≠ 0): check Σ_n s_n^(block) ≈ central FD
#   of total_ll over that block's packed params.  Also report at-MLE mean.
# ─────────────────────────────────────────────────────────────────────────
println("\n" * "="^70)
println("  (1) Finite-difference validation of the 6 nuisance score blocks")
println("="^70)

rng = MersenneTwister(7)
fd_eps = 1e-5

# Score-sum extractor for a given (pv, ps): returns the analytic Σ_n s_n for
# each nuisance block as a Dict.
function block_score_sums(pv, ps)
    bs = per_child_scores_v3_eps(pv, ps, data_nb;
        transition_groups = TRANSITION_GROUPS, include_nuisance_blocks = true)
    d = Dict{Symbol,Vector{Float64}}()
    d[:π_ε] = vec(sum(bs.π_ε; dims=2))
    d[:π0]  = vec(sum(bs.π0;  dims=2))
    d[:Q_P] = vec(sum(bs.Q_P; dims=2))
    for t in 1:T
        d[Symbol("probit_θ_$t")] = vec(sum(bs.probit_θ[t]; dims=2))
        d[Symbol("probit_I_$t")] = vec(sum(bs.probit_I[t]; dims=2))
    end
    if bs.has_adult_blk
        d[:adult] = vec(sum(bs.adult; dims=2))
    end
    return d
end

# Generic FD check: φ0 = packed params; rebuild(pv,ps,φ) → (pv',ps');
# score key in the Dict; perturb φ0 by random δ then FD + analytic at that point.
function fd_check(name, φ0, rebuild, score_key; n_test=min(8, length(φ0)))
    δ = 0.05 .* randn(rng, length(φ0))
    φp = φ0 .+ δ
    pv_p, ps_p = rebuild(φp)
    # analytic
    sums = block_score_sums(pv_p, ps_p)
    s_ana = sums[score_key]
    # FD over a subset of components (central differences)
    idxs = length(φ0) <= n_test ? collect(1:length(φ0)) :
           sort(randperm(rng, length(φ0))[1:n_test])
    rel_errs = Float64[]
    for j in idxs
        φa = copy(φp); φb = copy(φp)
        φa[j] += fd_eps; φb[j] -= fd_eps
        pva, psa = rebuild(φa); pvb, psb = rebuild(φb)
        lla = total_ll_at(pva, psa, data_nb)
        llb = total_ll_at(pvb, psb, data_nb)
        g_fd = (lla - llb) / (2fd_eps)
        g_an = s_ana[j]
        denom = max(abs(g_fd), abs(g_an), 1.0)
        push!(rel_errs, abs(g_fd - g_an) / denom)
    end
    maxre = maximum(rel_errs)
    @printf("  %-14s  n_φ=%-4d  max rel.err (over %d comps) = %.3e\n",
            name, length(φ0), length(idxs), maxre)
    return maxre
end

const RUN_FD = get(ENV, "SKIP_FD", "0") != "1"
sums_mle = block_score_sums(pv3, psk)
fd_results = Dict{String,Float64}()
if RUN_FD
    # at-MLE mean score (should be ≈ 0)
    println("\n  At-MLE mean |score| per block (should be ≈ 0):")
    for k in sort(collect(keys(sums_mle)); by=string)
        @printf("    %-14s  max|Σ_n s_n|/N = %.3e\n", k, maximum(abs.(sums_mle[k]))/N)
    end

    println("\n  FD checks at a perturbed point (rel.err should be < ~1e-3):")
    fd_results["π_ε"]  = fd_check("π_ε",  α_pi_ε0,  φ -> (set_pi_ε(pv3, φ), psk), :π_ε)
    fd_results["π0"]   = fd_check("π0",   α_pi00,   φ -> (set_pi0(pv3, φ),  psk), :π0)
    fd_results["Q_P"]  = fd_check("Q_P",  φ_QP0,    φ -> (set_Q_P(pv3, φ),  psk), :Q_P)
    # one representative early and one late period for θ and I emissions
    for t in (1, 4, 7)
        φ0 = emis_pack(psk.probit_θ[t], collect(data_nb.κ_θ[t]))
        fd_results["probit_θ_$t"] = fd_check("probit_θ[$t]", φ0,
            φ -> (pv3, set_probit_θ_t(psk, t, φ)), Symbol("probit_θ_$t"))
        φ0I = emis_pack(psk.probit_I[t], collect(data_nb.κ_I[t]))
        fd_results["probit_I_$t"] = fd_check("probit_I[$t]", φ0I,
            φ -> (pv3, set_probit_I_t(psk, t, φ)), Symbol("probit_I_$t"))
    end
    if !isempty(pv3.adult_emission.tau) && haskey(data_nb, :X_adult) && !isempty(data_nb.X_adult)
        fd_results["adult"] = fd_check("adult", φ_adult0,
            φ -> (set_adult(pv3, φ), psk), :adult)
    end
else
    println("\n  (FD validation skipped: SKIP_FD=1)")
end

# ─────────────────────────────────────────────────────────────────────────
# (2) 4-block vs 10-block asymptotic SE comparison
# ─────────────────────────────────────────────────────────────────────────
println("\n" * "="^70)
println("  (2) 4-block vs 10-block asymptotic SEs  (+ bootstrap SD)")
println("="^70)

println("\n  Running 4-block inference ...")
inf4  = compute_var_phi_v3_eps(pv3, psk, data_nb;
    transition_groups = TRANSITION_GROUPS, include_nuisance_blocks = false, verbose = true)
println("\n  Running 10-block inference ...")
inf10 = compute_var_phi_v3_eps(pv3, psk, data_nb;
    transition_groups = TRANSITION_GROUPS, include_nuisance_blocks = true, verbose = true)

# β₅/δ
r4  = delta_se_β5_over_δ(inf4,  pv3, psk)
r10 = delta_se_β5_over_δ(inf10, pv3, psk)

# Π̄_free per group
V4_list,  Pi_bar4  = delta_cov_Pi_bar_free(inf4,  pv3, psk; transition_groups = TRANSITION_GROUPS)
V10_list, Pi_bar10 = delta_cov_Pi_bar_free(inf10, pv3, psk; transition_groups = TRANSITION_GROUPS)

# Bootstrap (if available)
boot = nothing
boot_path = joinpath(REPO, "output", "bootstrap_estimation_v3_eps.jld2")
boot_se_ratio = NaN
boot_se_pibar = nothing  # Vector (per group) of Vector
if isfile(boot_path)
    try
        global boot = load(boot_path)
        # last completed budget run
        global boot_se_ratio = boot["runs_se_ratio"][end]
        global boot_se_pibar = boot["runs_se_pibar"][end]   # Vector(2) of Vector(60)
        println("\n  Bootstrap file found: B=$(boot["B"]), runs_n_ok=$(boot["runs_n_ok"]), " *
                "maxiter=$(boot["maxiter_list"])")
    catch e
        println("\n  (bootstrap file present but could not parse fully: $e)")
    end
end

println("\n  ── β₅/δ ──")
@printf("    point estimate            = %.4f\n", r4.point)
@printf("    4-block  asymptotic SE    = %.4f\n", r4.se)
@printf("    10-block asymptotic SE    = %.4f   (ratio 10/4 = %.3f)\n",
        r10.se, r10.se / r4.se)
if !isnan(boot_se_ratio)
    @printf("    bootstrap SD              = %.4f   (asymp10/boot = %.3f)\n",
            boot_se_ratio, r10.se / boot_se_ratio)
end

# Π̄_free: report per-group mean SE and a few representative entries.
# Free entries indexed: for kI in 1:r_I, kθ in 1:r_θ, j in 1:(r_θ-1) -> idx.
function se_vec(Vlist, g_t)  # g_t = a t in the group
    sqrt.(max.(diag(Vlist[g_t]), 0.0))
end

# representative entries: (j=1, kθ=1, kI=1), (j=1, kθ=4, kI=5), (j=2, kθ=2, kI=3)
function free_idx(j, kθ, kI)
    nfree_j = r_θ - 1
    (kI-1)*r_θ*nfree_j + (kθ-1)*nfree_j + j
end
reps = [(1,1,1), (1,r_θ,r_I), (2,2,3), (3,r_θ,1)]

group_rep_t = [TRANSITION_GROUPS[g][1] for g in 1:length(TRANSITION_GROUPS)]
println("\n  ── Π̄_free (per transition group) ──")
for (g, t) in enumerate(group_rep_t)
    se4  = se_vec(V4_list,  t)
    se10 = se_vec(V10_list, t)
    @printf("\n  Group %d  (periods %s):\n", g, string(TRANSITION_GROUPS[g]))
    @printf("    mean SE   : 4-block=%.4f  10-block=%.4f  (ratio=%.3f)\n",
            mean(se4), mean(se10), mean(se10)/mean(se4))
    @printf("    max  SE   : 4-block=%.4f  10-block=%.4f  (ratio=%.3f)\n",
            maximum(se4), maximum(se10), maximum(se10)/maximum(se4))
    if boot_se_pibar !== nothing && g <= length(boot_se_pibar)
        bse = boot_se_pibar[g]
        @printf("    mean boot SD = %.4f   (asymp10/boot = %.3f)\n",
                mean(bse), mean(se10)/mean(bse))
    end
    println("    representative entries  (j,θ,I) :  4blk   10blk   ratio" *
            (boot_se_pibar !== nothing ? "    boot" : ""))
    for (j, kθ, kI) in reps
        idx = free_idx(j, kθ, kI)
        bstr = ""
        if boot_se_pibar !== nothing && g <= length(boot_se_pibar)
            bstr = @sprintf("   %.4f", boot_se_pibar[g][idx])
        end
        @printf("      (%d,%d,%d) : %.4f  %.4f  %.3f%s\n",
                j, kθ, kI, se4[idx], se10[idx], se10[idx]/se4[idx], bstr)
    end
end

# ─────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────
out = joinpath(REPO, "output", "se_block_comparison.jld2")
jldsave(out;
    ratio_point = r4.point,
    se_ratio_4block = r4.se, se_ratio_10block = r10.se,
    boot_se_ratio = boot_se_ratio,
    V4_list, V10_list,
    se_pibar_4block  = [se_vec(V4_list,  group_rep_t[g]) for g in 1:length(TRANSITION_GROUPS)],
    se_pibar_10block = [se_vec(V10_list, group_rep_t[g]) for g in 1:length(TRANSITION_GROUPS)],
    boot_se_pibar = boot_se_pibar,
    Pi_bar = Pi_bar10,
    fd_max_rel_err = fd_results,
    at_mle_mean_score = Dict(string(k)=>maximum(abs.(v))/N for (k,v) in sums_mle),
    transition_groups = TRANSITION_GROUPS,
    rank_4block = inf4.rank_effective, rank_10block = inf10.rank_effective,
    nphi_4block = size(inf4.S,1), nphi_10block = size(inf10.S,1),
)
println("\n  Saved comparison to $out")
println("\nDone.")
