# Simple, independent estimation pipeline (2026-11-17), COLD START of every block.
#   MODE=stage1   : technology = additive ordered robit  F_ν(c_j + aθ + bI + γ_P θ^P + γ_ω ω)
#   MODE=pertheta : technology = per-skill ordered robit F_ν(c_j + μ_θ + β_θ I + …), exact
#                   embedding of the stage-1 technology; all other blocks from stage 1
#   MODE=nn       : technology = robit NN with K units (default 3), no log_w penalty; the
#                   hidden layer is first fitted to stage-1 posteriors from NN_STARTS random
#                   starts (best kept), then full EM from stage 1's other blocks
#   MODE=all      : stage1 then pertheta (the original two-stage run)
# Everything else (emissions, CCP with ε_Y, income, ASVAB, adult outcomes, priors) is
# initialized from scratch with the documented seed in stage 1.
#
# ENV: SEED (1), R_I (5), MODE (all), K (3), NN_STARTS (10), MAXITER (2000), TOL (1e-4), SMOKE=1
using LinearAlgebra, Random, Statistics, JLD2, Optim, Printf
const REPO = joinpath(@__DIR__, "..")
for f in ["parametric_models.jl", "parametric_models_v3.jl", "parametric_models_v3_eps.jl", "prepare_data.jl",
          "estimation.jl", "estimation_multiperiod.jl", "estimation_multiperiod_v3.jl",
          "estimation_multiperiod_v3_eps.jl", "parametric_models_v3_robit.jl", "investment_groups.jl"]
    include(joinpath(REPO, "src", f))
end

const SEED    = parse(Int, get(ENV, "SEED", "1"))
const MODE    = get(ENV, "MODE", "all")
const K_NN    = parse(Int, get(ENV, "K", "3"))
const NN_STARTS = parse(Int, get(ENV, "NN_STARTS", "10"))
const SMOKE   = get(ENV, "SMOKE", "0") == "1"
const MAXITER = SMOKE ? 3 : parse(Int, get(ENV, "MAXITER", "2000"))
const TOL     = parse(Float64, get(ENV, "TOL", "1e-4"))
const TG   = [[1, 2, 3], [4, 5, 6]]
println("SEED=$SEED  MODE=$MODE  K=$K_NN  MAXITER=$MAXITER  TOL=$TOL  SMOKE=$SMOKE"); flush(stdout)
include(joinpath(REPO, "scripts", "_ordrobit_data.jl"))
include(joinpath(REPO, "scripts", "_period_tech.jl"))
println("R_I=$R_I  INIT=$INIT  NO_PIAT3=$NO_PIAT3  GROUP_INV=$GROUP_INV  BINS=$BINS  TIDY=$TIDY  tag=$tag"); flush(stdout)
(; r_θ, r_I, T, N, κ_θ, κ_I, κ_P) = panel
outp(stage) = outp(stage, SEED)
function save_fit(path, pv, psk, γ, ξ, w, ll, stage; tg = TG)
    SMOKE && return
    jldsave(path; params_v3_eps = pv, params_skill = psk, gamma_agg = γ, xi_sum_z = ξ, w_z = w, best_ll = ll,
            stage, seed = SEED, n_trans_params = sum(s -> length(_pack_spec(s)), pv.trans_specs),
            data_dims = (; N, T, r_θ, r_I, r_P = R_P, r_ω = R_ω, r_Y = R_Y), age_ranges = AGES, transition_groups = tg)
    println("Saved → $path"); flush(stdout)
end
run_em(pv, psk; tg = TG) = expectation_maximization_v3_eps(data, pv, psk; transition_groups = tg,
                                                  maxiter = MAXITER, tol = TOL, verbose = true)

# ── data-driven start for the measurement blocks (no estimates borrowed) ──
# Per period: standardize each measure over observed children, average a child's
# z-scores, cut into equal-share quantile bins → hard pseudo-posteriors for θ_t and I_t.
# ASVAB the same → r_P terciles (ω, ε_Y uniform). One M-step on these pseudo-posteriors
# sets the skill/investment/ASVAB/adult emissions and π0.
function _score(Xs::Vector{Vector{Int}}, N)
    tot = zeros(N); cnt = zeros(Int, N)
    for x in Xs
        obs = findall(>(0), x); isempty(obs) && continue
        μ, σ = mean(x[obs]), std(x[obs]); σ = σ > 0 ? σ : 1.0
        for n in obs; tot[n] += (x[n] - μ) / σ; cnt[n] += 1; end
    end
    [cnt[n] > 0 ? tot[n] / cnt[n] : NaN for n in 1:N]
end
function _qbins(score, r)
    b = zeros(Int, length(score)); obs = findall(!isnan, score)
    ord = obs[sortperm(score[obs])]
    for (k, n) in enumerate(ord); b[n] = min(r, 1 + floor(Int, r * (k - 1) / length(ord))); end
    b                                                    # 0 = missing
end
# soft pseudo-posterior: 0.6 on the bin, 0.4 spread over adjacent bins (avoids perfect
# separation in the emission probits); uniform if the score is missing
function _soft(b, r)
    b == 0 && return fill(1 / r, r)
    w = zeros(r); nb = [k for k in (b - 1, b + 1) if 1 <= k <= r]
    w[b] = 0.6; for k in nb; w[k] = 0.4 / length(nb); end
    w
end
# keep the generic start for any measure whose pseudo-posterior fit is degenerate
function _guard(new::OrderedProbitEmission, old::OrderedProbitEmission)
    tau = copy(new.tau); alpha = copy(new.alpha); nbad = 0
    for m in eachindex(alpha)
        if !(0 < alpha[m] < 10) || any(!isfinite, tau[m])
            tau[m] = old.tau[m]; alpha[m] = old.alpha[m]; nbad += 1
        end
    end
    OrderedProbitEmission(tau, alpha), nbad
end
function data_driven_start(pv0, psk0)
    γ = zeros(T, r_θ, r_I, N)
    for t in 1:T
        bθ, bI = _qbins(_score(data.X_θ[t], N), r_θ), _qbins(_score(data.X_I[t], N), r_I)
        for n in 1:N
            wθ, wI = _soft(bθ[n], r_θ), _soft(bI[n], r_I)
            γ[t, :, :, n] .= wθ * wI'
        end
    end
    bP = _qbins(_score(data.X_P, N), R_P)
    r_Z = R_P * R_ω * R_Y; w_z = zeros(r_Z, N)
    for n in 1:N, kP in 1:R_P, kω in 1:R_ω, kε in 1:R_Y
        w_z[eps_regime_index(kP, kω, kε, R_ω, R_Y), n] = _soft(bP[n], R_P)[kP] / (R_ω * R_Y)
    end
    pθ, _ = parametric_mstep_emission_mp(psk0.probit_θ, γ, data, :theta, r_θ)
    pI, _ = parametric_mstep_emission_mp(psk0.probit_I, γ, data, :I, r_I)
    gθ = [_guard(pθ[t], psk0.probit_θ[t]) for t in 1:T]; gI = [_guard(pI[t], psk0.probit_I[t]) for t in 1:T]
    probit_θ = first.(gθ); probit_I = first.(gI)
    P_θ = [emission_matrices_from_probit(probit_θ[t], r_θ) for t in 1:T]
    P_I = [emission_matrices_from_probit(probit_I[t], r_I) for t in 1:T]
    pP, _ = mstep_Q_P_eps(pv0.probit_P, w_z, data, R_P, R_ω, R_Y)
    probit_P, nbadP = _guard(pP, pv0.probit_P); Q_P = emission_matrices_from_probit(probit_P, R_P)
    println("  data-driven start: measures reset to generic start (skill/invest/ASVAB): ",
            sum(last, gθ), "/", sum(last, gI), "/", nbadP)
    ae, am = mstep_adult_outcomes_eps(pv0.adult_emission, γ, w_z, data, r_θ, r_I, R_P, R_ω, R_Y)
    π0 = zeros(r_θ, R_P, R_ω)
    for n in 1:N, kP in 1:R_P
        wP = sum(w_z[eps_regime_index(kP, kω, kε, R_ω, R_Y), n] for kω in 1:R_ω, kε in 1:R_Y)
        for s in 1:r_θ, kω in 1:R_ω; π0[s, kP, kω] += sum(γ[1, s, :, n]) * wP / R_ω; end
    end
    π0 ./= sum(π0)
    println("  data-driven start: median skill loading t=1 ", round(median(probit_θ[1].alpha); digits = 2),
            ", ASVAB loadings ", round.(probit_P.alpha; digits = 2)); flush(stdout)
    (merge(pv0, (; probit_P, Q_P, adult_emission = ae, adult_matrices = am, π0,
                  π_Pω = dropdims(sum(π0; dims = 1); dims = 1))),
     merge(psk0, (; probit_θ, P_θ, probit_I, P_I)))
end

# ── stage 1: cold start, additive ordered robit ──
function stage1()
    rng = MersenneTwister(SEED)
    probit_θ = [initialize_ordered_probit(κ_θ[t], r_θ; rng) for t in 1:T]
    probit_I = [initialize_ordered_probit(κ_I[t], r_I; rng) for t in 1:T]
    probit_P = initialize_ordered_probit(collect(κ_P), R_P; rng)
    adult_emission = initialize_adult_outcome_emission(data.κ_adult, r_θ, R_ω; rng)
    pv0 = (; r_P = R_P, r_ω = R_ω, r_Y = R_Y,
           π0 = ones(r_θ, R_P, R_ω) / (r_θ * R_P * R_ω), π_Pω = ones(R_P, R_ω) / (R_P * R_ω), π_ε = fill(1 / R_Y, R_Y),
           trans_specs = [initialize_ordrobit_add_v3(r_θ; rng, ν = NU) for _ in TG],
           ccp_spec = initialize_ccp_v3_eps(r_I; rng),
           probit_P, Q_P = emission_matrices_from_probit(probit_P, R_P),
           income_model = initialize_income_model_eps(data.Y_income; delta_init = 0.15),
           adult_emission, adult_matrices = build_adult_matrices(adult_emission, data.κ_adult, r_θ, R_ω))
    psk0 = (; r_θ, r_I, T, P_θ = [emission_matrices_from_probit(probit_θ[t], r_θ) for t in 1:T],
            P_I = [emission_matrices_from_probit(probit_I[t], r_I) for t in 1:T], probit_θ, probit_I)
    if INIT == "data"
        pv0, psk0 = data_driven_start(pv0, psk0)
    end
    println("\n=== STAGE 1: additive ordered robit (cold start, INIT=$INIT) ==="); flush(stdout)
    t0 = time(); pv, psk, γ, ξ, w, ll = run_em(pv0, psk0)
    @printf("STAGE 1 done: LL = %.1f  (%.0f min)\n", ll, (time() - t0) / 60); flush(stdout)
    save_fit(outp("add"), pv, psk, γ, ξ, w, ll, "additive")
    (pv, psk, ξ)
end
load_stage1() = (F = load(outp("add")); (F["params_v3_eps"], F["params_skill"], F["xi_sum_z"]))

# ── stage 2a: per-skill ordered robit ──
function stage_pertheta(pv1, psk1)
    ts = [per_theta_from_add(s, r_θ) for s in pv1.trans_specs]
    for g in eachindex(TG)
        Πa = _make_build_fn(pv1.trans_specs[g], r_θ, r_I)(pack_ordrobit(pv1.trans_specs[g]), 2.0, 2.0)
        Πb = _make_build_fn(ts[g], r_θ, r_I)(pack_ordrobit(ts[g]), 2.0, 2.0)
        @printf("  embedding check group %d: max|ΔΠ| = %.2e\n", g, maximum(abs.(Πa .- Πb)))
    end
    println("\n=== STAGE 2a: per-skill ordered robit ==="); flush(stdout)
    t0 = time(); pv, psk, γ, ξ, w, ll = run_em(merge(pv1, (; trans_specs = ts)), psk1)
    @printf("STAGE 2a (per-skill) done: LL = %.1f  (%.0f min)\n", ll, (time() - t0) / 60); flush(stdout)
    save_fit(outp("pertheta"), pv, psk, γ, ξ, w, ll, "per-theta")
end

# ── stage 2b: robit NN, K units, hidden layer fitted to stage-1 posteriors first ──
function stage_nn(pv1, psk1, ξ1)
    ξt = collapse_xi_over_eps(ξ1, R_P, R_ω, R_Y, T - 1, r_θ, r_I)
    rng = MersenneTwister(1000 + SEED)
    ts = Vector{RobitNNTransitionV3}(undef, length(TG))
    for (g, grp) in enumerate(TG)
        best, bv = nothing, Inf
        for s in 1:NN_STARTS
            sp = mstep_transition_v3_grouped([initialize_robit_nn_v3(r_θ, K_NN; rng, ν = NU)], ξt, [grp],
                                             r_θ, r_I, R_P, R_ω)[1]
            Q = [reduce(.+, [_qsum_from_xi(ξt[z][t]) for t in grp]) for z in 1:R_P*R_ω]
            v = transition_negll_robit_nn_v3(pack_robit_nn_v3(sp), Q, r_θ, r_I, K_NN, NU, R_P, R_ω)
            v < bv && (best = sp; bv = v)
        end
        ts[g] = best
        @printf("  NN init group %d: best of %d starts, transition negll %.1f, max|log_w| %.1f\n",
                g, NN_STARTS, bv, maximum(abs.(best.log_w)))
    end
    println("\n=== STAGE 2b: robit NN K=$K_NN (λ_logw = 0) ==="); flush(stdout)
    t0 = time(); pv, psk, γ, ξ, w, ll = run_em(merge(pv1, (; trans_specs = ts)), psk1)
    @printf("STAGE 2b (NN K=%d) done: LL = %.1f  (%.0f min)  max|log_w| = %s\n", K_NN, ll, (time() - t0) / 60,
            string([round(maximum(abs.(s.log_w)); digits = 1) for s in pv.trans_specs])); flush(stdout)
    save_fit(outp("nnK$(K_NN)"), pv, psk, γ, ξ, w, ll, "robit NN K=$K_NN")
end

# ── stage 3: add a skill-specific squared investment term, from the per-skill fit ──
function stage_quad()
    F = load(outp("pertheta")); pv2, psk2 = F["params_v3_eps"], F["params_skill"]
    ts = [per_theta_quad_from_per_theta(s) for s in pv2.trans_specs]
    for g in eachindex(TG)
        Πa = _make_build_fn(pv2.trans_specs[g], r_θ, r_I)(pack_ordrobit(pv2.trans_specs[g]), 2.0, 2.0)
        Πb = _make_build_fn(ts[g], r_θ, r_I)(pack_ordrobit(ts[g]), 2.0, 2.0)
        @printf("  embedding check group %d: max|ΔΠ| = %.2e\n", g, maximum(abs.(Πa .- Πb)))
    end
    println("\n=== STAGE 3: per-skill ordered robit + I² (from per-skill fit, LL $(round(F["best_ll"]; digits = 1))) ==="); flush(stdout)
    t0 = time(); pv, psk, γ, ξ, w, ll = run_em(merge(pv2, (; trans_specs = ts)), psk2)
    @printf("STAGE 3 (per-skill + I²) done: LL = %.1f  (gain %+.1f over linear, %.0f min)\n", ll, ll - F["best_ll"], (time() - t0) / 60); flush(stdout)
    for (g, s) in enumerate(pv.trans_specs)
        @printf("  group %d: β_θ = %s  δ_θ = %s\n", g, round.(exp.(s.β_raw); digits = 3), round.(s.δ; digits = 3))
    end
    save_fit(outp("perthetaQ"), pv, psk, γ, ξ, w, ll, "per-theta + I^2")
end

# ── diagnostic: per-skill (then + I²) technology with every other block WARM from a given fit ──
function stage_warm()
    src = get(ENV, "WARM_FROM", joinpath(REPO, "output", "estimation_v3_eps_k3_rI4.jld2"))
    F = load(src); pvw, pskw = F["params_v3_eps"], F["params_skill"]
    ξt = collapse_xi_over_eps(F["xi_sum_z"], R_P, R_ω, R_Y, T - 1, r_θ, r_I)
    ts = OrderedRobitPerThetaV3[]
    for (g, grp) in enumerate(TG)
        add = mstep_transition_v3_grouped([initialize_ordrobit_add_v3(r_θ; rng = MersenneTwister(g), ν = NU)], ξt, [grp], r_θ, r_I, R_P, R_ω)[1]
        push!(ts, mstep_transition_v3_grouped([per_theta_from_add(add, r_θ)], ξt, [grp], r_θ, r_I, R_P, R_ω)[1])
    end
    println("\n=== WARM: per-skill technology, other blocks from $(basename(src)) (LL $(round(F["best_ll"]; digits = 1))) ==="); flush(stdout)
    t0 = time(); pv, psk, γ, ξ, w, ll = run_em(merge(pvw, (; trans_specs = ts)), pskw)
    @printf("WARM per-skill done: LL = %.1f  (%+.1f vs source, %.0f min)\n", ll, ll - F["best_ll"], (time() - t0) / 60); flush(stdout)
    save_fit(outp("warm_pertheta"), pv, psk, γ, ξ, w, ll, "warm per-theta")
    tq = [per_theta_quad_from_per_theta(s) for s in pv.trans_specs]
    t0 = time(); pvq, pskq, γq, ξq, wq, llq = run_em(merge(pv, (; trans_specs = tq)), psk)
    @printf("WARM per-skill + I² done: LL = %.1f  (%+.1f vs source, %.0f min)\n", llq, llq - F["best_ll"], (time() - t0) / 60); flush(stdout)
    save_fit(outp("warm_perthetaQ"), pvq, pskq, γq, ξq, wq, llq, "warm per-theta + I^2")
end

# ── flexible technology straight away: per-skill ordered robit from a cold start ──
# (same random initialization of every other block as stage 1; the per-skill start is a
#  random additive spec embedded exactly, so it is as uninformative as stage 1's)
function stage_direct()
    rng = MersenneTwister(SEED)
    probit_θ = [initialize_ordered_probit(κ_θ[t], r_θ; rng) for t in 1:T]
    probit_I = [initialize_ordered_probit(κ_I[t], r_I; rng) for t in 1:T]
    probit_P = initialize_ordered_probit(collect(κ_P), R_P; rng)
    adult_emission = initialize_adult_outcome_emission(data.κ_adult, r_θ, R_ω; rng)
    pv0 = (; r_P = R_P, r_ω = R_ω, r_Y = R_Y,
           π0 = ones(r_θ, R_P, R_ω) / (r_θ * R_P * R_ω), π_Pω = ones(R_P, R_ω) / (R_P * R_ω), π_ε = fill(1 / R_Y, R_Y),
           trans_specs = [per_theta_from_add(initialize_ordrobit_add_v3(r_θ; rng, ν = NU), r_θ) for _ in TG],
           ccp_spec = initialize_ccp_v3_eps(r_I; rng),
           probit_P, Q_P = emission_matrices_from_probit(probit_P, R_P),
           income_model = initialize_income_model_eps(data.Y_income; delta_init = 0.15),
           adult_emission, adult_matrices = build_adult_matrices(adult_emission, data.κ_adult, r_θ, R_ω))
    psk0 = (; r_θ, r_I, T, P_θ = [emission_matrices_from_probit(probit_θ[t], r_θ) for t in 1:T],
            P_I = [emission_matrices_from_probit(probit_I[t], r_I) for t in 1:T], probit_θ, probit_I)
    println("\n=== DIRECT: per-skill ordered robit from a cold start ==="); flush(stdout)
    t0 = time(); pv, psk, γ, ξ, w, ll = run_em(pv0, psk0)
    @printf("DIRECT (per-skill, cold) done: LL = %.1f  (%.0f min)\n", ll, (time() - t0) / 60); flush(stdout)
    save_fit(outp("direct"), pv, psk, γ, ξ, w, ll, "per-theta direct")
end

# ── period-specific technology: start from a grouped fit's period-by-period re-fit at its
#    converged posteriors, then iterate full EM with one technology per transition ──
function stage_period()
    F = load(joinpath(REPO, "output", ENV["PERIOD_FROM"])); pvg, pskg = F["params_v3_eps"], F["params_skill"]
    specs = period_specific_specs(pvg, F["xi_sum_z"], r_θ, r_I; TG)
    println("\n=== PERIOD-SPECIFIC technology, EM from $(ENV["PERIOD_FROM"]) (LL $(round(F["best_ll"]; digits = 1))) ==="); flush(stdout)
    t0 = time(); pv, psk, γ, ξ, w, ll = run_em(merge(pvg, (; trans_specs = specs)), pskg; tg = TG_PERIOD)
    @printf("PERIOD-SPECIFIC done: LL = %.1f  (%+.1f vs grouped, %.0f min)\n", ll, ll - F["best_ll"], (time() - t0) / 60); flush(stdout)
    save_fit(outp("period"), pv, psk, γ, ξ, w, ll, "per-theta, period-specific"; tg = TG_PERIOD)
end

if MODE == "period"
    stage_period()
elseif MODE == "direct"
    stage_direct()
elseif MODE == "stage1"
    stage1()
elseif MODE == "pertheta"
    pv1, psk1, _ = load_stage1(); stage_pertheta(pv1, psk1)
elseif MODE == "warm"
    stage_warm()
elseif MODE == "quad"
    stage_quad()
elseif MODE == "nn"
    pv1, psk1, ξ1 = load_stage1(); stage_nn(pv1, psk1, ξ1)
else   # "all"
    pv1, psk1, ξ1 = stage1(); stage_pertheta(pv1, psk1)
end
println("emission stall restarts triggered: ", EMISSION_STALL_COUNT[])
