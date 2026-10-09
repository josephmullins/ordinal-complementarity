# ═══════════════════════════════════════════════════════════════
# Multi-period EM — V3 + ε_Y extension
#
# Adds a child-level discrete income type ε_Y ∈ {1,...,r_Y}.
# Regime z = (θ^P, ω, ε_Y), r_Z = r_P · r_ω · r_Y.
#
# Key differences from V3:
#   • r_Z expands from r_P·r_ω → r_P·r_ω·r_Y.
#   • ε_Y does NOT enter transitions.  Π_z is still indexed only by
#     (kP, kω); we reuse `precompute_Pi_z` from V3 unchanged.
#   • CCP and income equation both depend on ε_Y.  No income binning
#     — CCPs are built per regime and shared across children.
#   • Regime prior is independent: π_Z = π_Pω ⊗ π_ε.
#   • π0(θ_1 | z) depends on (kP, kω) only, not kε.
#   • Missing income naturally contributes 1 to regime posterior.
#
# Depends on:
#   parametric_models.jl, parametric_models_v3.jl, parametric_models_v3_eps.jl
#   estimation_multiperiod.jl   — compute_emission_likelihood_t!, _qsum_from_xi
#   estimation_multiperiod_v3.jl — regime_z, precompute_Pi_z (reused),
#                                   forward_backward_v3_n!,
#                                   mstep_transition_v3_grouped (adapted)
# ═══════════════════════════════════════════════════════════════

using Random, LinearAlgebra, Printf

# ─────────────────────────────────────────────────────────────────
# Regime emission log-likelihood (ASVAB + income with ε_Y)
# ─────────────────────────────────────────────────────────────────

function regime_loglik_eps(n::Int, kP::Int, kω::Int, kε::Int,
                            Q_P::Vector{Matrix{Float64}},
                            income_model::IncomeModelEps,
                            data;
                            P_grid::AbstractVector,
                            ω_grid::AbstractVector,
                            Y_grid::AbstractVector)
    ll = 0.0
    # ASVAB emissions (depend only on kP)
    for m in eachindex(data.X_P)
        x = data.X_P[m][n]
        x == 0 && continue
        ll += log(max(Q_P[m][x, kP], 1e-300))
    end
    # Income (all T periods; ε_Y enters via δ·ε_Y)
    kP_val = P_grid[kP]; kω_val = ω_grid[kω]; kε_val = Y_grid[kε]
    for t in 1:data.T
        ll += income_loglik_eps(data.Y_income[n, t],
                                 kP_val, kω_val, kε_val, income_model)
    end
    return ll
end

# ─────────────────────────────────────────────────────────────────
# Precompute CCP arrays for all 27 regimes (no income binning)
# ─────────────────────────────────────────────────────────────────

"""
    precompute_CCP_z_eps(ccp_spec, r_P, r_ω, r_Y, r_θ, r_I; grids...)
        -> (CCP_init_pre, CCP_trans_pre)

For each regime z ∈ 1..r_Z, build
  CCP_init_pre[z]  :: (r_I, r_θ)
  CCP_trans_pre[z] :: (r_I, r_I, r_θ)
"""
function precompute_CCP_z_eps(ccp_spec::OrderedProbitCCPV3Eps,
                                r_P::Int, r_ω::Int, r_Y::Int,
                                r_θ::Int, r_I::Int;
                                θ_grid::AbstractVector,
                                I_grid::AbstractVector,
                                P_grid::AbstractVector,
                                ω_grid::AbstractVector,
                                Y_grid::AbstractVector)
    r_Z = r_P * r_ω * r_Y
    CCP_init_pre  = Vector{Matrix{Float64}}(undef, r_Z)
    CCP_trans_pre = Vector{Array{Float64,3}}(undef, r_Z)
    for kε in 1:r_Y, kω in 1:r_ω, kP in 1:r_P
        z = eps_regime_index(kP, kω, kε, r_ω, r_Y)
        CCP_init_pre[z]  = build_ccp_v3_eps_init(ccp_spec,
            P_grid[kP], ω_grid[kω], Y_grid[kε];
            θ_grid, r_I)
        CCP_trans_pre[z] = build_ccp_v3_eps_trans(ccp_spec,
            P_grid[kP], ω_grid[kω], Y_grid[kε];
            θ_grid, I_grid, r_I)
    end
    return CCP_init_pre, CCP_trans_pre
end

# ─────────────────────────────────────────────────────────────────
# E-step
# ─────────────────────────────────────────────────────────────────

"""
    expectation_step_v3_eps!(...) -> total_log_likelihood

Mutated outputs:
  γ_agg         — (T, r_θ, r_I, N)                    aggregated posteriors
  xi_sum_z_full — [z][t] (r_θ,r_I,r_θ,r_I), z ∈ 1..r_Z_full (27)
  w_z           — (r_Z_full, N)                        regime posteriors
  gamma_z_t1    — (r_θ, r_I, r_Z_full, N)              period-1 regime posteriors
  ccp_init_ss   — (r_θ, r_I, r_Z_full)                 CCP init SS per regime
  ccp_trans_ss  — (r_I, r_I, r_θ, r_Z_full, T-1)       CCP trans SS per regime
  pi0_ss        — (r_θ, r_P, r_ω)                      π0 accumulator (no ε_Y dim)
  pi_eps_ss     — (r_Y,)                                π_ε accumulator
"""
function expectation_step_v3_eps!(
    γ_agg::Array{Float64,4},
    xi_sum_z_full::Vector{Vector{Array{Float64,4}}},
    w_z::Matrix{Float64},
    gamma_z_t1::Array{Float64,4},
    ccp_init_ss::Array{Float64,3},
    ccp_trans_ss::Array{Float64,5},
    pi0_ss::Array{Float64,3},
    pi_eps_ss::Vector{Float64},
    Pi_z_pre::Vector{Vector{Array{Float64,3}}},      # [z_trans][t], z_trans ∈ 1..r_P·r_ω
    CCP_init_pre::Vector{Matrix{Float64}},            # [z_full]
    CCP_trans_pre::Vector{Array{Float64,3}},          # [z_full]
    params_v3_eps,
    params_skill,
    data;
    θ_grid::AbstractVector,
    I_grid::AbstractVector,
    P_grid::AbstractVector,
    ω_grid::AbstractVector,
    Y_grid::AbstractVector,
)
    (; r_P, r_ω, r_Y, π0, π_Pω, π_ε, Q_P, income_model) = params_v3_eps
    adult_matrices = haskey(params_v3_eps, :adult_matrices) ?
        params_v3_eps.adult_matrices : Array{Float64,3}[]
    has_adult = !isempty(adult_matrices) &&
                haskey(data, :X_adult) && !isempty(data.X_adult)
    (; r_θ, r_I, T) = params_skill
    N = data.N
    r_Z = r_P * r_ω * r_Y

    # Reset accumulators
    fill!(γ_agg, 0.0)
    for z in 1:r_Z, t in 1:T-1; fill!(xi_sum_z_full[z][t], 0.0); end
    fill!(w_z, 0.0); fill!(gamma_z_t1, 0.0)
    fill!(ccp_init_ss, 0.0); fill!(ccp_trans_ss, 0.0)
    fill!(pi0_ss, 0.0); fill!(pi_eps_ss, 0.0)

    # FB scratch (reused across children and regimes)
    α     = zeros(T, r_θ, r_I)
    β     = zeros(T, r_θ, r_I)
    scale = zeros(T)
    L_scratch = [zeros(r_θ, r_I) for _ in 1:T]

    γ_z_all = [zeros(T, r_θ, r_I)                                    for _ in 1:r_Z]
    ξ_z_all = [[zeros(r_θ, r_I, r_θ, r_I) for _ in 1:T-1]           for _ in 1:r_Z]
    CCP_nz = Vector{Array{Float64,3}}(undef, T-1)

    log_π_Pω = [log(max(π_Pω[kP, kω], 1e-300)) for kP in 1:r_P, kω in 1:r_ω]
    log_π_ε  = [log(max(π_ε[kε], 1e-300))       for kε in 1:r_Y]
    log_ll_z  = zeros(r_Z)
    π0_z_buf  = zeros(r_θ)
    π0_nz_buf = zeros(r_θ, r_I)

    total_ll = 0.0

    for n in 1:N
        # Emission likelihoods (same across regimes)
        for t in 1:T
            compute_emission_likelihood_t!(L_scratch[t], t, n, params_skill, data)
        end

        # ── Pass 1: forward-backward per regime ──
        for kε in 1:r_Y, kω in 1:r_ω, kP in 1:r_P
            z = eps_regime_index(kP, kω, kε, r_ω, r_Y)
            z_trans = regime_z(kP, kω, r_ω)     # transition index (1..r_P·r_ω)

            ll_regime = regime_loglik_eps(n, kP, kω, kε, Q_P, income_model, data;
                                           P_grid, ω_grid, Y_grid)
            log_ll_z[z] = log_π_Pω[kP, kω] + log_π_ε[kε] + ll_regime

            # π0 conditional on (kP, kω): P(θ_1 | kP, kω), independent of kε
            for kθ in 1:r_θ; @inbounds π0_z_buf[kθ] = max(π0[kθ, kP, kω], 0.0); end
            s = sum(π0_z_buf)
            if s > 0; π0_z_buf ./= s; else; fill!(π0_z_buf, 1.0/r_θ); end

            ccp_init = CCP_init_pre[z]
            for kI in 1:r_I, kθ in 1:r_θ
                @inbounds π0_nz_buf[kθ, kI] = π0_z_buf[kθ] * ccp_init[kI, kθ]
            end

            for t in 1:T-1
                CCP_nz[t] = CCP_trans_pre[z]
            end

            for t in 1:T-1; fill!(ξ_z_all[z][t], 0.0); end

            ll_skills = forward_backward_v3_n!(
                α, β, scale, γ_z_all[z], ξ_z_all[z],
                π0_nz_buf, Pi_z_pre[z_trans], CCP_nz, L_scratch, r_θ, r_I, T)
            log_ll_z[z] += ll_skills

            # Adult outcomes (depend on θ_T and ω only; ε_Y does not enter)
            if has_adult
                γ_z = γ_z_all[z]
                for m in eachindex(data.X_adult)
                    x = data.X_adult[m][n]
                    x == 0 && continue
                    Am = adult_matrices[m]
                    for kθ in 1:r_θ
                        θT_w = 0.0
                        for kI in 1:r_I
                            @inbounds θT_w += γ_z[T, kθ, kI]
                        end
                        θT_w < 1e-15 && continue
                        @inbounds log_ll_z[z] += θT_w * log(max(Am[x, kθ, kω], 1e-300))
                    end
                end
            end
        end

        # Regime posteriors
        log_max = maximum(log_ll_z)
        w_unnorm = exp.(log_ll_z .- log_max)
        w_sum = sum(w_unnorm)
        total_ll += log(w_sum) + log_max

        # ── Pass 2: weighted accumulation ──
        for kε in 1:r_Y, kω in 1:r_ω, kP in 1:r_P
            z = eps_regime_index(kP, kω, kε, r_ω, r_Y)
            wz = w_unnorm[z] / w_sum
            @inbounds w_z[z, n] = wz
            wz < 1e-15 && continue
            γ_z = γ_z_all[z]
            ξ_z = ξ_z_all[z]

            # γ_agg (aggregated over all 27 regimes)
            for t in 1:T, kI in 1:r_I, kθ in 1:r_θ
                @inbounds γ_agg[t, kθ, kI, n] += wz * γ_z[t, kθ, kI]
            end

            # Regime-specific ξ (stored at full-regime index)
            for t in 1:T-1
                ξ_t = ξ_z[t]; ξ_acc = xi_sum_z_full[z][t]
                for kI in 1:r_I, kθ in 1:r_θ, kI1 in 1:r_I, kθ1 in 1:r_θ
                    @inbounds ξ_acc[kθ1, kI1, kθ, kI] += wz * ξ_t[kθ1, kI1, kθ, kI]
                end
            end

            # Period-1 posteriors + CCP init SS
            for kI in 1:r_I, kθ in 1:r_θ
                @inbounds gamma_z_t1[kθ, kI, z, n] = γ_z[1, kθ, kI]
                @inbounds ccp_init_ss[kθ, kI, z] += wz * γ_z[1, kθ, kI]
            end

            # CCP trans SS (aggregate ξ → (kI1, kI, kθ1, z, t), summing over kθ)
            for t in 1:T-1
                ξ_t = ξ_z[t]
                for kθ1 in 1:r_θ, kI in 1:r_I, kI1 in 1:r_I
                    s = 0.0
                    for kθ in 1:r_θ; @inbounds s += ξ_t[kθ1, kI1, kθ, kI]; end
                    @inbounds ccp_trans_ss[kI1, kI, kθ1, z, t] += wz * s
                end
            end

            # π0 SS (accumulated over ε_Y — π0 has no ε_Y dimension)
            for kI in 1:r_I, kθ in 1:r_θ
                @inbounds pi0_ss[kθ, kP, kω] += wz * γ_z[1, kθ, kI]
            end

            # π_ε SS
            @inbounds pi_eps_ss[kε] += wz
        end
    end

    return total_ll
end

# ─────────────────────────────────────────────────────────────────
# M-step: priors
# ─────────────────────────────────────────────────────────────────

function mstep_pi0_eps(pi0_ss::Array{Float64,3})
    s = sum(pi0_ss)
    π0 = s > 0 ? pi0_ss ./ s : fill(1.0 / length(pi0_ss), size(pi0_ss))
    π_Pω = dropdims(sum(π0; dims=1); dims=1)   # (r_P, r_ω)
    return π0, π_Pω
end

function mstep_pi_eps(pi_eps_ss::Vector{Float64})
    s = sum(pi_eps_ss)
    s > 0 ? pi_eps_ss ./ s : fill(1.0 / length(pi_eps_ss), length(pi_eps_ss))
end

# ─────────────────────────────────────────────────────────────────
# M-step: ASVAB emissions (marginalize over (kω, kε))
# ─────────────────────────────────────────────────────────────────

function mstep_Q_P_eps(probit_P::OrderedProbitEmission,
                        w_z::Matrix{Float64}, data,
                        r_P::Int, r_ω::Int, r_Y::Int;
                        options=DEFAULT_OPTIM_OPTIONS)
    N = size(w_z, 2)
    W_P = zeros(r_P, N)
    for n in 1:N, kε in 1:r_Y, kω in 1:r_ω, kP in 1:r_P
        z = eps_regime_index(kP, kω, kε, r_ω, r_Y)
        @inbounds W_P[kP, n] += w_z[z, n]
    end

    M = length(data.X_P)
    new_tau = Vector{Vector{Float64}}(undef, M)
    new_alpha = Vector{Float64}(undef, M)
    Q_P_new = Vector{Matrix{Float64}}(undef, M)

    for m in 1:M
        κ_m = data.κ_P[m]
        n_cut = κ_m - 1
        x0 = vcat(_pack_cutpoints(probit_P.tau[m]), probit_P.alpha[m])
        obj = x -> _single_measure_negll(x, W_P, data.X_P[m], κ_m, r_P)
        xopt = try
            result = optimize(obj, x0, BFGS(), options; autodiff=:forward)
            Optim.minimizer(result)
        catch
            result = optimize(obj, x0, NelderMead(), options)
            Optim.minimizer(result)
        end
        new_tau[m] = _unpack_cutpoints(xopt[1:n_cut])
        new_alpha[m] = xopt[n_cut + 1]
        Q_P_new[m] = ordered_probit_matrix(new_tau[m], new_alpha[m], r_P)
    end

    return OrderedProbitEmission(new_tau, new_alpha), Q_P_new
end

# ─────────────────────────────────────────────────────────────────
# M-step: adult outcomes (marginalize regime over (kP, kε) → w_ω)
# ─────────────────────────────────────────────────────────────────

function mstep_adult_outcomes_eps(adult_emission::AdultOutcomeEmission,
                                    γ_agg::Array{Float64,4},
                                    w_z::Matrix{Float64},
                                    data,
                                    r_θ::Int, r_I::Int,
                                    r_P::Int, r_ω::Int, r_Y::Int;
                                    θ_grid::AbstractVector=1.0:r_θ,
                                    ω_grid::AbstractVector=1.0:r_ω,
                                    options=DEFAULT_OPTIM_OPTIONS)
    N = data.N; T = data.T
    κ_adult = data.κ_adult
    M_adult = length(κ_adult)
    SS_adult = [zeros(κ_adult[m], r_θ, r_ω) for m in 1:M_adult]

    for n in 1:N
        for kω in 1:r_ω
            w_omega = 0.0
            for kε in 1:r_Y, kP in 1:r_P
                z = eps_regime_index(kP, kω, kε, r_ω, r_Y)
                @inbounds w_omega += w_z[z, n]
            end
            w_omega < 1e-15 && continue
            for kθ in 1:r_θ
                θT_w = 0.0
                for kI in 1:r_I
                    @inbounds θT_w += γ_agg[T, kθ, kI, n]
                end
                weight = w_omega * θT_w
                weight < 1e-15 && continue
                for m in 1:M_adult
                    x = data.X_adult[m][n]
                    x == 0 && continue
                    @inbounds SS_adult[m][x, kθ, kω] += weight
                end
            end
        end
    end

    x0 = pack_adult_emission(adult_emission)
    obj = x -> adult_emission_negll(x, SS_adult, κ_adult, r_θ, r_ω; θ_grid, ω_grid)
    xopt = try
        result = optimize(obj, x0, LBFGS(), options; autodiff=:forward)
        Optim.minimizer(result)
    catch
        result = optimize(obj, x0, NelderMead(), options)
        Optim.minimizer(result)
    end
    new_ae = unpack_adult_emission(xopt, κ_adult)
    new_matrices = build_adult_matrices(new_ae, κ_adult, r_θ, r_ω; θ_grid, ω_grid)
    return new_ae, new_matrices
end

# ─────────────────────────────────────────────────────────────────
# M-step: transitions  (aggregate ξ over ε_Y → 9 transition regimes)
# ─────────────────────────────────────────────────────────────────

"""
Collapse xi_sum_z_full (r_Z = r_P·r_ω·r_Y entries) over ε_Y to produce
xi_sum_z_trans (r_P·r_ω entries) usable by the existing V3 transition M-step.
"""
function collapse_xi_over_eps(xi_sum_z_full::Vector{Vector{Array{Float64,4}}},
                                r_P::Int, r_ω::Int, r_Y::Int, T_minus_1::Int,
                                r_θ::Int, r_I::Int)
    r_Z_trans = r_P * r_ω
    xi_sum_z_trans = [[zeros(r_θ, r_I, r_θ, r_I) for _ in 1:T_minus_1] for _ in 1:r_Z_trans]
    for kω in 1:r_ω, kP in 1:r_P
        z_trans = regime_z(kP, kω, r_ω)
        for kε in 1:r_Y
            z_full = eps_regime_index(kP, kω, kε, r_ω, r_Y)
            for t in 1:T_minus_1
                xi_sum_z_trans[z_trans][t] .+= xi_sum_z_full[z_full][t]
            end
        end
    end
    return xi_sum_z_trans
end

# ─────────────────────────────────────────────────────────────────
# EM loop
# ─────────────────────────────────────────────────────────────────

"""
    expectation_maximization_v3_eps(data, params_v3_eps_init, params_skill_init;
                                      transition_groups, tol, maxiter, verbose, ...)

Returns (params_v3_eps, params_skill, γ_agg, xi_sum_z_full, w_z, ll_final).
"""
function expectation_maximization_v3_eps(
    data,
    params_v3_eps_init,
    params_skill_init;
    transition_groups::Vector{Vector{Int}} = [[1,2,3],[4,5,6]],
    tol::Float64 = 1e-4,
    maxiter::Int  = 500,
    verbose::Bool = true,
    mstep_options = DEFAULT_OPTIM_OPTIONS,
    λ_logw::Float64 = 0.0,   # ridge on robit-NN log_w (0 ⇒ unchanged behavior)
)
    (; r_θ, r_I, T) = params_skill_init
    (; r_P, r_ω, r_Y) = params_v3_eps_init
    N = data.N
    r_Z = r_P * r_ω * r_Y
    r_Z_trans = r_P * r_ω
    T_minus_1 = T - 1

    θ_grid = collect(1.0:r_θ); I_grid = collect(1.0:r_I)
    P_grid = collect(1.0:r_P); ω_grid = collect(1.0:r_ω)
    Y_grid = collect(1.0:r_Y)

    # Allocate
    γ_agg         = zeros(T, r_θ, r_I, N)
    xi_sum_z_full = [[zeros(r_θ, r_I, r_θ, r_I) for _ in 1:T_minus_1] for _ in 1:r_Z]
    w_z           = zeros(r_Z, N)
    gamma_z_t1    = zeros(r_θ, r_I, r_Z, N)
    ccp_init_ss   = zeros(r_θ, r_I, r_Z)
    ccp_trans_ss  = zeros(r_I, r_I, r_θ, r_Z, T_minus_1)
    pi0_ss        = zeros(r_θ, r_P, r_ω)
    pi_eps_ss     = zeros(r_Y)

    params_v3_eps = params_v3_eps_init
    params_skill  = params_skill_init

    err = Inf; iter = 0
    Pi_z_prev = nothing

    while err > tol && iter < maxiter
        # ── Precompute transition arrays (indexed by z_trans ∈ 1..9) ──
        Pi_z_pre = precompute_Pi_z(params_v3_eps.trans_specs, transition_groups,
                                     r_θ, r_I, r_P, r_ω;
                                     θ_grid, I_grid, P_grid, ω_grid)

        # ── Precompute CCP arrays (indexed by full regime z ∈ 1..27) ──
        CCP_init_pre, CCP_trans_pre = precompute_CCP_z_eps(
            params_v3_eps.ccp_spec, r_P, r_ω, r_Y, r_θ, r_I;
            θ_grid, I_grid, P_grid, ω_grid, Y_grid)

        # ── E-step ──
        ll = expectation_step_v3_eps!(
            γ_agg, xi_sum_z_full, w_z, gamma_z_t1,
            ccp_init_ss, ccp_trans_ss, pi0_ss, pi_eps_ss,
            Pi_z_pre, CCP_init_pre, CCP_trans_pre,
            params_v3_eps, params_skill, data;
            θ_grid, I_grid, P_grid, ω_grid, Y_grid)

        # ── M-step: child skill/investment emissions ──
        probit_θ_new, P_θ_new = parametric_mstep_emission_mp(
            params_skill.probit_θ, γ_agg, data, :theta, r_θ; options=mstep_options)
        probit_I_new, P_I_new = parametric_mstep_emission_mp(
            params_skill.probit_I, γ_agg, data, :I, r_I; options=mstep_options)

        # ── M-step: transitions (aggregate xi over ε_Y first) ──
        xi_sum_z_trans = collapse_xi_over_eps(xi_sum_z_full, r_P, r_ω, r_Y,
                                                T_minus_1, r_θ, r_I)
        new_trans_specs = mstep_transition_v3_grouped(
            params_v3_eps.trans_specs, xi_sum_z_trans, transition_groups,
            r_θ, r_I, r_P, r_ω;
            θ_grid, I_grid, P_grid, ω_grid, options=mstep_options, λ_logw=λ_logw)

        # ── M-step: CCP ──
        x0_ccp = pack_ccp_v3_eps(params_v3_eps.ccp_spec)
        obj_ccp = x -> ccp_negll_v3_eps(x, ccp_init_ss, ccp_trans_ss,
                                          r_θ, r_I, r_P, r_ω, r_Y, T;
                                          θ_grid, I_grid, P_grid, ω_grid, Y_grid)
        res_ccp = optimize(obj_ccp, x0_ccp, LBFGS(), mstep_options; autodiff=:forward)
        new_ccp_spec = unpack_ccp_v3_eps(Optim.minimizer(res_ccp), r_I)

        # ── M-step: π0, π_Pω, π_ε ──
        new_π0, new_π_Pω = mstep_pi0_eps(pi0_ss)
        new_π_ε          = mstep_pi_eps(pi_eps_ss)

        # ── M-step: Q_P (ordered probit) ──
        new_probit_P, new_Q_P = mstep_Q_P_eps(
            params_v3_eps.probit_P, w_z, data, r_P, r_ω, r_Y; options=mstep_options)

        # ── M-step: income model (adds δ) ──
        new_income = update_income_model_eps(
            w_z, data.Y_income, r_P, r_ω, r_Y;
            P_grid, ω_grid, Y_grid)

        # ── M-step: adult outcomes ──
        has_adult = haskey(data, :X_adult) && !isempty(data.X_adult) &&
                    haskey(data, :κ_adult) && !isempty(data.κ_adult)
        new_adult_emission = params_v3_eps.adult_emission
        new_adult_matrices = params_v3_eps.adult_matrices
        if has_adult && !isempty(params_v3_eps.adult_emission.tau)
            new_adult_emission, new_adult_matrices = mstep_adult_outcomes_eps(
                params_v3_eps.adult_emission, γ_agg, w_z, data,
                r_θ, r_I, r_P, r_ω, r_Y;
                θ_grid, ω_grid, options=mstep_options)
        end

        # ── Convergence: max|ΔΠ| (over the 9 transition regimes) ──
        Pi_z_new = precompute_Pi_z(new_trans_specs, transition_groups,
                                     r_θ, r_I, r_P, r_ω;
                                     θ_grid, I_grid, P_grid, ω_grid)
        if Pi_z_prev !== nothing
            err = 0.0
            for z in 1:r_Z_trans, t in 1:T_minus_1
                err = max(err, maximum(abs.(Pi_z_new[z][t] .- Pi_z_prev[z][t])))
            end
        end
        Pi_z_prev = Pi_z_new

        iter += 1
        if verbose && (iter ≤ 5 || iter % 25 == 0)
            σ_inc = round(exp(new_income.log_sigma); digits=3)
            println("  iter $iter: max|ΔΠ| = $(round(err; sigdigits=4)), " *
                    "ll = $(round(ll; digits=2)), " *
                    "δ = $(round(new_income.delta; digits=3)), " *
                    "β₅ = $(round(new_ccp_spec.beta5; digits=3)), " *
                    "σ_inc = $σ_inc")
        end

        params_v3_eps = (;
            r_P, r_ω, r_Y,
            π0 = new_π0, π_Pω = new_π_Pω, π_ε = new_π_ε,
            trans_specs = new_trans_specs,
            ccp_spec = new_ccp_spec,
            probit_P = new_probit_P,
            Q_P = new_Q_P,
            income_model = new_income,
            adult_emission = new_adult_emission,
            adult_matrices = new_adult_matrices,
        )
        params_skill = (; r_θ, r_I, T,
            P_θ = P_θ_new, P_I = P_I_new,
            probit_θ = probit_θ_new, probit_I = probit_I_new)
    end

    if verbose
        println("  Converged after $iter iterations (err = $(round(err; sigdigits=4)))")
    end

    # Final E-step for consistent LL at final parameters
    Pi_z_final = precompute_Pi_z(params_v3_eps.trans_specs, transition_groups,
                                   r_θ, r_I, r_P, r_ω;
                                   θ_grid, I_grid, P_grid, ω_grid)
    CCP_init_final, CCP_trans_final = precompute_CCP_z_eps(
        params_v3_eps.ccp_spec, r_P, r_ω, r_Y, r_θ, r_I;
        θ_grid, I_grid, P_grid, ω_grid, Y_grid)
    ll_final = expectation_step_v3_eps!(
        γ_agg, xi_sum_z_full, w_z, gamma_z_t1,
        ccp_init_ss, ccp_trans_ss, pi0_ss, pi_eps_ss,
        Pi_z_final, CCP_init_final, CCP_trans_final,
        params_v3_eps, params_skill, data;
        θ_grid, I_grid, P_grid, ω_grid, Y_grid)
    if verbose; println("  Final log-likelihood: $(round(ll_final; digits=2))"); end

    return params_v3_eps, params_skill, γ_agg, xi_sum_z_full, w_z, ll_final
end

# ─────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────

function display_results_v3_eps(params_v3_eps, params_skill, data)
    (; r_P, r_ω, r_Y, π_Pω, π_ε, ccp_spec, Q_P, income_model) = params_v3_eps
    (; r_θ, r_I, T) = params_skill

    println("\n" * "="^60)
    println("  V3 + ε_Y RESULTS  r_θ=$r_θ r_I=$r_I r_P=$r_P r_ω=$r_ω r_Y=$r_Y T=$T")
    println("="^60)

    println("\n  π_Pω[kP, kω]:")
    for kP in 1:r_P
        vals = ["$(round(π_Pω[kP,kω]; digits=3))" for kω in 1:r_ω]
        println("    kP=$kP:  $(join(vals, "  "))")
    end

    println("\n  π_ε[kε] = $(round.(π_ε; digits=3))")

    println("\n  Income model:")
    @printf("    β₀=%.3f  β_P=%.3f  β_ω=%.3f  δ=%.3f  σ=%.3f\n",
            income_model.beta0, income_model.betaP, income_model.betaω,
            income_model.delta, exp(income_model.log_sigma))

    println("\n  CCP (OrderedProbitCCPV3Eps):")
    @printf("    β₁(θ')=%.3f  β₃(θ^P)=%.3f  β₄(ω)=%.3f  β₅(ε_Y)=%.3f\n",
            ccp_spec.beta1, ccp_spec.beta3, ccp_spec.beta4, ccp_spec.beta5)
    println("    c = $(round.(ccp_spec.c; digits=3))")
end
