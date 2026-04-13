# ═══════════════════════════════════════════════════════════════
# Multi-period EM — Version 3: regime-extended model
#
# Adds time-invariant latent variables θ^P and ω via a
# mixture-over-regimes E-step.  For each child, run r_Z = r_P*r_ω
# forward-backward passes, then aggregate with regime posteriors.
#
# Depends on:
#   parametric_models.jl    — Φ, _ordered_probit_pmf, etc.
#   parametric_models_v3.jl — V3 structs
#   estimation_multiperiod.jl — compute_emission_likelihood_t!,
#       _qsum_from_xi, _ccpsum_from_xi, parametric_mstep_emission_mp
# ═══════════════════════════════════════════════════════════════

using Random, LinearAlgebra

# ─────────────────────────────────────────────────────────────────
# Regime index helpers
# ─────────────────────────────────────────────────────────────────

@inline regime_z(kP::Int, kω::Int, r_ω::Int) = (kP - 1) * r_ω + kω

@inline function regime_kPkω(z::Int, r_ω::Int)
    kω = ((z - 1) % r_ω) + 1
    kP = (z - kω) ÷ r_ω + 1
    return kP, kω
end

# ─────────────────────────────────────────────────────────────────
# Precompute regime-specific transition arrays
#
# Transitions depend on the regime (through alpha3, alpha4) but NOT
# on the child.  Build once per EM iteration, before the child loop.
# ─────────────────────────────────────────────────────────────────

"""
    precompute_Pi_z(trans_specs, transition_groups, r_θ, r_I, r_P, r_ω; grids...)
        -> Vector{Vector{Array{Float64,3}}}

Returns Pi_z[z][t] :: (r_θ, r_θ, r_I) for each regime z and transition t.
`trans_specs` is a vector of one RobitNNTransitionV3 per transition group.
Transitions in the same group share the same spec; each regime produces
a different Π by plugging in (P_grid[kP], ω_grid[kω]).
"""
function precompute_Pi_z(trans_specs::Vector{RobitNNTransitionV3},
                           transition_groups::Vector{Vector{Int}},
                           r_θ::Int, r_I::Int, r_P::Int, r_ω::Int;
                           θ_grid::AbstractVector=1.0:r_θ,
                           I_grid::AbstractVector=1.0:r_I,
                           P_grid::AbstractVector=1.0:r_P,
                           ω_grid::AbstractVector=1.0:r_ω)
    r_Z = r_P * r_ω
    T_minus_1 = sum(length, transition_groups)
    Pi_z = [Vector{Array{Float64,3}}(undef, T_minus_1) for _ in 1:r_Z]

    for (g, group) in enumerate(transition_groups)
        spec = trans_specs[g]
        vec_g = pack_robit_nn_v3(spec)
        K = spec.K; ν = spec.ν
        for kω in 1:r_ω, kP in 1:r_P
            z = regime_z(kP, kω, r_ω)
            Π = _build_robit_nn_v3_pi(vec_g, r_θ, r_I, K, ν,
                                        P_grid[kP], ω_grid[kω]; θ_grid, I_grid)
            for t in group
                Pi_z[z][t] = Π   # same Π for all t in group
            end
        end
    end
    return Pi_z
end

# ─────────────────────────────────────────────────────────────────
# Regime emission log-likelihood (ASVAB + income)
# ─────────────────────────────────────────────────────────────────

"""
    regime_loglik(n, kP, kω, Q_P, income_model, data; P_grid, ω_grid) -> Float64

Log-likelihood of parental measures and income for child n at regime (kP, kω).
"""
function regime_loglik(n::Int, kP::Int, kω::Int,
                        Q_P::Vector{Matrix{Float64}},
                        income_model::IncomeModel,
                        data;
                        P_grid::AbstractVector, ω_grid::AbstractVector)
    ll = 0.0
    # ASVAB emissions
    for m in eachindex(data.X_P)
        x = data.X_P[m][n]
        x == 0 && continue
        ll += log(max(Q_P[m][x, kP], 1e-300))
    end
    # Income (all T periods)
    kP_val = P_grid[kP]; kω_val = ω_grid[kω]
    for t in 1:data.T
        ll += income_loglik(data.Y_income[n, t], kP_val, kω_val, income_model)
    end
    return ll
end

# ─────────────────────────────────────────────────────────────────
# Forward-backward for one child at one regime
# ─────────────────────────────────────────────────────────────────

"""
    forward_backward_v3_n!(α, β, scale, γ_z, ξ_z,
                            π0_nz, Pi_z, CCP_nz, L_scratch,
                            r_θ, r_I, T) -> log_likelihood

Run scaled forward-backward for one child at one regime.

Arguments:
  α, β   — (T, r_θ, r_I) scratch
  scale  — (T,) scratch
  γ_z    — (T, r_θ, r_I)  written in place
  ξ_z    — Vector{T-1} of (r_θ, r_I, r_θ, r_I), accumulated (caller must reset)
  π0_nz  — (r_θ, r_I) joint initial = π0_marginal_z[kθ] * CCP_init[kI, kθ]
  Pi_z   — Vector{T-1} of (r_θ, r_θ, r_I) regime-specific transitions
  CCP_nz — Vector{T-1} of (r_I, r_I, r_θ) child+regime-specific CCPs
  L_scratch — Vector{T} of (r_θ, r_I) emission likelihoods (precomputed)
"""
function forward_backward_v3_n!(
    α::Array{Float64,3}, β::Array{Float64,3}, scale::Vector{Float64},
    γ_z::Array{Float64,3}, ξ_z::Vector{Array{Float64,4}},
    π0_nz::Matrix{Float64},
    Pi_z::Vector{Array{Float64,3}},
    CCP_nz::Vector{Array{Float64,3}},
    L_scratch::Vector{Matrix{Float64}},
    r_θ::Int, r_I::Int, T::Int
)
    # ── Forward pass: t = 1 ──
    s = 0.0
    for kI in 1:r_I, kθ in 1:r_θ
        @inbounds v = π0_nz[kθ, kI] * L_scratch[1][kθ, kI]
        @inbounds α[1, kθ, kI] = v
        s += v
    end
    scale[1] = s
    if s > 0
        for kI in 1:r_I, kθ in 1:r_θ; @inbounds α[1, kθ, kI] /= s; end
    end

    # ── Forward pass: t = 2, …, T ──
    for t in 1:T-1
        Π_t   = Pi_z[t]
        CCP_t = CCP_nz[t]
        L_next = L_scratch[t+1]

        # Stage 1: A[kθ1, kI] = Σ_kθ α[t, kθ, kI] * Π[kθ1, kθ, kI]
        for kI in 1:r_I, kθ1 in 1:r_θ; @inbounds α[t+1, kθ1, kI] = 0.0; end
        for kI in 1:r_I, kθ in 1:r_θ, kθ1 in 1:r_θ
            @inbounds α[t+1, kθ1, kI] += α[t, kθ, kI] * Π_t[kθ1, kθ, kI]
        end

        # Stage 2: temp[kθ1, kI1] = Σ_kI A[kθ1, kI] * CCP[kI1, kI, kθ1]
        #   (reuse β[t+1] as scratch — overwritten in backward pass)
        for kI1 in 1:r_I, kθ1 in 1:r_θ; @inbounds β[t+1, kθ1, kI1] = 0.0; end
        for kI in 1:r_I, kθ1 in 1:r_θ, kI1 in 1:r_I
            @inbounds β[t+1, kθ1, kI1] += α[t+1, kθ1, kI] * CCP_t[kI1, kI, kθ1]
        end

        # Multiply by emission likelihood and scale
        s = 0.0
        for kI1 in 1:r_I, kθ1 in 1:r_θ
            @inbounds α[t+1, kθ1, kI1] = β[t+1, kθ1, kI1] * L_next[kθ1, kI1]
            @inbounds s += α[t+1, kθ1, kI1]
        end
        scale[t+1] = s
        if s > 0
            for kI1 in 1:r_I, kθ1 in 1:r_θ; @inbounds α[t+1, kθ1, kI1] /= s; end
        end
    end

    # ── Backward pass ──
    for kI in 1:r_I, kθ in 1:r_θ; @inbounds β[T, kθ, kI] = 1.0; end

    for t in T-1:-1:1
        Π_t   = Pi_z[t]
        CCP_t = CCP_nz[t]
        L_next = L_scratch[t+1]
        sc = scale[t+1]

        for kI in 1:r_I, kθ in 1:r_θ
            val = 0.0
            for kθ1 in 1:r_θ
                inner = 0.0
                for kI1 in 1:r_I
                    @inbounds inner += CCP_t[kI1, kI, kθ1] * L_next[kθ1, kI1] * β[t+1, kθ1, kI1]
                end
                @inbounds val += Π_t[kθ1, kθ, kI] * inner
            end
            @inbounds β[t, kθ, kI] = val / sc
        end
    end

    # ── Marginal posteriors γ_z ──
    for t in 1:T
        s = 0.0
        for kI in 1:r_I, kθ in 1:r_θ
            @inbounds v = α[t, kθ, kI] * β[t, kθ, kI]
            @inbounds γ_z[t, kθ, kI] = v
            s += v
        end
        if s > 0
            for kI in 1:r_I, kθ in 1:r_θ; @inbounds γ_z[t, kθ, kI] /= s; end
        end
    end

    # ── Pairwise posteriors: accumulate into ξ_z ──
    for t in 1:T-1
        Π_t   = Pi_z[t]
        CCP_t = CCP_nz[t]
        L_next = L_scratch[t+1]
        sc = scale[t+1]
        ξ_t = ξ_z[t]

        for kI in 1:r_I, kθ in 1:r_θ
            @inbounds a_val = α[t, kθ, kI]
            a_val == 0.0 && continue
            for kθ1 in 1:r_θ
                @inbounds pi_val = a_val * Π_t[kθ1, kθ, kI]
                pi_val == 0.0 && continue
                for kI1 in 1:r_I
                    @inbounds ξ_t[kθ1, kI1, kθ, kI] +=
                        pi_val * CCP_t[kI1, kI, kθ1] *
                        L_next[kθ1, kI1] * β[t+1, kθ1, kI1] / sc
                end
            end
        end
    end

    # ── Log-likelihood ──
    ll = 0.0
    for t in 1:T
        ll += scale[t] > 0 ? log(scale[t]) : -1e10
    end
    return ll
end

# ─────────────────────────────────────────────────────────────────
# E-step
# ─────────────────────────────────────────────────────────────────

"""
    expectation_step_v3!(γ_agg, xi_sum_z, w_z, gamma_z_t1, ccp_xi_z, pi0_ss,
                          Pi_z_pre, params_v3, params_skill, data; grids...)
        -> total_log_likelihood

Full V3 E-step.  For each child, run r_Z forward-backward passes (with
precomputed emissions — one compute per child, not per regime), then
aggregate with regime posteriors.

Mutated outputs:
  γ_agg      — (T, r_θ, r_I, N)              aggregated posteriors
  xi_sum_z   — [z][t] (r_θ,r_I,r_θ,r_I)      regime-specific pairwise SS
  w_z        — (r_Z, N)                       regime posteriors
  gamma_z_t1 — (r_θ, r_I, r_Z, N)             period-1 regime posteriors (for CCP/π0 M-step)
  ccp_xi_z   — (r_I, r_I, r_θ, r_Z, T-1, N)   per-child CCP SS (for CCP M-step)
  pi0_ss     — (r_θ, r_P, r_ω)                π0 accumulator
"""
function expectation_step_v3!(
    γ_agg::Array{Float64,4},
    xi_sum_z::Vector{Vector{Array{Float64,4}}},   # [z][t]
    w_z::Matrix{Float64},
    gamma_z_t1::Array{Float64,4},
    ccp_xi_z::Array{Float64,6},
    pi0_ss::Array{Float64,3},
    Pi_z_pre::Vector{Vector{Array{Float64,3}}},   # [z][t] precomputed
    params_v3,          # (; r_P, r_ω, π0, π_Z, ccp_spec, Q_P, income_model, trans_specs)
    params_skill,       # (; r_θ, r_I, T, P_θ, P_I)
    data;
    θ_grid::AbstractVector = 1.0:params_skill.r_θ,
    I_grid::AbstractVector = 1.0:params_skill.r_I,
    P_grid::AbstractVector = 1.0:params_v3.r_P,
    ω_grid::AbstractVector = 1.0:params_v3.r_ω,
)
    (; r_P, r_ω, π0, π_Z, ccp_spec, Q_P, income_model) = params_v3
    (; r_θ, r_I, T) = params_skill
    N = data.N
    r_Z = r_P * r_ω

    # Reset accumulators
    fill!(γ_agg, 0.0)
    for z in 1:r_Z, t in 1:T-1; fill!(xi_sum_z[z][t], 0.0); end
    fill!(w_z, 0.0); fill!(gamma_z_t1, 0.0); fill!(ccp_xi_z, 0.0); fill!(pi0_ss, 0.0)

    # FB scratch (reused across children and regimes)
    α     = zeros(T, r_θ, r_I)
    β     = zeros(T, r_θ, r_I)
    scale = zeros(T)
    L_scratch = [zeros(r_θ, r_I) for _ in 1:T]

    # Per-regime scratch for one child (stored for all z before weighting)
    γ_z_all = [zeros(T, r_θ, r_I)                                   for _ in 1:r_Z]
    ξ_z_all = [[zeros(r_θ, r_I, r_θ, r_I) for _ in 1:T-1]          for _ in 1:r_Z]

    log_π_Z   = [log(max(π_Z[kP, kω], 1e-300)) for kP in 1:r_P, kω in 1:r_ω]
    log_ll_z  = zeros(r_Z)
    π0_z_buf  = zeros(r_θ)
    π0_nz_buf = zeros(r_θ, r_I)

    total_ll = 0.0

    for n in 1:N
        # ── Compute emission likelihoods ONCE (same for all regimes) ──
        for t in 1:T
            compute_emission_likelihood_t!(L_scratch[t], t, n, params_skill, data)
        end

        # ── Pass 1: forward-backward for each regime ──
        for kω in 1:r_ω, kP in 1:r_P
            z = regime_z(kP, kω, r_ω)
            kP_val = P_grid[kP]; kω_val = ω_grid[kω]

            # Regime emission log-likelihood
            ll_regime = regime_loglik(n, kP, kω, Q_P, income_model, data;
                                       P_grid, ω_grid)
            log_ll_z[z] = log_π_Z[kP, kω] + ll_regime

            # π0 conditional on z: P(θ_1 | z) ∝ π0[kθ, kP, kω]
            for kθ in 1:r_θ; @inbounds π0_z_buf[kθ] = max(π0[kθ, kP, kω], 0.0); end
            s = sum(π0_z_buf)
            if s > 0; π0_z_buf ./= s; else; fill!(π0_z_buf, 1.0/r_θ); end

            # Build initial joint: π0_nz[kθ,kI] = P(θ_1|z) * CCP_init(I_1|θ_1,I_0=0,z,y_1)
            ccp_init = build_ccp_v3_init(ccp_spec, kP_val, kω_val,
                                           data.Y_income[n, 1]; θ_grid, r_I)
            for kI in 1:r_I, kθ in 1:r_θ
                @inbounds π0_nz_buf[kθ, kI] = π0_z_buf[kθ] * ccp_init[kI, kθ]
            end

            # Build per-transition CCPs
            CCP_nz = Vector{Array{Float64,3}}(undef, T-1)
            for t in 1:T-1
                CCP_nz[t] = build_ccp_v3_trans(ccp_spec, kP_val, kω_val,
                                                 data.Y_income[n, t+1];
                                                 θ_grid, I_grid, r_I)
            end

            # Reset ξ scratch for this regime
            for t in 1:T-1; fill!(ξ_z_all[z][t], 0.0); end

            # Forward-backward
            ll_skills = forward_backward_v3_n!(
                α, β, scale, γ_z_all[z], ξ_z_all[z],
                π0_nz_buf, Pi_z_pre[z], CCP_nz, L_scratch, r_θ, r_I, T)

            log_ll_z[z] += ll_skills
        end

        # ── Regime posteriors ──
        log_max = maximum(log_ll_z)
        w_unnorm = exp.(log_ll_z .- log_max)
        w_sum = sum(w_unnorm)
        total_ll += log(w_sum) + log_max

        # ── Pass 2: weighted accumulation ──
        for kω in 1:r_ω, kP in 1:r_P
            z = regime_z(kP, kω, r_ω)
            wz = w_unnorm[z] / w_sum
            @inbounds w_z[z, n] = wz
            wz < 1e-15 && continue

            γ_z = γ_z_all[z]
            ξ_z = ξ_z_all[z]

            # Aggregated marginal posteriors
            for t in 1:T, kI in 1:r_I, kθ in 1:r_θ
                @inbounds γ_agg[t, kθ, kI, n] += wz * γ_z[t, kθ, kI]
            end

            # Regime-specific pairwise posteriors (accumulated over n)
            for t in 1:T-1
                ξ_t = ξ_z[t]; ξ_acc = xi_sum_z[z][t]
                for kI in 1:r_I, kθ in 1:r_θ, kI1 in 1:r_I, kθ1 in 1:r_θ
                    @inbounds ξ_acc[kθ1, kI1, kθ, kI] += wz * ξ_t[kθ1, kI1, kθ, kI]
                end
            end

            # Period-1 posteriors (for CCP M-step initial and π0)
            for kI in 1:r_I, kθ in 1:r_θ
                @inbounds gamma_z_t1[kθ, kI, z, n] = γ_z[1, kθ, kI]
            end

            # Per-child CCP sufficient stats: _ccpsum_from_xi per (z, t)
            for t in 1:T-1
                ξ_t = ξ_z[t]
                for kθ1 in 1:r_θ, kI in 1:r_I, kI1 in 1:r_I
                    s = 0.0
                    for kθ in 1:r_θ; @inbounds s += ξ_t[kθ1, kI1, kθ, kI]; end
                    @inbounds ccp_xi_z[kI1, kI, kθ1, z, t, n] = s
                end
            end

            # π0 SS
            for kI in 1:r_I, kθ in 1:r_θ
                @inbounds pi0_ss[kθ, kP, kω] += wz * γ_z[1, kθ, kI]
            end
        end
    end

    return total_ll
end

# ─────────────────────────────────────────────────────────────────
# M-step components
# ─────────────────────────────────────────────────────────────────

"""Update π0[kθ, kP, kω] and derive π_Z."""
function mstep_pi0(pi0_ss::Array{Float64,3})
    s = sum(pi0_ss)
    π0 = s > 0 ? pi0_ss ./ s : fill(1.0 / length(pi0_ss), size(pi0_ss))
    π_Z = dropdims(sum(π0; dims=1); dims=1)   # (r_P, r_ω)
    return π0, π_Z
end

"""Update Q_P[m] nonparametrically from regime posteriors."""
function mstep_Q_P(w_z::Matrix{Float64}, data, r_P::Int, r_ω::Int)
    N = size(w_z, 2)
    Q_P_new = Vector{Matrix{Float64}}(undef, length(data.X_P))

    for m in eachindex(data.X_P)
        κ_m = data.κ_P[m]
        Q = zeros(κ_m, r_P)
        denom = zeros(r_P)

        for n in 1:N
            x = data.X_P[m][n]
            x == 0 && continue
            for kω in 1:r_ω, kP in 1:r_P
                z = regime_z(kP, kω, r_ω)
                @inbounds wz = w_z[z, n]
                @inbounds Q[x, kP] += wz
                @inbounds denom[kP] += wz
            end
        end

        for kP in 1:r_P
            if denom[kP] > 0
                Q[:, kP] ./= denom[kP]
            else
                Q[:, kP] .= 1.0 / κ_m
            end
        end
        Q_P_new[m] = Q
    end
    return Q_P_new
end

"""Grouped transition M-step for V3 robit NN."""
function mstep_transition_v3_grouped(
    trans_specs::Vector{RobitNNTransitionV3},
    xi_sum_z::Vector{Vector{Array{Float64,4}}},
    transition_groups::Vector{Vector{Int}},
    r_θ::Int, r_I::Int, r_P::Int, r_ω::Int;
    θ_grid::AbstractVector=collect(1.0:r_θ),
    I_grid::AbstractVector=collect(1.0:r_I),
    P_grid::AbstractVector=collect(1.0:r_P),
    ω_grid::AbstractVector=collect(1.0:r_ω),
    options=DEFAULT_OPTIM_OPTIONS,
)
    r_Z = r_P * r_ω
    new_specs = Vector{RobitNNTransitionV3}(undef, length(transition_groups))

    for (g, group) in enumerate(transition_groups)
        # Pool Q_sum across periods in group, per regime
        Q_sum_z = Vector{Array{Float64,3}}(undef, r_Z)
        for z in 1:r_Z
            Q_sum_z[z] = _qsum_from_xi(xi_sum_z[z][group[1]])
            for t in group[2:end]
                Q_sum_z[z] = Q_sum_z[z] .+ _qsum_from_xi(xi_sum_z[z][t])
            end
        end

        # Optimise
        K = trans_specs[g].K; ν = trans_specs[g].ν
        x0 = pack_robit_nn_v3(trans_specs[g])
        obj = x -> transition_negll_robit_nn_v3(x, Q_sum_z, r_θ, r_I, K, ν,
                                                  r_P, r_ω; θ_grid, I_grid, P_grid, ω_grid)
        result = optimize(obj, x0, LBFGS(), options; autodiff=:forward)
        new_specs[g] = unpack_robit_nn_v3(Optim.minimizer(result), r_θ, K; ν)
    end
    return new_specs
end

"""CCP M-step: optimise OrderedProbitCCPV3 over all (n, z, t)."""
function mstep_ccp_v3(ccp_spec::OrderedProbitCCPV3,
                       ccp_xi_z::Array{Float64,6},
                       gamma_z_t1::Array{Float64,4},
                       w_z::Matrix{Float64},
                       Y_income::Matrix{Float64},
                       r_θ::Int, r_I::Int, r_P::Int, r_ω::Int, T::Int;
                       θ_grid::AbstractVector=collect(1.0:r_θ),
                       I_grid::AbstractVector=collect(1.0:r_I),
                       P_grid::AbstractVector=collect(1.0:r_P),
                       ω_grid::AbstractVector=collect(1.0:r_ω),
                       options=DEFAULT_OPTIM_OPTIONS)
    x0 = pack_ccp_v3(ccp_spec)
    obj = x -> ccp_negll_v3(x, ccp_xi_z, gamma_z_t1, w_z, Y_income,
                              r_θ, r_I, r_P, r_ω, T; θ_grid, I_grid, P_grid, ω_grid)
    result = optimize(obj, x0, LBFGS(), options; autodiff=:forward)
    return unpack_ccp_v3(Optim.minimizer(result), r_I)
end

# ─────────────────────────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────────────────────────

"""
    initialize_params_v3(data, config; rng) -> (params_v3, params_skill)

`data` is the output of `prepare_panel_data` merged with `(; r_P, r_ω)`.
`config` NamedTuple with optional fields: K (default 4), ν (default 7.0).
"""
function initialize_params_v3(data, config::NamedTuple=NamedTuple();
                                rng::AbstractRNG=Random.default_rng())
    (; r_θ, r_I, T, κ_θ, κ_I, κ_P, r_P, r_ω) = data
    K = get(config, :K, 4)
    ν = get(config, :ν, DEFAULT_ROBIT_DF)
    n_groups = get(config, :n_groups, 2)

    # Ordered probit emissions for child skills and investment
    probit_θ = [initialize_ordered_probit(κ_θ[t], r_θ; rng) for t in 1:T]
    P_θ      = [emission_matrices_from_probit(probit_θ[t], r_θ) for t in 1:T]
    probit_I = [initialize_ordered_probit(κ_I[t], r_I; rng) for t in 1:T]
    P_I      = [emission_matrices_from_probit(probit_I[t], r_I) for t in 1:T]

    # Transition specs: one per group
    trans_specs = [initialize_robit_nn_v3(r_θ, K; rng, ν) for _ in 1:n_groups]

    # CCP
    ccp_spec = initialize_ccp_v3(r_I; rng)

    # Initial distribution (uniform)
    π0 = ones(r_θ, r_P, r_ω) / (r_θ * r_P * r_ω)
    π_Z = ones(r_P, r_ω) / (r_P * r_ω)

    # Parental emissions (near-uniform)
    Q_P = [begin
        κ_m = κ_P[m]
        Q = 0.5 * ones(κ_m, r_P) / κ_m .+ 0.5 * rand(rng, κ_m, r_P)
        Q ./= sum(Q; dims=1)
        Q
    end for m in eachindex(κ_P)]

    # Income model (from sample moments)
    income_model = initialize_income_model(data.Y_income)

    params_v3 = (;
        r_P, r_ω,
        π0, π_Z,
        trans_specs,
        ccp_spec,
        Q_P,
        income_model,
    )
    params_skill = (; r_θ, r_I, T, P_θ, P_I, probit_θ, probit_I)

    return params_v3, params_skill
end

# ─────────────────────────────────────────────────────────────────
# EM loop
# ─────────────────────────────────────────────────────────────────

"""
    expectation_maximization_v3(data, params_v3_init, params_skill_init;
                                  transition_groups, tol, maxiter, verbose, mstep_options)
    -> (params_v3, params_skill, γ_agg, xi_sum_z, w_z, ll_final)
"""
function expectation_maximization_v3(
    data,
    params_v3_init,
    params_skill_init;
    transition_groups::Vector{Vector{Int}} = [[1,2,3],[4,5,6]],
    tol::Float64 = 1e-5,
    maxiter::Int  = 500,
    verbose::Bool = true,
    mstep_options = DEFAULT_OPTIM_OPTIONS,
)
    (; r_θ, r_I, T) = params_skill_init
    (; r_P, r_ω) = params_v3_init
    N = data.N
    r_Z = r_P * r_ω
    T_minus_1 = T - 1

    # Grids
    θ_grid = collect(1.0:r_θ);  I_grid = collect(1.0:r_I)
    P_grid = collect(1.0:r_P);  ω_grid = collect(1.0:r_ω)

    # Allocate
    γ_agg      = zeros(T, r_θ, r_I, N)
    xi_sum_z   = [[zeros(r_θ, r_I, r_θ, r_I) for _ in 1:T_minus_1] for _ in 1:r_Z]
    w_z        = zeros(r_Z, N)
    gamma_z_t1 = zeros(r_θ, r_I, r_Z, N)
    ccp_xi_z   = zeros(r_I, r_I, r_θ, r_Z, T_minus_1, N)
    pi0_ss     = zeros(r_θ, r_P, r_ω)

    params_v3    = params_v3_init
    params_skill = params_skill_init

    err = Inf; iter = 0
    Pi_z_prev = nothing   # for convergence check

    while err > tol && iter < maxiter
        # ── Precompute regime transition arrays ──
        Pi_z_pre = precompute_Pi_z(params_v3.trans_specs, transition_groups,
                                     r_θ, r_I, r_P, r_ω; θ_grid, I_grid, P_grid, ω_grid)

        # ── E-step ──
        ll = expectation_step_v3!(
            γ_agg, xi_sum_z, w_z, gamma_z_t1, ccp_xi_z, pi0_ss,
            Pi_z_pre, params_v3, params_skill, data;
            θ_grid, I_grid, P_grid, ω_grid)

        # ── M-step: child skill/investment emissions ──
        probit_θ_new, P_θ_new = parametric_mstep_emission_mp(
            params_skill.probit_θ, γ_agg, data, :theta, r_θ; options=mstep_options)
        probit_I_new, P_I_new = parametric_mstep_emission_mp(
            params_skill.probit_I, γ_agg, data, :I, r_I; options=mstep_options)

        # ── M-step: transition (grouped) ──
        new_trans_specs = mstep_transition_v3_grouped(
            params_v3.trans_specs, xi_sum_z, transition_groups,
            r_θ, r_I, r_P, r_ω; θ_grid, I_grid, P_grid, ω_grid, options=mstep_options)

        # ── M-step: CCP ──
        new_ccp_spec = mstep_ccp_v3(
            params_v3.ccp_spec, ccp_xi_z, gamma_z_t1, w_z, data.Y_income,
            r_θ, r_I, r_P, r_ω, T; θ_grid, I_grid, P_grid, ω_grid, options=mstep_options)

        # ── M-step: π0, π_Z ──
        new_π0, new_π_Z = mstep_pi0(pi0_ss)

        # ── M-step: Q_P ──
        new_Q_P = mstep_Q_P(w_z, data, r_P, r_ω)

        # ── M-step: income model ──
        new_income = update_income_model(w_z, data.Y_income, r_P, r_ω; P_grid, ω_grid)

        # ── Convergence: max|ΔΠ| ──
        Pi_z_new = precompute_Pi_z(new_trans_specs, transition_groups,
                                     r_θ, r_I, r_P, r_ω; θ_grid, I_grid, P_grid, ω_grid)
        if Pi_z_prev !== nothing
            err = 0.0
            for z in 1:r_Z, t in 1:T_minus_1
                err = max(err, maximum(abs.(Pi_z_new[z][t] .- Pi_z_prev[z][t])))
            end
        end
        Pi_z_prev = Pi_z_new

        iter += 1
        if verbose && (iter ≤ 5 || iter % 25 == 0)
            σ_inc = round(exp(new_income.log_sigma); digits=3)
            println("  iter $iter: max|ΔΠ| = $(round(err; sigdigits=4)), " *
                    "ll = $(round(ll; digits=2)), σ_inc = $σ_inc")
        end

        # Update params
        params_v3 = (;
            r_P, r_ω,
            π0 = new_π0, π_Z = new_π_Z,
            trans_specs = new_trans_specs,
            ccp_spec = new_ccp_spec,
            Q_P = new_Q_P,
            income_model = new_income,
        )
        params_skill = (; r_θ, r_I, T,
            P_θ = P_θ_new, P_I = P_I_new,
            probit_θ = probit_θ_new, probit_I = probit_I_new)
    end

    if verbose
        println("  Converged after $iter iterations (err = $(round(err; sigdigits=4)))")
    end

    # Final E-step
    Pi_z_final = precompute_Pi_z(params_v3.trans_specs, transition_groups,
                                   r_θ, r_I, r_P, r_ω; θ_grid, I_grid, P_grid, ω_grid)
    ll_final = expectation_step_v3!(
        γ_agg, xi_sum_z, w_z, gamma_z_t1, ccp_xi_z, pi0_ss,
        Pi_z_final, params_v3, params_skill, data;
        θ_grid, I_grid, P_grid, ω_grid)
    if verbose; println("  Final log-likelihood: $(round(ll_final; digits=2))"); end

    return params_v3, params_skill, γ_agg, xi_sum_z, w_z, ll_final
end

# ─────────────────────────────────────────────────────────────────
# Funnel restarts
# ─────────────────────────────────────────────────────────────────

"""
    run_em_restarts_v3(data; n_initial, n_semifinal, phase1_iters, phase2_iters,
                        maxiter, tol, verbose, config, transition_groups, mstep_options)
"""
function run_em_restarts_v3(data;
    n_initial::Int    = 10,
    n_semifinal::Int  = 3,
    phase1_iters::Int = 15,
    phase2_iters::Int = 75,
    maxiter::Int      = 500,
    tol::Float64      = 1e-4,
    verbose::Bool     = true,
    config::NamedTuple = NamedTuple(),
    transition_groups::Vector{Vector{Int}} = [[1,2,3],[4,5,6]],
    mstep_options     = DEFAULT_OPTIM_OPTIONS,
)
    em_kw = (; transition_groups, mstep_options)

    # Phase 1: screen
    println("  Phase 1: screening $n_initial candidates × $phase1_iters iters")
    candidates = []
    for i in 1:n_initial
        rng = MersenneTwister(42 + i)
        pv3, psk = initialize_params_v3(data, config; rng)
        pv3, psk, _, _, _, ll = expectation_maximization_v3(
            data, pv3, psk; maxiter=phase1_iters, verbose=false, em_kw...)
        push!(candidates, (; pv3, psk, ll))
        println("    Candidate $i/$n_initial: ll = $(round(ll; digits=2))")
    end
    sort!(candidates; by=c -> c.ll, rev=true)
    resize!(candidates, n_semifinal)
    println("  Phase 1 done — top $n_semifinal: ll ∈ " *
            "[$(round(candidates[end].ll; digits=2)), $(round(candidates[1].ll; digits=2))]")

    # Phase 2: refine
    println("  Phase 2: refining × $phase2_iters iters")
    for j in eachindex(candidates)
        pv3, psk, _, _, _, ll = expectation_maximization_v3(
            data, candidates[j].pv3, candidates[j].psk;
            maxiter=phase2_iters, verbose=false, em_kw...)
        candidates[j] = (; pv3, psk, ll)
        println("    Candidate $j: ll = $(round(ll; digits=2))")
    end
    sort!(candidates; by=c -> c.ll, rev=true)
    println("  Phase 2 done — best ll = $(round(candidates[1].ll; digits=2))")

    # Phase 3: converge
    println("  Phase 3: converging to tol=$tol")
    pv3, psk, γ_agg, xi_sum_z, w_z, ll_final = expectation_maximization_v3(
        data, candidates[1].pv3, candidates[1].psk;
        maxiter=maxiter, tol=tol, verbose=verbose, em_kw...)

    return pv3, psk, γ_agg, xi_sum_z, w_z, ll_final
end

# ─────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────

function display_results_v3(params_v3, params_skill, data)
    (; r_P, r_ω, π0, π_Z, trans_specs, ccp_spec, Q_P, income_model) = params_v3
    (; r_θ, r_I, T) = params_skill

    println("\n" * "="^60)
    println("  V3 RESULTS  r_θ=$r_θ r_I=$r_I r_P=$r_P r_ω=$r_ω T=$T")
    println("="^60)

    println("\n  Regime prior π_Z[kP, kω]:")
    for kP in 1:r_P
        vals = ["$(round(π_Z[kP,kω]; digits=4))" for kω in 1:r_ω]
        println("    kP=$kP:  $(join(vals, "  "))")
    end

    println("\n  P(θ₁|z) from π0:")
    for kP in 1:r_P, kω in 1:r_ω
        p = π0[:, kP, kω]; s = sum(p); p_norm = s > 0 ? p ./ s : p
        println("    (kP=$kP,kω=$kω): $(round.(p_norm; digits=3))")
    end

    println("\n  Income model: β₀=$(round(income_model.beta0;digits=3)), " *
            "β_P=$(round(income_model.betaP;digits=3)), " *
            "β_ω=$(round(income_model.betaω;digits=3)), " *
            "σ=$(round(exp(income_model.log_sigma);digits=3))")

    println("\n  CCP V3: β₁=$(round(ccp_spec.beta1;digits=3)), " *
            "β₂=$(round(ccp_spec.beta2;digits=3)), " *
            "β₃=$(round(ccp_spec.beta3;digits=3)), " *
            "β₄=$(round(ccp_spec.beta4;digits=3)), " *
            "β₅=$(round(ccp_spec.beta5;digits=3))")

    for (g, spec) in enumerate(trans_specs)
        println("\n  Transition group $g:")
        println("    α₁(θ):   $(round.(spec.alpha1; digits=3))")
        println("    α₂(I):   $(round.(exp.(spec.alpha2_raw); digits=3))")
        println("    α₃(θ^P): $(round.(spec.alpha3; digits=3))")
        println("    α₄(ω):   $(round.(spec.alpha4; digits=3))")
    end

    println("\n  Q_P (ASVAB emissions):")
    for m in eachindex(Q_P)
        name = m ≤ length(data.measure_names_P) ? data.measure_names_P[m] : "ASVAB_$m"
        vals = ["$(round(Q_P[m][end,kP]; digits=3))" for kP in 1:r_P]
        println("    $name  P(high|kP): $(join(vals, "  "))")
    end
end
