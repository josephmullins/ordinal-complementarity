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
                Pi_z[z][t] = Π
            end
        end
    end
    return Pi_z
end

function precompute_Pi_z(trans_specs::Vector{OrderedProbitTransitionV3},
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
        vec_g = pack_op_trans_v3(spec)
        for kω in 1:r_ω, kP in 1:r_P
            z = regime_z(kP, kω, r_ω)
            Π = _build_op_trans_v3_pi(vec_g, r_θ, r_I, P_grid[kP], ω_grid[kω];
                                        θ_grid, I_grid)
            for t in group
                Pi_z[z][t] = Π
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
    precompute_CCP_z(ccp_spec, r_Z, r_P, r_ω, bin_values, r_θ, r_I;
                      θ_grid, I_grid, P_grid, ω_grid)
        -> (CCP_init_pre, CCP_trans_pre)

Precompute CCP arrays for all (regime, income bin) combinations.

Returns:
  CCP_init_pre  — [z][b] :: Matrix{Float64} (r_I, r_θ)
  CCP_trans_pre — [z][b] :: Array{Float64,3} (r_I, r_I, r_θ)
"""
function precompute_CCP_z(ccp_spec::OrderedProbitCCPV3,
                            r_Z::Int, r_P::Int, r_ω::Int,
                            bin_values::Vector{Float64},
                            r_θ::Int, r_I::Int;
                            θ_grid::AbstractVector, I_grid::AbstractVector,
                            P_grid::AbstractVector, ω_grid::AbstractVector)
    n_bins = length(bin_values)
    CCP_init_pre  = [[Matrix{Float64}(undef, 0, 0) for _ in 1:n_bins] for _ in 1:r_Z]
    CCP_trans_pre = [[Array{Float64,3}(undef, 0, 0, 0) for _ in 1:n_bins] for _ in 1:r_Z]

    for kω in 1:r_ω, kP in 1:r_P
        z = regime_z(kP, kω, r_ω)
        kP_val = P_grid[kP]; kω_val = ω_grid[kω]
        for b in 1:n_bins
            y_b = bin_values[b]
            # For missing bin (y_b=0.0), build_ccp treats NaN specially
            # but bin_values already maps missing → 0.0 so β₅·0 = 0. Correct.
            CCP_init_pre[z][b] = build_ccp_v3_init(ccp_spec, kP_val, kω_val,
                                                      y_b; θ_grid, r_I)
            CCP_trans_pre[z][b] = build_ccp_v3_trans(ccp_spec, kP_val, kω_val,
                                                       y_b; θ_grid, I_grid, r_I)
        end
    end
    return CCP_init_pre, CCP_trans_pre
end

"""
    expectation_step_v3!(γ_agg, xi_sum_z, w_z, gamma_z_t1, ccp_init_ss, ccp_trans_ss,
                          pi0_ss, Pi_z_pre, CCP_init_pre, CCP_trans_pre,
                          bin_map, params_v3, params_skill, data; grids...)
        -> total_log_likelihood

Full V3 E-step with precomputed CCP arrays (no per-child allocation).

Mutated outputs:
  γ_agg         — (T, r_θ, r_I, N)                    aggregated posteriors
  xi_sum_z      — [z][t] (r_θ,r_I,r_θ,r_I)            regime-specific pairwise SS
  w_z           — (r_Z, N)                             regime posteriors
  gamma_z_t1    — (r_θ, r_I, r_Z, N)                   period-1 regime posteriors
  ccp_init_ss   — (r_θ, r_I, r_Z, n_bins)             binned initial CCP SS
  ccp_trans_ss  — (r_I, r_I, r_θ, r_Z, T-1, n_bins)   binned transition CCP SS
  pi0_ss        — (r_θ, r_P, r_ω)                      π0 accumulator
"""
function expectation_step_v3!(
    γ_agg::Array{Float64,4},
    xi_sum_z::Vector{Vector{Array{Float64,4}}},   # [z][t]
    w_z::Matrix{Float64},
    gamma_z_t1::Array{Float64,4},
    ccp_init_ss::Array{Float64,4},
    ccp_trans_ss::Array{Float64,6},
    pi0_ss::Array{Float64,3},
    Pi_z_pre::Vector{Vector{Array{Float64,3}}},   # [z][t] precomputed
    CCP_init_pre::Vector{Vector{Matrix{Float64}}},   # [z][b]
    CCP_trans_pre::Vector{Vector{Array{Float64,3}}},  # [z][b]
    bin_map::Matrix{Int},
    params_v3,          # (; r_P, r_ω, π0, π_Z, ccp_spec, Q_P, income_model, trans_specs)
    params_skill,       # (; r_θ, r_I, T, P_θ, P_I)
    data;
    θ_grid::AbstractVector = 1.0:params_skill.r_θ,
    I_grid::AbstractVector = 1.0:params_skill.r_I,
    P_grid::AbstractVector = 1.0:params_v3.r_P,
    ω_grid::AbstractVector = 1.0:params_v3.r_ω,
)
    (; r_P, r_ω, π0, π_Z, Q_P, income_model) = params_v3
    adult_matrices = haskey(params_v3, :adult_matrices) ? params_v3.adult_matrices : Array{Float64,3}[]
    has_adult = !isempty(adult_matrices) && haskey(data, :X_adult) && !isempty(data.X_adult)
    (; r_θ, r_I, T) = params_skill
    N = data.N
    r_Z = r_P * r_ω

    # Reset accumulators
    fill!(γ_agg, 0.0)
    for z in 1:r_Z, t in 1:T-1; fill!(xi_sum_z[z][t], 0.0); end
    fill!(w_z, 0.0); fill!(gamma_z_t1, 0.0)
    fill!(ccp_init_ss, 0.0); fill!(ccp_trans_ss, 0.0); fill!(pi0_ss, 0.0)

    # FB scratch (reused across children and regimes)
    α     = zeros(T, r_θ, r_I)
    β     = zeros(T, r_θ, r_I)
    scale = zeros(T)
    L_scratch = [zeros(r_θ, r_I) for _ in 1:T]

    # Per-regime scratch for one child (stored for all z before weighting)
    γ_z_all = [zeros(T, r_θ, r_I)                                   for _ in 1:r_Z]
    ξ_z_all = [[zeros(r_θ, r_I, r_θ, r_I) for _ in 1:T-1]          for _ in 1:r_Z]

    # Reusable CCP vector (just pointers into precomputed arrays)
    CCP_nz = Vector{Array{Float64,3}}(undef, T-1)

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

            # Regime emission log-likelihood
            ll_regime = regime_loglik(n, kP, kω, Q_P, income_model, data;
                                       P_grid, ω_grid)
            log_ll_z[z] = log_π_Z[kP, kω] + ll_regime

            # π0 conditional on z: P(θ_1 | z) ∝ π0[kθ, kP, kω]
            for kθ in 1:r_θ; @inbounds π0_z_buf[kθ] = max(π0[kθ, kP, kω], 0.0); end
            s = sum(π0_z_buf)
            if s > 0; π0_z_buf ./= s; else; fill!(π0_z_buf, 1.0/r_θ); end

            # Build initial joint from precomputed CCP
            ccp_init = CCP_init_pre[z][bin_map[n, 1]]
            for kI in 1:r_I, kθ in 1:r_θ
                @inbounds π0_nz_buf[kθ, kI] = π0_z_buf[kθ] * ccp_init[kI, kθ]
            end

            # Look up precomputed CCPs for transitions
            for t in 1:T-1
                CCP_nz[t] = CCP_trans_pre[z][bin_map[n, t+1]]
            end

            # Reset ξ scratch for this regime
            for t in 1:T-1; fill!(ξ_z_all[z][t], 0.0); end

            # Forward-backward
            ll_skills = forward_backward_v3_n!(
                α, β, scale, γ_z_all[z], ξ_z_all[z],
                π0_nz_buf, Pi_z_pre[z], CCP_nz, L_scratch, r_θ, r_I, T)

            log_ll_z[z] += ll_skills

            # Adult outcome log-likelihood (uses smoothed γ at final period T)
            if has_adult
                γ_z = γ_z_all[z]
                for m in eachindex(data.X_adult)
                    x = data.X_adult[m][n]
                    x == 0 && continue   # missing
                    Am = adult_matrices[m]
                    for kθ in 1:r_θ
                        # Marginalize over I at the final period
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

            # Period-1 posteriors for CCP and π0
            b1 = bin_map[n, 1]
            for kI in 1:r_I, kθ in 1:r_θ
                @inbounds gamma_z_t1[kθ, kI, z, n] = γ_z[1, kθ, kI]
                @inbounds ccp_init_ss[kθ, kI, z, b1] += wz * γ_z[1, kθ, kI]
            end

            # Binned CCP transition SS (aggregate ξ into bins)
            for t in 1:T-1
                b = bin_map[n, t+1]
                ξ_t = ξ_z[t]
                for kθ1 in 1:r_θ, kI in 1:r_I, kI1 in 1:r_I
                    s = 0.0
                    for kθ in 1:r_θ; @inbounds s += ξ_t[kθ1, kI1, kθ, kI]; end
                    @inbounds ccp_trans_ss[kI1, kI, kθ1, z, t, b] += wz * s
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

"""
    mstep_Q_P(probit_P, w_z, data, r_P, r_ω; options) -> (probit_P_new, Q_P_new)

Ordered probit M-step for ASVAB emissions.  Marginalizes w_z over ω to get
per-child weights W_P[kP, n], then optimizes each measure's ordered probit.
"""
function mstep_Q_P(probit_P::OrderedProbitEmission,
                    w_z::Matrix{Float64}, data, r_P::Int, r_ω::Int;
                    options=DEFAULT_OPTIM_OPTIONS)
    N = size(w_z, 2)

    # Marginalize regime posteriors over ω to get P(θ^P = kP | data_n)
    W_P = zeros(r_P, N)
    for n in 1:N, kω in 1:r_ω, kP in 1:r_P
        z = regime_z(kP, kω, r_ω)
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

"""
    mstep_adult_outcomes(adult_emission, γ_agg, w_z, data, r_θ, r_I, r_P, r_ω;
                          θ_grid, ω_grid, options) -> (new_adult_emission, new_adult_matrices)

M-step for adult outcome emissions. Builds sufficient statistics SS[m][k, kθ, kω]
from smoothed posteriors γ at final period T, weighted by regime posteriors w_z.
Then optimizes ordered probit parameters via L-BFGS.
"""
function mstep_adult_outcomes(adult_emission::AdultOutcomeEmission,
                               γ_agg::Array{Float64,4},
                               w_z::Matrix{Float64},
                               data, r_θ::Int, r_I::Int, r_P::Int, r_ω::Int;
                               θ_grid::AbstractVector=1.0:r_θ,
                               ω_grid::AbstractVector=1.0:r_ω,
                               options=DEFAULT_OPTIM_OPTIONS)
    N = data.N
    T = data.T
    κ_adult = data.κ_adult
    M_adult = length(κ_adult)

    # Build sufficient statistics: SS[m][k, kθ, kω]
    # SS[m][k, kθ, kω] = Σ_n 1[x_m=k] * w_ω(kω, n) * (Σ_kI γ_agg[T, kθ, kI, n])
    # where w_ω(kω, n) = Σ_kP w_z(z(kP,kω), n) marginalizes over θ^P.
    # Since γ_agg = Σ_z w_z * γ^z and Σ_z w_z = 1, and the adult outcome
    # emissions depend on (kθ, kω) but not kP, we weight γ_agg by the
    # marginalized regime posterior w_ω. This is exact when γ^z is independent
    # of the regime (approximately true for well-identified models).
    SS_adult = [zeros(κ_adult[m], r_θ, r_ω) for m in 1:M_adult]
    for n in 1:N
        for kω in 1:r_ω
            # Marginalize w_z over kP to get w_ω[kω, n]
            w_omega = 0.0
            for kP in 1:r_P
                z = regime_z(kP, kω, r_ω)
                @inbounds w_omega += w_z[z, n]
            end
            w_omega < 1e-15 && continue
            for kθ in 1:r_θ
                # Marginalize γ_agg over I at final period
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

    # Optimize
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

"""Grouped transition M-step for V3 ordered probit."""
function mstep_transition_v3_grouped(
    trans_specs::Vector{OrderedProbitTransitionV3},
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
    new_specs = Vector{OrderedProbitTransitionV3}(undef, length(transition_groups))

    for (g, group) in enumerate(transition_groups)
        Q_sum_z = Vector{Array{Float64,3}}(undef, r_Z)
        for z in 1:r_Z
            Q_sum_z[z] = _qsum_from_xi(xi_sum_z[z][group[1]])
            for t in group[2:end]
                Q_sum_z[z] = Q_sum_z[z] .+ _qsum_from_xi(xi_sum_z[z][t])
            end
        end

        x0 = pack_op_trans_v3(trans_specs[g])
        obj = x -> transition_negll_op_v3(x, Q_sum_z, r_θ, r_I,
                                            r_P, r_ω; θ_grid, I_grid, P_grid, ω_grid)
        result = optimize(obj, x0, LBFGS(), options; autodiff=:forward)
        new_specs[g] = unpack_op_trans_v3(Optim.minimizer(result), r_θ)
    end
    return new_specs
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
    λ_logw::Float64=0.0,   # ridge on log_w (weight decay) to tame the
                            # robit-NN compensating-pair degeneracy.  0 ⇒ off.
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

        # Optimise. The last K packed entries are log_w; penalize them
        # (toward 0 ⇒ w toward 1) to remove the flat compensating-pair ridge.
        K = trans_specs[g].K; ν = trans_specs[g].ν
        x0 = pack_robit_nn_v3(trans_specs[g])
        obj = x -> transition_negll_robit_nn_v3(x, Q_sum_z, r_θ, r_I, K, ν,
                                                  r_P, r_ω; θ_grid, I_grid, P_grid, ω_grid) +
                   λ_logw * sum(abs2, @view x[(end - K + 1):end])
        result = optimize(obj, x0, LBFGS(), options; autodiff=:forward)
        new_specs[g] = unpack_robit_nn_v3(Optim.minimizer(result), r_θ, K; ν)
    end
    return new_specs
end

## Note: CCP M-step is now done inline in the EM loop using
## ccp_negll_v3_binned with pre-aggregated SS from the E-step.

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
    transition_type = get(config, :transition, :robit_nn)
    trans_specs = if transition_type == :ordered_probit
        [initialize_op_trans_v3(r_θ; rng) for _ in 1:n_groups]
    else
        [initialize_robit_nn_v3(r_θ, K; rng, ν) for _ in 1:n_groups]
    end

    # CCP
    ccp_spec = initialize_ccp_v3(r_I; rng)

    # Initial distribution (uniform)
    π0 = ones(r_θ, r_P, r_ω) / (r_θ * r_P * r_ω)
    π_Z = ones(r_P, r_ω) / (r_P * r_ω)

    # Parental emissions (ordered probit)
    probit_P = initialize_ordered_probit(collect(κ_P), r_P; rng)
    Q_P = emission_matrices_from_probit(probit_P, r_P)

    # Income model (from sample moments)
    income_model = initialize_income_model(data.Y_income)

    # Adult outcome emissions (ordered probit on θ_T × ω)
    κ_adult = haskey(data, :κ_adult) ? data.κ_adult : Int[]
    if !isempty(κ_adult)
        adult_emission = initialize_adult_outcome_emission(κ_adult, r_θ, r_ω; rng)
        adult_matrices = build_adult_matrices(adult_emission, κ_adult, r_θ, r_ω)
    else
        adult_emission = AdultOutcomeEmission(Vector{Float64}[], Float64[], Float64[])
        adult_matrices = Array{Float64,3}[]
    end

    params_v3 = (;
        r_P, r_ω,
        π0, π_Z,
        trans_specs,
        ccp_spec,
        probit_P,
        Q_P,
        income_model,
        adult_emission,
        adult_matrices,
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
    n_income_bins::Int = 20,
    fix_betaω::Bool = false,
)
    (; r_θ, r_I, T) = params_skill_init
    (; r_P, r_ω) = params_v3_init
    N = data.N
    r_Z = r_P * r_ω
    T_minus_1 = T - 1

    # Grids
    θ_grid = collect(1.0:r_θ);  I_grid = collect(1.0:r_I)
    P_grid = collect(1.0:r_P);  ω_grid = collect(1.0:r_ω)

    # Bin income ONCE (depends only on data, not parameters)
    bin_map, bin_values = bin_income(data.Y_income; n_bins=n_income_bins)
    n_total_bins = length(bin_values)

    # Allocate
    γ_agg        = zeros(T, r_θ, r_I, N)
    xi_sum_z     = [[zeros(r_θ, r_I, r_θ, r_I) for _ in 1:T_minus_1] for _ in 1:r_Z]
    w_z          = zeros(r_Z, N)
    gamma_z_t1   = zeros(r_θ, r_I, r_Z, N)
    ccp_init_ss  = zeros(r_θ, r_I, r_Z, n_total_bins)
    ccp_trans_ss = zeros(r_I, r_I, r_θ, r_Z, T_minus_1, n_total_bins)
    pi0_ss       = zeros(r_θ, r_P, r_ω)

    params_v3    = params_v3_init
    params_skill = params_skill_init

    err = Inf; iter = 0
    Pi_z_prev = nothing   # for convergence check

    while err > tol && iter < maxiter
        # ── Precompute regime transition arrays ──
        Pi_z_pre = precompute_Pi_z(params_v3.trans_specs, transition_groups,
                                     r_θ, r_I, r_P, r_ω; θ_grid, I_grid, P_grid, ω_grid)

        # ── Precompute CCP arrays per (regime, income bin) ──
        CCP_init_pre, CCP_trans_pre = precompute_CCP_z(
            params_v3.ccp_spec, r_Z, r_P, r_ω, bin_values, r_θ, r_I;
            θ_grid, I_grid, P_grid, ω_grid)

        # ── E-step ──
        ll = expectation_step_v3!(
            γ_agg, xi_sum_z, w_z, gamma_z_t1, ccp_init_ss, ccp_trans_ss, pi0_ss,
            Pi_z_pre, CCP_init_pre, CCP_trans_pre, bin_map,
            params_v3, params_skill, data;
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

        # ── M-step: CCP (using pre-aggregated binned SS) ──
        x0_ccp = pack_ccp_v3(params_v3.ccp_spec)
        obj_ccp = x -> ccp_negll_v3_binned(x, ccp_init_ss, ccp_trans_ss, bin_values,
                                             r_θ, r_I, r_P, r_ω, T;
                                             θ_grid, I_grid, P_grid, ω_grid)
        res_ccp = optimize(obj_ccp, x0_ccp, LBFGS(), mstep_options; autodiff=:forward)
        new_ccp_spec = unpack_ccp_v3(Optim.minimizer(res_ccp), r_I)

        # ── M-step: π0, π_Z ──
        new_π0, new_π_Z = mstep_pi0(pi0_ss)

        # ── M-step: Q_P (ordered probit) ──
        new_probit_P, new_Q_P = mstep_Q_P(params_v3.probit_P, w_z, data, r_P, r_ω;
                                            options=mstep_options)

        # ── M-step: income model ──
        new_income = update_income_model(w_z, data.Y_income, r_P, r_ω;
                                          P_grid, ω_grid, fix_betaω=fix_betaω)

        # ── M-step: adult outcome emissions ──
        has_adult_data = haskey(data, :X_adult) && !isempty(data.X_adult) &&
                         haskey(data, :κ_adult) && !isempty(data.κ_adult)
        if has_adult_data && haskey(params_v3, :adult_emission) &&
           !isempty(params_v3.adult_emission.tau)
            new_adult_emission, new_adult_matrices = mstep_adult_outcomes(
                params_v3.adult_emission, γ_agg, w_z, data, r_θ, r_I, r_P, r_ω;
                θ_grid, ω_grid, options=mstep_options)
        else
            new_adult_emission = haskey(params_v3, :adult_emission) ?
                params_v3.adult_emission :
                AdultOutcomeEmission(Vector{Float64}[], Float64[], Float64[])
            new_adult_matrices = haskey(params_v3, :adult_matrices) ?
                params_v3.adult_matrices : Array{Float64,3}[]
        end

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

    # Final E-step
    Pi_z_final = precompute_Pi_z(params_v3.trans_specs, transition_groups,
                                   r_θ, r_I, r_P, r_ω; θ_grid, I_grid, P_grid, ω_grid)
    CCP_init_final, CCP_trans_final = precompute_CCP_z(
        params_v3.ccp_spec, r_Z, r_P, r_ω, bin_values, r_θ, r_I;
        θ_grid, I_grid, P_grid, ω_grid)
    ll_final = expectation_step_v3!(
        γ_agg, xi_sum_z, w_z, gamma_z_t1, ccp_init_ss, ccp_trans_ss, pi0_ss,
        Pi_z_final, CCP_init_final, CCP_trans_final, bin_map,
        params_v3, params_skill, data;
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
    n_income_bins::Int = 20,
)
    em_kw = (; transition_groups, mstep_options, n_income_bins)

    # Phase 1: screen
    println("  Phase 1: screening $n_initial candidates × $phase1_iters iters")
    candidates = []
    for i in 1:n_initial
        rng = MersenneTwister(42 + i)
        pv3, psk = initialize_params_v3(data, config; rng)
        try
            pv3, psk, _, _, _, ll = expectation_maximization_v3(
                data, pv3, psk; maxiter=phase1_iters, verbose=false, em_kw...)
            push!(candidates, (; pv3, psk, ll))
            println("    Candidate $i/$n_initial: ll = $(round(ll; digits=2))")
        catch e
            println("    Candidate $i/$n_initial: failed ($(typeof(e)))")
        end
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
        if spec isa RobitNNTransitionV3
            println("    α₁(θ):   $(round.(spec.alpha1; digits=3))")
            println("    α₂(I):   $(round.(exp.(spec.alpha2_raw); digits=3))")
            println("    α₃(θ^P): $(round.(spec.alpha3; digits=3))")
            println("    α₄(ω):   $(round.(spec.alpha4; digits=3))")
        elseif spec isa OrderedProbitTransitionV3
            println("    β₁(θ):   $(round(spec.beta1; digits=3))")
            println("    β₂(I):   $(round(spec.beta2; digits=3))")
            println("    β₃(θI):  $(round(spec.beta3; digits=3))")
            println("    β₄(θ^P): $(round(spec.beta4; digits=3))")
            println("    β₅(ω):   $(round(spec.beta5; digits=3))")
            println("    c:       $(round.(spec.c; digits=3))")
        end
    end

    println("\n  Q_P (ASVAB emissions):")
    for m in eachindex(Q_P)
        name = m ≤ length(data.measure_names_P) ? data.measure_names_P[m] : "ASVAB_$m"
        vals = ["$(round(Q_P[m][end,kP]; digits=3))" for kP in 1:r_P]
        println("    $name  P(high|kP): $(join(vals, "  "))")
    end
end

# ─────────────────────────────────────────────────────────────────
# V3 Inference: complementarity test on regime-marginalized Π̄
# ─────────────────────────────────────────────────────────────────

"""
    marginalize_Pi_v3(trans_specs, transition_groups, π_Z, r_θ, r_I, r_P, r_ω;
                       θ_grid, I_grid, P_grid, ω_grid) -> Vector{Array{Float64,3}}

Compute Π̄[t] = Σ_z π_Z(z) · Π(z, t) for each transition.
Returns T-1 arrays of size (r_θ, r_θ, r_I).
"""
function marginalize_Pi_v3(trans_specs::AbstractVector,
                            transition_groups::Vector{Vector{Int}},
                            π_Z::Matrix{Float64},
                            r_θ::Int, r_I::Int, r_P::Int, r_ω::Int;
                            θ_grid::AbstractVector=1.0:r_θ,
                            I_grid::AbstractVector=1.0:r_I,
                            P_grid::AbstractVector=1.0:r_P,
                            ω_grid::AbstractVector=1.0:r_ω)
    T_minus_1 = sum(length, transition_groups)
    Pi_bar = Vector{Array{Float64,3}}(undef, T_minus_1)

    for (g, group) in enumerate(transition_groups)
        spec = trans_specs[g]
        φ_g = _pack_spec(spec)
        build_fn = _make_build_fn(spec, r_θ, r_I; θ_grid, I_grid)
        Π_bar_g = zeros(r_θ, r_θ, r_I)
        for kω in 1:r_ω, kP in 1:r_P
            w = π_Z[kP, kω]
            w < 1e-15 && continue
            Π_z = build_fn(φ_g, P_grid[kP], ω_grid[kω])
            Π_bar_g .+= w .* Π_z
        end
        for t in group
            Pi_bar[t] = Π_bar_g
        end
    end
    return Pi_bar
end

"""
    compute_psi_all_v3(xi_sum_z, w_z, π_Z, r_θ, r_I, r_P, r_ω, T, N)
        -> Array{Float64, 5}   (T-1, r_θ, r_θ, r_I, N)

Compute per-child pairwise posteriors marginalized over regimes and I':
    ψ[t, θ', θ, I, n] = Σ_z w(z,n) · Σ_{I'} ξ_z[t][θ', I', θ, I]

These are the regime-marginalized analogue of V2's ψ_all, and can be
fed directly into `get_score(ψ_t, Π̄)`.

Note: requires running a fresh E-step that stores per-child ξ_z.
Since the current E-step only stores ξ_sum_z (summed over n), we
recompute from γ and the forward-backward. Instead, we use a simpler
approach: run a dedicated pass that computes ψ directly.
"""
function compute_psi_all_v3(params_v3, params_skill, data,
                              transition_groups::Vector{Vector{Int}};
                              n_income_bins::Int=20)
    (; r_P, r_ω, π0, π_Z, ccp_spec, Q_P, income_model, trans_specs) = params_v3
    (; r_θ, r_I, T) = params_skill
    N = data.N
    r_Z = r_P * r_ω

    # Bin income and precompute
    bin_map, bin_values = bin_income(data.Y_income; n_bins=n_income_bins)
    θ_grid = collect(1.0:r_θ); I_grid = collect(1.0:r_I)
    P_grid = collect(1.0:r_P); ω_grid = collect(1.0:r_ω)

    Pi_z_pre = precompute_Pi_z(trans_specs, transition_groups,
                                 r_θ, r_I, r_P, r_ω; θ_grid, I_grid, P_grid, ω_grid)
    CCP_init_pre, CCP_trans_pre = precompute_CCP_z(
        ccp_spec, r_Z, r_P, r_ω, bin_values, r_θ, r_I;
        θ_grid, I_grid, P_grid, ω_grid)

    ψ_all = zeros(T-1, r_θ, r_θ, r_I, N)

    # Scratch
    α     = zeros(T, r_θ, r_I)
    β     = zeros(T, r_θ, r_I)
    scale = zeros(T)
    L_scratch = [zeros(r_θ, r_I) for _ in 1:T]
    γ_z   = zeros(T, r_θ, r_I)
    ξ_z   = [zeros(r_θ, r_I, r_θ, r_I) for _ in 1:T-1]
    CCP_nz = Vector{Array{Float64,3}}(undef, T-1)

    log_π_Z   = [log(max(π_Z[kP, kω], 1e-300)) for kP in 1:r_P, kω in 1:r_ω]
    log_ll_z  = zeros(r_Z)
    π0_z_buf  = zeros(r_θ)
    π0_nz_buf = zeros(r_θ, r_I)

    for n in 1:N
        for t in 1:T
            compute_emission_likelihood_t!(L_scratch[t], t, n, params_skill, data)
        end

        # Pass 1: forward-backward per regime, store ξ_z per regime
        ξ_all_z = [[zeros(r_θ, r_I, r_θ, r_I) for _ in 1:T-1] for _ in 1:r_Z]

        for kω in 1:r_ω, kP in 1:r_P
            z = regime_z(kP, kω, r_ω)

            ll_regime = regime_loglik(n, kP, kω, Q_P, income_model, data;
                                       P_grid, ω_grid)
            log_ll_z[z] = log_π_Z[kP, kω] + ll_regime

            for kθ in 1:r_θ; @inbounds π0_z_buf[kθ] = max(π0[kθ, kP, kω], 0.0); end
            s = sum(π0_z_buf)
            if s > 0; π0_z_buf ./= s; else; fill!(π0_z_buf, 1.0/r_θ); end

            ccp_init = CCP_init_pre[z][bin_map[n, 1]]
            for kI in 1:r_I, kθ in 1:r_θ
                @inbounds π0_nz_buf[kθ, kI] = π0_z_buf[kθ] * ccp_init[kI, kθ]
            end

            for t in 1:T-1
                CCP_nz[t] = CCP_trans_pre[z][bin_map[n, t+1]]
            end

            for t in 1:T-1; fill!(ξ_all_z[z][t], 0.0); end

            ll_skills = forward_backward_v3_n!(
                α, β, scale, γ_z, ξ_all_z[z],
                π0_nz_buf, Pi_z_pre[z], CCP_nz, L_scratch, r_θ, r_I, T)
            log_ll_z[z] += ll_skills
        end

        # Regime posteriors
        log_max = maximum(log_ll_z)
        w_unnorm = exp.(log_ll_z .- log_max)
        w_sum = sum(w_unnorm)

        # Accumulate ψ: marginalize over z and I'
        for z in 1:r_Z
            wz = w_unnorm[z] / w_sum
            wz < 1e-15 && continue
            for t in 1:T-1
                ξ_t = ξ_all_z[z][t]
                for kI in 1:r_I, kθ in 1:r_θ, kθ1 in 1:r_θ
                    # ψ[t, θ', θ, I, n] = Σ_z w(z,n) · Σ_{I'} ξ_z[θ', I', θ, I]
                    s = 0.0
                    for kI1 in 1:r_I
                        @inbounds s += ξ_t[kθ1, kI1, kθ, kI]
                    end
                    @inbounds ψ_all[t, kθ1, kθ, kI, n] += wz * s
                end
            end
        end
    end

    return ψ_all
end

"""
    _pi_bar_from_phi(build_fn, φ_g, π_Z_vec, r_P, r_ω; mode=:free)

Generic: maps transition params → Π̄ (marginalized over regimes).
`build_fn(φ, kP_val, kω_val)` returns a single-regime Π array (r_θ, r_θ, r_I).
mode=:free returns (r_θ-1)*r_θ*r_I free entries; mode=:full returns all r_θ*r_θ*r_I.
ForwardDiff-compatible.
"""
function _pi_bar_from_phi(build_fn, φ_g::AbstractVector, π_Z_vec::AbstractVector,
                           r_θ::Int, r_I::Int, r_P::Int, r_ω::Int;
                           P_grid::AbstractVector=1.0:r_P,
                           ω_grid::AbstractVector=1.0:r_ω,
                           mode::Symbol=:free)
    T_el = promote_type(eltype(φ_g), eltype(π_Z_vec))
    Π_bar = zeros(T_el, r_θ, r_θ, r_I)
    for kω in 1:r_ω, kP in 1:r_P
        z = (kP - 1) * r_ω + kω
        w = π_Z_vec[z]
        Π_z = build_fn(φ_g, P_grid[kP], ω_grid[kω])
        Π_bar .+= w .* Π_z
    end
    if mode == :full
        return vec(Π_bar)
    else
        v = Vector{T_el}(undef, (r_θ - 1) * r_θ * r_I)
        idx = 0
        for kI in 1:r_I, kθ in 1:r_θ, j in 1:(r_θ - 1)
            idx += 1
            v[idx] = Π_bar[j, kθ, kI]
        end
        return v
    end
end

"""
    _make_build_fn(spec, r_θ, r_I; θ_grid, I_grid)

Returns a ForwardDiff-compatible `(φ, kP_val, kω_val) -> Π` function
appropriate for the given spec type.
"""
function _make_build_fn(spec::RobitNNTransitionV3, r_θ::Int, r_I::Int;
                         θ_grid::AbstractVector=1.0:r_θ, I_grid::AbstractVector=1.0:r_I)
    K = spec.K; ν = spec.ν
    return (φ, kP_val, kω_val) -> _build_robit_nn_v3_pi(φ, r_θ, r_I, K, ν,
                                                           kP_val, kω_val; θ_grid, I_grid)
end

function _make_build_fn(spec::OrderedProbitTransitionV3, r_θ::Int, r_I::Int;
                         θ_grid::AbstractVector=1.0:r_θ, I_grid::AbstractVector=1.0:r_I)
    return (φ, kP_val, kω_val) -> _build_op_trans_v3_pi(φ, r_θ, r_I, kP_val, kω_val;
                                                           θ_grid, I_grid)
end

"""Pack transition spec into parameter vector (dispatches on type)."""
_pack_spec(s::RobitNNTransitionV3) = pack_robit_nn_v3(s)
_pack_spec(s::OrderedProbitTransitionV3) = pack_op_trans_v3(s)

"""
    get_score_parametric_v3(ψ_all, Pi_bar, params_v3, transition_groups)
        -> Vector{Matrix{Float64}}

Parametric score dL/dφ_g for each transition group, using the chain rule:
  dL/dφ_n = J_full' · dL/dΠ̄_vec_n

where dL/dΠ̄[k,θ,I,n] = ψ[k,θ,I,n] / Π̄[k,θ,I] (score w.r.t. all Π̄ entries)
and J_full = ∂vec(Π̄)/∂φ_g (Jacobian from robit params to full marginalized Π̄).

Returns one (n_φ × N) score matrix per group, pooled across transitions in the group.
"""
function get_score_parametric_v3(ψ_all::Array{Float64,5},
                                  Pi_bar::Vector{Array{Float64,3}},
                                  params_v3,
                                  transition_groups::Vector{Vector{Int}})
    (; r_P, r_ω, trans_specs, π_Z) = params_v3
    r_θ, _, r_I = size(Pi_bar[1])
    N = size(ψ_all, 5)
    r_Z = r_P * r_ω

    θ_grid = collect(1.0:r_θ); I_grid = collect(1.0:r_I)
    P_grid = collect(1.0:r_P); ω_grid = collect(1.0:r_ω)
    π_Z_vec = Float64[π_Z[kP, kω] for kP in 1:r_P for kω in 1:r_ω]

    score_list = Vector{Matrix{Float64}}(undef, length(transition_groups))

    for (g, group) in enumerate(transition_groups)
        t_rep = group[1]
        spec_g = trans_specs[g]
        φ_g = _pack_spec(spec_g)
        build_fn = _make_build_fn(spec_g, r_θ, r_I)
        n_φ = length(φ_g)

        # Floor Π̄
        Π_g = copy(Pi_bar[t_rep])
        Π_g .= max.(Π_g, 1e-15)
        for kI in 1:r_I, kθ in 1:r_θ
            Π_g[:, kθ, kI] ./= sum(Π_g[:, kθ, kI])
        end

        # Jacobian ∂vec(Π̄)/∂φ_g  (n_pi_full × n_φ)
        J_full = ForwardDiff.jacobian(
            φ -> _pi_bar_from_phi(build_fn, φ, π_Z_vec, r_θ, r_I, r_P, r_ω;
                                    P_grid, ω_grid, mode=:full),
            φ_g)

        # Score w.r.t. all Π̄ entries: dL/dΠ̄[k,θ,I,n] = ψ[k,θ,I,n] / Π̄[k,θ,I]
        n_pi = r_θ * r_θ * r_I
        pi_idx = LinearIndices((r_θ, r_θ, r_I))

        # Pool across transitions in group, then chain rule
        dL_dφ = zeros(n_φ, N)
        for t in group
            dL_dΠ = zeros(n_pi, N)
            for n in 1:N, kI in 1:r_I, kθ in 1:r_θ, k in 1:r_θ
                ip = pi_idx[k, kθ, kI]
                @inbounds dL_dΠ[ip, n] = ψ_all[t, k, kθ, kI, n] / Π_g[k, kθ, kI]
            end
            # Chain rule: dL/dφ += J_full' · dL/dΠ̄
            dL_dφ .+= J_full' * dL_dΠ
        end

        score_list[g] = dL_dφ
    end

    return score_list
end

"""
    compute_Var_free_v3(ψ_all, Pi_bar, params_v3, data, transition_groups)
        -> (Var_free_list, Σ_φ_list)

Parametric delta-method Var(Π̄_free) per group:
  1. Parametric score dL/dφ via chain rule (correct — no nonparametric score)
  2. Σ_φ = cov(score_φ') — parametric score covariance
  3. Var(φ̂) = inv(Σ_φ) / N
  4. J_free = ∂Π̄_free/∂φ via ForwardDiff
  5. Var(Π̄_free) = J_free · Var(φ̂) · J_free'
"""
function compute_Var_free_v3(ψ_all::Array{Float64,5},
                               Pi_bar::Vector{Array{Float64,3}},
                               params_v3,
                               data,
                               transition_groups::Vector{Vector{Int}})
    (; r_θ, r_I, T, N) = data
    (; r_P, r_ω, trans_specs, π_Z) = params_v3
    T_minus_1 = T - 1

    θ_grid = collect(1.0:r_θ); I_grid = collect(1.0:r_I)
    P_grid = collect(1.0:r_P); ω_grid = collect(1.0:r_ω)
    π_Z_vec = Float64[π_Z[kP, kω] for kP in 1:r_P for kω in 1:r_ω]

    # Get parametric scores
    score_list = get_score_parametric_v3(ψ_all, Pi_bar, params_v3, transition_groups)

    Var_free_list = Vector{Matrix{Float64}}(undef, T_minus_1)
    Σ_φ_list = Vector{Matrix{Float64}}(undef, length(transition_groups))

    for (g, group) in enumerate(transition_groups)
        spec_g = trans_specs[g]
        φ_g = _pack_spec(spec_g)
        build_fn = _make_build_fn(spec_g, r_θ, r_I)

        # Identify active parameters (non-zero Jacobian columns)
        J_free_full = ForwardDiff.jacobian(
            φ -> _pi_bar_from_phi(build_fn, φ, π_Z_vec, r_θ, r_I, r_P, r_ω;
                                    P_grid, ω_grid, mode=:free),
            φ_g)
        col_norms = [norm(J_free_full[:, j]) for j in 1:length(φ_g)]
        active = findall(n -> n > 1e-6, col_norms)
        n_active = length(active)
        n_φ = length(φ_g)
        if n_active < n_φ
            @info "Group $g: $n_active / $n_φ active parameters ($(n_φ - n_active) dead)"
        end

        # Restrict score and Jacobian to active parameters
        dL_active = score_list[g][active, :]
        J_free = J_free_full[:, active]

        # Parametric score covariance (active params only)
        Σ_φ = cov(dL_active')
        Σ_φ_list[g] = Σ_φ

        # Var(φ̂)
        r = rank(Σ_φ)
        if r < n_active
            @warn "Group $g: parametric score covariance has rank $r / $n_active; using pinv"
            Var_φ = pinv(Σ_φ) / N
        else
            Var_φ = inv(Σ_φ) / N
        end

        # Delta method
        Var_free_g = J_free * Var_φ * J_free'

        for t in group
            Var_free_list[t] = Var_free_g
        end
    end

    return Var_free_list, Σ_φ_list
end

"""
    get_test_stat_v3_grouped(Var_free_list, Pi_bar, data, transition_groups; α=0.05)

Complementarity test on regime-marginalized Π̄ using parametric
Var(Π̄_free) from the delta method.
"""
function get_test_stat_v3_grouped(Var_free_list::Vector{Matrix{Float64}},
                                   Pi_bar::Vector{Array{Float64,3}},
                                   data,
                                   transition_groups::Vector{Vector{Int}};
                                   α::Float64=0.05, rng::AbstractRNG=Random.default_rng())
    # Grouped OC test on the regime-marginalized Π̄ (minimal adjacent constraint
    # set; scale=1 since Var_free is the delta-method covariance, already O(1/N)).
    (; r_θ, r_I, T, N) = data
    S_mat = get_S_adjacent(r_θ, r_I)

    results = Vector{NamedTuple}(undef, length(transition_groups))

    for (g, group) in enumerate(transition_groups)
        t_rep = group[1]

        Π_g = copy(Pi_bar[t_rep])
        Π_g .= max.(Π_g, 1e-15)
        for kI in 1:r_I, kθ in 1:r_θ
            Π_g[:, kθ, kI] ./= sum(Π_g[:, kθ, kI])
        end
        Π_vec = Π_g[1:end-1, :, :][:]
        Var_free = Var_free_list[t_rep]

        V = S_mat * Var_free * S_mat'        # variance of the DID conditions (O(1/N))
        r = mmm_test(S_mat * Π_vec, V; scale = 1.0, α, rng)
        results[g] = (; group, r.T_n, r.c_α, r.reject, r.Ω, r.n_used, Var_free)
    end

    return results
end

"""
    survivor_se_v3_grouped(Var_free_list, Pi_bar, transition_groups)
        -> Vector{Array{Float64,3}}

Survivor function SEs from parametric Var(Π̄_free).
Returns SE[t] of size (r_θ-1, r_θ, r_I) for each transition.
"""
function survivor_se_v3_grouped(Var_free_list::Vector{Matrix{Float64}},
                                 Pi_bar::Vector{Array{Float64,3}},
                                 transition_groups::Vector{Vector{Int}})
    T_minus_1 = length(Var_free_list)
    SE_list = Vector{Array{Float64,3}}(undef, T_minus_1)

    for (g, group) in enumerate(transition_groups)
        t_rep = group[1]
        Π_g = Pi_bar[t_rep]
        SE_list[t_rep] = survivor_se_all(Var_free_list[t_rep], Π_g)
        for t in group[2:end]
            SE_list[t] = SE_list[t_rep]
        end
    end

    return SE_list
end
