# ═══════════════════════════════════════════════════════════════
# Parametric inference for V3 (no ε_Y) via Fisher's identity
#
# See /inference_notes.md for the design doc.
#
# Per-child score computation for each parameter block, using
# smoothed posteriors (γ_z, ξ_z, w_z) from one forward-backward pass
# per regime per child.  OPG-based Fisher and delta-method variance.
#
# This mirrors inference_v3_eps.jl but adapted for V3's parameter
# structure:
#   - Regime z = (kP, kω), r_Z = r_P · r_ω = 9 (not 27)
#   - CCP: OrderedProbitCCPV3 with income bins (not ε_Y grid)
#   - Income: IncomeModel (4 params, no δ)
#   - Prior: π_Z (r_P × r_ω), not factored
#   - No π_ε block
#
# Priority blocks: transition, CCP, income, π_Z.
# Nuisance blocks (π0, Q_P, probit_θ[t], probit_I[t], adult)
# are stubbed; SEs are therefore conditional on those at their MLE.
#
# Depends on:
#   parametric_models.jl, parametric_models_v3.jl
#   estimation_multiperiod.jl            — compute_emission_likelihood_t!
#   estimation_multiperiod_v3.jl         — forward_backward_v3_n!,
#                                          precompute_Pi_z, regime_z,
#                                          regime_loglik, regime_kPkω,
#                                          precompute_CCP_z, bin_income,
#                                          _pi_bar_from_phi, _make_build_fn,
#                                          _pack_spec
#   inference_common.jl                  — softmax_to_probs, probs_to_softmax,
#                                          opg_fisher, var_phi_hat,
#                                          delta_cov, ses_from_cov
# ═══════════════════════════════════════════════════════════════

using LinearAlgebra, Random, ForwardDiff

# ─────────────────────────────────────────────────────────────────
# Precomputed per-regime Jacobians for transition and CCP blocks
# ─────────────────────────────────────────────────────────────────

"""
    precompute_J_Pi_z_v3(trans_specs, transition_groups, r_θ, r_I, r_P, r_ω; grids...)
        -> (J_list, Π_list)

For each group g and each regime z = (kP-1)*r_ω + kω, compute:
  J_list[g][z]  :: Matrix (r_θ·r_θ·r_I, n_φ_g)   = ∂vec(Π_z)/∂φ_g
  Π_list[g][z]  :: Array{Float64,3} (r_θ, r_θ, r_I)  = Π_z at φ̂_g
"""
function precompute_J_Pi_z_v3(trans_specs::Vector{RobitNNTransitionV3},
                               transition_groups::Vector{Vector{Int}},
                               r_θ::Int, r_I::Int, r_P::Int, r_ω::Int;
                               θ_grid::AbstractVector=1.0:r_θ,
                               I_grid::AbstractVector=1.0:r_I,
                               P_grid::AbstractVector=1.0:r_P,
                               ω_grid::AbstractVector=1.0:r_ω)
    r_Z = r_P * r_ω
    n_groups = length(transition_groups)
    J_list = Vector{Vector{Matrix{Float64}}}(undef, n_groups)
    Π_list = Vector{Vector{Array{Float64,3}}}(undef, n_groups)

    for (g, _) in enumerate(transition_groups)
        spec = trans_specs[g]
        φ_g = pack_robit_nn_v3(spec)
        K = spec.K; ν = spec.ν
        J_list[g] = Vector{Matrix{Float64}}(undef, r_Z)
        Π_list[g] = Vector{Array{Float64,3}}(undef, r_Z)

        for kω in 1:r_ω, kP in 1:r_P
            z = regime_z(kP, kω, r_ω)
            Π_z = _build_robit_nn_v3_pi(φ_g, r_θ, r_I, K, ν,
                                          P_grid[kP], ω_grid[kω];
                                          θ_grid, I_grid)
            J_z = ForwardDiff.jacobian(
                φ -> vec(_build_robit_nn_v3_pi(φ, r_θ, r_I, K, ν,
                                                  P_grid[kP], ω_grid[kω];
                                                  θ_grid, I_grid)),
                φ_g)
            J_list[g][z] = J_z
            Π_list[g][z] = Π_z
        end
    end
    return J_list, Π_list
end

# ─────────────────────────────────────────────────────────────────
# CCP builders from packed parameter vector (ForwardDiff-compatible)
# ─────────────────────────────────────────────────────────────────

"""
Build CCP_init directly from a packed parameter vector (ForwardDiff-compatible).
Pack order matches `pack_ccp_v3`: [c_reparam(r_I-1), β₁, β₃, β₄, β₅].  β₂ = 0.
Income enters via bin value y_b (continuous).
"""
function _ccp_v3_init_from_vec(vec_φ::AbstractVector{Tv},
                                kP_val::Real, kω_val::Real, y_b::Real;
                                θ_grid::AbstractVector, r_I::Int) where Tv
    nc = r_I - 1
    c  = _unpack_cutpoints(vec_φ[1:nc])
    β1 = vec_φ[nc + 1]
    β3 = vec_φ[nc + 2]
    β4 = vec_φ[nc + 3]
    β5 = vec_φ[nc + 4]
    r_θ = length(θ_grid)
    M = zeros(promote_type(Tv, Float64), r_I, r_θ)
    regime_part = β3 * kP_val + β4 * kω_val + β5 * y_b
    for kθ in 1:r_θ
        η = β1 * θ_grid[kθ] + regime_part
        M[:, kθ] = _ordered_probit_pmf(c, η, r_I)
    end
    return M
end

"""
Build CCP_trans directly from a packed parameter vector (ForwardDiff-compatible).
β₂ (I_old) = 0.  Income enters via bin value y_b.
"""
function _ccp_v3_trans_from_vec(vec_φ::AbstractVector{Tv},
                                 kP_val::Real, kω_val::Real, y_b::Real;
                                 θ_grid::AbstractVector,
                                 I_grid::AbstractVector, r_I::Int) where Tv
    nc = r_I - 1
    c  = _unpack_cutpoints(vec_φ[1:nc])
    β1 = vec_φ[nc + 1]
    β3 = vec_φ[nc + 2]
    β4 = vec_φ[nc + 3]
    β5 = vec_φ[nc + 4]
    r_θ = length(θ_grid)
    CCP = zeros(promote_type(Tv, Float64), r_I, r_I, r_θ)
    regime_part = β3 * kP_val + β4 * kω_val + β5 * y_b
    for kθ1 in 1:r_θ, kI in 1:r_I
        η = β1 * θ_grid[kθ1] + regime_part   # β₂·I_old = 0
        CCP[:, kI, kθ1] = _ordered_probit_pmf(c, η, r_I)
    end
    return CCP
end

"""
    precompute_J_CCP_z_v3(ccp_spec, r_P, r_ω, bin_values, r_θ, r_I; grids...)
        -> (J_init, J_trans, CCP_init, CCP_trans)

For each regime z and income bin b, compute:
  J_init[z][b]  :: Matrix (r_I·r_θ, n_φ_ccp)     = ∂vec(CCP_init_z_b)/∂φ_ccp
  J_trans[z][b] :: Matrix (r_I·r_I·r_θ, n_φ_ccp) = ∂vec(CCP_trans_z_b)/∂φ_ccp
  CCP_init[z][b]  :: Matrix (r_I, r_θ)
  CCP_trans[z][b] :: Array{Float64,3} (r_I, r_I, r_θ)
"""
function precompute_J_CCP_z_v3(ccp_spec::OrderedProbitCCPV3,
                                r_P::Int, r_ω::Int,
                                bin_values::Vector{Float64},
                                r_θ::Int, r_I::Int;
                                θ_grid::AbstractVector=1.0:r_θ,
                                I_grid::AbstractVector=1.0:r_I,
                                P_grid::AbstractVector=1.0:r_P,
                                ω_grid::AbstractVector=1.0:r_ω)
    r_Z = r_P * r_ω
    n_bins = length(bin_values)
    φ_ccp = pack_ccp_v3(ccp_spec)

    J_init    = [[Matrix{Float64}(undef, 0, 0) for _ in 1:n_bins] for _ in 1:r_Z]
    J_trans   = [[Matrix{Float64}(undef, 0, 0) for _ in 1:n_bins] for _ in 1:r_Z]
    CCP_init  = [[Matrix{Float64}(undef, 0, 0) for _ in 1:n_bins] for _ in 1:r_Z]
    CCP_trans = [[Array{Float64,3}(undef, 0, 0, 0) for _ in 1:n_bins] for _ in 1:r_Z]

    for kω in 1:r_ω, kP in 1:r_P
        z = regime_z(kP, kω, r_ω)
        kP_val = P_grid[kP]; kω_val = ω_grid[kω]

        for b in 1:n_bins
            y_b = bin_values[b]

            CCP_init[z][b] = _ccp_v3_init_from_vec(φ_ccp, kP_val, kω_val, y_b;
                                                     θ_grid, r_I)
            CCP_trans[z][b] = _ccp_v3_trans_from_vec(φ_ccp, kP_val, kω_val, y_b;
                                                      θ_grid, I_grid, r_I)
            J_init[z][b] = ForwardDiff.jacobian(
                φ -> vec(_ccp_v3_init_from_vec(φ, kP_val, kω_val, y_b;
                                                θ_grid, r_I)),
                φ_ccp)
            J_trans[z][b] = ForwardDiff.jacobian(
                φ -> vec(_ccp_v3_trans_from_vec(φ, kP_val, kω_val, y_b;
                                                 θ_grid, I_grid, r_I)),
                φ_ccp)
        end
    end
    return J_init, J_trans, CCP_init, CCP_trans
end

# ─────────────────────────────────────────────────────────────────
# Per-child score for a single block — inlined for speed
# ─────────────────────────────────────────────────────────────────

"""
Accumulate child-n transition score into `s_trans_g` for group g.
    s_trans_g[:] += Σ_z w(z,n) · Σ_{t∈group} Σ_{k,θ,I} ξ_marg[k,θ,I]/Π_z[k,θ,I] · J_z[cell,:]

where ξ_marg[k,θ,I] = Σ_{I'} ξ_z[t, k, I', θ, I].
"""
function _score_trans_contrib_v3!(
    s_trans_g::AbstractVector{Float64},     # length n_φ_g
    w_vec::Vector{Float64},                  # length r_Z (regime posterior for child n)
    ξ_z_all::Vector{Vector{Array{Float64,4}}},  # ξ_z_all[z][t]
    J_list_g::Vector{Matrix{Float64}},       # J_list_g[z] :: (r_θ·r_θ·r_I × n_φ_g)
    Π_list_g::Vector{Array{Float64,3}},      # Π_list_g[z]
    group::Vector{Int},
    r_P::Int, r_ω::Int,
    r_θ::Int, r_I::Int,
)
    cell_lin = LinearIndices((r_θ, r_θ, r_I))   # (k, θ, I) → linear vec idx
    nφ = length(s_trans_g)
    for kω in 1:r_ω, kP in 1:r_P
        z = regime_z(kP, kω, r_ω)
        wz = w_vec[z]
        wz < 1e-15 && continue
        Π_z = Π_list_g[z]
        J_z = J_list_g[z]

        for t in group
            ξ_t = ξ_z_all[z][t]
            for kI in 1:r_I, kθ in 1:r_θ, k in 1:r_θ
                @inbounds πv = Π_z[k, kθ, kI]
                πv < 1e-300 && continue
                # Marginalize over I' (ξ[kθ1, kI1, kθ, kI])
                m = 0.0
                for kI1 in 1:r_I
                    @inbounds m += ξ_t[k, kI1, kθ, kI]
                end
                m < 1e-15 && continue
                cell = cell_lin[k, kθ, kI]
                coef = wz * m / πv
                @inbounds for p in 1:nφ
                    s_trans_g[p] += coef * J_z[cell, p]
                end
            end
        end
    end
    return nothing
end

"""
CCP score for child n.  Combines the t=1 (init) and t>=2 (trans) contributions.
CCP Jacobians vary by (z, income_bin) — bin_map_n[t] gives the income bin for
child n at period t.

Init contribution (t=1): use bin_map_n[1] for income bin.
Trans contribution (t=1..T-1 in ξ indexing): use bin_map_n[t+1] for income bin
  (ξ[t] corresponds to the transition from period t to t+1, and the CCP for
   I_{t+1} uses income at period t+1).
"""
function _score_ccp_contrib_v3!(
    s_ccp::AbstractVector{Float64},         # length n_φ_ccp
    w_vec::Vector{Float64},
    γ_z_all::Vector{Array{Float64,3}},       # γ_z_all[z] :: (T, r_θ, r_I)
    ξ_z_all::Vector{Vector{Array{Float64,4}}},
    J_init_z::Vector{Vector{Matrix{Float64}}},   # J_init_z[z][b]
    J_trans_z::Vector{Vector{Matrix{Float64}}},   # J_trans_z[z][b]
    CCP_init_z::Vector{Vector{Matrix{Float64}}},  # CCP_init_z[z][b]
    CCP_trans_z::Vector{Vector{Array{Float64,3}}}, # CCP_trans_z[z][b]
    bin_map_n::AbstractVector{Int},           # bin_map[n, :] — length T
    r_P::Int, r_ω::Int,
    r_θ::Int, r_I::Int, T::Int,
)
    init_cell_lin  = LinearIndices((r_I, r_θ))               # (I, θ) for CCP_init
    trans_cell_lin = LinearIndices((r_I, r_I, r_θ))          # (I', I_old, θ')
    nφ = length(s_ccp)

    for kω in 1:r_ω, kP in 1:r_P
        z = regime_z(kP, kω, r_ω)
        wz = w_vec[z]
        wz < 1e-15 && continue

        γ_z = γ_z_all[z]

        # ── Initial CCP contribution (t = 1) ──
        b1 = bin_map_n[1]
        C_init = CCP_init_z[z][b1]
        Ji = J_init_z[z][b1]
        for kI in 1:r_I, kθ in 1:r_θ
            @inbounds g = γ_z[1, kθ, kI]
            g < 1e-15 && continue
            @inbounds c = C_init[kI, kθ]
            c < 1e-300 && continue
            coef = wz * g / c
            cell = init_cell_lin[kI, kθ]
            @inbounds for p in 1:nφ
                s_ccp[p] += coef * Ji[cell, p]
            end
        end

        # ── Transition CCP contributions (t = 1..T-1 in ξ) ──
        for t in 1:T-1
            b_tp1 = bin_map_n[t+1]
            C_trans = CCP_trans_z[z][b_tp1]
            Jt = J_trans_z[z][b_tp1]
            ξ_t = ξ_z_all[z][t]
            for kθ1 in 1:r_θ, kI in 1:r_I, kI1 in 1:r_I
                @inbounds c = C_trans[kI1, kI, kθ1]
                c < 1e-300 && continue
                # Marginalize ξ over kθ (second-to-last dim)
                m = 0.0
                for kθ in 1:r_θ
                    @inbounds m += ξ_t[kθ1, kI1, kθ, kI]
                end
                m < 1e-15 && continue
                coef = wz * m / c
                cell = trans_cell_lin[kI1, kI, kθ1]
                @inbounds for p in 1:nφ
                    s_ccp[p] += coef * Jt[cell, p]
                end
            end
        end
    end
    return nothing
end

"""
Income score for child n.
    s_inc = Σ_z w(z,n) Σ_{t: y_{nt} observed} ∂log φ(y_{nt}; μ_z, σ)/∂(β₀, β_P, β_ω, log_σ)

with μ_z = β₀ + β_P·kP + β_ω·kω.
"""
function _score_inc_contrib_v3!(
    s_inc::AbstractVector{Float64},           # length 4
    w_vec::Vector{Float64},
    y_n::AbstractVector{Float64},              # length T — log income, NaN = missing
    income_model::IncomeModel,
    r_P::Int, r_ω::Int,
    P_grid::AbstractVector, ω_grid::AbstractVector,
    T::Int,
)
    σ = exp(income_model.log_sigma)
    σ2 = σ * σ
    for kω in 1:r_ω, kP in 1:r_P
        z = regime_z(kP, kω, r_ω)
        wz = w_vec[z]
        wz < 1e-15 && continue
        kP_val = P_grid[kP]; kω_val = ω_grid[kω]
        μ = income_model.beta0 + income_model.betaP * kP_val +
            income_model.betaω * kω_val
        for t in 1:T
            y = y_n[t]
            isnan(y) && continue
            r = y - μ                  # residual
            # ∂log N/∂β₀ = r/σ²
            # ∂log N/∂β_P = (r/σ²) · kP_val
            # ∂log N/∂β_ω = (r/σ²) · kω_val
            # ∂log N/∂log σ = (r²/σ²) - 1
            inv_σ2 = r / σ2
            @inbounds s_inc[1] += wz * inv_σ2
            @inbounds s_inc[2] += wz * inv_σ2 * kP_val
            @inbounds s_inc[3] += wz * inv_σ2 * kω_val
            @inbounds s_inc[4] += wz * (r * r / σ2 - 1)
        end
    end
    return nothing
end

"""
π_Z score for child n (softmax-reparameterized).
π_Z(kP, kω) = softmax([0; α_Z])[k], k = (kP-1)·r_ω + kω.
Bin 1 is the reference.

Score:
    s_{α_j} = w_Z(j+1, n) - π_Z_flat(j+1)
"""
function _score_pi_Z_contrib_v3!(
    s_π_Z::AbstractVector{Float64},          # length (r_P·r_ω - 1)
    w_vec::Vector{Float64},                   # length r_Z
    π_Z_flat::Vector{Float64},                # length r_P·r_ω, reading order (kP-1)*r_ω + kω
    r_P::Int, r_ω::Int,
)
    K_Z = r_P * r_ω
    # w_vec is already indexed by z = (kP-1)*r_ω + kω, same as π_Z_flat
    # Score: s[j] = w_vec[j+1] - π_Z_flat[j+1]
    for j in 1:(K_Z - 1)
        idx = j + 1
        @inbounds s_π_Z[j] += w_vec[idx] - π_Z_flat[idx]
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────
# Main entry point: per-child scores (priority blocks)
# ─────────────────────────────────────────────────────────────────

"""
    per_child_scores_v3(params_v3, params_skill, data;
                          transition_groups = [[1,2,3],[4,5,6]],
                          n_income_bins = 20)
        -> NamedTuple (trans_g::Vector{Matrix}, ccp::Matrix, inc::Matrix,
                       π_Z::Matrix, indices, n_groups, dims)

Computes per-child parametric scores for the priority blocks:
transition (per group), CCP, income, and π_Z.  Remaining blocks
(π0, Q_P, probit_θ[t], probit_I[t], adult) are stubbed (not
computed); their omission gives conditional SEs.

Per-child matrices have shape (n_φ_block × N).  `indices` provides
the cumulative index ranges when stacked.
"""
function per_child_scores_v3(params_v3, params_skill, data;
                               transition_groups::Vector{Vector{Int}} = [[1,2,3],[4,5,6]],
                               n_income_bins::Int = 20)
    (; r_P, r_ω, π0, π_Z, ccp_spec, Q_P, income_model, trans_specs) = params_v3
    (; r_θ, r_I, T) = params_skill
    N = data.N
    r_Z = r_P * r_ω

    θ_grid = collect(1.0:r_θ); I_grid = collect(1.0:r_I)
    P_grid = collect(1.0:r_P); ω_grid = collect(1.0:r_ω)

    # Bin income
    bin_map, bin_values = bin_income(data.Y_income; n_bins=n_income_bins)

    # Precompute per-regime Jacobians (constant across children)
    J_Π_z, Π_z_list = precompute_J_Pi_z_v3(trans_specs, transition_groups,
                                             r_θ, r_I, r_P, r_ω;
                                             θ_grid, I_grid, P_grid, ω_grid)
    J_CCP_init, J_CCP_trans, CCP_init_J, CCP_trans_J =
        precompute_J_CCP_z_v3(ccp_spec, r_P, r_ω, bin_values, r_θ, r_I;
                               θ_grid, I_grid, P_grid, ω_grid)

    # Transitions: Pi_z_pre for forward-backward
    Pi_z_pre = precompute_Pi_z(trans_specs, transition_groups,
                                r_θ, r_I, r_P, r_ω;
                                θ_grid, I_grid, P_grid, ω_grid)

    # CCP arrays for forward-backward
    CCP_init_pre_FB, CCP_trans_pre_FB = precompute_CCP_z(
        ccp_spec, r_Z, r_P, r_ω, bin_values, r_θ, r_I;
        θ_grid, I_grid, P_grid, ω_grid)

    # Priors
    π_Z_flat = Float64[π_Z[kP, kω] for kP in 1:r_P for kω in 1:r_ω]
    log_π_Z = [log(max(π_Z[kP, kω], 1e-300)) for kP in 1:r_P, kω in 1:r_ω]

    # Score storage per block (n_φ_block × N)
    n_groups = length(transition_groups)
    n_φ_g = length(pack_robit_nn_v3(trans_specs[1]))
    n_φ_ccp = length(pack_ccp_v3(ccp_spec))
    n_φ_inc = 4
    n_φ_π_Z = r_P * r_ω - 1

    s_trans_g = [zeros(n_φ_g, N) for _ in 1:n_groups]
    s_ccp     = zeros(n_φ_ccp, N)
    s_inc     = zeros(n_φ_inc, N)
    s_π_Z     = zeros(n_φ_π_Z, N)

    # Per-child scratch
    α     = zeros(T, r_θ, r_I)
    β     = zeros(T, r_θ, r_I)
    scale = zeros(T)
    L_scratch = [zeros(r_θ, r_I) for _ in 1:T]
    γ_z_all = [zeros(T, r_θ, r_I) for _ in 1:r_Z]
    ξ_z_all = [[zeros(r_θ, r_I, r_θ, r_I) for _ in 1:T-1] for _ in 1:r_Z]
    CCP_nz  = Vector{Array{Float64,3}}(undef, T-1)
    π0_z_buf  = zeros(r_θ)
    π0_nz_buf = zeros(r_θ, r_I)
    log_ll_z  = zeros(r_Z)

    # Check for adult outcome data
    adult_matrices = haskey(params_v3, :adult_matrices) ? params_v3.adult_matrices : Array{Float64,3}[]
    has_adult = !isempty(adult_matrices) && haskey(data, :X_adult) && !isempty(data.X_adult)

    for n in 1:N
        # Emission likelihoods (same across regimes)
        for t in 1:T
            compute_emission_likelihood_t!(L_scratch[t], t, n, params_skill, data)
        end

        # Pass 1: forward-backward per regime
        for kω in 1:r_ω, kP in 1:r_P
            z = regime_z(kP, kω, r_ω)

            ll_regime = regime_loglik(n, kP, kω, Q_P, income_model, data;
                                       P_grid, ω_grid)
            log_ll_z[z] = log_π_Z[kP, kω] + ll_regime

            # π0(θ_1 | kP, kω)
            for kθ in 1:r_θ; @inbounds π0_z_buf[kθ] = max(π0[kθ, kP, kω], 0.0); end
            s = sum(π0_z_buf)
            if s > 0; π0_z_buf ./= s; else; fill!(π0_z_buf, 1.0/r_θ); end

            ccp_init_arr = CCP_init_pre_FB[z][bin_map[n, 1]]
            for kI in 1:r_I, kθ in 1:r_θ
                @inbounds π0_nz_buf[kθ, kI] = π0_z_buf[kθ] * ccp_init_arr[kI, kθ]
            end

            for t in 1:T-1
                CCP_nz[t] = CCP_trans_pre_FB[z][bin_map[n, t+1]]
            end

            for t in 1:T-1; fill!(ξ_z_all[z][t], 0.0); end

            ll_skills = forward_backward_v3_n!(
                α, β, scale, γ_z_all[z], ξ_z_all[z],
                π0_nz_buf, Pi_z_pre[z], CCP_nz, L_scratch, r_θ, r_I, T)
            log_ll_z[z] += ll_skills

            # Adult outcome contribution to regime log-likelihood (for correct w_z)
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
        w_vec = w_unnorm ./ w_sum

        # ── Transition scores ──
        for (g, group) in enumerate(transition_groups)
            s_view = @view s_trans_g[g][:, n]
            _score_trans_contrib_v3!(s_view, w_vec, ξ_z_all,
                                     J_Π_z[g], Π_z_list[g], group,
                                     r_P, r_ω, r_θ, r_I)
        end

        # ── CCP score ──
        s_view_ccp = @view s_ccp[:, n]
        bin_map_n = @view bin_map[n, :]
        _score_ccp_contrib_v3!(s_view_ccp, w_vec, γ_z_all, ξ_z_all,
                                J_CCP_init, J_CCP_trans, CCP_init_J, CCP_trans_J,
                                bin_map_n,
                                r_P, r_ω, r_θ, r_I, T)

        # ── Income score ──
        s_view_inc = @view s_inc[:, n]
        y_n = @view data.Y_income[n, :]
        _score_inc_contrib_v3!(s_view_inc, w_vec, y_n, income_model,
                                r_P, r_ω, P_grid, ω_grid, T)

        # ── π_Z score ──
        s_view_π = @view s_π_Z[:, n]
        _score_pi_Z_contrib_v3!(s_view_π, w_vec, π_Z_flat, r_P, r_ω)
    end

    return (
        trans_g  = s_trans_g,
        ccp      = s_ccp,
        inc      = s_inc,
        π_Z      = s_π_Z,
        n_groups = n_groups,
        dims     = (; r_θ, r_I, r_P, r_ω, T, N,
                     n_φ_g, n_φ_ccp, n_φ_inc, n_φ_π_Z),
    )
end

# ─────────────────────────────────────────────────────────────────
# Score stacking
# ─────────────────────────────────────────────────────────────────

"""
    stack_scores_v3(block_scores)
        -> (S::Matrix{Float64}, ranges::NamedTuple)

Stack per-block scores into a single (n_φ_total × N) matrix.
`ranges.trans_g[g]`, `ranges.ccp`, `ranges.inc`, `ranges.π_Z` give the
row index ranges for each block.
"""
function stack_scores_v3(block_scores)
    dims = block_scores.dims
    n_groups = block_scores.n_groups
    N = dims.N

    n_φ_g = dims.n_φ_g
    n_φ_ccp = dims.n_φ_ccp
    n_φ_inc = dims.n_φ_inc
    n_φ_π_Z = dims.n_φ_π_Z
    n_total = n_groups * n_φ_g + n_φ_ccp + n_φ_inc + n_φ_π_Z

    S = zeros(n_total, N)
    idx = 1
    trans_ranges = Vector{UnitRange{Int}}(undef, n_groups)
    for g in 1:n_groups
        trans_ranges[g] = idx:(idx + n_φ_g - 1)
        S[trans_ranges[g], :] = block_scores.trans_g[g]
        idx += n_φ_g
    end
    ccp_range = idx:(idx + n_φ_ccp - 1)
    S[ccp_range, :] = block_scores.ccp
    idx += n_φ_ccp

    inc_range = idx:(idx + n_φ_inc - 1)
    S[inc_range, :] = block_scores.inc
    idx += n_φ_inc

    π_Z_range = idx:(idx + n_φ_π_Z - 1)
    S[π_Z_range, :] = block_scores.π_Z
    idx += n_φ_π_Z

    return S, (; trans_g = trans_ranges, ccp = ccp_range,
                 inc = inc_range, π_Z = π_Z_range)
end

# ─────────────────────────────────────────────────────────────────
# Top-level Var(φ̂) with stacked scores
# ─────────────────────────────────────────────────────────────────

"""
    compute_var_phi_v3(params_v3, params_skill, data;
                        transition_groups=[[1,2,3],[4,5,6]],
                        n_income_bins=20,
                        active_tol=1e-6)
        -> (scores, S, ranges, Σ_s, Var_φ, active, inactive,
            cond_num, rank_effective, score_check_max)

Run the full per-child score computation, stack, compute OPG Fisher,
and invert to get Var(φ̂).

Identifies inactive parameters via near-zero score column norms and
drops them before inversion.  Var_φ is returned as an n_φ_total ×
n_φ_total matrix with zeros in the inactive rows/columns.
"""
function compute_var_phi_v3(params_v3, params_skill, data;
                             transition_groups::Vector{Vector{Int}}=[[1,2,3],[4,5,6]],
                             n_income_bins::Int=20,
                             active_tol::Float64=1e-6,
                             verbose::Bool=false)
    scores = per_child_scores_v3(params_v3, params_skill, data;
                                   transition_groups, n_income_bins)
    S, ranges = stack_scores_v3(scores)
    N = scores.dims.N
    n_φ = size(S, 1)

    # Sanity: mean score should be near zero at the MLE
    score_mean = sum(S; dims=2)[:, 1] ./ N
    score_check_max = maximum(abs.(score_mean))

    # Identify active parameters via row norm of S
    row_norms = [norm(S[j, :]) / sqrt(N) for j in 1:n_φ]
    active   = findall(n -> n > active_tol, row_norms)
    inactive = findall(n -> n ≤ active_tol, row_norms)

    if verbose
        println("Score rows: $n_φ total, $(length(active)) active, $(length(inactive)) inactive")
        println("Max |mean score| across params: $score_check_max  (should be ~0 at MLE)")
    end

    # Work on active block
    S_active = S[active, :]
    Σ_s_active = (S_active * S_active') / N

    # Symmetrize for numerical stability
    Σ_s_active = (Σ_s_active + Σ_s_active') / 2

    # Condition number
    cond_num = try
        cond(Σ_s_active)
    catch
        NaN
    end

    # SVD-based pseudo-inverse with rank truncation.
    rank_tol = 1e-8
    F = svd(Σ_s_active)
    sv_max = maximum(F.S)
    sv_cut = rank_tol * sv_max
    inv_S = [s > sv_cut ? 1.0 / s : 0.0 for s in F.S]
    Var_φ_active = F.V * Diagonal(inv_S) * F.U' ./ N
    # Resymmetrize
    Var_φ_active = (Var_φ_active + Var_φ_active') / 2

    rank_effective = count(>(sv_cut), F.S)

    if verbose
        println("Effective rank: $rank_effective / $(length(F.S))")
    end

    # Embed active block back into full n_φ × n_φ matrix
    Var_φ = zeros(n_φ, n_φ)
    for (i_local, i_global) in enumerate(active)
        for (j_local, j_global) in enumerate(active)
            Var_φ[i_global, j_global] = Var_φ_active[i_local, j_local]
        end
    end

    # Symmetrize Σ_s for returning
    Σ_s = (S * S') / N
    Σ_s = (Σ_s + Σ_s') / 2

    return (; scores, S, ranges, Σ_s, Var_φ, active, inactive,
             cond_num, rank_effective, score_check_max)
end

# ─────────────────────────────────────────────────────────────────
# Delta method wrappers for quantities of interest
# ─────────────────────────────────────────────────────────────────

"""
    delta_cov_survivor_bar_v3(inf_result, params_v3, params_skill;
                               transition_groups=[[1,2,3],[4,5,6]])
        -> Vector{NamedTuple} (one entry per group)

Full covariance of the regime-marginalized survivor-function free
entries, computed via the delta method.  Uses transition params and
π_Z from the joint Var(φ̂).

Per-group:
  Π̄_g = Σ_{kP,kω} π_Z(kP, kω) · Π_{kP, kω}(φ_g_trans)
  S̄_free[θ, I, k] = Σ_{j>k} Π̄_g[j, θ, I]

For each group we compute the Jacobian w.r.t. (φ_g_trans, α_π_Z) and
apply delta_cov(J, V_block).
"""
function delta_cov_survivor_bar_v3(inf_result, params_v3, params_skill;
                                    transition_groups::Vector{Vector{Int}}=[[1,2,3],[4,5,6]])
    (; r_P, r_ω, π_Z, trans_specs) = params_v3
    (; r_θ, r_I) = params_skill
    P_grid = collect(1.0:r_P); ω_grid = collect(1.0:r_ω)
    θ_grid = collect(1.0:r_θ); I_grid = collect(1.0:r_I)

    n_cut = r_θ - 1
    n_free = n_cut * r_θ * r_I

    # Softmax reparameterization of π_Z
    K_Z = r_P * r_ω
    π_Z_flat = Float64[π_Z[kP, kω] for kP in 1:r_P for kω in 1:r_ω]
    α_Z_hat = probs_to_softmax(π_Z_flat; ref=1)

    results = []

    for (g, _) in enumerate(transition_groups)
        spec_g = trans_specs[g]
        K = spec_g.K; ν = spec_g.ν
        φ_g_hat = pack_robit_nn_v3(spec_g)
        n_φ_g = length(φ_g_hat)
        n_α = length(α_Z_hat)

        # Combined parameter vector: [φ_g ; α_Z]
        θ_combined = vcat(φ_g_hat, α_Z_hat)

        # Map combined → survivor-free vector
        function survivor_free_map(θ_vec)
            φ_g = θ_vec[1:n_φ_g]
            α   = θ_vec[(n_φ_g + 1):end]
            π   = softmax_to_probs(α)   # length K_Z
            # Π̄ marginalized
            TE  = eltype(θ_vec)
            Π_bar = zeros(TE, r_θ, r_θ, r_I)
            for kω in 1:r_ω, kP in 1:r_P
                w = π[(kP - 1) * r_ω + kω]
                Π_z = _build_robit_nn_v3_pi(φ_g, r_θ, r_I, K, ν,
                                              P_grid[kP], ω_grid[kω];
                                              θ_grid, I_grid)
                Π_bar .+= w .* Π_z
            end
            # Survivor free entries: S(k|θ,I) for k=1..n_cut
            #   S(k|θ,I) = Σ_{j>k} Π̄(j, θ, I)
            S_flat = Vector{TE}(undef, n_free)
            idx = 0
            for kI in 1:r_I, kθ in 1:r_θ, k in 1:n_cut
                idx += 1
                sum_s = zero(TE)
                for j in (k + 1):r_θ
                    sum_s += Π_bar[j, kθ, kI]
                end
                S_flat[idx] = sum_s
            end
            return S_flat
        end

        # Jacobian via ForwardDiff
        J_full = ForwardDiff.jacobian(survivor_free_map, θ_combined)

        # Var(φ_g, α_Z) — extract from inf_result.Var_φ
        trans_range = inf_result.ranges.trans_g[g]
        π_Z_range   = inf_result.ranges.π_Z
        combined_idx = vcat(collect(trans_range), collect(π_Z_range))
        V_combined = inf_result.Var_φ[combined_idx, combined_idx]

        Var_S = delta_cov(J_full, V_combined)
        ses   = sqrt.(max.(diag(Var_S), 0.0))

        # Compute the survivor estimate at the MLE
        S_hat = survivor_free_map(θ_combined)

        push!(results, (; group = g, S_hat, Var_S, ses, n_cut, n_free))
    end

    return results
end
