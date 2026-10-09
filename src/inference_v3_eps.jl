# ═══════════════════════════════════════════════════════════════
# Parametric inference for V3+ε_Y via Fisher's identity
#
# See /inference_notes.md for the design doc.
#
# Per-child score computation for each parameter block, using
# smoothed posteriors (γ_z, ξ_z, w_z) from one forward-backward pass
# per regime per child.  OPG-based Fisher and delta-method variance.
#
# This initial version covers the four blocks directly needed for:
#   - Π̄_free complementarity-test inference  : transition + π_Pω
#   - β₅/δ delta-method SE                   : CCP + income
# Nuisance blocks (π_ε, π0, Q_P, probit_θ[t], probit_I[t], adult)
# are stubbed; SEs are therefore conditional on those at their MLE.
#
# Depends on:
#   parametric_models.jl, parametric_models_v3.jl, parametric_models_v3_eps.jl
#   estimation_multiperiod.jl            — compute_emission_likelihood_t!
#   estimation_multiperiod_v3.jl         — forward_backward_v3_n!,
#                                          precompute_Pi_z, regime_z,
#                                          _pi_bar_from_phi, _make_build_fn,
#                                          _pack_spec
#   estimation_multiperiod_v3_eps.jl     — expectation_step_v3_eps! helpers,
#                                          regime_loglik_eps,
#                                          precompute_CCP_z_eps
#   inference_common.jl                  — softmax_to_probs, opg_fisher,
#                                          var_phi_hat, delta_cov
# ═══════════════════════════════════════════════════════════════

using LinearAlgebra, Random, ForwardDiff

# ─────────────────────────────────────────────────────────────────
# Precomputed per-regime Jacobians for transition and CCP blocks
# ─────────────────────────────────────────────────────────────────

"""
    precompute_J_Pi_z_eps(trans_specs, transition_groups, r_θ, r_I, r_P, r_ω; grids...)
        -> (J_list, Π_list)

For each group g and each transition regime z_trans = (kP-1)*r_ω + kω,
compute:
  J_list[g][z_trans]  :: Matrix (r_θ·r_θ·r_I, n_φ_g)   = ∂vec(Π_z)/∂φ_g
  Π_list[g][z_trans]  :: Array{Float64,3} (r_θ, r_θ, r_I)  = Π_z at φ̂_g

Transitions are shared across ε_Y, so these are indexed by (g, z_trans).
"""
function precompute_J_Pi_z_eps(trans_specs::Vector{RobitNNTransitionV3},
                                transition_groups::Vector{Vector{Int}},
                                r_θ::Int, r_I::Int, r_P::Int, r_ω::Int;
                                θ_grid::AbstractVector=1.0:r_θ,
                                I_grid::AbstractVector=1.0:r_I,
                                P_grid::AbstractVector=1.0:r_P,
                                ω_grid::AbstractVector=1.0:r_ω)
    r_Z_trans = r_P * r_ω
    n_groups  = length(transition_groups)
    J_list = Vector{Vector{Matrix{Float64}}}(undef, n_groups)
    Π_list = Vector{Vector{Array{Float64,3}}}(undef, n_groups)

    for (g, _) in enumerate(transition_groups)
        spec = trans_specs[g]
        φ_g = pack_robit_nn_v3(spec)
        K = spec.K; ν = spec.ν
        J_list[g] = Vector{Matrix{Float64}}(undef, r_Z_trans)
        Π_list[g] = Vector{Array{Float64,3}}(undef, r_Z_trans)

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

"""
Build CCP_init directly from a packed parameter vector (ForwardDiff-compatible).
Pack order matches `pack_ccp_v3_eps`: [c_reparam(r_I-1), β₁, β₃, β₄, β₅].  β₂ = 0.
"""
function _ccp_eps_init_from_vec(vec_φ::AbstractVector{Tv},
                                 kP_val::Real, kω_val::Real, kε_val::Real;
                                 θ_grid::AbstractVector, r_I::Int) where Tv
    nc = r_I - 1
    c  = _unpack_cutpoints(vec_φ[1:nc])
    β1 = vec_φ[nc + 1]
    β3 = vec_φ[nc + 2]
    β4 = vec_φ[nc + 3]
    β5 = vec_φ[nc + 4]
    r_θ = length(θ_grid)
    M = zeros(promote_type(Tv, Float64), r_I, r_θ)
    regime_part = β3 * kP_val + β4 * kω_val + β5 * kε_val
    for kθ in 1:r_θ
        η = β1 * θ_grid[kθ] + regime_part
        M[:, kθ] = _ordered_probit_pmf(c, η, r_I)
    end
    return M
end

"""
Build CCP_trans directly from a packed parameter vector (ForwardDiff-compatible).
β₂ (I_old) = 0.
"""
function _ccp_eps_trans_from_vec(vec_φ::AbstractVector{Tv},
                                  kP_val::Real, kω_val::Real, kε_val::Real;
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
    regime_part = β3 * kP_val + β4 * kω_val + β5 * kε_val
    for kθ1 in 1:r_θ, kI in 1:r_I
        η = β1 * θ_grid[kθ1] + regime_part   # β₂·I_old = 0
        CCP[:, kI, kθ1] = _ordered_probit_pmf(c, η, r_I)
    end
    return CCP
end

"""
    precompute_J_CCP_z_eps(ccp_spec, r_P, r_ω, r_Y, r_θ, r_I; grids...)
        -> (J_init, J_trans, CCP_init, CCP_trans)

For each full regime z = (kP, kω, kε), compute:
  J_init[z]  :: Matrix (r_I·r_θ, n_φ_ccp)     = ∂vec(CCP_init_z)/∂φ_ccp
  J_trans[z] :: Matrix (r_I·r_I·r_θ, n_φ_ccp) = ∂vec(CCP_trans_z)/∂φ_ccp
  CCP_init[z]  :: Matrix (r_I, r_θ)
  CCP_trans[z] :: Array{Float64,3} (r_I, r_I, r_θ)
"""
function precompute_J_CCP_z_eps(ccp_spec::OrderedProbitCCPV3Eps,
                                 r_P::Int, r_ω::Int, r_Y::Int,
                                 r_θ::Int, r_I::Int;
                                 θ_grid::AbstractVector=1.0:r_θ,
                                 I_grid::AbstractVector=1.0:r_I,
                                 P_grid::AbstractVector=1.0:r_P,
                                 ω_grid::AbstractVector=1.0:r_ω,
                                 Y_grid::AbstractVector=1.0:r_Y)
    r_Z = r_P * r_ω * r_Y
    φ_ccp = pack_ccp_v3_eps(ccp_spec)

    J_init    = Vector{Matrix{Float64}}(undef, r_Z)
    J_trans   = Vector{Matrix{Float64}}(undef, r_Z)
    CCP_init  = Vector{Matrix{Float64}}(undef, r_Z)
    CCP_trans = Vector{Array{Float64,3}}(undef, r_Z)

    for kε in 1:r_Y, kω in 1:r_ω, kP in 1:r_P
        z = eps_regime_index(kP, kω, kε, r_ω, r_Y)
        kP_val = P_grid[kP]; kω_val = ω_grid[kω]; kε_val = Y_grid[kε]

        CCP_init[z] = _ccp_eps_init_from_vec(φ_ccp, kP_val, kω_val, kε_val;
                                              θ_grid, r_I)
        CCP_trans[z] = _ccp_eps_trans_from_vec(φ_ccp, kP_val, kω_val, kε_val;
                                                θ_grid, I_grid, r_I)
        J_init[z] = ForwardDiff.jacobian(
            φ -> vec(_ccp_eps_init_from_vec(φ, kP_val, kω_val, kε_val;
                                              θ_grid, r_I)),
            φ_ccp)
        J_trans[z] = ForwardDiff.jacobian(
            φ -> vec(_ccp_eps_trans_from_vec(φ, kP_val, kω_val, kε_val;
                                               θ_grid, I_grid, r_I)),
            φ_ccp)
    end
    return J_init, J_trans, CCP_init, CCP_trans
end

# ─────────────────────────────────────────────────────────────────
# Per-child score for a single block — inlined for speed
# ─────────────────────────────────────────────────────────────────

# Each `_score_X_contrib!` mutates a pre-allocated score slice for child n.

"""
Accumulate child-n transition score into `s_trans_g` for group g.
    s_trans_g[:, n] += Σ_z w(z,n) · Σ_{t∈group} Σ_{k,θ,I} ξ_marg[k,θ,I]/Π_z[k,θ,I] · J_z[cell,:]

where ξ_marg[k,θ,I] = Σ_{I'} ξ_z[t, k, I', θ, I].
"""
function _score_trans_contrib!(
    s_trans_g::AbstractVector{Float64},     # length n_φ_g
    w_vec::Vector{Float64},                  # length r_Z (regime posterior for child n)
    ξ_z_all::Vector{Vector{Array{Float64,4}}},  # ξ_z_all[z][t]
    J_list_g::Vector{Matrix{Float64}},       # J_list_g[z_trans] :: (r_θ·r_θ·r_I × n_φ_g)
    Π_list_g::Vector{Array{Float64,3}},      # Π_list_g[z_trans]
    group::Vector{Int},
    r_P::Int, r_ω::Int, r_Y::Int,
    r_θ::Int, r_I::Int,
)
    cell_lin = LinearIndices((r_θ, r_θ, r_I))   # (k, θ, I) → linear vec idx
    nφ = length(s_trans_g)
    for kε in 1:r_Y, kω in 1:r_ω, kP in 1:r_P
        z_full  = eps_regime_index(kP, kω, kε, r_ω, r_Y)
        wz = w_vec[z_full]
        wz < 1e-15 && continue
        z_trans = regime_z(kP, kω, r_ω)
        Π_z = Π_list_g[z_trans]
        J_z = J_list_g[z_trans]

        for t in group
            ξ_t = ξ_z_all[z_full][t]
            for kI in 1:r_I, kθ in 1:r_θ, k in 1:r_θ
                @inbounds πv = Π_z[k, kθ, kI]
                πv < 1e-300 && continue
                # Marginalize over I' (fourth index of ξ stores I', third stores θ, second I,
                # first k=θ'). Pattern matches forward_backward_v3_n! storage:
                # ξ[kθ1, kI1, kθ, kI].
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
CCP score for child n.  Combines the t=1 (init) and t≥2 (trans) contributions.

Init contribution:
    s_ccp_init = Σ_z w(z,n) Σ_{k,θ,I} γ_z[1, k, I] / CCP_init_z[I, k] · J_init_z[(I, k), :]

Wait — in the complete-data LL the initial-CCP term is log CCP_init_z(I | θ, I_0=0, z).
Its score under the posterior is Σ_{θ, I} γ_z[1, θ, I] · ∂log CCP_init_z(I|θ)/∂φ.

Trans contribution:
    s_ccp_trans = Σ_z w(z,n) Σ_{t} Σ_{θ', I', I} [Σ_θ ξ_z[t, θ', I', θ, I]]
                                  · ∂log CCP_trans_z(I' | θ', I)/∂φ

`CCP_init_z` is a (r_I × r_θ) matrix (row = I, col = θ).
`CCP_trans_z` is a (r_I × r_I × r_θ) array (first = I', second = I_old, third = θ').
"""
function _score_ccp_contrib!(
    s_ccp::AbstractVector{Float64},         # length n_φ_ccp
    w_vec::Vector{Float64},
    γ_z_all::Vector{Array{Float64,3}},       # γ_z_all[z] :: (T, r_θ, r_I)
    ξ_z_all::Vector{Vector{Array{Float64,4}}},
    J_init_z::Vector{Matrix{Float64}},       # per full regime
    J_trans_z::Vector{Matrix{Float64}},
    CCP_init_z::Vector{Matrix{Float64}},
    CCP_trans_z::Vector{Array{Float64,3}},
    r_P::Int, r_ω::Int, r_Y::Int,
    r_θ::Int, r_I::Int, T::Int,
)
    init_cell_lin  = LinearIndices((r_I, r_θ))               # (I, θ)  for CCP_init
    trans_cell_lin = LinearIndices((r_I, r_I, r_θ))          # (I', I_old, θ')
    nφ = length(s_ccp)

    for kε in 1:r_Y, kω in 1:r_ω, kP in 1:r_P
        z = eps_regime_index(kP, kω, kε, r_ω, r_Y)
        wz = w_vec[z]
        wz < 1e-15 && continue

        γ_z = γ_z_all[z]
        C_init_z  = CCP_init_z[z]
        C_trans_z = CCP_trans_z[z]
        Ji = J_init_z[z]
        Jt = J_trans_z[z]

        # ── Initial CCP contribution (t = 1) ──
        for kI in 1:r_I, kθ in 1:r_θ
            @inbounds g = γ_z[1, kθ, kI]
            g < 1e-15 && continue
            @inbounds c = C_init_z[kI, kθ]
            c < 1e-300 && continue
            coef = wz * g / c
            cell = init_cell_lin[kI, kθ]
            @inbounds for p in 1:nφ
                s_ccp[p] += coef * Ji[cell, p]
            end
        end

        # ── Transition CCP contributions (t = 1..T-1 in ξ) ──
        for t in 1:T-1
            ξ_t = ξ_z_all[z][t]
            for kθ1 in 1:r_θ, kI in 1:r_I, kI1 in 1:r_I
                @inbounds c = C_trans_z[kI1, kI, kθ1]
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
    s_inc = Σ_z w(z,n) Σ_{t: y_{nt} observed} ∂log φ(y_{nt}; μ_z, σ)/∂(β₀, β_P, β_ω, δ, log_σ)

with μ_z = β₀ + β_P·kP + β_ω·kω + δ·kε.
"""
function _score_inc_contrib!(
    s_inc::AbstractVector{Float64},           # length 5
    w_vec::Vector{Float64},
    y_n::AbstractVector{Float64},              # length T — log income, NaN = missing
    income_model::IncomeModelEps,
    r_P::Int, r_ω::Int, r_Y::Int,
    P_grid::AbstractVector, ω_grid::AbstractVector, Y_grid::AbstractVector,
    T::Int,
)
    σ = exp(income_model.log_sigma)
    σ2 = σ * σ
    for kε in 1:r_Y, kω in 1:r_ω, kP in 1:r_P
        z = eps_regime_index(kP, kω, kε, r_ω, r_Y)
        wz = w_vec[z]
        wz < 1e-15 && continue
        kP_val = P_grid[kP]; kω_val = ω_grid[kω]; kε_val = Y_grid[kε]
        μ = income_model.beta0 + income_model.betaP * kP_val +
            income_model.betaω * kω_val + income_model.delta * kε_val
        for t in 1:T
            y = y_n[t]
            isnan(y) && continue
            r = y - μ                  # residual
            # ∂log N/∂β₀ = r/σ²
            # ∂log N/∂β_P = (r/σ²) · kP_val
            # ∂log N/∂β_ω = (r/σ²) · kω_val
            # ∂log N/∂δ  = (r/σ²) · kε_val
            # ∂log N/∂log σ = (r²/σ²) - 1
            inv_σ2 = r / σ2
            @inbounds s_inc[1] += wz * inv_σ2
            @inbounds s_inc[2] += wz * inv_σ2 * kP_val
            @inbounds s_inc[3] += wz * inv_σ2 * kω_val
            @inbounds s_inc[4] += wz * inv_σ2 * kε_val
            @inbounds s_inc[5] += wz * (r * r / σ2 - 1)
        end
    end
    return nothing
end

"""
π_Pω score for child n (softmax-reparameterized).
π_Pω(kP, kω) = softmax([0; α_Pω])[k], k = (kP-1)·r_ω + kω + 1 if (kP, kω) != (1, 1) else π_Pω(1,1) = softmax[1].

Complete-data log-likelihood term: log π_Pω(kP, kω) where (kP, kω) is the realized regime.
Score:
    s_{α_j} = ∂ log π_Pω / ∂α_j   evaluated at posterior weight w(z_trans, n) = Σ_{kε} w(z,n)
           = Σ_{(kP,kω)} w_Pω(kP, kω, n) · ∂ log π_Pω(kP, kω) / ∂α_j

Under the softmax reparameterization with α ∈ ℝ^(K-1) and reference bin = 1:
  ∂ π_Pω(k) / ∂α_j =  π_Pω(k) · (δ_{k, j+1} - π_Pω(j+1))
  ⇒ ∂ log π_Pω(k) / ∂α_j = δ_{k, j+1} - π_Pω(j+1)

(with K = r_P · r_ω).
"""
function _score_pi_Pω_contrib!(
    s_π_Pω::AbstractVector{Float64},          # length (r_P·r_ω - 1)
    w_vec::Vector{Float64},
    π_Pω_flat::Vector{Float64},                # length r_P·r_ω, reading order (kP, kω) -> (kP-1)*r_ω + kω
    r_P::Int, r_ω::Int, r_Y::Int,
)
    K_Pω = r_P * r_ω
    # Marginalize w over ε_Y: w_Pω[z_trans] = Σ_{kε} w_vec[(kP-1)*r_ω*r_Y + (kω-1)*r_Y + kε]
    w_Pω = zeros(K_Pω)
    for kε in 1:r_Y, kω in 1:r_ω, kP in 1:r_P
        z = eps_regime_index(kP, kω, kε, r_ω, r_Y)
        @inbounds w_Pω[(kP - 1) * r_ω + kω] += w_vec[z]
    end
    # ∂ log π_Pω(k) / ∂α_j = δ_{k, j+1} - π_Pω(j+1)
    # Score over all k weighted by w_Pω(k):
    #   Σ_k w_Pω(k) · (δ_{k, j+1} - π_Pω(j+1))
    #     = w_Pω(j+1) - (Σ_k w_Pω(k)) · π_Pω(j+1)
    # Since Σ_k w_Pω(k) = 1 for this child:
    #     = w_Pω(j+1) - π_Pω(j+1)
    for j in 1:(K_Pω - 1)
        idx = j + 1
        @inbounds s_π_Pω[j] += w_Pω[idx] - π_Pω_flat[idx]
    end
    return nothing
end

# ═════════════════════════════════════════════════════════════════
# NUISANCE BLOCKS (additive opt-in via include_nuisance_blocks=true)
#
#   §4.5 π_ε      — softmax over income type ε_Y (length r_Y-1)
#   §4.6 π0       — per-(kP,kω) softmax over θ_1  (length r_θ-1 each)
#   §4.7 Q_P      — ASVAB ordered-probit emissions
#   §4.8 probit_θ[t], probit_I[t] — child skill/investment emissions
#   §4.9 adult    — adult-outcome ordered-probit emissions (θ_T, ω)
#
# Each per-child score is the per-n term of −∂(block negll)/∂φ_block,
# weighted by the smoothed posteriors at the MLE.  ForwardDiff is used
# on the same small per-cell builders the M-step optimizes, so the
# Fisher information is in identical coordinates to the existing blocks.
# ═════════════════════════════════════════════════════════════════

# ── §4.5  π_ε  (softmax, reference bin = 1) ──────────────────────────
"""
π_ε score for child n.  Complete-data term: log π_ε(kε).
Posterior weight on kε is w_ε(kε,n) = Σ_{kP,kω} w(z,n).
Under softmax (ref bin 1):  ∂log π_ε(k)/∂α_j = δ_{k,j+1} − π_ε(j+1).
Summed over k weighted by w_ε (which sums to 1 per child):
    s_{α_j} = w_ε(j+1) − π_ε(j+1).
"""
function _score_pi_ε_contrib!(
    s_π_ε::AbstractVector{Float64},          # length r_Y - 1
    w_vec::Vector{Float64},
    π_ε_flat::Vector{Float64},                # length r_Y
    r_P::Int, r_ω::Int, r_Y::Int,
)
    w_ε = zeros(r_Y)
    for kε in 1:r_Y, kω in 1:r_ω, kP in 1:r_P
        z = eps_regime_index(kP, kω, kε, r_ω, r_Y)
        @inbounds w_ε[kε] += w_vec[z]
    end
    for j in 1:(r_Y - 1)
        @inbounds s_π_ε[j] += w_ε[j + 1] - π_ε_flat[j + 1]
    end
    return nothing
end

# ── §4.6  π0(θ_1 | kP, kω)  (per-cell softmax, reference bin = 1) ─────
"""
π0 score for child n.  Parameterized as the CONDITIONAL distribution
P(θ_1 | kP, kω); the marginal P(kP, kω) is the separate π_Pω block, so
the joint π0_ss simplex is reparameterized as (π_Pω) × (π0_cond) — a
bijection, no double counting.

Complete-data term for cell (kP,kω): Σ_{θ_1} γ̃_z[θ_1] · log π0(θ_1|kP,kω)
where γ̃ over (kP,kω) sums (within this child) over kε and θ_1, and the
posterior weight on θ_1 is  Σ_{kε,kI} w(z,n) γ_z[1, θ_1, kI].

Under per-cell softmax (ref θ_1 = 1):
    ∂log π0(s|·)/∂α_j = δ_{s,j+1} − π0(j+1|·)
    s_{α_j(kP,kω)} = Σ_s W0(s) (δ_{s,j+1} − π0(j+1))
                   = W0(j+1) − (Σ_s W0(s))·π0(j+1)
where W0(s) = Σ_{kε,kI} w(z,n) γ_z[1, s, kI].  (Σ_s W0 = total posterior
weight on this (kP,kω) cell for this child, NOT 1.)

`s_π0` is laid out cell-major: cell c = (kP-1)*r_ω + kω, each with
(r_θ-1) entries → offset (c-1)*(r_θ-1).
"""
function _score_pi0_contrib!(
    s_π0::AbstractVector{Float64},            # length r_P*r_ω*(r_θ-1)
    w_vec::Vector{Float64},
    γ_z_all::Vector{Array{Float64,3}},         # γ_z_all[z] :: (T, r_θ, r_I)
    π0_cond::Array{Float64,2},                 # (r_θ, r_P*r_ω)  conditional P(θ_1|cell)
    r_P::Int, r_ω::Int, r_Y::Int,
    r_θ::Int, r_I::Int,
)
    n_cell = r_P * r_ω
    nc = r_θ - 1
    W0 = zeros(r_θ, n_cell)
    for kε in 1:r_Y, kω in 1:r_ω, kP in 1:r_P
        z = eps_regime_index(kP, kω, kε, r_ω, r_Y)
        wz = w_vec[z]
        wz < 1e-15 && continue
        cell = (kP - 1) * r_ω + kω
        γ_z = γ_z_all[z]
        for kI in 1:r_I, s in 1:r_θ
            @inbounds W0[s, cell] += wz * γ_z[1, s, kI]
        end
    end
    for cell in 1:n_cell
        tot = 0.0
        for s in 1:r_θ; @inbounds tot += W0[s, cell]; end
        tot < 1e-15 && continue
        off = (cell - 1) * nc
        for j in 1:nc
            idx = j + 1
            @inbounds s_π0[off + j] += W0[idx, cell] - tot * π0_cond[idx, cell]
        end
    end
    return nothing
end

# ── §4.7  Q_P  (ASVAB ordered-probit emissions) ──────────────────────
"""
Q_P score for child n.  For each measure m (skip x==0):
    s = Σ_{kP} w_P(kP,n) · ∂log Q^m(x | kP)/∂φ_{Q^m}
with w_P(kP,n) = Σ_{kω,kε} w(z,n).  Packed per measure:
[cutpoint_reparam(κ_m-1), alpha].  Jacobians precomputed (constant).
`J_logQ[m][kP]` :: (κ_m × n_φ_m) = ∂ log Q^m(·|kP)/∂φ_m.
"""
function _score_Q_P_contrib!(
    s_Q::AbstractVector{Float64},             # length Σ_m n_φ_m
    w_vec::Vector{Float64},
    X_P::Vector{Vector{Int}}, n::Int,
    J_logQ::Vector{Vector{Matrix{Float64}}},  # J_logQ[m][kP]
    blk_off::Vector{Int}, blk_len::Vector{Int},
    r_P::Int, r_ω::Int, r_Y::Int,
)
    w_P = zeros(r_P)
    for kε in 1:r_Y, kω in 1:r_ω, kP in 1:r_P
        z = eps_regime_index(kP, kω, kε, r_ω, r_Y)
        @inbounds w_P[kP] += w_vec[z]
    end
    M = length(X_P)
    for m in 1:M
        x = X_P[m][n]
        x == 0 && continue
        off = blk_off[m]; L = blk_len[m]
        for kP in 1:r_P
            wp = w_P[kP]
            wp < 1e-15 && continue
            Jm = J_logQ[m][kP]      # (κ_m × L)
            @inbounds for p in 1:L
                s_Q[off + p] += wp * Jm[x, p]
            end
        end
    end
    return nothing
end

# ── §4.8  probit_θ[t] / probit_I[t]  (child emissions) ───────────────
"""
Child-emission score for child n, one period t, one latent axis.
W[k] = marginal posterior on the latent (Σ over the other latent) from
γ_agg[t,·,·,n].  For each measure m (skip x==0):
    s += Σ_k W[k] · ∂log P^m(x|k)/∂φ_m
`J_logP[m][k]` :: (κ_m × n_φ_m).
"""
function _score_emission_t_contrib!(
    s_blk::AbstractVector{Float64},           # length Σ_m n_φ_m  (this t)
    Wlat::Vector{Float64},                     # length r_lat (marginal posterior)
    X_t::Vector{Vector{Int}}, n::Int,
    J_logP::Vector{Vector{Matrix{Float64}}},   # J_logP[m][k]
    blk_off::Vector{Int}, blk_len::Vector{Int},
    r_lat::Int,
)
    M = length(X_t)
    for m in 1:M
        x = X_t[m][n]
        x == 0 && continue
        off = blk_off[m]; L = blk_len[m]
        for k in 1:r_lat
            wk = Wlat[k]
            wk < 1e-15 && continue
            Jm = J_logP[m][k]
            @inbounds for p in 1:L
                s_blk[off + p] += wk * Jm[x, p]
            end
        end
    end
    return nothing
end

# ── §4.9  adult emissions ────────────────────────────────────────────
"""
Adult-emission score for child n.  Depends on (θ_T, ω).

The E-step's adult log-likelihood enters log_ll_z[z] BEFORE the regime
posterior is formed, using the per-regime smoothed marginal
  θT_w_z(kθ) = Σ_kI γ_z[T, kθ, kI].
By the envelope theorem the observed-data score is therefore
  s = Σ_z w(z) Σ_m Σ_kθ θT_w_z(kθ) · ∂log A^m(x | kθ, ω(z))/∂φ_A.
Since A depends on ω only, we accumulate the per-regime weight
  Wθω(kθ,kω) = Σ_{z: ω(z)=kω} w(z)·θT_w_z(kθ)
(NOT the product w_ω(kω)·Σ_z w(z)θT_w_z(kθ), which would double-mix the
ω coupling).  The block is packed jointly (`pack_adult_emission`).
`J_logA[m][kθ,kω]` :: (κ_m × n_φ_adult) = ∂ log A^m(·|kθ,kω)/∂φ_adult.
"""
function _score_adult_contrib!(
    s_adult::AbstractVector{Float64},         # length n_φ_adult
    w_vec::Vector{Float64},
    γ_z_all::Vector{Array{Float64,3}},         # γ_z_all[z] :: (T, r_θ, r_I)
    X_adult::Vector{Vector{Int}}, n::Int,
    J_logA::Vector{Matrix{Matrix{Float64}}},   # J_logA[m][kθ,kω]
    r_P::Int, r_ω::Int, r_Y::Int, r_θ::Int, r_I::Int, T::Int,
)
    # Wθω[kθ, kω] = Σ_{z: ω=kω} w(z) Σ_kI γ_z[T, kθ, kI]
    Wθω = zeros(r_θ, r_ω)
    for kε in 1:r_Y, kω in 1:r_ω, kP in 1:r_P
        z = eps_regime_index(kP, kω, kε, r_ω, r_Y)
        wz = w_vec[z]
        wz < 1e-15 && continue
        γ_z = γ_z_all[z]
        @inbounds for kθ in 1:r_θ
            acc = 0.0
            for kI in 1:r_I; acc += γ_z[T, kθ, kI]; end
            Wθω[kθ, kω] += wz * acc
        end
    end
    nφ = length(s_adult)
    M = length(X_adult)
    for m in 1:M
        x = X_adult[m][n]
        x == 0 && continue
        for kω in 1:r_ω
            for kθ in 1:r_θ
                wk = Wθω[kθ, kω]
                wk < 1e-15 && continue
                Jm = J_logA[m][kθ, kω]
                @inbounds for p in 1:nφ
                    s_adult[p] += wk * Jm[x, p]
                end
            end
        end
    end
    return nothing
end

# ── Precompute Jacobians of log-emission matrices (constant in n) ────

"""
∂ log Q^m(·|kP) / ∂φ_m for an ordered-probit emission packed
[cutpoint_reparam(κ-1), alpha].  Returns J_logQ[m][kP] :: (κ_m × (κ_m)).
Also returns per-measure block offsets/lengths in the stacked vector.
"""
function _precompute_J_logemission(probit::OrderedProbitEmission,
                                     κ_list::Vector{Int}, r_lat::Int)
    M = length(κ_list)
    J = Vector{Vector{Matrix{Float64}}}(undef, M)
    blk_off = Vector{Int}(undef, M)
    blk_len = Vector{Int}(undef, M)
    off = 0
    for m in 1:M
        κ = κ_list[m]; nc = κ - 1
        L = nc + 1
        blk_off[m] = off; blk_len[m] = L; off += L
        φ_m = vcat(_pack_cutpoints(probit.tau[m]), probit.alpha[m])
        J[m] = Vector{Matrix{Float64}}(undef, r_lat)
        for k in 1:r_lat
            J[m][k] = ForwardDiff.jacobian(
                v -> begin
                    tau = _unpack_cutpoints(v[1:nc]); a = v[nc + 1]
                    P = ordered_probit_matrix(tau, a, r_lat)
                    log.(P[:, k])
                end, φ_m)
        end
    end
    return J, blk_off, blk_len, off
end

"""
∂ log A^m(·|kθ,kω) / ∂φ_adult for the jointly-packed adult emission.
Returns J_logA[m][kθ,kω] :: (κ_m × n_φ_adult) and n_φ_adult.
"""
function _precompute_J_logadult(adult_emission::AdultOutcomeEmission,
                                  κ_adult::Vector{Int}, r_θ::Int, r_ω::Int;
                                  θ_grid::AbstractVector=1.0:r_θ,
                                  ω_grid::AbstractVector=1.0:r_ω)
    φ = pack_adult_emission(adult_emission)
    nφ = length(φ)
    M = length(κ_adult)
    # per-measure offset of [cutpoints(κ_m-1), α_θ, α_ω] in the packed vector
    moff = zeros(Int, M); off = 0
    for m in 1:M; moff[m] = off; off += (κ_adult[m] - 1) + 2; end
    J = Vector{Matrix{Matrix{Float64}}}(undef, M)
    for m in 1:M
        nc = κ_adult[m] - 1; o = moff[m]
        Jm = Matrix{Matrix{Float64}}(undef, r_θ, r_ω)
        for kω in 1:r_ω, kθ in 1:r_θ
            Jm[kθ, kω] = ForwardDiff.jacobian(
                v -> begin
                    tau_m = _unpack_cutpoints(v[o+1 : o+nc])
                    a_θ   = v[o+nc+1]; a_ω = v[o+nc+2]
                    Am = adult_outcome_matrix(tau_m, a_θ, a_ω, r_θ, r_ω,
                                              κ_adult[m]; θ_grid, ω_grid)
                    log.(Am[:, kθ, kω])
                end, φ)
        end
        J[m] = Jm
    end
    return J, nφ
end

# ─────────────────────────────────────────────────────────────────
# Main entry point: per-child scores (priority blocks)
# ─────────────────────────────────────────────────────────────────

"""
    per_child_scores_v3_eps(params_v3_eps, params_skill, data;
                              transition_groups = [[1,2,3],[4,5,6]])
        -> NamedTuple (trans_g::Vector{Matrix}, ccp::Matrix, inc::Matrix,
                       π_Pω::Matrix, indices, n_groups, dims)

Computes per-child parametric scores for the priority blocks:
transition (per group), CCP, income, and π_Pω.  Remaining blocks
(π_ε, π0, Q_P, probit_θ[t], probit_I[t], adult) are stubbed (not
computed); their omission gives conditional SEs.

Per-child matrices have shape (n_φ_block × N).  `indices` provides
the cumulative index ranges when stacked.
"""
function per_child_scores_v3_eps(params_v3_eps, params_skill, data;
                                   transition_groups::Vector{Vector{Int}} = [[1,2,3],[4,5,6]],
                                   include_nuisance_blocks::Bool = false)
    (; r_P, r_ω, r_Y, π0, π_Pω, π_ε, ccp_spec, Q_P, income_model, trans_specs) = params_v3_eps
    (; r_θ, r_I, T) = params_skill
    N = data.N
    r_Z = r_P * r_ω * r_Y

    θ_grid = collect(1.0:r_θ); I_grid = collect(1.0:r_I)
    P_grid = collect(1.0:r_P); ω_grid = collect(1.0:r_ω)
    Y_grid = collect(1.0:r_Y)

    # Precompute per-regime Jacobians (constant across children)
    J_Π_z, Π_z_list = precompute_J_Pi_z_eps(trans_specs, transition_groups,
                                             r_θ, r_I, r_P, r_ω;
                                             θ_grid, I_grid, P_grid, ω_grid)
    J_CCP_init, J_CCP_trans, CCP_init, CCP_trans =
        precompute_J_CCP_z_eps(ccp_spec, r_P, r_ω, r_Y, r_θ, r_I;
                                 θ_grid, I_grid, P_grid, ω_grid, Y_grid)

    # Transitions: Pi_z_pre for forward-backward
    Pi_z_pre = precompute_Pi_z(trans_specs, transition_groups,
                                r_θ, r_I, r_P, r_ω;
                                θ_grid, I_grid, P_grid, ω_grid)

    # CCP arrays for forward-backward (use existing precomputer)
    CCP_init_pre_FB, CCP_trans_pre_FB = precompute_CCP_z_eps(
        ccp_spec, r_P, r_ω, r_Y, r_θ, r_I;
        θ_grid, I_grid, P_grid, ω_grid, Y_grid)

    # Priors
    π_Pω_flat = vec(collect(π_Pω'))   # reading: (kP, kω) -> (kP-1)*r_ω + kω
    #  (ensure we construct in the same order that _score_pi_Pω_contrib! expects)
    π_Pω_flat = Float64[π_Pω[kP, kω] for kP in 1:r_P for kω in 1:r_ω]

    log_π_Pω = [log(max(π_Pω[kP, kω], 1e-300)) for kP in 1:r_P, kω in 1:r_ω]
    log_π_ε  = [log(max(π_ε[kε], 1e-300))       for kε in 1:r_Y]

    # ── Nuisance-block reparameterizations / precomputed Jacobians ──
    π_ε_flat = collect(Float64.(π_ε))
    # π0 conditional P(θ_1 | kP, kω), laid out cell-major (kP-1)*r_ω + kω
    π0_cond = zeros(r_θ, r_P * r_ω)
    if include_nuisance_blocks
        for kω in 1:r_ω, kP in 1:r_P
            cell = (kP - 1) * r_ω + kω
            s = 0.0
            for kθ in 1:r_θ; s += max(π0[kθ, kP, kω], 0.0); end
            if s > 0
                for kθ in 1:r_θ; π0_cond[kθ, cell] = max(π0[kθ, kP, kω], 0.0) / s; end
            else
                fill!(@view(π0_cond[:, cell]), 1.0 / r_θ)
            end
        end
    end

    # Q_P emission Jacobians
    if include_nuisance_blocks
        J_logQ, Qoff, Qlen, n_φ_QP = _precompute_J_logemission(
            params_v3_eps.probit_P, collect(data.κ_P), r_P)
        # child skill / investment emission Jacobians per period
        J_logθ = Vector{Any}(undef, T); θoff = Vector{Vector{Int}}(undef, T)
        θlen = Vector{Vector{Int}}(undef, T); n_φ_θ = zeros(Int, T)
        J_logI = Vector{Any}(undef, T); Ioff = Vector{Vector{Int}}(undef, T)
        Ilen = Vector{Vector{Int}}(undef, T); n_φ_I = zeros(Int, T)
        for t in 1:T
            J_logθ[t], θoff[t], θlen[t], n_φ_θ[t] = _precompute_J_logemission(
                params_skill.probit_θ[t], collect(data.κ_θ[t]), r_θ)
            J_logI[t], Ioff[t], Ilen[t], n_φ_I[t] = _precompute_J_logemission(
                params_skill.probit_I[t], collect(data.κ_I[t]), r_I)
        end
        # adult emission Jacobians
        has_adult_blk = haskey(params_v3_eps, :adult_emission) &&
                        !isempty(params_v3_eps.adult_emission.tau) &&
                        haskey(data, :X_adult) && !isempty(data.X_adult)
        if has_adult_blk
            J_logA, n_φ_adult = _precompute_J_logadult(
                params_v3_eps.adult_emission, collect(data.κ_adult), r_θ, r_ω;
                θ_grid=θ_grid, ω_grid=ω_grid)
        else
            J_logA = Vector{Matrix{Matrix{Float64}}}(); n_φ_adult = 0
        end
    else
        n_φ_QP = 0; n_φ_θ = zeros(Int, T); n_φ_I = zeros(Int, T); n_φ_adult = 0
        has_adult_blk = false
    end

    # Score storage per block (n_φ_block × N)
    n_groups = length(transition_groups)
    n_φ_g = length(pack_robit_nn_v3(trans_specs[1]))
    n_φ_ccp = length(pack_ccp_v3_eps(ccp_spec))
    n_φ_inc = 5
    n_φ_π_Pω = r_P * r_ω - 1
    n_φ_π_ε  = r_Y - 1
    n_φ_π0   = r_P * r_ω * (r_θ - 1)

    s_trans_g = [zeros(n_φ_g, N) for _ in 1:n_groups]
    s_ccp     = zeros(n_φ_ccp, N)
    s_inc     = zeros(n_φ_inc, N)
    s_π_Pω    = zeros(n_φ_π_Pω, N)
    # Nuisance block score storage (empty unless include_nuisance_blocks)
    s_π_ε  = include_nuisance_blocks ? zeros(n_φ_π_ε, N)  : zeros(0, N)
    s_π0   = include_nuisance_blocks ? zeros(n_φ_π0, N)   : zeros(0, N)
    s_Q_P  = include_nuisance_blocks ? zeros(n_φ_QP, N)   : zeros(0, N)
    s_θ    = include_nuisance_blocks ? [zeros(n_φ_θ[t], N) for t in 1:T] : Matrix{Float64}[]
    s_I    = include_nuisance_blocks ? [zeros(n_φ_I[t], N) for t in 1:T] : Matrix{Float64}[]
    s_adult = (include_nuisance_blocks && has_adult_blk) ? zeros(n_φ_adult, N) : zeros(0, N)

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
    γ_agg_n   = zeros(T, r_θ, r_I)   # per-child aggregated marginals Σ_z w(z,n) γ_z

    # Build γ_z_init_full: P(θ_1 | z) for every full regime (duplicates across kε)
    #                   — used to seed forward-backward.
    # Actually, π0 doesn't depend on kε, so we reuse the (kP, kω) slice across ε_Y.

    for n in 1:N
        # Emission likelihoods (same across regimes)
        for t in 1:T
            compute_emission_likelihood_t!(L_scratch[t], t, n, params_skill, data)
        end

        # Pass 1: forward-backward per regime
        for kε in 1:r_Y, kω in 1:r_ω, kP in 1:r_P
            z = eps_regime_index(kP, kω, kε, r_ω, r_Y)
            z_trans = regime_z(kP, kω, r_ω)

            ll_regime = regime_loglik_eps(n, kP, kω, kε, Q_P, income_model, data;
                                           P_grid, ω_grid, Y_grid)
            log_ll_z[z] = log_π_Pω[kP, kω] + log_π_ε[kε] + ll_regime

            # π0(θ_1 | kP, kω) — independent of kε
            for kθ in 1:r_θ; @inbounds π0_z_buf[kθ] = max(π0[kθ, kP, kω], 0.0); end
            s = sum(π0_z_buf)
            if s > 0; π0_z_buf ./= s; else; fill!(π0_z_buf, 1.0/r_θ); end

            ccp_init_arr = CCP_init_pre_FB[z]
            for kI in 1:r_I, kθ in 1:r_θ
                @inbounds π0_nz_buf[kθ, kI] = π0_z_buf[kθ] * ccp_init_arr[kI, kθ]
            end

            for t in 1:T-1
                CCP_nz[t] = CCP_trans_pre_FB[z]
            end

            for t in 1:T-1; fill!(ξ_z_all[z][t], 0.0); end

            ll_skills = forward_backward_v3_n!(
                α, β, scale, γ_z_all[z], ξ_z_all[z],
                π0_nz_buf, Pi_z_pre[z_trans], CCP_nz, L_scratch, r_θ, r_I, T)
            log_ll_z[z] += ll_skills

            # Adult outcome contribution to regime log-likelihood (for correct w_z)
            if haskey(params_v3_eps, :adult_matrices) &&
               !isempty(params_v3_eps.adult_matrices) &&
               haskey(data, :X_adult) && !isempty(data.X_adult)
                adult_matrices = params_v3_eps.adult_matrices
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

        # Aggregated per-child marginals (needed only for nuisance emission blocks)
        if include_nuisance_blocks
            fill!(γ_agg_n, 0.0)
            for kε in 1:r_Y, kω in 1:r_ω, kP in 1:r_P
                z = eps_regime_index(kP, kω, kε, r_ω, r_Y)
                wz = w_vec[z]
                wz < 1e-15 && continue
                γ_z = γ_z_all[z]
                @inbounds for kI in 1:r_I, kθ in 1:r_θ, t in 1:T
                    γ_agg_n[t, kθ, kI] += wz * γ_z[t, kθ, kI]
                end
            end
        end

        # ── Transition scores ──
        for (g, group) in enumerate(transition_groups)
            s_view = @view s_trans_g[g][:, n]
            _score_trans_contrib!(s_view, w_vec, ξ_z_all,
                                   J_Π_z[g], Π_z_list[g], group,
                                   r_P, r_ω, r_Y, r_θ, r_I)
        end

        # ── CCP score ──
        s_view_ccp = @view s_ccp[:, n]
        _score_ccp_contrib!(s_view_ccp, w_vec, γ_z_all, ξ_z_all,
                             J_CCP_init, J_CCP_trans, CCP_init, CCP_trans,
                             r_P, r_ω, r_Y, r_θ, r_I, T)

        # ── Income score ──
        s_view_inc = @view s_inc[:, n]
        y_n = @view data.Y_income[n, :]
        _score_inc_contrib!(s_view_inc, w_vec, y_n, income_model,
                             r_P, r_ω, r_Y, P_grid, ω_grid, Y_grid, T)

        # ── π_Pω score ──
        s_view_π = @view s_π_Pω[:, n]
        _score_pi_Pω_contrib!(s_view_π, w_vec, π_Pω_flat, r_P, r_ω, r_Y)

        # ── Nuisance blocks (opt-in) ──
        if include_nuisance_blocks
            # §4.5 π_ε
            _score_pi_ε_contrib!(@view(s_π_ε[:, n]), w_vec, π_ε_flat, r_P, r_ω, r_Y)
            # §4.6 π0 (conditional)
            _score_pi0_contrib!(@view(s_π0[:, n]), w_vec, γ_z_all, π0_cond,
                                r_P, r_ω, r_Y, r_θ, r_I)
            # §4.7 Q_P
            _score_Q_P_contrib!(@view(s_Q_P[:, n]), w_vec, data.X_P, n,
                                J_logQ, Qoff, Qlen, r_P, r_ω, r_Y)
            # §4.8 probit_θ[t], probit_I[t]  (use γ_agg marginals built here)
            for t in 1:T
                Wθ = zeros(r_θ); WI = zeros(r_I)
                @inbounds for kI in 1:r_I, kθ in 1:r_θ
                    g = γ_agg_n[t, kθ, kI]
                    Wθ[kθ] += g; WI[kI] += g
                end
                _score_emission_t_contrib!(@view(s_θ[t][:, n]), Wθ, data.X_θ[t], n,
                                           J_logθ[t], θoff[t], θlen[t], r_θ)
                _score_emission_t_contrib!(@view(s_I[t][:, n]), WI, data.X_I[t], n,
                                           J_logI[t], Ioff[t], Ilen[t], r_I)
            end
            # §4.9 adult  (per-regime ω coupling — matches E-step objective)
            if has_adult_blk
                _score_adult_contrib!(@view(s_adult[:, n]), w_vec, γ_z_all,
                                      data.X_adult, n, J_logA,
                                      r_P, r_ω, r_Y, r_θ, r_I, T)
            end
        end
    end

    return (
        trans_g  = s_trans_g,
        ccp      = s_ccp,
        inc      = s_inc,
        π_Pω     = s_π_Pω,
        # nuisance blocks (empty matrices if not requested)
        π_ε      = s_π_ε,
        π0       = s_π0,
        Q_P      = s_Q_P,
        probit_θ = s_θ,        # Vector{Matrix} (per period) or empty Vector
        probit_I = s_I,
        adult    = s_adult,
        include_nuisance_blocks = include_nuisance_blocks,
        has_adult_blk = has_adult_blk,
        n_groups = n_groups,
        dims     = (; r_θ, r_I, r_P, r_ω, r_Y, T, N,
                     n_φ_g, n_φ_ccp, n_φ_inc, n_φ_π_Pω,
                     n_φ_π_ε, n_φ_π0, n_φ_QP,
                     n_φ_θ = collect(n_φ_θ), n_φ_I = collect(n_φ_I),
                     n_φ_adult),
    )
end

# ─────────────────────────────────────────────────────────────────
# Score stacking
# ─────────────────────────────────────────────────────────────────

"""
    stack_scores_v3_eps(block_scores)
        -> (S::Matrix{Float64}, ranges::NamedTuple)

Stack per-block scores into a single (n_φ_total × N) matrix.
`ranges.trans_g[g]`, `ranges.ccp`, `ranges.inc`, `ranges.π_Pω` give the
row index ranges for each block.
"""
function stack_scores_v3_eps(block_scores)
    dims = block_scores.dims
    n_groups = block_scores.n_groups
    N = dims.N

    n_φ_g = dims.n_φ_g
    n_φ_ccp = dims.n_φ_ccp
    n_φ_inc = dims.n_φ_inc
    n_φ_π_Pω = dims.n_φ_π_Pω

    inc_nb = get(block_scores, :include_nuisance_blocks, false)
    has_adult_blk = get(block_scores, :has_adult_blk, false)

    # Leading (always-present) block sizes — positions match the 4-block path
    n_lead = n_groups * n_φ_g + n_φ_ccp + n_φ_inc + n_φ_π_Pω

    # Trailing nuisance block sizes
    n_nb = 0
    if inc_nb
        n_nb += dims.n_φ_π_ε + dims.n_φ_π0 + dims.n_φ_QP +
                sum(dims.n_φ_θ) + sum(dims.n_φ_I)
        has_adult_blk && (n_nb += dims.n_φ_adult)
    end
    n_total = n_lead + n_nb

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

    π_Pω_range = idx:(idx + n_φ_π_Pω - 1)
    S[π_Pω_range, :] = block_scores.π_Pω
    idx += n_φ_π_Pω

    # ── Trailing nuisance blocks (only when requested) ──
    ranges = (; trans_g = trans_ranges, ccp = ccp_range,
                inc = inc_range, π_Pω = π_Pω_range)
    if inc_nb
        π_ε_range = idx:(idx + dims.n_φ_π_ε - 1)
        S[π_ε_range, :] = block_scores.π_ε; idx += dims.n_φ_π_ε

        π0_range = idx:(idx + dims.n_φ_π0 - 1)
        S[π0_range, :] = block_scores.π0; idx += dims.n_φ_π0

        QP_range = idx:(idx + dims.n_φ_QP - 1)
        S[QP_range, :] = block_scores.Q_P; idx += dims.n_φ_QP

        θ_ranges = Vector{UnitRange{Int}}(undef, dims.T)
        for t in 1:dims.T
            θ_ranges[t] = idx:(idx + dims.n_φ_θ[t] - 1)
            S[θ_ranges[t], :] = block_scores.probit_θ[t]; idx += dims.n_φ_θ[t]
        end
        I_ranges = Vector{UnitRange{Int}}(undef, dims.T)
        for t in 1:dims.T
            I_ranges[t] = idx:(idx + dims.n_φ_I[t] - 1)
            S[I_ranges[t], :] = block_scores.probit_I[t]; idx += dims.n_φ_I[t]
        end
        if has_adult_blk
            adult_range = idx:(idx + dims.n_φ_adult - 1)
            S[adult_range, :] = block_scores.adult; idx += dims.n_φ_adult
        else
            adult_range = idx:(idx - 1)
        end
        ranges = merge(ranges, (; π_ε = π_ε_range, π0 = π0_range, Q_P = QP_range,
                                  probit_θ = θ_ranges, probit_I = I_ranges,
                                  adult = adult_range))
    end

    return S, ranges
end

# ─────────────────────────────────────────────────────────────────
# Top-level Var(φ̂) with stacked scores
# ─────────────────────────────────────────────────────────────────

"""
    compute_var_phi_v3_eps(params_v3_eps, params_skill, data;
                            transition_groups=[[1,2,3],[4,5,6]],
                            active_tol=1e-6)
        -> (scores, S, ranges, Σ_s, Var_φ, active, inactive)

Run the full per-child score computation, stack, compute OPG Fisher,
and invert to get Var(φ̂).

Identifies inactive parameters via near-zero score column norms and
drops them before inversion.  Var_φ is returned as an n_φ_total ×
n_φ_total matrix with zeros in the inactive rows/columns.

Returned fields:
  scores   — NamedTuple of per-block score matrices (n_φ_block × N)
  S        — stacked score matrix (n_φ_total × N)
  ranges   — NamedTuple of index ranges
  Σ_s      — OPG Fisher estimate (n_φ_total × n_φ_total)
  Var_φ    — Σ_s⁻¹ / N on active block; zeros elsewhere
  active   — Vector{Int} of active parameter indices
  inactive — Vector{Int} of dropped parameter indices
  cond_num — condition number of Σ_s_active
  score_check — Σ_n s_n / N (should be ≈ 0 at MLE)
"""
function compute_var_phi_v3_eps(params_v3_eps, params_skill, data;
                                 transition_groups::Vector{Vector{Int}}=[[1,2,3],[4,5,6]],
                                 active_tol::Float64=1e-6,
                                 include_nuisance_blocks::Bool=false,
                                 verbose::Bool=false)
    scores = per_child_scores_v3_eps(params_v3_eps, params_skill, data;
                                       transition_groups, include_nuisance_blocks)
    S, ranges = stack_scores_v3_eps(scores)
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
    # Drop singular values below rank_tol · max(σ).
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
    delta_se_β5_over_δ(inf_result, params_v3_eps) -> (point, se, Var_block)

β₅ (CCP loading on ε_Y) lives at position (r_I - 1) + 4 inside the CCP
block; δ (income loading on ε_Y) lives at position 4 inside the income
block.  Extract the 2x2 submatrix of Var(φ̂) for (β₅, δ) and apply the
delta method.
"""
function delta_se_β5_over_δ(inf_result, params_v3_eps, params_skill)
    ccp = params_v3_eps.ccp_spec
    inc = params_v3_eps.income_model
    r_I = params_skill.r_I
    β5_local = (r_I - 1) + 4
    δ_local  = 4        # 4th income param: (β₀, β_P, β_ω, δ, log_σ)

    β5_global = inf_result.ranges.ccp[β5_local]
    δ_global  = inf_result.ranges.inc[δ_local]

    V_full = inf_result.Var_φ
    V_block = V_full[[β5_global, δ_global], [β5_global, δ_global]]

    β5_hat = ccp.beta5
    δ_hat  = inc.delta
    J = [1 / δ_hat, -β5_hat / δ_hat^2]
    var_ratio = J' * V_block * J
    se = sqrt(max(var_ratio, 0.0))
    return (
        point   = β5_hat / δ_hat,
        se      = se,
        V_block = V_block,
    )
end

"""
    delta_cov_survivor_bar(inf_result, params_v3_eps, params_skill;
                            transition_groups=[[1,2,3],[4,5,6]],
                            k_cut::Int = params_skill.r_θ - 1)
        -> Vector{NamedTuple} (one entry per group)

Full covariance of the regime-marginalized survivor-function free
entries, computed via the delta method.  Uses transition params and
π_Pω from the joint Var(φ̂).

Per-group:
  Π̄_g = Σ_{kP,kω} π_Pω(kP, kω) · Π_{kP, kω}(φ_g_trans)
  S̄_free[θ, I, k] = Σ_{j>k} Π̄_g[j, θ, I]

For each group we compute the Jacobian w.r.t. (φ_g_trans, α_π_Pω) and
apply delta_cov(J, V_block).
"""
function delta_cov_survivor_bar(inf_result, params_v3_eps, params_skill;
                                 transition_groups::Vector{Vector{Int}}=[[1,2,3],[4,5,6]])
    (; r_P, r_ω, r_Y, π_Pω, trans_specs) = params_v3_eps
    (; r_θ, r_I) = params_skill
    P_grid = collect(1.0:r_P); ω_grid = collect(1.0:r_ω)
    θ_grid = collect(1.0:r_θ); I_grid = collect(1.0:r_I)

    n_cut = r_θ - 1
    n_free = n_cut * r_θ * r_I

    # Softmax reparameterization of π_Pω
    K_Pω = r_P * r_ω
    π_Pω_flat = Float64[π_Pω[kP, kω] for kP in 1:r_P for kω in 1:r_ω]
    α_Pω_hat = probs_to_softmax(π_Pω_flat; ref=1)

    results = []

    for (g, _) in enumerate(transition_groups)
        spec_g = trans_specs[g]
        K = spec_g.K; ν = spec_g.ν
        φ_g_hat = pack_robit_nn_v3(spec_g)
        n_φ_g = length(φ_g_hat)
        n_α = length(α_Pω_hat)

        # Combined parameter vector: [φ_g ; α_Pω]
        θ_combined = vcat(φ_g_hat, α_Pω_hat)

        # Map combined → survivor-free vector
        function survivor_free_map(θ_vec)
            φ_g = θ_vec[1:n_φ_g]
            α   = θ_vec[(n_φ_g + 1):end]
            π   = softmax_to_probs(α)   # length K_Pω
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

        # Var(φ_g, α_Pω) — extract from inf_result.Var_φ
        trans_range = inf_result.ranges.trans_g[g]
        π_Pω_range  = inf_result.ranges.π_Pω
        combined_idx = vcat(collect(trans_range), collect(π_Pω_range))
        V_combined = inf_result.Var_φ[combined_idx, combined_idx]

        Var_S = delta_cov(J_full, V_combined)
        ses   = sqrt.(max.(diag(Var_S), 0.0))

        # Compute the survivor estimate at the MLE
        S_hat = survivor_free_map(θ_combined)

        push!(results, (; group = g, S_hat, Var_S, ses, n_cut, n_free))
    end

    return results
end

"""
    delta_cov_Pi_bar_free(inf_result, params_v3_eps, params_skill;
                           transition_groups=[[1,2,3],[4,5,6]])
        -> (Var_free_list, Pi_bar)

Variance of the free entries of Π̄ (first r_θ-1 rows, vectorized).
Returns:
  Var_free_list[t] — indexed by t_rep for each group (compatible with get_test_stat_v3_grouped)
  Pi_bar[t]        — Π̄ arrays
"""
function delta_cov_Pi_bar_free(inf_result, params_v3_eps, params_skill;
                                transition_groups::Vector{Vector{Int}}=[[1,2,3],[4,5,6]])
    (; r_P, r_ω, r_Y, π_Pω, trans_specs) = params_v3_eps
    (; r_θ, r_I, T) = params_skill
    P_grid = collect(1.0:r_P); ω_grid = collect(1.0:r_ω)
    θ_grid = collect(1.0:r_θ); I_grid = collect(1.0:r_I)

    n_free = (r_θ - 1) * r_θ * r_I
    K_Pω = r_P * r_ω
    π_Pω_flat = Float64[π_Pω[kP, kω] for kP in 1:r_P for kω in 1:r_ω]
    α_Pω_hat = probs_to_softmax(π_Pω_flat; ref=1)

    Var_free_list = Vector{Matrix{Float64}}(undef, T - 1)
    Pi_bar = Vector{Array{Float64,3}}(undef, T - 1)

    for (g, group) in enumerate(transition_groups)
        spec_g = trans_specs[g]
        K = spec_g.K; ν = spec_g.ν
        φ_g_hat = pack_robit_nn_v3(spec_g)
        n_φ_g = length(φ_g_hat)
        θ_combined = vcat(φ_g_hat, α_Pω_hat)

        function pi_bar_free_map(θ_vec)
            φ_g = θ_vec[1:n_φ_g]
            α   = θ_vec[(n_φ_g + 1):end]
            π   = softmax_to_probs(α)
            TE  = eltype(θ_vec)
            Π_bar = zeros(TE, r_θ, r_θ, r_I)
            for kω in 1:r_ω, kP in 1:r_P
                w = π[(kP - 1) * r_ω + kω]
                Π_z = _build_robit_nn_v3_pi(φ_g, r_θ, r_I, K, ν,
                                              P_grid[kP], ω_grid[kω];
                                              θ_grid, I_grid)
                Π_bar .+= w .* Π_z
            end
            v = Vector{TE}(undef, n_free)
            idx = 0
            for kI in 1:r_I, kθ in 1:r_θ, j in 1:(r_θ - 1)
                idx += 1
                v[idx] = Π_bar[j, kθ, kI]
            end
            return v
        end

        J_full = ForwardDiff.jacobian(pi_bar_free_map, θ_combined)
        trans_range = inf_result.ranges.trans_g[g]
        π_Pω_range  = inf_result.ranges.π_Pω
        combined_idx = vcat(collect(trans_range), collect(π_Pω_range))
        V_combined = inf_result.Var_φ[combined_idx, combined_idx]
        Var_free = delta_cov(J_full, V_combined)

        Π_bar_float = Float64.(pi_bar_free_map(θ_combined))
        Π_bar_full = zeros(r_θ, r_θ, r_I)
        for kω in 1:r_ω, kP in 1:r_P
            w = π_Pω_flat[(kP - 1) * r_ω + kω]
            Π_z = _build_robit_nn_v3_pi(φ_g_hat, r_θ, r_I, K, ν,
                                          P_grid[kP], ω_grid[kω]; θ_grid, I_grid)
            Π_bar_full .+= w .* Π_z
        end

        for t in group
            Var_free_list[t] = Var_free
            Pi_bar[t] = Π_bar_full
        end
    end

    return Var_free_list, Pi_bar
end
