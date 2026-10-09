# ═══════════════════════════════════════════════════════════════
# V3 + ε_Y parametric models
#
#   Adds a child-level discrete "income type" latent ε_Y ∈ {1,...,r_Y}.
#   Regime is now z = (θ^P, ω, ε_Y), r_Z = r_P · r_ω · r_Y.
#   ε_Y enters:
#     (a) the income equation:  log y_nt = β₀ + β_P·θ^P + β_ω·ω + δ·ε_Y + σ·ν
#     (b) the CCP index:        η = β₁·θ' + β₃·θ^P + β₄·ω + β₅·ε_Y
#   ε_Y does NOT enter skill transitions (see feedback memory).
#
#   Depends on helpers in parametric_models.jl and parametric_models_v3.jl:
#     _unpack_cutpoints, _pack_cutpoints, _ordered_probit_pmf,
#     _ordered_probit_logprob, Φ, RobitNNTransitionV3, AdultOutcomeEmission.
# ═══════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────
# Regime indexing helpers
#   z = (kP-1)*r_ω*r_Y + (kω-1)*r_Y + kε
# ─────────────────────────────────────────────────────────────────

@inline function eps_regime_index(kP::Int, kω::Int, kε::Int, r_ω::Int, r_Y::Int)
    (kP - 1) * r_ω * r_Y + (kω - 1) * r_Y + kε
end

@inline function eps_regime_decode(z::Int, r_ω::Int, r_Y::Int)
    z0 = z - 1
    kε = mod(z0, r_Y) + 1
    kω = div(mod(z0, r_ω * r_Y), r_Y) + 1
    kP = div(z0, r_ω * r_Y) + 1
    return kP, kω, kε
end

# ─────────────────────────────────────────────────────────────────
# OrderedProbitCCPV3Eps
#
#   P(I' ≤ k | θ', I_old, θ^P, ω, ε_Y) =
#       Φ(c_k − β₁θ' − β₂ I_old − β₃θ^P − β₄ω − β₅ ε_Y)
#
#   β₂ (I_old) fixed at zero (per V3 convention).
#   At t=1, I_old = 0.
#   Pack: [c_reparam(r_I-1), β₁, β₃, β₄, β₅]  →  (r_I-1) + 4 params.
# ─────────────────────────────────────────────────────────────────

struct OrderedProbitCCPV3Eps
    c::Vector{Float64}   # r_I - 1 increasing cutpoints
    beta1::Float64        # θ'
    beta2::Float64        # I_old   (fixed = 0)
    beta3::Float64        # θ^P
    beta4::Float64        # ω
    beta5::Float64        # ε_Y
end

function pack_ccp_v3_eps(s::OrderedProbitCCPV3Eps)
    vcat(_pack_cutpoints(s.c), s.beta1, s.beta3, s.beta4, s.beta5)
end

function unpack_ccp_v3_eps(v::AbstractVector, r_I::Int)
    nc = r_I - 1
    c  = _unpack_cutpoints(v[1:nc])
    OrderedProbitCCPV3Eps(c, v[nc+1], 0.0, v[nc+2], v[nc+3], v[nc+4])
end

"""
    build_ccp_v3_eps_init(spec, kP_val, kω_val, kε_val;
                          θ_grid, r_I) -> Matrix (r_I, r_θ)

Build CCP_init[kI, kθ] = P(I_1 = kI | θ_1 = kθ, I_0 = 0, regime=(P,ω,ε_Y)).
"""
function build_ccp_v3_eps_init(spec::OrderedProbitCCPV3Eps,
                                kP_val::Real, kω_val::Real, kε_val::Real;
                                θ_grid::AbstractVector, r_I::Int)
    r_θ = length(θ_grid)
    M = zeros(r_I, r_θ)
    regime_part = spec.beta3 * kP_val + spec.beta4 * kω_val + spec.beta5 * kε_val
    for kθ in 1:r_θ
        η = spec.beta1 * θ_grid[kθ] + regime_part
        M[:, kθ] = _ordered_probit_pmf(spec.c, η, r_I)
    end
    return M
end

"""
    build_ccp_v3_eps_trans(spec, kP_val, kω_val, kε_val;
                           θ_grid, I_grid, r_I) -> Array (r_I, r_I, r_θ)

Build CCP_trans[I', I_old, θ'] for transitions t ≥ 2.  β₂·I_old = 0 per V3.
"""
function build_ccp_v3_eps_trans(spec::OrderedProbitCCPV3Eps,
                                 kP_val::Real, kω_val::Real, kε_val::Real;
                                 θ_grid::AbstractVector,
                                 I_grid::AbstractVector, r_I::Int)
    r_θ = length(θ_grid)
    CCP = zeros(r_I, r_I, r_θ)
    regime_part = spec.beta3 * kP_val + spec.beta4 * kω_val + spec.beta5 * kε_val
    for kθ1 in 1:r_θ, kI in 1:r_I
        η = spec.beta1 * θ_grid[kθ1] + spec.beta2 * I_grid[kI] + regime_part
        CCP[:, kI, kθ1] = _ordered_probit_pmf(spec.c, η, r_I)
    end
    return CCP
end

"""
    ccp_negll_v3_eps(vec, init_ss, trans_ss, r_θ, r_I, r_P, r_ω, r_Y, T;
                      grids...)

Negative expected complete-data log-likelihood.
The SS are already aggregated per regime z = (kP, kω, kε):
  init_ss   — (r_θ, r_I, r_Z)           weighted γ at t = 1
  trans_ss  — (r_I, r_I, r_θ, r_Z, T-1) weighted ξ per transition

No income bins: ε_Y is discrete and enters through the regime.
Pack matches `pack_ccp_v3_eps`: [c_reparam(r_I-1), β₁, β₃, β₄, β₅].
"""
function ccp_negll_v3_eps(vec::AbstractVector{Tv},
                           init_ss::Array{Float64,3},     # (r_θ, r_I, r_Z)
                           trans_ss::Array{Float64,5},    # (r_I, r_I, r_θ, r_Z, T-1)
                           r_θ::Int, r_I::Int,
                           r_P::Int, r_ω::Int, r_Y::Int, Tp::Int;
                           θ_grid::AbstractVector=1.0:r_θ,
                           I_grid::AbstractVector=1.0:r_I,
                           P_grid::AbstractVector=1.0:r_P,
                           ω_grid::AbstractVector=1.0:r_ω,
                           Y_grid::AbstractVector=1.0:r_Y) where Tv
    nc = r_I - 1
    c  = _unpack_cutpoints(vec[1:nc])
    β1, β3, β4, β5 = vec[nc+1], vec[nc+2], vec[nc+3], vec[nc+4]
    negll = zero(Tv)
    for kε in 1:r_Y, kω in 1:r_ω, kP in 1:r_P
        z = eps_regime_index(kP, kω, kε, r_ω, r_Y)
        rp = β3 * P_grid[kP] + β4 * ω_grid[kω] + β5 * Y_grid[kε]

        # ── Initial CCP ──
        for kθ in 1:r_θ
            η = β1 * θ_grid[kθ] + rp
            for kI in 1:r_I
                @inbounds g = init_ss[kθ, kI, z]
                g < 1e-15 && continue
                negll -= g * _ordered_probit_logprob(c, η, kI, nc)
            end
        end

        # ── Transition CCPs (β₂·I_old = 0) ──
        for t in 1:Tp-1
            for kθ1 in 1:r_θ
                η = β1 * θ_grid[kθ1] + rp  # β₂ · I_grid[kI] drops out (β₂ = 0)
                for kI in 1:r_I, kI1 in 1:r_I
                    @inbounds ξv = trans_ss[kI1, kI, kθ1, z, t]
                    ξv < 1e-15 && continue
                    negll -= ξv * _ordered_probit_logprob(c, η, kI1, nc)
                end
            end
        end
    end
    return negll
end

"""
    initialize_ccp_v3_eps(r_I; rng, beta5_init=0.3) -> OrderedProbitCCPV3Eps

Initial β₅ > 0 anchors ε_Y labeling (higher ε_Y ⇒ higher P(high I)).
"""
function initialize_ccp_v3_eps(r_I::Int; rng::AbstractRNG=Random.default_rng(),
                                beta5_init::Float64 = 0.3)
    nc = r_I - 1
    c = collect(range(-1.0, 1.0; length=nc)) .+ 0.02 .* randn(rng, nc)
    sort!(c)
    OrderedProbitCCPV3Eps(c,
        0.1 + 0.02 * randn(rng),   # β₁ (θ')
        0.0,                         # β₂ (I_old) — fixed
        0.02 * randn(rng),          # β₃ (θ^P)
        0.02 * randn(rng),          # β₄ (ω)
        beta5_init)                 # β₅ (ε_Y) — positive anchor
end

# ─────────────────────────────────────────────────────────────────
# IncomeModelEps
#
#   log y_nt = β₀ + β_P·θ^P + β_ω·ω + δ·ε_Y + σ·ν_nt,   ν ∼ N(0,1)
#
#   M-step is closed-form WLS with 4 regressors.
# ─────────────────────────────────────────────────────────────────

struct IncomeModelEps
    beta0::Float64
    betaP::Float64
    betaω::Float64
    delta::Float64           # coefficient on ε_Y
    log_sigma::Float64
end

function income_loglik_eps(y::Float64, kP_val::Real, kω_val::Real, kε_val::Real,
                            m::IncomeModelEps)
    isnan(y) && return 0.0
    μ = m.beta0 + m.betaP * kP_val + m.betaω * kω_val + m.delta * kε_val
    σ = exp(m.log_sigma)
    return -0.5 * ((y - μ) / σ)^2 - m.log_sigma - 0.9189385332046727  # log √(2π)
end

"""
    update_income_model_eps(w_z, Y_income, r_P, r_ω, r_Y; grids...) -> IncomeModelEps

Closed-form WLS with weights from the regime posterior w_z[z, n].
"""
function update_income_model_eps(w_z::Matrix{Float64},
                                  Y_income::Matrix{Float64},
                                  r_P::Int, r_ω::Int, r_Y::Int;
                                  P_grid::AbstractVector=1.0:r_P,
                                  ω_grid::AbstractVector=1.0:r_ω,
                                  Y_grid::AbstractVector=1.0:r_Y)
    N, Tp = size(Y_income)
    XtWX = zeros(4, 4)
    XtWy = zeros(4)

    for n in 1:N
        for kε in 1:r_Y, kω in 1:r_ω, kP in 1:r_P
            z = eps_regime_index(kP, kω, kε, r_ω, r_Y)
            wz = w_z[z, n]
            wz < 1e-15 && continue
            x1 = P_grid[kP]; x2 = ω_grid[kω]; x3 = Y_grid[kε]
            for t in 1:Tp
                y = Y_income[n, t]
                isnan(y) && continue
                # Symmetric build of normal equations: x = [1, x1, x2, x3]
                xv = (1.0, x1, x2, x3)
                for i in 1:4, j in 1:4
                    XtWX[i, j] += wz * xv[i] * xv[j]
                end
                for i in 1:4
                    XtWy[i] += wz * xv[i] * y
                end
            end
        end
    end

    λ = 1e-8 * max(XtWX[1,1], 1.0)
    for i in 1:4; XtWX[i,i] += λ; end
    β = XtWX \ XtWy

    # Residual variance
    ssr = 0.0; w_total = 0.0
    for n in 1:N
        for kε in 1:r_Y, kω in 1:r_ω, kP in 1:r_P
            z = eps_regime_index(kP, kω, kε, r_ω, r_Y)
            wz = w_z[z, n]
            wz < 1e-15 && continue
            μ_z = β[1] + β[2]*P_grid[kP] + β[3]*ω_grid[kω] + β[4]*Y_grid[kε]
            for t in 1:Tp
                y = Y_income[n, t]
                isnan(y) && continue
                ssr += wz * (y - μ_z)^2
                w_total += wz
            end
        end
    end
    σ² = w_total > 0 ? ssr / w_total : 1.0

    return IncomeModelEps(β[1], β[2], β[3], β[4], 0.5 * log(max(σ², 1e-10)))
end

"""
    initialize_income_model_eps(Y_income; delta_init=0.15) -> IncomeModelEps

β₀ = sample mean of log y (observed).  σ = sample sd.  δ > 0 anchors ε_Y.
"""
function initialize_income_model_eps(Y_income::Matrix{Float64};
                                      delta_init::Float64 = 0.15)
    obs = filter(!isnan, vec(Y_income))
    if isempty(obs)
        return IncomeModelEps(0.0, 0.0, 0.0, delta_init, 0.0)
    end
    m = sum(obs) / length(obs)
    s = sqrt(max(sum((obs .- m).^2) / length(obs), 1e-6))
    # Center β₀ so μ at ε_Y = (r_Y+1)/2 approximates the sample mean
    # β₀ + δ·((r_Y+1)/2) = m   ⇒   β₀ = m − δ·mid
    return IncomeModelEps(m, 0.0, 0.0, delta_init, log(s))
end
