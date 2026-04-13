# ═══════════════════════════════════════════════════════════════
# V3 parametric models: regime-extended structs for (θ^P, ω)
#
# Depends on helpers from parametric_models.jl:
#   Φ, Φ_t, _unpack_survivor_cutoffs, _pack_survivor_cutoffs,
#   _unpack_cutpoints, _pack_cutpoints, _ordered_probit_pmf,
#   _transition_negll_precomputed, DEFAULT_ROBIT_DF, DEFAULT_OPTIM_OPTIONS
# ═══════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────
# RobitNNTransitionV3
# ─────────────────────────────────────────────────────────────────

"""
    RobitNNTransitionV3

Robit NN transition with four inputs: (θ, I, θ^P, ω).

    h_k = α₁ₖ·θ + α₂ₖ·I + α₃ₖ·θ^P + α₄ₖ·ω + bₖ
    F̄(j|θ,I,θ^P,ω) = F_ν(c_j + Σ_k wₖ · Φ(hₖ))

Pack layout: [c_raw(n_cut), alpha1(K), alpha2_raw(K), alpha3(K), alpha4(K), b(K), log_w(K)]
Total: n_cut + 6K parameters.
"""
struct RobitNNTransitionV3
    c_raw::Vector{Float64}       # n_cut survivor cutpoint reparameterisation
    alpha1::Vector{Float64}      # K: θ loadings
    alpha2_raw::Vector{Float64}  # K: I loadings (exp for positivity)
    alpha3::Vector{Float64}      # K: θ^P loadings
    alpha4::Vector{Float64}      # K: ω loadings
    b::Vector{Float64}           # K: biases
    log_w::Vector{Float64}       # K: log mixture weights
    K::Int
    ν::Float64
end

function pack_robit_nn_v3(s::RobitNNTransitionV3)
    vcat(s.c_raw, s.alpha1, s.alpha2_raw, s.alpha3, s.alpha4, s.b, s.log_w)
end

function unpack_robit_nn_v3(v::AbstractVector, r_θ::Int, K::Int; ν::Float64=DEFAULT_ROBIT_DF)
    nc = r_θ - 1; i = 0
    c_raw  = v[i+1:i+nc]; i += nc
    a1     = v[i+1:i+K];  i += K
    a2_raw = v[i+1:i+K];  i += K
    a3     = v[i+1:i+K];  i += K
    a4     = v[i+1:i+K];  i += K
    b      = v[i+1:i+K];  i += K
    lw     = v[i+1:i+K]
    RobitNNTransitionV3(collect(c_raw), collect(a1), collect(a2_raw),
                        collect(a3), collect(a4), collect(b), collect(lw), K, ν)
end

"""
    _build_robit_nn_v3_pi(vec, r_θ, r_I, K, ν, kP_val, kω_val; θ_grid, I_grid)
        -> Array{T,3}  (r_θ, r_θ, r_I)

Build Π[θ',θ,I] for a single regime where θ^P = kP_val and ω = kω_val.
ForwardDiff-compatible.
"""
function _build_robit_nn_v3_pi(vec::AbstractVector{T}, r_θ::Int, r_I::Int,
                                K::Int, ν::Float64, kP_val::Real, kω_val::Real;
                                θ_grid::AbstractVector=1.0:r_θ,
                                I_grid::AbstractVector=1.0:r_I) where T
    n_cut = r_θ - 1; i = 0
    c      = _unpack_survivor_cutoffs(vec[i+1:i+n_cut]); i += n_cut
    alpha1 = vec[i+1:i+K];       i += K
    alpha2 = exp.(vec[i+1:i+K]); i += K   # enforce > 0
    alpha3 = vec[i+1:i+K];       i += K
    alpha4 = vec[i+1:i+K];       i += K
    b      = vec[i+1:i+K];       i += K
    w      = exp.(vec[i+1:i+K])            # enforce > 0

    Π = zeros(T, r_θ, r_θ, r_I)
    for kI in 1:r_I, kθ in 1:r_θ
        mu = zero(T)
        for h in 1:K
            mu += w[h] * Φ(alpha1[h]*θ_grid[kθ] + alpha2[h]*I_grid[kI] +
                           alpha3[h]*kP_val      + alpha4[h]*kω_val + b[h])
        end
        S_bar = Vector{T}(undef, n_cut)
        for k in 1:n_cut
            S_bar[k] = Φ_t(c[k] + mu, ν)
        end
        pmf = Vector{T}(undef, r_θ)
        pmf[1] = one(T) - S_bar[1]
        for k in 2:n_cut
            pmf[k] = S_bar[k-1] - S_bar[k]
        end
        pmf[r_θ] = S_bar[n_cut]
        pmf .= max.(pmf, T(1e-12))
        pmf ./= sum(pmf)
        Π[:, kθ, kI] = pmf
    end
    return Π
end

"""
    transition_negll_robit_nn_v3(vec, Q_sum_z, r_θ, r_I, K, ν, r_P, r_ω; grids...)

Negative complete-data log-likelihood for V3 robit NN transition, summed over
all regimes z.  `Q_sum_z[z]` is (r_θ, r_θ, r_I) for z = (kP-1)*r_ω + kω.
Reuses `_transition_negll_precomputed` from parametric_models.jl.
ForwardDiff-compatible.
"""
function transition_negll_robit_nn_v3(vec::AbstractVector{T},
                                       Q_sum_z::Vector{Array{Float64,3}},
                                       r_θ::Int, r_I::Int, K::Int, ν::Float64,
                                       r_P::Int, r_ω::Int;
                                       θ_grid::AbstractVector=1.0:r_θ,
                                       I_grid::AbstractVector=1.0:r_I,
                                       P_grid::AbstractVector=1.0:r_P,
                                       ω_grid::AbstractVector=1.0:r_ω) where T
    negll = zero(T)
    for kω in 1:r_ω, kP in 1:r_P
        z = (kP - 1) * r_ω + kω
        Π_z = _build_robit_nn_v3_pi(vec, r_θ, r_I, K, ν,
                                      P_grid[kP], ω_grid[kω]; θ_grid, I_grid)
        negll += _transition_negll_precomputed(Π_z, Q_sum_z[z])
    end
    return negll
end

"""
    initialize_robit_nn_v3(r_θ, K; rng, ν) -> RobitNNTransitionV3

Same spread as V2 initialiser; alpha3, alpha4 start near zero.
"""
function initialize_robit_nn_v3(r_θ::Int, K::Int;
                                  rng::AbstractRNG=Random.default_rng(),
                                  ν::Float64=DEFAULT_ROBIT_DF)
    n_cut = r_θ - 1
    c = collect(range(0.5, -0.5; length=n_cut)) .+ 0.02 .* randn(rng, n_cut)
    sort!(c; rev=true)
    c_raw      = _pack_survivor_cutoffs(c)
    alpha1     = collect(range(-0.5, 0.5; length=K)) .+ 0.1 .* randn(rng, K)
    alpha2_raw = collect(range(-1.5, -0.5; length=K)) .+ 0.1 .* randn(rng, K)
    alpha3     = 0.05 .* randn(rng, K)
    alpha4     = 0.05 .* randn(rng, K)
    b          = collect(range(-1.0, 1.0; length=K)) .+ 0.1 .* randn(rng, K)
    log_w      = fill(-1.0, K) .+ 0.1 .* randn(rng, K)
    RobitNNTransitionV3(c_raw, alpha1, alpha2_raw, alpha3, alpha4, b, log_w, K, ν)
end

# ─────────────────────────────────────────────────────────────────
# OrderedProbitCCPV3
# ─────────────────────────────────────────────────────────────────

"""
    OrderedProbitCCPV3

Ordered probit CCP with five inputs:

    P(I' ≤ k | θ', I_old, θ^P, ω, log y) = Φ(cₖ − β₁θ' − β₂ I_old − β₃θ^P − β₄ω − β₅ log y)

At t=1: I_old = 0.  Missing income (NaN): β₅ term dropped.
One spec shared across all periods; the only period-varying input is log y.

Pack: [c_reparam(r_I-1), β₁, β₂, β₃, β₄, β₅]  =  (r_I-1) + 5 params.
"""
struct OrderedProbitCCPV3
    c::Vector{Float64}   # r_I-1 increasing cutpoints
    beta1::Float64        # θ'
    beta2::Float64        # I_old
    beta3::Float64        # θ^P
    beta4::Float64        # ω
    beta5::Float64        # log y
end

function pack_ccp_v3(s::OrderedProbitCCPV3)
    vcat(_pack_cutpoints(s.c), s.beta1, s.beta2, s.beta3, s.beta4, s.beta5)
end

function unpack_ccp_v3(v::AbstractVector, r_I::Int)
    nc = r_I - 1
    c = _unpack_cutpoints(v[1:nc])
    OrderedProbitCCPV3(c, v[nc+1], v[nc+2], v[nc+3], v[nc+4], v[nc+5])
end

"""
    build_ccp_v3_init(spec, kP_val, kω_val, y_1; θ_grid, r_I) -> Matrix{Float64} (r_I, r_θ)

Build CCP_init[kI, kθ] = P(I_1=kI | θ_1=kθ, I_0=0, regime, y_1).
"""
function build_ccp_v3_init(spec::OrderedProbitCCPV3, kP_val::Real, kω_val::Real,
                            y_1::Float64; θ_grid::AbstractVector, r_I::Int)
    r_θ = length(θ_grid)
    M = zeros(r_I, r_θ)
    regime_part = spec.beta3 * kP_val + spec.beta4 * kω_val
    inc = isnan(y_1) ? 0.0 : spec.beta5 * y_1
    for kθ in 1:r_θ
        η = spec.beta1 * θ_grid[kθ] + regime_part + inc
        M[:, kθ] = _ordered_probit_pmf(spec.c, η, r_I)
    end
    return M
end

"""
    build_ccp_v3_trans(spec, kP_val, kω_val, y_tp1; θ_grid, I_grid, r_I) -> Array{Float64,3} (r_I, r_I, r_θ)

Build CCP[I', I_old, θ'] for one transition where income = y_{t+1}.
"""
function build_ccp_v3_trans(spec::OrderedProbitCCPV3, kP_val::Real, kω_val::Real,
                             y_tp1::Float64; θ_grid::AbstractVector,
                             I_grid::AbstractVector, r_I::Int)
    r_θ = length(θ_grid)
    CCP = zeros(r_I, r_I, r_θ)
    regime_part = spec.beta3 * kP_val + spec.beta4 * kω_val
    inc = isnan(y_tp1) ? 0.0 : spec.beta5 * y_tp1
    for kθ1 in 1:r_θ, kI in 1:r_I
        η = spec.beta1 * θ_grid[kθ1] + spec.beta2 * I_grid[kI] + regime_part + inc
        CCP[:, kI, kθ1] = _ordered_probit_pmf(spec.c, η, r_I)
    end
    return CCP
end

"""
    ccp_negll_v3(vec, ccp_xi_z, gamma_z_t1, w_z, Y_income, r_θ, r_I, r_P, r_ω, T; grids...)

Negative expected complete-data log-likelihood for the V3 CCP.
Iterates over all (child, regime, period) triples because income is continuous.
ForwardDiff-compatible (9 parameters for r_I=5).
"""
function ccp_negll_v3(vec::AbstractVector{Tv},
                       ccp_xi_z::Array{Float64,6},    # (r_I, r_I, r_θ, r_Z, T-1, N)
                       gamma_z_t1::Array{Float64,4},   # (r_θ, r_I, r_Z, N)
                       w_z::Matrix{Float64},            # (r_Z, N)
                       Y_income::Matrix{Float64},       # (N, T)
                       r_θ::Int, r_I::Int, r_P::Int, r_ω::Int, Tp::Int;
                       θ_grid::AbstractVector=1.0:r_θ,
                       I_grid::AbstractVector=1.0:r_I,
                       P_grid::AbstractVector=1.0:r_P,
                       ω_grid::AbstractVector=1.0:r_ω) where Tv
    nc = r_I - 1
    c  = _unpack_cutpoints(vec[1:nc])
    β1, β2, β3, β4, β5 = vec[nc+1], vec[nc+2], vec[nc+3], vec[nc+4], vec[nc+5]

    N = size(Y_income, 1)
    negll = zero(Tv)

    for n in 1:N
        for kω in 1:r_ω, kP in 1:r_P
            z = (kP - 1) * r_ω + kω
            w = w_z[z, n]
            w < 1e-15 && continue

            rp = β3 * P_grid[kP] + β4 * ω_grid[kω]  # regime part

            # ── Initial CCP (I_1 | θ_1, I_0=0, z, y_1) ──
            y1 = Y_income[n, 1]
            ip1 = isnan(y1) ? zero(Tv) : β5 * y1
            for kθ in 1:r_θ
                η = β1 * θ_grid[kθ] + rp + ip1
                pmf = _ordered_probit_pmf(c, η, r_I)
                for kI in 1:r_I
                    @inbounds g = gamma_z_t1[kθ, kI, z, n]
                    g < 1e-15 && continue
                    negll -= w * g * log(max(pmf[kI], Tv(1e-15)))
                end
            end

            # ── Transition CCPs (I_{t+1} | θ_{t+1}, I_t, z, y_{t+1}) ──
            for t in 1:Tp-1
                y_tp1 = Y_income[n, t+1]
                ip = isnan(y_tp1) ? zero(Tv) : β5 * y_tp1
                for kθ1 in 1:r_θ
                    base_η = β1 * θ_grid[kθ1] + rp + ip
                    for kI in 1:r_I
                        η = base_η + β2 * I_grid[kI]
                        pmf = _ordered_probit_pmf(c, η, r_I)
                        for kI1 in 1:r_I
                            @inbounds ξv = ccp_xi_z[kI1, kI, kθ1, z, t, n]
                            ξv < 1e-15 && continue
                            negll -= w * ξv * log(max(pmf[kI1], Tv(1e-15)))
                        end
                    end
                end
            end
        end
    end
    return negll
end

"""
    initialize_ccp_v3(r_I; rng) -> OrderedProbitCCPV3
"""
function initialize_ccp_v3(r_I::Int; rng::AbstractRNG=Random.default_rng())
    nc = r_I - 1
    c = collect(range(-1.0, 1.0; length=nc)) .+ 0.02 .* randn(rng, nc)
    sort!(c)
    OrderedProbitCCPV3(c,
        0.1 + 0.02 * randn(rng),   # β₁ (θ')
        0.1 + 0.02 * randn(rng),   # β₂ (I_old)
        0.02 * randn(rng),          # β₃ (θ^P)
        0.02 * randn(rng),          # β₄ (ω)
        0.0)                         # β₅ (log y) — start at zero
end

# ─────────────────────────────────────────────────────────────────
# IncomeModel
# ─────────────────────────────────────────────────────────────────

"""
    IncomeModel

    log(y_nt) = β₀ + β_P·θ^P + β_ω·ω + εₙₜ,   εₙₜ ∼ N(0, σ²)

Missing income (NaN) contributes log-likelihood = 0.
M-step is closed-form weighted OLS.
"""
struct IncomeModel
    beta0::Float64
    betaP::Float64
    betaω::Float64
    log_sigma::Float64     # log(σ), so σ = exp(log_sigma) > 0
end

function income_loglik(y::Float64, kP_val::Real, kω_val::Real, m::IncomeModel)
    isnan(y) && return 0.0
    μ = m.beta0 + m.betaP * kP_val + m.betaω * kω_val
    σ = exp(m.log_sigma)
    return -0.5 * ((y - μ) / σ)^2 - m.log_sigma - 0.9189385332046727  # 0.5*log(2π)
end

"""
    update_income_model(w_z, Y_income, r_P, r_ω; P_grid, ω_grid) -> IncomeModel

Closed-form WLS.  Weights from regime posteriors w_z[z, n].
"""
function update_income_model(w_z::Matrix{Float64}, Y_income::Matrix{Float64},
                               r_P::Int, r_ω::Int;
                               P_grid::AbstractVector=1.0:r_P,
                               ω_grid::AbstractVector=1.0:r_ω)
    N, Tp = size(Y_income)
    XtWX = zeros(3, 3)
    XtWy = zeros(3)

    for n in 1:N, kω in 1:r_ω, kP in 1:r_P
        z = (kP - 1) * r_ω + kω
        wz = w_z[z, n]
        wz < 1e-15 && continue
        x1 = P_grid[kP]; x2 = ω_grid[kω]
        for t in 1:Tp
            y = Y_income[n, t]
            isnan(y) && continue
            # Accumulate X'WX and X'Wy for design row [1, x1, x2]
            XtWX[1,1] += wz;       XtWX[1,2] += wz*x1;       XtWX[1,3] += wz*x2
            XtWX[2,1] += wz*x1;    XtWX[2,2] += wz*x1*x1;    XtWX[2,3] += wz*x1*x2
            XtWX[3,1] += wz*x2;    XtWX[3,2] += wz*x1*x2;    XtWX[3,3] += wz*x2*x2
            XtWy[1] += wz*y;       XtWy[2] += wz*x1*y;        XtWy[3] += wz*x2*y
        end
    end

    # Regularise and solve
    λ = 1e-8 * max(XtWX[1,1], 1.0)
    for i in 1:3; XtWX[i,i] += λ; end
    β = XtWX \ XtWy

    # Residual variance
    ssr = 0.0; w_total = 0.0
    for n in 1:N, kω in 1:r_ω, kP in 1:r_P
        z = (kP - 1) * r_ω + kω
        wz = w_z[z, n]
        wz < 1e-15 && continue
        μ_z = β[1] + β[2]*P_grid[kP] + β[3]*ω_grid[kω]
        for t in 1:Tp
            y = Y_income[n, t]
            isnan(y) && continue
            ssr += wz * (y - μ_z)^2
            w_total += wz
        end
    end
    σ² = w_total > 0 ? ssr / w_total : 1.0

    return IncomeModel(β[1], β[2], β[3], 0.5 * log(max(σ², 1e-10)))
end

"""
    initialize_income_model(Y_income) -> IncomeModel

Initialise from sample moments of observed log-incomes.
"""
function initialize_income_model(Y_income::Matrix{Float64})
    obs = filter(!isnan, vec(Y_income))
    if isempty(obs)
        return IncomeModel(0.0, 0.0, 0.0, 0.0)
    end
    m = sum(obs) / length(obs)
    s = sqrt(max(sum((obs .- m).^2) / length(obs), 1e-6))
    return IncomeModel(m, 0.0, 0.0, log(s))
end
