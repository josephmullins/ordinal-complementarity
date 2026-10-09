# ═══════════════════════════════════════════════════════════════
# V3 parametric models: regime-extended structs for (θ^P, ω)
#
# Depends on helpers from parametric_models.jl:
#   Φ, Φ_t, _unpack_survivor_cutoffs, _pack_survivor_cutoffs,
#   _unpack_cutpoints, _pack_cutpoints, _ordered_probit_pmf,
#   _transition_negll_precomputed, DEFAULT_ROBIT_DF, DEFAULT_OPTIM_OPTIONS
# ═══════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────
# OrderedProbitTransitionV3 — simple parametric baseline for warm-start
# ─────────────────────────────────────────────────────────────────

"""
    OrderedProbitTransitionV3

    P(θ'≤k | θ,I,θ^P,ω) = Φ(c_k - β₁θ - β₂I - β₃θI - β₄θ^P - β₅ω)

Pack: [c_reparam(r_θ-1), β₁, β₂, β₃, β₄, β₅] = (r_θ-1)+5 params.
"""
struct OrderedProbitTransitionV3
    c::Vector{Float64}   # r_θ-1 increasing cutpoints
    beta1::Float64        # θ
    beta2::Float64        # I
    beta3::Float64        # θ·I
    beta4::Float64        # θ^P
    beta5::Float64        # ω
end

function pack_op_trans_v3(s::OrderedProbitTransitionV3)
    vcat(_pack_cutpoints(s.c), s.beta1, s.beta2, s.beta3, s.beta4, s.beta5)
end

function unpack_op_trans_v3(v::AbstractVector, r_θ::Int)
    nc = r_θ - 1
    c = _unpack_cutpoints(v[1:nc])
    OrderedProbitTransitionV3(c, v[nc+1], v[nc+2], v[nc+3], v[nc+4], v[nc+5])
end

"""
Build Π[θ',θ,I] for a single regime (kP_val, kω_val). ForwardDiff-compatible.
"""
function _build_op_trans_v3_pi(vec::AbstractVector{T}, r_θ::Int, r_I::Int,
                                kP_val::Real, kω_val::Real;
                                θ_grid::AbstractVector=1.0:r_θ,
                                I_grid::AbstractVector=1.0:r_I) where T
    nc = r_θ - 1
    c  = _unpack_cutpoints(vec[1:nc])
    β1, β2, β3, β4, β5 = vec[nc+1], vec[nc+2], vec[nc+3], vec[nc+4], vec[nc+5]

    regime_part = β4 * kP_val + β5 * kω_val
    Π = zeros(T, r_θ, r_θ, r_I)
    for kI in 1:r_I, kθ in 1:r_θ
        η = β1 * θ_grid[kθ] + β2 * I_grid[kI] + β3 * θ_grid[kθ] * I_grid[kI] + regime_part
        Π[:, kθ, kI] = _ordered_probit_pmf(c, η, r_θ)
    end
    return Π
end

"""
Negative log-likelihood summed over all regimes. ForwardDiff-compatible.
"""
function transition_negll_op_v3(vec::AbstractVector{T},
                                 Q_sum_z::Vector{Array{Float64,3}},
                                 r_θ::Int, r_I::Int,
                                 r_P::Int, r_ω::Int;
                                 θ_grid::AbstractVector=1.0:r_θ,
                                 I_grid::AbstractVector=1.0:r_I,
                                 P_grid::AbstractVector=1.0:r_P,
                                 ω_grid::AbstractVector=1.0:r_ω) where T
    negll = zero(T)
    for kω in 1:r_ω, kP in 1:r_P
        z = (kP - 1) * r_ω + kω
        Π_z = _build_op_trans_v3_pi(vec, r_θ, r_I, P_grid[kP], ω_grid[kω];
                                      θ_grid, I_grid)
        negll += _transition_negll_precomputed(Π_z, Q_sum_z[z])
    end
    return negll
end

function initialize_op_trans_v3(r_θ::Int; rng::AbstractRNG=Random.default_rng())
    nc = r_θ - 1
    c = collect(range(-1.0, 1.0; length=nc)) .+ 0.02 .* randn(rng, nc)
    sort!(c)
    OrderedProbitTransitionV3(c,
        0.1 + 0.02 * randn(rng),   # β₁ (θ)
        0.1 + 0.02 * randn(rng),   # β₂ (I)
        0.0,                         # β₃ (θI)
        0.02 * randn(rng),          # β₄ (θ^P)
        0.02 * randn(rng))          # β₅ (ω)
end

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
    # β₂ (I_old) is fixed at zero — not optimized
    vcat(_pack_cutpoints(s.c), s.beta1, s.beta3, s.beta4, s.beta5)
end

function unpack_ccp_v3(v::AbstractVector, r_I::Int)
    nc = r_I - 1
    c = _unpack_cutpoints(v[1:nc])
    # β₂ fixed at zero
    OrderedProbitCCPV3(c, v[nc+1], 0.0, v[nc+2], v[nc+3], v[nc+4])
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
    _ordered_probit_logprob(c, η, k, nc)

Log-probability of category `k` from an ordered probit with cutpoints `c`
and index `η`.  `nc = length(c) = r_I - 1`.  No allocation.
ForwardDiff-compatible.
"""
@inline function _ordered_probit_logprob(c, η, k::Int, nc::Int)
    if k == 1
        p = Φ(c[1] - η)
    elseif k > nc
        p = one(typeof(η)) - Φ(c[nc] - η)
    else
        p = Φ(c[k] - η) - Φ(c[k-1] - η)
    end
    return log(max(p, typeof(η)(1e-12)))
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
                for kI in 1:r_I
                    @inbounds g = gamma_z_t1[kθ, kI, z, n]
                    g < 1e-15 && continue
                    negll -= w * g * _ordered_probit_logprob(c, η, kI, nc)
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
                        for kI1 in 1:r_I
                            @inbounds ξv = ccp_xi_z[kI1, kI, kθ1, z, t, n]
                            ξv < 1e-15 && continue
                            negll -= w * ξv * _ordered_probit_logprob(c, η, kI1, nc)
                        end
                    end
                end
            end
        end
    end
    return negll
end

# ─────────────────────────────────────────────────────────────────
# Binned CCP objective: pre-aggregate sufficient statistics by
# discretized income to eliminate the N loop from the optimizer.
# ─────────────────────────────────────────────────────────────────

"""
    bin_income(Y_income; n_bins=20) -> (bin_map, bin_values)

Discretize income into quantile-based bins.  Missing (NaN) gets its own
bin (index n_bins+1) with value 0.0 (income term dropped).

Returns:
  bin_map   — N×T Int matrix of bin indices
  bin_values — length n_bins+1 vector of bin midpoints (mean within bin)
"""
function bin_income(Y_income::Matrix{Float64}; n_bins::Int=20)
    N, Tp = size(Y_income)
    obs = filter(!isnan, vec(Y_income))

    # Handle all-missing income: everyone goes to the missing bin
    if isempty(obs)
        bin_map = fill(1, N, Tp)  # single bin (the missing bin)
        bin_values = [0.0]         # missing bin value
        return bin_map, bin_values
    end

    # Quantile bin edges
    edges = quantile(obs, range(0, 1; length=n_bins+1))
    # Ensure distinct edges
    for i in 2:length(edges)
        edges[i] = max(edges[i], edges[i-1] + 1e-10)
    end

    bin_map = zeros(Int, N, Tp)
    bin_sums = zeros(n_bins)
    bin_counts = zeros(Int, n_bins)

    for t in 1:Tp, n in 1:N
        y = Y_income[n, t]
        if isnan(y)
            bin_map[n, t] = n_bins + 1  # missing bin
        else
            # Binary search for bin
            b = searchsortedlast(edges, y)
            b = clamp(b, 1, n_bins)
            bin_map[n, t] = b
            bin_sums[b] += y
            bin_counts[b] += 1
        end
    end

    bin_values = zeros(n_bins + 1)
    for b in 1:n_bins
        bin_values[b] = bin_counts[b] > 0 ? bin_sums[b] / bin_counts[b] : edges[b]
    end
    bin_values[n_bins + 1] = 0.0  # missing: income term will be zero

    return bin_map, bin_values
end

"""
    aggregate_ccp_ss(ccp_xi_z, gamma_z_t1, w_z, bin_map, n_total_bins, r_θ, r_I, r_Z, T)
        -> (init_ss, trans_ss)

Pre-aggregate CCP sufficient statistics by income bin.

Returns:
  init_ss  — (r_θ, r_I, r_Z, n_total_bins)  weighted gamma at t=1
  trans_ss — (r_I, r_I, r_θ, r_Z, T-1, n_total_bins)  weighted xi
"""
function aggregate_ccp_ss(ccp_xi_z::Array{Float64,6},
                           gamma_z_t1::Array{Float64,4},
                           w_z::Matrix{Float64},
                           bin_map::Matrix{Int},
                           n_total_bins::Int,
                           r_θ::Int, r_I::Int, r_Z::Int, T::Int)
    N = size(w_z, 2)

    init_ss  = zeros(r_θ, r_I, r_Z, n_total_bins)
    trans_ss = zeros(r_I, r_I, r_θ, r_Z, T-1, n_total_bins)

    for n in 1:N
        # Initial period
        b1 = bin_map[n, 1]
        for z in 1:r_Z
            @inbounds wz = w_z[z, n]
            wz < 1e-15 && continue
            for kI in 1:r_I, kθ in 1:r_θ
                @inbounds init_ss[kθ, kI, z, b1] += wz * gamma_z_t1[kθ, kI, z, n]
            end
        end

        # Transition periods
        for t in 1:T-1
            b = bin_map[n, t+1]
            for z in 1:r_Z
                @inbounds wz = w_z[z, n]
                wz < 1e-15 && continue
                for kθ1 in 1:r_θ, kI in 1:r_I, kI1 in 1:r_I
                    @inbounds trans_ss[kI1, kI, kθ1, z, t, b] +=
                        wz * ccp_xi_z[kI1, kI, kθ1, z, t, n]
                end
            end
        end
    end

    return init_ss, trans_ss
end

"""
    ccp_negll_v3_binned(vec, init_ss, trans_ss, bin_values,
                         r_θ, r_I, r_P, r_ω, T; grids...)

Binned CCP negative log-likelihood.  Loops over income bins instead of
children — ~100x fewer iterations than the per-child version.
ForwardDiff-compatible.
"""
function ccp_negll_v3_binned(vec::AbstractVector{Tv},
                              init_ss::Array{Float64,4},     # (r_θ, r_I, r_Z, n_bins)
                              trans_ss::Array{Float64,6},    # (r_I, r_I, r_θ, r_Z, T-1, n_bins)
                              bin_values::Vector{Float64},
                              r_θ::Int, r_I::Int, r_P::Int, r_ω::Int, Tp::Int;
                              θ_grid::AbstractVector=1.0:r_θ,
                              I_grid::AbstractVector=1.0:r_I,
                              P_grid::AbstractVector=1.0:r_P,
                              ω_grid::AbstractVector=1.0:r_ω) where Tv
    nc = r_I - 1
    c  = _unpack_cutpoints(vec[1:nc])
    # β₂ (I_old) dropped — pack is [c_reparam, β₁, β₃, β₄, β₅]
    β1, β3, β4, β5 = vec[nc+1], vec[nc+2], vec[nc+3], vec[nc+4]

    n_total_bins = length(bin_values)
    negll = zero(Tv)

    for kω in 1:r_ω, kP in 1:r_P
        z = (kP - 1) * r_ω + kω
        rp = β3 * P_grid[kP] + β4 * ω_grid[kω]

        for b in 1:n_total_bins
            y_b = bin_values[b]
            ip = β5 * y_b  # missing bin has y_b=0, so ip=0

            # ── Initial CCP ──
            for kθ in 1:r_θ
                η = β1 * θ_grid[kθ] + rp + ip
                for kI in 1:r_I
                    @inbounds g = init_ss[kθ, kI, z, b]
                    g < 1e-15 && continue
                    negll -= g * _ordered_probit_logprob(c, η, kI, nc)
                end
            end

            # ── Transition CCPs ──
            for t in 1:Tp-1
                for kθ1 in 1:r_θ
                    η = β1 * θ_grid[kθ1] + rp + ip
                    for kI in 1:r_I
                        for kI1 in 1:r_I
                            @inbounds ξv = trans_ss[kI1, kI, kθ1, z, t, b]
                            ξv < 1e-15 && continue
                            negll -= ξv * _ordered_probit_logprob(c, η, kI1, nc)
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
        0.0,                         # β₂ (I_old) — fixed at zero
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
                               ω_grid::AbstractVector=1.0:r_ω,
                               fix_betaω::Bool=false)
    N, Tp = size(Y_income)

    if fix_betaω
        # Regress on [1, x1] only (no ω)
        XtWX = zeros(2, 2)
        XtWy = zeros(2)
        for n in 1:N, kω in 1:r_ω, kP in 1:r_P
            z = (kP - 1) * r_ω + kω
            wz = w_z[z, n]
            wz < 1e-15 && continue
            x1 = P_grid[kP]
            for t in 1:Tp
                y = Y_income[n, t]
                isnan(y) && continue
                XtWX[1,1] += wz;    XtWX[1,2] += wz*x1
                XtWX[2,1] += wz*x1; XtWX[2,2] += wz*x1*x1
                XtWy[1] += wz*y;    XtWy[2] += wz*x1*y
            end
        end
        λ = 1e-8 * max(XtWX[1,1], 1.0)
        for i in 1:2; XtWX[i,i] += λ; end
        β2 = XtWX \ XtWy
        β = [β2[1], β2[2], 0.0]
    else
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
                XtWX[1,1] += wz;       XtWX[1,2] += wz*x1;       XtWX[1,3] += wz*x2
                XtWX[2,1] += wz*x1;    XtWX[2,2] += wz*x1*x1;    XtWX[2,3] += wz*x1*x2
                XtWX[3,1] += wz*x2;    XtWX[3,2] += wz*x1*x2;    XtWX[3,3] += wz*x2*x2
                XtWy[1] += wz*y;       XtWy[2] += wz*x1*y;        XtWy[3] += wz*x2*y
            end
        end
        λ = 1e-8 * max(XtWX[1,1], 1.0)
        for i in 1:3; XtWX[i,i] += λ; end
        β = XtWX \ XtWy
    end

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

# ─────────────────────────────────────────────────────────────────
# AdultOutcomeEmission — ordered probit for adult outcomes of (θ_T, ω)
# ─────────────────────────────────────────────────────────────────

"""
    AdultOutcomeEmission

Ordered probit emission for adult outcomes that depend on both final-period
child skill θ_T and child heterogeneity ω:

    P(x ≤ k | θ_T = j, ω = v) = Φ(τ_k - α_θ·j - α_ω·v)

Each of the M adult measures has its own cutpoints τ[m], loading α_θ[m], α_ω[m].
"""
struct AdultOutcomeEmission
    tau::Vector{Vector{Float64}}   # cutpoints per measure (length(tau[m]) = κ_m - 1)
    alpha_theta::Vector{Float64}   # loading on θ_T per measure
    alpha_omega::Vector{Float64}   # loading on ω per measure
end

"""
    adult_outcome_matrix(tau_m, alpha_theta_m, alpha_omega_m, r_θ, r_ω, κ_m;
                          θ_grid, ω_grid) -> Array{T,3}

Build emission probabilities P(x=k | θ_T=j, ω=v). Size (κ_m, r_θ, r_ω).
ForwardDiff-compatible.
"""
function adult_outcome_matrix(tau_m::AbstractVector{Tt}, alpha_theta_m::Ta, alpha_omega_m::Tb,
                               r_θ::Int, r_ω::Int, κ_m::Int;
                               θ_grid::AbstractVector=1.0:r_θ,
                               ω_grid::AbstractVector=1.0:r_ω) where {Tt, Ta, Tb}
    RT = promote_type(Tt, Ta, Tb, Float64)
    P = zeros(RT, κ_m, r_θ, r_ω)
    for v in 1:r_ω, j in 1:r_θ
        η = alpha_theta_m * θ_grid[j] + alpha_omega_m * ω_grid[v]
        F_prev = zero(RT)
        for k in 1:κ_m-1
            F_curr = Φ(tau_m[k] - η)
            P[k, j, v] = max(F_curr - F_prev, RT(1e-12))
            F_prev = F_curr
        end
        P[κ_m, j, v] = max(one(RT) - F_prev, RT(1e-12))
        # Renormalize for safety
        s = zero(RT)
        for k in 1:κ_m; s += P[k, j, v]; end
        for k in 1:κ_m; P[k, j, v] /= s; end
    end
    return P
end

"""
Build all adult outcome emission matrices (one 3D array per measure).
"""
function build_adult_matrices(ae::AdultOutcomeEmission, κ_adult::Vector{Int},
                               r_θ::Int, r_ω::Int;
                               θ_grid::AbstractVector=1.0:r_θ,
                               ω_grid::AbstractVector=1.0:r_ω)
    M = length(κ_adult)
    [adult_outcome_matrix(ae.tau[m], ae.alpha_theta[m], ae.alpha_omega[m],
                           r_θ, r_ω, κ_adult[m]; θ_grid, ω_grid) for m in 1:M]
end

"""
    pack_adult_emission(ae::AdultOutcomeEmission) -> Vector{Float64}

Pack into a single vector for optimization: for each measure m,
[cutpoint_reparam(κ_m-1), alpha_theta_m, alpha_omega_m].
"""
function pack_adult_emission(ae::AdultOutcomeEmission)
    v = Float64[]
    for m in eachindex(ae.tau)
        append!(v, _pack_cutpoints(ae.tau[m]))
        push!(v, ae.alpha_theta[m])
        push!(v, ae.alpha_omega[m])
    end
    return v
end

"""
    unpack_adult_emission(v, κ_adult) -> AdultOutcomeEmission
"""
function unpack_adult_emission(v::AbstractVector, κ_adult::Vector{Int})
    M = length(κ_adult)
    tau = Vector{Vector{Float64}}(undef, M)
    alpha_theta = Vector{Float64}(undef, M)
    alpha_omega = Vector{Float64}(undef, M)
    idx = 1
    for m in 1:M
        nc = κ_adult[m] - 1
        tau[m] = _unpack_cutpoints(v[idx:idx+nc-1])
        idx += nc
        alpha_theta[m] = v[idx]; idx += 1
        alpha_omega[m] = v[idx]; idx += 1
    end
    return AdultOutcomeEmission(tau, alpha_theta, alpha_omega)
end

"""
    initialize_adult_outcome_emission(κ_adult, r_θ, r_ω; rng) -> AdultOutcomeEmission

Initialize with equally spaced cutpoints and small positive loadings.
"""
function initialize_adult_outcome_emission(κ_adult::Vector{Int}, r_θ::Int, r_ω::Int;
                                            rng::AbstractRNG=Random.default_rng())
    M = length(κ_adult)
    tau = Vector{Vector{Float64}}(undef, M)
    alpha_theta = Vector{Float64}(undef, M)
    alpha_omega = Vector{Float64}(undef, M)
    for m in 1:M
        nc = κ_adult[m] - 1
        if nc == 1
            tau[m] = [0.0 + 0.1 * randn(rng)]
        else
            tau[m] = collect(range(-1.5, 1.5; length=nc)) .+ 0.05 .* randn(rng, nc)
        end
        alpha_theta[m] = 0.3 + 0.1 * abs(randn(rng))
        alpha_omega[m] = 0.3 + 0.1 * abs(randn(rng))
    end
    return AdultOutcomeEmission(tau, alpha_theta, alpha_omega)
end

"""
    adult_emission_negll(vec, SS_adult, κ_adult, r_θ, r_ω; θ_grid, ω_grid) -> negll

Negative log-likelihood for adult outcome emissions given sufficient statistics.
SS_adult[m] is a (κ_m, r_θ, r_ω) array of weighted counts.
"""
function adult_emission_negll(vec::AbstractVector{T}, SS_adult::Vector{Array{Float64,3}},
                               κ_adult::Vector{Int}, r_θ::Int, r_ω::Int;
                               θ_grid::AbstractVector=1.0:r_θ,
                               ω_grid::AbstractVector=1.0:r_ω) where T
    M = length(κ_adult)
    negll = zero(T)
    idx = 1
    for m in 1:M
        nc = κ_adult[m] - 1
        tau_m = _unpack_cutpoints(vec[idx:idx+nc-1])
        idx += nc
        alpha_theta_m = vec[idx]; idx += 1
        alpha_omega_m = vec[idx]; idx += 1
        Pm = adult_outcome_matrix(tau_m, alpha_theta_m, alpha_omega_m,
                                   r_θ, r_ω, κ_adult[m]; θ_grid, ω_grid)
        for v in 1:r_ω, j in 1:r_θ, k in 1:κ_adult[m]
            w = SS_adult[m][k, j, v]
            if w > 1e-15
                negll -= w * log(max(Pm[k, j, v], T(1e-15)))
            end
        end
    end
    return negll
end
