# ─────────────────────────────────────────────────────────────────
# Parametric emission and transition models
#
# Provides ordered probit emissions and linear probability model (LPM)
# transitions as alternatives to the nonparametric discrete tables.
# All parametric models convert to dense matrices so the E-step is
# unchanged — only the M-step differs (Optim.jl instead of closed-form).
# ─────────────────────────────────────────────────────────────────

using Random
using Distributions: Normal, TDist, cdf, pdf
using Optim
using ForwardDiff

const Φ(x) = cdf(Normal(), x)

# ─── Robit (t-CDF) link function ─────────────────────────────────
# ForwardDiff-compatible t-CDF with fixed degrees of freedom.
# The standard Distributions.cdf(TDist(ν), x) is not AD-compatible
# because beta_inc doesn't support Dual numbers. We provide a custom
# ForwardDiff rule: d/dx F_ν(x) = f_ν(x) (the t-density).

const DEFAULT_ROBIT_DF = 7.0   # ν = 7: mostly linear in [0.1, 0.9]

"""
    Φ_t(x, ν=DEFAULT_ROBIT_DF)

Student-t CDF (robit link). ForwardDiff-compatible for differentiation
w.r.t. `x` with fixed `ν`. Heavier tails than probit → flatter (more
linear) in the interior of the probability space.
"""
Φ_t(x::Real, ν::Real=DEFAULT_ROBIT_DF) = cdf(TDist(ν), x)

function Φ_t(x::ForwardDiff.Dual{T,V,N}, ν::Real=DEFAULT_ROBIT_DF) where {T,V,N}
    xval = ForwardDiff.value(x)
    d = TDist(ν)
    return ForwardDiff.Dual{T}(cdf(d, xval), pdf(d, xval) * ForwardDiff.partials(x))
end

# ═════════════════════════════════════════════════════════════════
# 1. Structs
# ═════════════════════════════════════════════════════════════════

"""
    OrderedProbitEmission

Ordered probit emission model for a set of measures.
For measure m with κ_m categories and latent state j ∈ {1,...,r}:

    P(X_m = k | state = j) = Φ(τ_{m,k} - α_m·j) - Φ(τ_{m,k-1} - α_m·j)

Fields:
- `tau[m]`: Vector of κ_m-1 ordered cutpoints for measure m
- `alpha[m]`: Loading (discrimination) for measure m; α>0 gives monotonicity
"""
struct OrderedProbitEmission
    tau::Vector{Vector{Float64}}
    alpha::Vector{Float64}
end

"""
    FlexibleLPMTransition

Flexible linear probability model for the survivor function (Model A).
For cutoff k = 1,...,r_θ-1:

    S̄(k|θ,I) = c_k + γ₀(θ) + γ₁(θ)·I

- `c`: r_θ-1 cutoff intercepts
- `gamma0`: r_θ θ-specific shifts (gamma0[1] = 0 for identification)
- `gamma1`: r_θ θ-specific investment slopes

Complementarity ⟺ γ₁(θ) non-decreasing in θ.
"""
struct FlexibleLPMTransition
    c::Vector{Float64}
    gamma0::Vector{Float64}
    gamma1::Vector{Float64}
end

"""
    ParsimoniousLPMTransition

Parsimonious linear probability model (Model B, nested in Model A).

    S̄(k|θ,I) = c_k + γ₁·I + γ₂·θ + γ₃·θ·I

Complementarity ⟺ γ₃ ≥ 0.
"""
struct ParsimoniousLPMTransition
    c::Vector{Float64}
    gamma1::Float64
    gamma2::Float64
    gamma3::Float64
end

"""
    OrderedProbitTransition

Ordered probit model for the transition (production function).

    P(θ' ≤ k | θ, I) = Φ(c_k − β₁·θ − β₂·I − β₃·θ·I)

where c₁ < c₂ < ... < c_{r_θ-1} are increasing cutpoints and Φ is the
standard normal CDF. The PMF is obtained by differencing the CDF.

Unlike the LPM, Φ automatically keeps probabilities in [0,1], so no
clamping is needed. However, the nonlinearity of Φ means β₃ ≥ 0 is
necessary but not sufficient for ordinal complementarity — the test
must still be applied to the resulting Π matrix.
"""
struct OrderedProbitTransition
    c::Vector{Float64}
    beta1::Float64
    beta2::Float64
    beta3::Float64
end

"""
    MixtureProbitTransition

Mixture of ordered probits with shared cutpoints for the transition (production function).

    F̄(k|θ,I) = Σ_h w_h(θ) · [1 - Φ(c_k - a_h · I)]

where:
- c₁ < c₂ < ... < c_{r_θ-1} are shared increasing cutpoints (CDF convention)
- a_h = exp(ã_h) > 0 are component-specific slopes
- w_h(θ) = softmax(λ_h(θ)) are θ-dependent mixing weights (λ₁ ≡ 0 for identification)

Each component is a valid ordered probit survivor function; the convex combination
is automatically a valid distribution. The number of components K controls flexibility:
K=1 reduces to an ordered probit without θ-dependence.

Fields:
- `c_raw`: r_θ-1 cutpoint log-diffs (reparameterized for increasing order)
- `a_raw`: K log-slopes (enforce a_h > 0 via exp)
- `lambda`: (K-1) × r_θ softmax logits (component 1 normalized to 0)
- `K`: number of mixture components
"""
struct MixtureProbitTransition
    c_raw::Vector{Float64}
    a_raw::Vector{Float64}
    lambda::Matrix{Float64}
    K::Int
end

"""
    MonotoneNNTransition

Non-separable monotone neural network for the transition (production function).

    F̄(k|θ,I) = Φ(c_k + Σ_h w_h · Φ(α_{h,1}·θ + α_{h,2}·I + b_h))

where:
- c₁ > c₂ > ... > c_{r_θ-1} are decreasing cutpoints (survivor convention)
- w_h = exp(log_w_h) > 0 are positive output weights
- α_{h,2} = exp(ã_{h,2}) > 0 enforces monotonicity in I
- α_{h,1} is unconstrained (θ enters inside hidden units)

Each hidden unit takes both θ and I as inputs. Valid distributions come from
Φ + decreasing cutpoints. Monotonicity in I is automatic since α_{h,2} > 0
and w_h > 0.

Parameter count: (r_θ - 1) + 4K.

Fields:
- `c_raw`: r_θ-1 cutpoint log-diffs (reparameterized for decreasing order)
- `alpha1`: K unconstrained θ-weights
- `alpha2_raw`: K log-slopes for I (enforce α_{h,2} > 0 via exp)
- `b`: K bias terms
- `log_w`: K log-weights (enforce w_h > 0 via exp)
- `K`: number of hidden units
"""
struct MonotoneNNTransition
    c_raw::Vector{Float64}
    alpha1::Vector{Float64}
    alpha2_raw::Vector{Float64}
    b::Vector{Float64}
    log_w::Vector{Float64}
    K::Int
end

"""
    RobitNNTransition

Monotone neural network transition using the robit (Student-t CDF) link
instead of probit (normal CDF). Same architecture as MonotoneNNTransition:

    F̄(k|θ,I) = F_ν(c_k + Σ_h w_h · Φ(α_{h,1}·θ + α_{h,2}·I + b_h))

where F_ν is the CDF of a Student-t with ν degrees of freedom. The inner
activations still use Φ (normal CDF) — only the outer link is replaced.

The t-CDF is flatter (more linear) in the interior of [0,1] than Φ,
curving sharply only near 0 and 1. This means complementarity properties
of the index function translate more directly to the survivor function.

Parameter count: (r_θ - 1) + 4K  (same as MonotoneNNTransition; ν is fixed).

Fields:
- `c_raw`: r_θ-1 cutpoint log-diffs (reparameterized for decreasing order)
- `alpha1`: K unconstrained θ-weights
- `alpha2_raw`: K log-slopes for I (enforce α_{h,2} > 0 via exp)
- `b`: K bias terms
- `log_w`: K log-weights (enforce w_h > 0 via exp)
- `K`: number of hidden units
- `ν`: degrees of freedom for the t-CDF link (fixed, not estimated)
"""
struct RobitNNTransition
    c_raw::Vector{Float64}
    alpha1::Vector{Float64}
    alpha2_raw::Vector{Float64}
    b::Vector{Float64}
    log_w::Vector{Float64}
    K::Int
    ν::Float64
end

"""
    OrderedProbitCCP

Ordered probit model for the investment CCP (conditional choice probability).

    P(I' ≤ k | θ', I) = Φ(c_k − β₁·θ' − β₂·I − β₃·θ'·I)

where c₁ < c₂ < ... < c_{r_I-1} are increasing cutpoints and Φ is the
standard normal CDF. Produces the CCP array CCP[I', I, θ'] of size r_I × r_I × r_θ.
"""
struct OrderedProbitCCP
    c::Vector{Float64}
    beta1::Float64
    beta2::Float64
    beta3::Float64
end

"""
    OrderedProbitCCPSimple

Ordered probit model for the investment CCP without lagged investment dependence.

    P(I' ≤ k | θ') = Φ(c_k − β₁·θ')

Produces the CCP array CCP[I', I, θ'] of size r_I × r_I × r_θ, where the
distribution is the same for all values of I (broadcast across axis 2).
"""
struct OrderedProbitCCPSimple
    c::Vector{Float64}
    beta1::Float64
end

"""
    OrderedProbitInitialI

Ordered probit model for the initial investment distribution conditional on θ.

    P(I ≤ k | θ) = Φ(c_k − β₁·θ)

Used to factor the initial distribution as π0[θ, I] = p(θ) · P(I|θ),
where p(θ) is updated nonparametrically and P(I|θ) is this ordered probit.
"""
struct OrderedProbitInitialI
    c::Vector{Float64}
    beta1::Float64
end

# ═════════════════════════════════════════════════════════════════
# 2. Conversion: parametric params → dense matrices
# ═════════════════════════════════════════════════════════════════

"""
    ordered_probit_matrix(tau, alpha, r) -> Matrix{Float64}

Build a κ×r emission matrix from ordered probit parameters for one measure.
`tau` has κ-1 cutpoints, `alpha` is the loading, `r` is the number of latent states.
Latent states are mapped to grid values 1, 2, ..., r.
"""
function ordered_probit_matrix(tau::AbstractVector{T}, alpha::T2, r::Int) where {T, T2}
    RT = promote_type(T, T2, Float64)
    κ = length(tau) + 1
    P = zeros(RT, κ, r)
    for j in 1:r
        z = alpha * j
        F_prev = zero(RT)
        for k in 1:κ-1
            F_curr = Φ(tau[k] - z)
            P[k, j] = max(F_curr - F_prev, RT(1e-12))
            F_prev = F_curr
        end
        P[κ, j] = max(one(RT) - F_prev, RT(1e-12))
    end
    # Renormalize for safety
    for j in 1:r
        P[:, j] ./= sum(P[:, j])
    end
    return P
end

"""
    emission_matrices_from_probit(spec, r) -> Vector{Matrix}

Convert an OrderedProbitEmission to dense emission matrices.
"""
function emission_matrices_from_probit(spec::OrderedProbitEmission, r::Int)
    return [ordered_probit_matrix(spec.tau[m], spec.alpha[m], r)
            for m in eachindex(spec.alpha)]
end

"""
    survivor_to_pmf(S_bar) -> Vector

Given survivor function values S̄(1),...,S̄(r_θ-1), return the PMF vector
of length r_θ. Clamps to [0,1] and enforces monotonicity.
ForwardDiff-compatible (no mutation).
"""
function survivor_to_pmf(S_bar::AbstractVector{T}) where T
    r_θ = length(S_bar) + 1

    # Clamp to [0,1]
    S = clamp.(S_bar, zero(T), one(T))

    # Enforce non-increasing: S[1] ≥ S[2] ≥ ...
    S_mono = similar(S)
    S_mono[1] = S[1]
    for k in 2:length(S)
        S_mono[k] = min(S[k], S_mono[k-1])
    end

    # Convert to PMF
    pmf = Vector{T}(undef, r_θ)
    pmf[1] = one(T) - S_mono[1]
    for k in 2:r_θ-1
        pmf[k] = S_mono[k-1] - S_mono[k]
    end
    pmf[r_θ] = S_mono[end]

    # Floor and renormalize
    pmf .= max.(pmf, T(1e-12))
    pmf ./= sum(pmf)
    return pmf
end

"""
    transition_from_flexible_lpm(spec, r_θ, r_I; I_grid) -> Array

Build Π[θ',θ,I] from a FlexibleLPMTransition.
Default I_grid: 1, 2, ..., r_I. ForwardDiff-compatible.
"""
function transition_from_flexible_lpm(spec::FlexibleLPMTransition,
                                       r_θ::Int, r_I::Int;
                                       I_grid::AbstractVector=1.0:r_I)
    T = promote_type(eltype(spec.c), eltype(spec.gamma0), eltype(spec.gamma1), Float64)
    Π = zeros(T, r_θ, r_θ, r_I)
    for kI in 1:r_I, kθ in 1:r_θ
        S_bar = [spec.c[k] + spec.gamma0[kθ] + spec.gamma1[kθ] * I_grid[kI]
                 for k in 1:r_θ-1]
        Π[:, kθ, kI] = survivor_to_pmf(S_bar)
    end
    return Π
end

"""
    transition_from_parsimonious_lpm(spec, r_θ, r_I; θ_grid, I_grid) -> Array

Build Π[θ',θ,I] from a ParsimoniousLPMTransition.
Default grids: 1, 2, ..., r_θ and 1, 2, ..., r_I. ForwardDiff-compatible.
"""
function transition_from_parsimonious_lpm(spec::ParsimoniousLPMTransition,
                                           r_θ::Int, r_I::Int;
                                           θ_grid::AbstractVector=1.0:r_θ,
                                           I_grid::AbstractVector=1.0:r_I)
    T = promote_type(eltype(spec.c), typeof(spec.gamma1), typeof(spec.gamma2),
                     typeof(spec.gamma3), Float64)
    Π = zeros(T, r_θ, r_θ, r_I)
    for kI in 1:r_I, kθ in 1:r_θ
        th = θ_grid[kθ]
        inv = I_grid[kI]
        S_bar = [spec.c[k] + spec.gamma1 * inv + spec.gamma2 * th + spec.gamma3 * th * inv
                 for k in 1:r_θ-1]
        Π[:, kθ, kI] = survivor_to_pmf(S_bar)
    end
    return Π
end

"""
    transition_from_ordered_probit(spec, r_θ, r_I; θ_grid, I_grid) -> Array

Build Π[θ',θ,I] from an OrderedProbitTransition.
    P(θ' ≤ k | θ, I) = Φ(c_k − β₁·θ − β₂·I − β₃·θ·I)
    P(θ' = k | θ, I) = Φ(c_k − η) − Φ(c_{k-1} − η),  η = β₁·θ + β₂·I + β₃·θ·I

Default grids: 1, 2, ..., r_θ and 1, 2, ..., r_I. ForwardDiff-compatible.
"""
function transition_from_ordered_probit(spec::OrderedProbitTransition,
                                         r_θ::Int, r_I::Int;
                                         θ_grid::AbstractVector=1.0:r_θ,
                                         I_grid::AbstractVector=1.0:r_I)
    T = promote_type(eltype(spec.c), typeof(spec.beta1), typeof(spec.beta2),
                     typeof(spec.beta3), Float64)
    Π = zeros(T, r_θ, r_θ, r_I)
    for kI in 1:r_I, kθ in 1:r_θ
        η = spec.beta1 * θ_grid[kθ] + spec.beta2 * I_grid[kI] + spec.beta3 * θ_grid[kθ] * I_grid[kI]
        Π[:, kθ, kI] = _ordered_probit_pmf(spec.c, η, r_θ)
    end
    return Π
end

"""
    transition_from_mixture_probit(spec, r_θ, r_I; I_grid) -> Array

Build Π[θ',θ,I] from a MixtureProbitTransition.
    F̄(k|θ,I) = Σ_h w_h(θ) · [1 - Φ(c_k - a_h · I)]
Default I_grid: 1, 2, ..., r_I. ForwardDiff-compatible.
"""
function transition_from_mixture_probit(spec::MixtureProbitTransition,
                                         r_θ::Int, r_I::Int;
                                         I_grid::AbstractVector=1.0:r_I)
    c = _unpack_cutpoints(spec.c_raw)
    a_raw = spec.a_raw
    lambda = spec.lambda
    K = spec.K

    # Build using the flat-vector version for consistency
    flatvec = vcat(spec.c_raw, a_raw, Base.vec(lambda))
    return _build_mixture_probit_pi(flatvec, r_θ, r_I, K; I_grid)
end

"""
    transition_from_monotone_nn(spec, r_θ, r_I; θ_grid, I_grid) -> Array

Build Π[θ',θ,I] from a MonotoneNNTransition.
    F̄(k|θ,I) = Φ(c_k + Σ_h w_h · Φ(α_{h,1}·θ + α_{h,2}·I + b_h))
Default grids: 1, 2, ..., r_θ and 1, 2, ..., r_I. ForwardDiff-compatible.
"""
function transition_from_monotone_nn(spec::MonotoneNNTransition,
                                       r_θ::Int, r_I::Int;
                                       θ_grid::AbstractVector=1.0:r_θ,
                                       I_grid::AbstractVector=1.0:r_I)
    flatvec = pack_monotone_nn(spec)
    return _build_monotone_nn_pi(flatvec, r_θ, r_I, spec.K; θ_grid, I_grid)
end

"""
    transition_from_robit_nn(spec, r_θ, r_I; θ_grid, I_grid) -> Array

Build Π[θ',θ,I] from a RobitNNTransition.
    F̄(k|θ,I) = F_ν(c_k + Σ_h w_h · Φ(α_{h,1}·θ + α_{h,2}·I + b_h))
where F_ν is the Student-t CDF with ν degrees of freedom.
"""
function transition_from_robit_nn(spec::RobitNNTransition,
                                    r_θ::Int, r_I::Int;
                                    θ_grid::AbstractVector=1.0:r_θ,
                                    I_grid::AbstractVector=1.0:r_I)
    flatvec = pack_robit_nn(spec)
    return _build_robit_nn_pi(flatvec, r_θ, r_I, spec.K, spec.ν; θ_grid, I_grid)
end

"""
    ccp_from_ordered_probit(spec, r_θ, r_I; θ_grid, I_grid) -> Array

Build CCP[I', I, θ'] from an OrderedProbitCCP.
    P(I' ≤ k | θ', I) = Φ(c_k − β₁·θ' − β₂·I − β₃·θ'·I)

Default grids: 1, 2, ..., r_θ and 1, 2, ..., r_I. ForwardDiff-compatible.
"""
function ccp_from_ordered_probit(spec::OrderedProbitCCP,
                                  r_θ::Int, r_I::Int;
                                  θ_grid::AbstractVector=1.0:r_θ,
                                  I_grid::AbstractVector=1.0:r_I)
    T = promote_type(eltype(spec.c), typeof(spec.beta1), typeof(spec.beta2),
                     typeof(spec.beta3), Float64)
    CCP = zeros(T, r_I, r_I, r_θ)
    for kθ1 in 1:r_θ, kI in 1:r_I
        η = spec.beta1 * θ_grid[kθ1] + spec.beta2 * I_grid[kI] + spec.beta3 * θ_grid[kθ1] * I_grid[kI]
        CCP[:, kI, kθ1] = _ordered_probit_pmf(spec.c, η, r_I)
    end
    return CCP
end

"""
    ccp_from_ordered_probit_simple(spec, r_θ, r_I; θ_grid) -> Array

Build CCP[I', I, θ'] from an OrderedProbitCCPSimple.
    P(I' ≤ k | θ') = Φ(c_k − β₁·θ')

The distribution is the same for all I (broadcast across axis 2).
"""
function ccp_from_ordered_probit_simple(spec::OrderedProbitCCPSimple,
                                         r_θ::Int, r_I::Int;
                                         θ_grid::AbstractVector=1.0:r_θ)
    T = promote_type(eltype(spec.c), typeof(spec.beta1), Float64)
    CCP = zeros(T, r_I, r_I, r_θ)
    for kθ1 in 1:r_θ
        η = spec.beta1 * θ_grid[kθ1]
        pmf = _ordered_probit_pmf(spec.c, η, r_I)
        for kI in 1:r_I
            CCP[:, kI, kθ1] = pmf
        end
    end
    return CCP
end

"""
    initial_I_from_ordered_probit(spec, r_θ, r_I; θ_grid) -> Matrix

Build P(I|θ) matrix of size r_I × r_θ from an OrderedProbitInitialI.
    P(I ≤ k | θ) = Φ(c_k − β₁·θ)
"""
function initial_I_from_ordered_probit(spec::OrderedProbitInitialI,
                                        r_θ::Int, r_I::Int;
                                        θ_grid::AbstractVector=1.0:r_θ)
    T = promote_type(eltype(spec.c), typeof(spec.beta1), Float64)
    P_I_θ = zeros(T, r_I, r_θ)
    for kθ in 1:r_θ
        η = spec.beta1 * θ_grid[kθ]
        P_I_θ[:, kθ] = _ordered_probit_pmf(spec.c, η, r_I)
    end
    return P_I_θ
end

"""
    π0_from_initial_I(p_θ, P_I_θ) -> Matrix

Build π0[θ, I] = p(θ) · P(I|θ) from marginal p_θ and conditional P_I_θ.
p_θ is a vector of length r_θ, P_I_θ is r_I × r_θ.
"""
function π0_from_initial_I(p_θ::AbstractVector, P_I_θ::AbstractMatrix)
    r_θ = length(p_θ)
    r_I = size(P_I_θ, 1)
    π0 = zeros(r_θ, r_I)
    for kI in 1:r_I, kθ in 1:r_θ
        π0[kθ, kI] = p_θ[kθ] * P_I_θ[kI, kθ]
    end
    return π0
end

"""
    _ordered_probit_pmf(c, η, r_θ) -> Vector

Compute the PMF for an ordered probit with cutpoints `c` (length r_θ-1)
and linear index `η`. ForwardDiff-compatible.
    P(k) = Φ(c_k − η) − Φ(c_{k-1} − η)
"""
function _ordered_probit_pmf(c::AbstractVector{Tc}, η::Tη, r_θ::Int) where {Tc, Tη}
    T = promote_type(Tc, Tη, Float64)
    pmf = Vector{T}(undef, r_θ)
    F_prev = zero(T)
    for k in 1:r_θ-1
        F_curr = Φ(c[k] - η)
        pmf[k] = max(F_curr - F_prev, T(1e-12))
        F_prev = F_curr
    end
    pmf[r_θ] = max(one(T) - F_prev, T(1e-12))
    # Renormalize for safety
    pmf ./= sum(pmf)
    return pmf
end

# ═════════════════════════════════════════════════════════════════
# 3. Pack / Unpack (struct ↔ flat vector for Optim.jl)
# ═════════════════════════════════════════════════════════════════

# Cutpoint reparameterization: store [τ₁, log(τ₂-τ₁), log(τ₃-τ₂), ...]
# so unconstrained optimization maintains increasing ordering.

function _pack_cutpoints(tau::AbstractVector)
    if length(tau) == 0
        return Float64[]
    elseif length(tau) == 1
        return [tau[1]]
    else
        return vcat(tau[1], log.(diff(tau)))
    end
end

function _unpack_cutpoints(v::AbstractVector)
    if length(v) == 0
        return Float64[]
    elseif length(v) == 1
        return [v[1]]
    else
        tau = similar(v)
        tau[1] = v[1]
        for k in 2:length(v)
            tau[k] = tau[k-1] + exp(v[k])
        end
        return tau
    end
end

# Survivor-function cutoff reparameterization: store [c₁, log(c₁-c₂), log(c₂-c₃), ...]
# so unconstrained optimization maintains decreasing ordering (c₁ ≥ c₂ ≥ ...).

function _pack_survivor_cutoffs(c::AbstractVector)
    if length(c) <= 1
        return collect(Float64, c)
    else
        diffs = [-c[k+1] + c[k] for k in 1:length(c)-1]  # positive differences
        return vcat(c[1], log.(max.(diffs, 1e-10)))
    end
end

function _unpack_survivor_cutoffs(v::AbstractVector{T}) where T
    if length(v) <= 1
        return collect(v)
    else
        c = similar(v)
        c[1] = v[1]
        for k in 2:length(v)
            c[k] = c[k-1] - exp(v[k])
        end
        return c
    end
end

"""
    pack_probit(spec) -> Vector{Float64}
    unpack_probit(vec, κ_list) -> OrderedProbitEmission

Flatten: for each measure m, [reparameterized_cutpoints..., alpha_m].
"""
function pack_probit(spec::OrderedProbitEmission)
    parts = Float64[]
    for m in eachindex(spec.alpha)
        append!(parts, _pack_cutpoints(spec.tau[m]))
        push!(parts, spec.alpha[m])
    end
    return parts
end

function unpack_probit(vec::AbstractVector, κ_list::Vector{Int})
    M = length(κ_list)
    tau = Vector{Vector{eltype(vec)}}(undef, M)
    alpha = Vector{eltype(vec)}(undef, M)
    offset = 0
    for m in 1:M
        n_cut = κ_list[m] - 1
        tau[m] = _unpack_cutpoints(vec[offset+1:offset+n_cut])
        alpha[m] = vec[offset + n_cut + 1]
        offset += n_cut + 1
    end
    return OrderedProbitEmission(tau, alpha)
end

"""
    pack_flexible_lpm(spec) -> Vector{Float64}
    unpack_flexible_lpm(vec, r_θ) -> FlexibleLPMTransition

Flatten: [reparameterized_c..., gamma0[2:end]..., gamma1...].
Cutoffs c are reparameterized via log-differences to enforce c₁ ≥ c₂ ≥ ...
"""
function pack_flexible_lpm(spec::FlexibleLPMTransition)
    return vcat(_pack_survivor_cutoffs(spec.c), spec.gamma0[2:end], spec.gamma1)
end

function unpack_flexible_lpm(vec::AbstractVector, r_θ::Int)
    n_cut = r_θ - 1
    c = _unpack_survivor_cutoffs(vec[1:n_cut])
    gamma0_free = vec[n_cut+1:n_cut+(r_θ-1)]
    gamma0 = vcat(zero(eltype(vec)), gamma0_free)
    gamma1 = vec[n_cut+(r_θ-1)+1:end]
    return FlexibleLPMTransition(c, gamma0, gamma1)
end

"""
    pack_parsimonious_lpm(spec) -> Vector{Float64}
    unpack_parsimonious_lpm(vec, r_θ) -> ParsimoniousLPMTransition

Flatten: [reparameterized_c..., gamma1, gamma2, gamma3].
Cutoffs c are reparameterized via log-differences to enforce c₁ ≥ c₂ ≥ ...
"""
function pack_parsimonious_lpm(spec::ParsimoniousLPMTransition)
    return vcat(_pack_survivor_cutoffs(spec.c), spec.gamma1, spec.gamma2, spec.gamma3)
end

function unpack_parsimonious_lpm(vec::AbstractVector, r_θ::Int)
    n_cut = r_θ - 1
    c = _unpack_survivor_cutoffs(vec[1:n_cut])
    return ParsimoniousLPMTransition(c, vec[n_cut+1], vec[n_cut+2], vec[n_cut+3])
end

"""
    pack_mixture_probit(spec) -> Vector{Float64}
    unpack_mixture_probit(vec, r_θ, K) -> MixtureProbitTransition

Flatten: [reparameterized_c..., a_raw..., lambda...].
Cutpoints c are *increasing* (CDF convention), reparameterized via
log-differences. lambda is (K-1) × r_θ, stored column-major.
"""
function pack_mixture_probit(spec::MixtureProbitTransition)
    return vcat(spec.c_raw, spec.a_raw, vec(spec.lambda))
end

function unpack_mixture_probit(v::AbstractVector, r_θ::Int, K::Int)
    n_cut = r_θ - 1
    c_raw = v[1:n_cut]
    a_raw = v[n_cut+1:n_cut+K]
    n_lambda = (K - 1) * r_θ
    lambda = reshape(v[n_cut+K+1:n_cut+K+n_lambda], K - 1, r_θ)
    return MixtureProbitTransition(collect(c_raw), collect(a_raw), collect(lambda), K)
end

"""
    pack_monotone_nn(spec) -> Vector{Float64}
    unpack_monotone_nn(vec, r_θ, K) -> MonotoneNNTransition

Flatten: [reparameterized_c..., alpha1..., alpha2_raw..., b..., log_w...].
Cutpoints c are *decreasing* (survivor convention), reparameterized via
log-differences. Total length: (r_θ-1) + 4K.
"""
function pack_monotone_nn(spec::MonotoneNNTransition)
    return vcat(spec.c_raw, spec.alpha1, spec.alpha2_raw, spec.b, spec.log_w)
end

function unpack_monotone_nn(v::AbstractVector, r_θ::Int, K::Int)
    n_cut = r_θ - 1
    c_raw = v[1:n_cut]
    alpha1 = v[n_cut+1:n_cut+K]
    alpha2_raw = v[n_cut+K+1:n_cut+2K]
    b = v[n_cut+2K+1:n_cut+3K]
    log_w = v[n_cut+3K+1:n_cut+4K]
    return MonotoneNNTransition(collect(c_raw), collect(alpha1), collect(alpha2_raw),
                                 collect(b), collect(log_w), K)
end

"""
    pack_robit_nn(spec) -> Vector{Float64}
    unpack_robit_nn(vec, r_θ, K; ν) -> RobitNNTransition

Same layout as monotone NN: [reparameterized_c..., alpha1..., alpha2_raw..., b..., log_w...].
ν is not packed (it is fixed, not estimated).
"""
function pack_robit_nn(spec::RobitNNTransition)
    return vcat(spec.c_raw, spec.alpha1, spec.alpha2_raw, spec.b, spec.log_w)
end

function unpack_robit_nn(v::AbstractVector, r_θ::Int, K::Int; ν::Float64=DEFAULT_ROBIT_DF)
    n_cut = r_θ - 1
    c_raw = v[1:n_cut]
    alpha1 = v[n_cut+1:n_cut+K]
    alpha2_raw = v[n_cut+K+1:n_cut+2K]
    b = v[n_cut+2K+1:n_cut+3K]
    log_w = v[n_cut+3K+1:n_cut+4K]
    return RobitNNTransition(collect(c_raw), collect(alpha1), collect(alpha2_raw),
                              collect(b), collect(log_w), K, ν)
end

"""
    pack_ordered_probit_transition(spec) -> Vector{Float64}
    unpack_ordered_probit_transition(vec, r_θ) -> OrderedProbitTransition

Flatten: [reparameterized_c..., beta1, beta2, beta3].
Cutpoints c are *increasing* (CDF convention), reparameterized via
log-differences to enforce c₁ ≤ c₂ ≤ ... (same as emission cutpoints).
"""
function pack_ordered_probit_transition(spec::OrderedProbitTransition;
                                        monotone_I::Bool=false)
    b2_packed = monotone_I ? log(max(spec.beta2, 1e-10)) : spec.beta2
    return vcat(_pack_cutpoints(spec.c), spec.beta1, b2_packed, spec.beta3)
end

function unpack_ordered_probit_transition(vec::AbstractVector, r_θ::Int;
                                           monotone_I::Bool=false)
    n_cut = r_θ - 1
    c = _unpack_cutpoints(vec[1:n_cut])
    b2 = monotone_I ? exp(vec[n_cut+2]) : vec[n_cut+2]
    return OrderedProbitTransition(c, vec[n_cut+1], b2, vec[n_cut+3])
end

"""
    pack_ordered_probit_ccp(spec) -> Vector{Float64}
    unpack_ordered_probit_ccp(vec, r_I) -> OrderedProbitCCP

Flatten: [reparameterized_c..., beta1, beta2, beta3].
Cutpoints c are *increasing* (CDF convention), reparameterized via
log-differences to enforce c₁ ≤ c₂ ≤ ...
"""
function pack_ordered_probit_ccp(spec::OrderedProbitCCP)
    return vcat(_pack_cutpoints(spec.c), spec.beta1, spec.beta2, spec.beta3)
end

function unpack_ordered_probit_ccp(vec::AbstractVector, r_I::Int)
    n_cut = r_I - 1
    c = _unpack_cutpoints(vec[1:n_cut])
    return OrderedProbitCCP(c, vec[n_cut+1], vec[n_cut+2], vec[n_cut+3])
end

"""
    pack_ordered_probit_ccp_simple(spec) -> Vector{Float64}
    unpack_ordered_probit_ccp_simple(vec, r_I) -> OrderedProbitCCPSimple

Flatten: [reparameterized_c..., beta1].
"""
function pack_ordered_probit_ccp_simple(spec::OrderedProbitCCPSimple)
    return vcat(_pack_cutpoints(spec.c), spec.beta1)
end

function unpack_ordered_probit_ccp_simple(vec::AbstractVector, r_I::Int)
    n_cut = r_I - 1
    c = _unpack_cutpoints(vec[1:n_cut])
    return OrderedProbitCCPSimple(c, vec[n_cut+1])
end

"""
    pack_ordered_probit_initial_I(spec) -> Vector{Float64}
    unpack_ordered_probit_initial_I(vec, r_I) -> OrderedProbitInitialI

Flatten: [reparameterized_c..., beta1].
"""
function pack_ordered_probit_initial_I(spec::OrderedProbitInitialI)
    return vcat(_pack_cutpoints(spec.c), spec.beta1)
end

function unpack_ordered_probit_initial_I(vec::AbstractVector, r_I::Int)
    n_cut = r_I - 1
    c = _unpack_cutpoints(vec[1:n_cut])
    return OrderedProbitInitialI(c, vec[n_cut+1])
end

# ═════════════════════════════════════════════════════════════════
# 4. M-step objective functions
# ═════════════════════════════════════════════════════════════════

"""
    _marginal_weight(Q, n, k, axis) -> Float64

Compute the marginal posterior weight for latent state k at observation n,
marginalizing out the other latent dimensions.

axis = :theta → sum over (θ', I): ∑_{θ',I} Q[θ', k, I, n]
axis = :theta_prime → sum over (θ, I): ∑_{θ,I} Q[k, θ, I, n]
axis = :I → sum over (θ', θ): ∑_{θ',θ} Q[θ', θ, k, n]
"""
function _marginal_weight(Q::AbstractArray{<:Real,4}, n::Int, k::Int, axis::Symbol)
    if axis == :theta
        return @views sum(Q[:, k, :, n])
    elseif axis == :theta_prime
        return @views sum(Q[k, :, :, n])
    elseif axis == :I
        return @views sum(Q[:, :, k, n])
    else
        error("Unknown axis: $axis")
    end
end

"""
    _precompute_marginal_weights(Q, axis, r) -> Matrix{Float64}

Precompute all marginal weights W[k, n] for the given axis.
This avoids redundant summation during Optim iterations (weights depend
on Q, not on the emission parameters being optimized).
"""
function _precompute_marginal_weights(Q::Array{Float64,4}, axis::Symbol, r::Int)
    N = size(Q, 4)
    W = zeros(r, N)
    if axis == :theta
        for n in 1:N, k in 1:r
            @views W[k, n] = sum(Q[:, k, :, n])
        end
    elseif axis == :theta_prime
        for n in 1:N, k in 1:r
            @views W[k, n] = sum(Q[k, :, :, n])
        end
    elseif axis == :I
        for n in 1:N, k in 1:r
            @views W[k, n] = sum(Q[:, :, k, n])
        end
    else
        error("Unknown axis: $axis")
    end
    return W
end

"""
    _build_probit_matrices(vec, κ_list, r) -> Vector{Matrix}

Build emission matrices from a flat parameter vector, ForwardDiff-compatible.
Avoids constructing OrderedProbitEmission (which has concrete Float64 fields).
"""
function _build_probit_matrices(vec::AbstractVector{T}, κ_list::Vector{Int}, r::Int) where T
    M = length(κ_list)
    P_list = Vector{Matrix{T}}(undef, M)
    offset = 0
    for m in 1:M
        n_cut = κ_list[m] - 1
        tau_raw = vec[offset+1:offset+n_cut]
        tau_m = _unpack_cutpoints(tau_raw)
        alpha_m = vec[offset + n_cut + 1]
        P_list[m] = ordered_probit_matrix(tau_m, alpha_m, r)
        offset += n_cut + 1
    end
    return P_list
end

"""
    emission_negll(vec, Q, X_measures, κ_list, axis, r) -> scalar

Negative expected complete-data log-likelihood for ordered probit emissions.
Used as the objective for Optim.jl in the M-step. ForwardDiff-compatible.
"""
function emission_negll(vec::AbstractVector{T}, Q::AbstractArray{<:Real,4},
                        X_measures::Vector{Vector{Int}},
                        κ_list::Vector{Int}, axis::Symbol, r::Int) where T
    P_list = _build_probit_matrices(vec, κ_list, r)
    negll = zero(T)
    N = size(Q, 4)
    for m in eachindex(X_measures)
        for n in 1:N
            x = X_measures[m][n]
            x == 0 && continue
            for k in 1:r
                w = _marginal_weight(Q, n, k, axis)
                if w > 1e-15
                    negll -= w * log(max(P_list[m][x, k], T(1e-15)))
                end
            end
        end
    end
    return negll
end

"""
    _single_measure_negll(vec, W, X_m, κ, r) -> scalar

Negative expected log-likelihood for a single ordered probit measure.
`W[k,n]` are precomputed marginal weights; `vec = [tau_reparam..., alpha]`.
ForwardDiff-compatible.
"""
function _single_measure_negll(vec::AbstractVector{T}, W::Matrix{Float64},
                                X_m::Vector{Int}, κ::Int, r::Int) where T
    n_cut = κ - 1
    tau = _unpack_cutpoints(vec[1:n_cut])
    alpha = vec[n_cut + 1]
    P = ordered_probit_matrix(tau, alpha, r)

    negll = zero(T)
    N = size(W, 2)
    @inbounds for n in 1:N
        x = X_m[n]
        x == 0 && continue
        for k in 1:r
            w = W[k, n]
            if w > 1e-15
                negll -= w * log(max(P[x, k], T(1e-15)))
            end
        end
    end
    return negll
end

"""
    _build_flexible_lpm_pi(vec, r_θ, r_I; I_grid) -> Array

Build Π from a flat parameter vector for flexible LPM. ForwardDiff-compatible.
"""
function _build_flexible_lpm_pi(vec::AbstractVector{T}, r_θ::Int, r_I::Int;
                                 I_grid::AbstractVector=1.0:r_I) where T
    n_cut = r_θ - 1
    c = _unpack_survivor_cutoffs(vec[1:n_cut])
    gamma0_free = vec[n_cut+1:n_cut+(r_θ-1)]
    gamma0 = vcat(zero(T), gamma0_free)
    gamma1 = vec[n_cut+(r_θ-1)+1:end]

    Π = zeros(T, r_θ, r_θ, r_I)
    for kI in 1:r_I, kθ in 1:r_θ
        S_bar = [c[k] + gamma0[kθ] + gamma1[kθ] * I_grid[kI] for k in 1:r_θ-1]
        Π[:, kθ, kI] = survivor_to_pmf(S_bar)
    end
    return Π
end

"""
    _build_parsimonious_lpm_pi(vec, r_θ, r_I; θ_grid, I_grid) -> Array

Build Π from a flat parameter vector for parsimonious LPM. ForwardDiff-compatible.
"""
function _build_parsimonious_lpm_pi(vec::AbstractVector{T}, r_θ::Int, r_I::Int;
                                     θ_grid::AbstractVector=1.0:r_θ,
                                     I_grid::AbstractVector=1.0:r_I) where T
    n_cut = r_θ - 1
    c = _unpack_survivor_cutoffs(vec[1:n_cut])
    g1, g2, g3 = vec[n_cut+1], vec[n_cut+2], vec[n_cut+3]

    Π = zeros(T, r_θ, r_θ, r_I)
    for kI in 1:r_I, kθ in 1:r_θ
        th = θ_grid[kθ]
        inv = I_grid[kI]
        S_bar = [c[k] + g1 * inv + g2 * th + g3 * th * inv for k in 1:r_θ-1]
        Π[:, kθ, kI] = survivor_to_pmf(S_bar)
    end
    return Π
end

"""
    transition_negll_flexible(vec, Q, r_θ, r_I; I_grid) -> scalar

Negative expected complete-data log-likelihood for flexible LPM transition.
"""
function transition_negll_flexible(vec::AbstractVector{T}, Q::AbstractArray{<:Real,4},
                                    r_θ::Int, r_I::Int;
                                    I_grid::AbstractVector=1.0:r_I) where T
    Π = _build_flexible_lpm_pi(vec, r_θ, r_I; I_grid)
    return _transition_negll(Π, Q)
end

"""
    transition_negll_parsimonious(vec, Q, r_θ, r_I; θ_grid, I_grid) -> scalar

Negative expected complete-data log-likelihood for parsimonious LPM transition.
"""
function transition_negll_parsimonious(vec::AbstractVector{T}, Q::AbstractArray{<:Real,4},
                                        r_θ::Int, r_I::Int;
                                        θ_grid::AbstractVector=1.0:r_θ,
                                        I_grid::AbstractVector=1.0:r_I) where T
    Π = _build_parsimonious_lpm_pi(vec, r_θ, r_I; θ_grid, I_grid)
    return _transition_negll(Π, Q)
end

function _transition_negll(Π::AbstractArray{T,3}, Q::AbstractArray{<:Real,4}) where T
    negll = zero(T)
    r1, r2, r3 = size(Π)
    N = size(Q, 4)
    for n in 1:N, kI in 1:r3, kθ in 1:r2, kθ1 in 1:r1
        w = Q[kθ1, kθ, kI, n]
        if w > 1e-15
            negll -= w * log(max(Π[kθ1, kθ, kI], T(1e-15)))
        end
    end
    return negll
end

"""
    _precompute_Q_sum(Q) -> Array{Float64,3}

Sum Q over observations: Q_sum[θ', θ, I] = Σ_n Q[θ', θ, I, n].
Used to avoid iterating over N inside the transition objective.
"""
function _precompute_Q_sum(Q::Array{Float64,4})
    return dropdims(sum(Q; dims=4); dims=4)
end

"""
    _transition_negll_precomputed(Π, Q_sum) -> scalar

Transition negll using precomputed Q_sum (no loop over N).
"""
function _transition_negll_precomputed(Π::AbstractArray{T,3},
                                        Q_sum::Array{Float64,3}) where T
    negll = zero(T)
    r1, r2, r3 = size(Π)
    @inbounds for kI in 1:r3, kθ in 1:r2, kθ1 in 1:r1
        w = Q_sum[kθ1, kθ, kI]
        if w > 1e-15
            negll -= w * log(max(Π[kθ1, kθ, kI], T(1e-15)))
        end
    end
    return negll
end

function transition_negll_flexible_fast(vec::AbstractVector{T}, Q_sum::Array{Float64,3},
                                         r_θ::Int, r_I::Int;
                                         I_grid::AbstractVector=1.0:r_I) where T
    Π = _build_flexible_lpm_pi(vec, r_θ, r_I; I_grid)
    return _transition_negll_precomputed(Π, Q_sum)
end

function transition_negll_parsimonious_fast(vec::AbstractVector{T}, Q_sum::Array{Float64,3},
                                             r_θ::Int, r_I::Int;
                                             θ_grid::AbstractVector=1.0:r_θ,
                                             I_grid::AbstractVector=1.0:r_I) where T
    Π = _build_parsimonious_lpm_pi(vec, r_θ, r_I; θ_grid, I_grid)
    return _transition_negll_precomputed(Π, Q_sum)
end

"""
    _build_ordered_probit_pi(vec, r_θ, r_I; θ_grid, I_grid) -> Array

Build Π from a flat parameter vector for ordered probit transition.
vec = [reparameterized_cutpoints..., beta1, beta2, beta3].
ForwardDiff-compatible.
"""
function _build_ordered_probit_pi(vec::AbstractVector{T}, r_θ::Int, r_I::Int;
                                   θ_grid::AbstractVector=1.0:r_θ,
                                   I_grid::AbstractVector=1.0:r_I,
                                   monotone_I::Bool=false) where T
    n_cut = r_θ - 1
    c = _unpack_cutpoints(vec[1:n_cut])
    b1 = vec[n_cut+1]
    b2 = monotone_I ? exp(vec[n_cut+2]) : vec[n_cut+2]
    b3 = vec[n_cut+3]

    Π = zeros(T, r_θ, r_θ, r_I)
    for kI in 1:r_I, kθ in 1:r_θ
        η = b1 * θ_grid[kθ] + b2 * I_grid[kI] + b3 * θ_grid[kθ] * I_grid[kI]
        Π[:, kθ, kI] = _ordered_probit_pmf(c, η, r_θ)
    end
    return Π
end

"""
    _build_mixture_probit_pi(vec, r_θ, r_I, K; I_grid) -> Array

Build Π from a flat parameter vector for mixture probit transition.
vec = [reparameterized_cutpoints..., a_raw..., lambda...].
ForwardDiff-compatible.
"""
function _build_mixture_probit_pi(vec::AbstractVector{T}, r_θ::Int, r_I::Int, K::Int;
                                   I_grid::AbstractVector=1.0:r_I) where T
    n_cut = r_θ - 1
    c = _unpack_cutpoints(vec[1:n_cut])
    a = exp.(vec[n_cut+1:n_cut+K])

    n_lambda = (K - 1) * r_θ
    lambda_free = reshape(vec[n_cut+K+1:n_cut+K+n_lambda], K - 1, r_θ)

    Π = zeros(T, r_θ, r_θ, r_I)
    for kI in 1:r_I, kθ in 1:r_θ
        # Softmax weights for this θ
        logits = vcat(zero(T), lambda_free[:, kθ])
        max_l = maximum(logits)
        exp_l = exp.(logits .- max_l)
        w = exp_l ./ sum(exp_l)

        # Mixture survivor: F̄(k|θ,I) = Σ_h w_h · [1 - Φ(c_k - a_h·I)]
        S_bar = zeros(T, n_cut)
        for k in 1:n_cut
            for h in 1:K
                S_bar[k] += w[h] * (one(T) - Φ(c[k] - a[h] * I_grid[kI]))
            end
        end

        # PMF from survivor
        pmf = Vector{T}(undef, r_θ)
        pmf[1] = one(T) - S_bar[1]
        for k in 2:n_cut
            pmf[k] = S_bar[k-1] - S_bar[k]
        end
        pmf[r_θ] = S_bar[n_cut]

        # Floor and renormalize for numerical safety
        pmf .= max.(pmf, T(1e-12))
        pmf ./= sum(pmf)
        Π[:, kθ, kI] = pmf
    end
    return Π
end

function transition_negll_mixture_probit_fast(vec::AbstractVector{T}, Q_sum::Array{Float64,3},
                                               r_θ::Int, r_I::Int, K::Int;
                                               I_grid::AbstractVector=1.0:r_I) where T
    Π = _build_mixture_probit_pi(vec, r_θ, r_I, K; I_grid)
    return _transition_negll_precomputed(Π, Q_sum)
end

"""
    _build_monotone_nn_pi(vec, r_θ, r_I, K; θ_grid, I_grid) -> Array

Build Π from a flat parameter vector for monotone NN transition.
vec = [reparameterized_cutpoints..., alpha1..., alpha2_raw..., b..., log_w...].
ForwardDiff-compatible.

    F̄(k|θ,I) = Φ(c_k + Σ_h w_h · Φ(α_{h,1}·θ + α_{h,2}·I + b_h))
"""
function _build_monotone_nn_pi(vec::AbstractVector{T}, r_θ::Int, r_I::Int, K::Int;
                                 θ_grid::AbstractVector=1.0:r_θ,
                                 I_grid::AbstractVector=1.0:r_I) where T
    n_cut = r_θ - 1
    c = _unpack_survivor_cutoffs(vec[1:n_cut])
    alpha1 = vec[n_cut+1:n_cut+K]
    alpha2 = exp.(vec[n_cut+K+1:n_cut+2K])
    b = vec[n_cut+2K+1:n_cut+3K]
    w = exp.(vec[n_cut+3K+1:n_cut+4K])

    Π = zeros(T, r_θ, r_θ, r_I)
    for kI in 1:r_I, kθ in 1:r_θ
        # μ(θ,I) = Σ_h w_h · Φ(α_{h,1}·θ + α_{h,2}·I + b_h)
        mu = zero(T)
        for h in 1:K
            mu += w[h] * Φ(alpha1[h] * θ_grid[kθ] + alpha2[h] * I_grid[kI] + b[h])
        end

        # Survivor: F̄(k) = Φ(c_k + μ), decreasing cutpoints → decreasing F̄
        S_bar = Vector{T}(undef, n_cut)
        for k in 1:n_cut
            S_bar[k] = Φ(c[k] + mu)
        end

        # PMF from survivor
        pmf = Vector{T}(undef, r_θ)
        pmf[1] = one(T) - S_bar[1]
        for k in 2:n_cut
            pmf[k] = S_bar[k-1] - S_bar[k]
        end
        pmf[r_θ] = S_bar[n_cut]

        # Floor and renormalize for numerical safety
        pmf .= max.(pmf, T(1e-12))
        pmf ./= sum(pmf)
        Π[:, kθ, kI] = pmf
    end
    return Π
end

function transition_negll_monotone_nn_fast(vec::AbstractVector{T}, Q_sum::Array{Float64,3},
                                             r_θ::Int, r_I::Int, K::Int;
                                             θ_grid::AbstractVector=1.0:r_θ,
                                             I_grid::AbstractVector=1.0:r_I) where T
    Π = _build_monotone_nn_pi(vec, r_θ, r_I, K; θ_grid, I_grid)
    return _transition_negll_precomputed(Π, Q_sum)
end

"""
    _build_robit_nn_pi(vec, r_θ, r_I, K, ν; θ_grid, I_grid) -> Array

Build Π from a flat parameter vector for robit NN transition.
Same layout as monotone NN, but the outer link is the t-CDF F_ν instead of Φ.
ForwardDiff-compatible (via custom Φ_t dual rule).

    F̄(k|θ,I) = F_ν(c_k + Σ_h w_h · Φ(α_{h,1}·θ + α_{h,2}·I + b_h))
"""
function _build_robit_nn_pi(vec::AbstractVector{T}, r_θ::Int, r_I::Int, K::Int,
                              ν::Float64=DEFAULT_ROBIT_DF;
                              θ_grid::AbstractVector=1.0:r_θ,
                              I_grid::AbstractVector=1.0:r_I) where T
    n_cut = r_θ - 1
    c = _unpack_survivor_cutoffs(vec[1:n_cut])
    alpha1 = vec[n_cut+1:n_cut+K]
    alpha2 = exp.(vec[n_cut+K+1:n_cut+2K])
    b = vec[n_cut+2K+1:n_cut+3K]
    w = exp.(vec[n_cut+3K+1:n_cut+4K])

    Π = zeros(T, r_θ, r_θ, r_I)
    for kI in 1:r_I, kθ in 1:r_θ
        # μ(θ,I) = Σ_h w_h · Φ(α_{h,1}·θ + α_{h,2}·I + b_h)
        mu = zero(T)
        for h in 1:K
            mu += w[h] * Φ(alpha1[h] * θ_grid[kθ] + alpha2[h] * I_grid[kI] + b[h])
        end

        # Survivor: F̄(k) = F_ν(c_k + μ), using t-CDF instead of Φ
        S_bar = Vector{T}(undef, n_cut)
        for k in 1:n_cut
            S_bar[k] = Φ_t(c[k] + mu, ν)
        end

        # PMF from survivor
        pmf = Vector{T}(undef, r_θ)
        pmf[1] = one(T) - S_bar[1]
        for k in 2:n_cut
            pmf[k] = S_bar[k-1] - S_bar[k]
        end
        pmf[r_θ] = S_bar[n_cut]

        # Floor and renormalize for numerical safety
        pmf .= max.(pmf, T(1e-12))
        pmf ./= sum(pmf)
        Π[:, kθ, kI] = pmf
    end
    return Π
end

function transition_negll_robit_nn_fast(vec::AbstractVector{T}, Q_sum::Array{Float64,3},
                                          r_θ::Int, r_I::Int, K::Int,
                                          ν::Float64=DEFAULT_ROBIT_DF;
                                          θ_grid::AbstractVector=1.0:r_θ,
                                          I_grid::AbstractVector=1.0:r_I) where T
    Π = _build_robit_nn_pi(vec, r_θ, r_I, K, ν; θ_grid, I_grid)
    return _transition_negll_precomputed(Π, Q_sum)
end

function transition_negll_ordered_probit_fast(vec::AbstractVector{T}, Q_sum::Array{Float64,3},
                                               r_θ::Int, r_I::Int;
                                               θ_grid::AbstractVector=1.0:r_θ,
                                               I_grid::AbstractVector=1.0:r_I,
                                               monotone_I::Bool=false) where T
    Π = _build_ordered_probit_pi(vec, r_θ, r_I; θ_grid, I_grid, monotone_I)
    return _transition_negll_precomputed(Π, Q_sum)
end

"""
    _build_ordered_probit_ccp(vec, r_θ, r_I; θ_grid, I_grid) -> Array

Build CCP[I', I, θ'] from a flat parameter vector for ordered probit CCP.
vec = [reparameterized_cutpoints..., beta1, beta2, beta3].
ForwardDiff-compatible.
"""
function _build_ordered_probit_ccp(vec::AbstractVector{T}, r_θ::Int, r_I::Int;
                                    θ_grid::AbstractVector=1.0:r_θ,
                                    I_grid::AbstractVector=1.0:r_I) where T
    n_cut = r_I - 1
    c = _unpack_cutpoints(vec[1:n_cut])
    b1, b2, b3 = vec[n_cut+1], vec[n_cut+2], vec[n_cut+3]

    CCP = zeros(T, r_I, r_I, r_θ)
    for kθ1 in 1:r_θ, kI in 1:r_I
        η = b1 * θ_grid[kθ1] + b2 * I_grid[kI] + b3 * θ_grid[kθ1] * I_grid[kI]
        CCP[:, kI, kθ1] = _ordered_probit_pmf(c, η, r_I)
    end
    return CCP
end

"""
    _ccp_negll_precomputed(CCP, CCP_sum) -> scalar

CCP negll using precomputed CCP_sum[I', I, θ'] (no loop over N).
"""
function _ccp_negll_precomputed(CCP::AbstractArray{T,3},
                                 CCP_sum::Array{Float64,3}) where T
    negll = zero(T)
    r_I1, r_I2, r_θ = size(CCP)
    @inbounds for kθ1 in 1:r_θ, kI in 1:r_I2, kI1 in 1:r_I1
        w = CCP_sum[kI1, kI, kθ1]
        if w > 1e-15
            negll -= w * log(max(CCP[kI1, kI, kθ1], T(1e-15)))
        end
    end
    return negll
end

function ccp_negll_ordered_probit_fast(vec::AbstractVector{T}, CCP_sum::Array{Float64,3},
                                        r_θ::Int, r_I::Int;
                                        θ_grid::AbstractVector=1.0:r_θ,
                                        I_grid::AbstractVector=1.0:r_I) where T
    CCP = _build_ordered_probit_ccp(vec, r_θ, r_I; θ_grid, I_grid)
    return _ccp_negll_precomputed(CCP, CCP_sum)
end

"""
    _build_ordered_probit_ccp_simple(vec, r_θ, r_I; θ_grid) -> Array

Build CCP[I', I, θ'] from a flat parameter vector for simple ordered probit CCP.
vec = [reparameterized_cutpoints..., beta1].
ForwardDiff-compatible.
"""
function _build_ordered_probit_ccp_simple(vec::AbstractVector{T}, r_θ::Int, r_I::Int;
                                           θ_grid::AbstractVector=1.0:r_θ) where T
    n_cut = r_I - 1
    c = _unpack_cutpoints(vec[1:n_cut])
    b1 = vec[n_cut+1]

    CCP = zeros(T, r_I, r_I, r_θ)
    for kθ1 in 1:r_θ
        η = b1 * θ_grid[kθ1]
        pmf = _ordered_probit_pmf(c, η, r_I)
        for kI in 1:r_I
            CCP[:, kI, kθ1] = pmf
        end
    end
    return CCP
end

function ccp_negll_ordered_probit_simple_fast(vec::AbstractVector{T}, CCP_sum::Array{Float64,3},
                                               r_θ::Int, r_I::Int;
                                               θ_grid::AbstractVector=1.0:r_θ) where T
    CCP = _build_ordered_probit_ccp_simple(vec, r_θ, r_I; θ_grid)
    return _ccp_negll_precomputed(CCP, CCP_sum)
end

"""
    _build_ordered_probit_initial_I(vec, r_θ, r_I; θ_grid) -> Matrix

Build P(I|θ) matrix of size r_I × r_θ from a flat parameter vector.
vec = [reparameterized_cutpoints..., beta1].
ForwardDiff-compatible.
"""
function _build_ordered_probit_initial_I(vec::AbstractVector{T}, r_θ::Int, r_I::Int;
                                          θ_grid::AbstractVector=1.0:r_θ) where T
    n_cut = r_I - 1
    c = _unpack_cutpoints(vec[1:n_cut])
    b1 = vec[n_cut+1]

    P_I_θ = zeros(T, r_I, r_θ)
    for kθ in 1:r_θ
        η = b1 * θ_grid[kθ]
        P_I_θ[:, kθ] = _ordered_probit_pmf(c, η, r_I)
    end
    return P_I_θ
end

"""
    _initial_I_negll_precomputed(P_I_θ, π0_marginals) -> scalar

Negative log-likelihood for initial I distribution.
π0_marginals[I, θ] = Σ_n γ_n(θ, I, t=1) — the sufficient statistics for (θ, I) at t=1.
"""
function _initial_I_negll_precomputed(P_I_θ::AbstractMatrix{T},
                                       π0_marginals::Matrix{Float64}) where T
    negll = zero(T)
    r_I, r_θ = size(P_I_θ)
    @inbounds for kθ in 1:r_θ, kI in 1:r_I
        w = π0_marginals[kI, kθ]
        if w > 1e-15
            negll -= w * log(max(P_I_θ[kI, kθ], T(1e-15)))
        end
    end
    return negll
end

function initial_I_negll_ordered_probit_fast(vec::AbstractVector{T}, π0_marginals::Matrix{Float64},
                                              r_θ::Int, r_I::Int;
                                              θ_grid::AbstractVector=1.0:r_θ) where T
    P_I_θ = _build_ordered_probit_initial_I(vec, r_θ, r_I; θ_grid)
    return _initial_I_negll_precomputed(P_I_θ, π0_marginals)
end

# ═════════════════════════════════════════════════════════════════
# 5. Wrapper M-step functions
# ═════════════════════════════════════════════════════════════════

const DEFAULT_OPTIM_OPTIONS = Optim.Options(iterations=100, g_tol=1e-6)

"""
    parametric_mstep_emission(spec, Q, X_measures, κ_list, axis, r; ...) -> (new_spec, P_list)

Optimize ordered probit parameters to maximize E[LL | Q].
Each measure is optimized independently (they share the same marginal
weights but have separate parameters). This keeps the parameter dimension
small for ForwardDiff and avoids redundant weight computation.
Returns the updated spec and the corresponding dense emission matrices.
"""
function parametric_mstep_emission(spec::OrderedProbitEmission,
                                    Q::Array{Float64,4},
                                    X_measures::Vector{Vector{Int}},
                                    κ_list::Vector{Int},
                                    axis::Symbol, r::Int;
                                    options=DEFAULT_OPTIM_OPTIONS)
    # Precompute marginal weights once (does not depend on emission params)
    W = _precompute_marginal_weights(Q, axis, r)

    M = length(κ_list)
    new_tau = Vector{Vector{Float64}}(undef, M)
    new_alpha = Vector{Float64}(undef, M)
    P_list = Vector{Matrix{Float64}}(undef, M)

    for m in 1:M
        n_cut = κ_list[m] - 1
        x0 = vcat(_pack_cutpoints(spec.tau[m]), spec.alpha[m])
        obj = x -> _single_measure_negll(x, W, X_measures[m], κ_list[m], r)
        result = optimize(obj, x0, Newton(), options; autodiff=:forward)
        xopt = Optim.minimizer(result)
        new_tau[m] = _unpack_cutpoints(xopt[1:n_cut])
        new_alpha[m] = xopt[n_cut + 1]
        P_list[m] = ordered_probit_matrix(new_tau[m], new_alpha[m], r)
    end

    new_spec = OrderedProbitEmission(new_tau, new_alpha)
    return new_spec, P_list
end

"""
    parametric_mstep_transition(spec::FlexibleLPMTransition, Q, r_θ, r_I; ...)

Optimize flexible LPM parameters. Returns (new_spec, Π).
"""
function parametric_mstep_transition(spec::FlexibleLPMTransition,
                                      Q::Array{Float64,4},
                                      r_θ::Int, r_I::Int;
                                      I_grid::AbstractVector=collect(1.0:r_I),
                                      options=DEFAULT_OPTIM_OPTIONS)
    Q_sum = _precompute_Q_sum(Q)
    x0 = pack_flexible_lpm(spec)
    obj = x -> transition_negll_flexible_fast(x, Q_sum, r_θ, r_I; I_grid)
    result = optimize(obj, x0, LBFGS(), options; autodiff=:forward)
    new_spec = unpack_flexible_lpm(Optim.minimizer(result), r_θ)
    Π = transition_from_flexible_lpm(new_spec, r_θ, r_I; I_grid)
    return new_spec, Π
end

"""
    parametric_mstep_transition(spec::ParsimoniousLPMTransition, Q, r_θ, r_I; ...)

Optimize parsimonious LPM parameters. Returns (new_spec, Π).
"""
function parametric_mstep_transition(spec::ParsimoniousLPMTransition,
                                      Q::Array{Float64,4},
                                      r_θ::Int, r_I::Int;
                                      θ_grid::AbstractVector=collect(1.0:r_θ),
                                      I_grid::AbstractVector=collect(1.0:r_I),
                                      options=DEFAULT_OPTIM_OPTIONS)
    Q_sum = _precompute_Q_sum(Q)
    x0 = pack_parsimonious_lpm(spec)
    obj = x -> transition_negll_parsimonious_fast(x, Q_sum, r_θ, r_I; θ_grid, I_grid)
    result = optimize(obj, x0, LBFGS(), options; autodiff=:forward)
    new_spec = unpack_parsimonious_lpm(Optim.minimizer(result), r_θ)
    Π = transition_from_parsimonious_lpm(new_spec, r_θ, r_I; θ_grid, I_grid)
    return new_spec, Π
end

"""
    parametric_mstep_transition(spec::OrderedProbitTransition, Q, r_θ, r_I; ...)

Optimize ordered probit transition parameters. Returns (new_spec, Π).
"""
function parametric_mstep_transition(spec::OrderedProbitTransition,
                                      Q::Array{Float64,4},
                                      r_θ::Int, r_I::Int;
                                      θ_grid::AbstractVector=collect(1.0:r_θ),
                                      I_grid::AbstractVector=collect(1.0:r_I),
                                      monotone_I::Bool=false,
                                      options=DEFAULT_OPTIM_OPTIONS)
    Q_sum = _precompute_Q_sum(Q)
    x0 = pack_ordered_probit_transition(spec; monotone_I)
    obj = x -> transition_negll_ordered_probit_fast(x, Q_sum, r_θ, r_I; θ_grid, I_grid, monotone_I)
    result = optimize(obj, x0, Newton(), options; autodiff=:forward)
    new_spec = unpack_ordered_probit_transition(Optim.minimizer(result), r_θ; monotone_I)
    Π = transition_from_ordered_probit(new_spec, r_θ, r_I; θ_grid, I_grid)
    return new_spec, Π
end

"""
    parametric_mstep_transition(spec::MixtureProbitTransition, Q, r_θ, r_I; ...)

Optimize mixture probit transition parameters. Returns (new_spec, Π).
"""
function parametric_mstep_transition(spec::MixtureProbitTransition,
                                      Q::Array{Float64,4},
                                      r_θ::Int, r_I::Int;
                                      I_grid::AbstractVector=collect(1.0:r_I),
                                      options=DEFAULT_OPTIM_OPTIONS,
                                      kwargs...)
    Q_sum = _precompute_Q_sum(Q)
    x0 = pack_mixture_probit(spec)
    obj = x -> transition_negll_mixture_probit_fast(x, Q_sum, r_θ, r_I, spec.K; I_grid)
    result = optimize(obj, x0, LBFGS(), options; autodiff=:forward)
    new_spec = unpack_mixture_probit(Optim.minimizer(result), r_θ, spec.K)
    Π = _build_mixture_probit_pi(Optim.minimizer(result), r_θ, r_I, spec.K; I_grid)
    return new_spec, Π
end

"""
    parametric_mstep_transition(spec::MonotoneNNTransition, Q, r_θ, r_I; ...)

Optimize monotone NN transition parameters. Returns (new_spec, Π).
"""
function parametric_mstep_transition(spec::MonotoneNNTransition,
                                      Q::Array{Float64,4},
                                      r_θ::Int, r_I::Int;
                                      θ_grid::AbstractVector=collect(1.0:r_θ),
                                      I_grid::AbstractVector=collect(1.0:r_I),
                                      options=DEFAULT_OPTIM_OPTIONS,
                                      kwargs...)
    Q_sum = _precompute_Q_sum(Q)
    x0 = pack_monotone_nn(spec)
    obj = x -> transition_negll_monotone_nn_fast(x, Q_sum, r_θ, r_I, spec.K; θ_grid, I_grid)
    result = optimize(obj, x0, LBFGS(), options; autodiff=:forward)
    new_spec = unpack_monotone_nn(Optim.minimizer(result), r_θ, spec.K)
    Π = _build_monotone_nn_pi(Optim.minimizer(result), r_θ, r_I, spec.K; θ_grid, I_grid)
    return new_spec, Π
end

"""
    parametric_mstep_transition(spec::RobitNNTransition, Q, r_θ, r_I; ...)

Optimize robit NN transition parameters. Returns (new_spec, Π).
"""
function parametric_mstep_transition(spec::RobitNNTransition,
                                      Q::Array{Float64,4},
                                      r_θ::Int, r_I::Int;
                                      θ_grid::AbstractVector=collect(1.0:r_θ),
                                      I_grid::AbstractVector=collect(1.0:r_I),
                                      options=DEFAULT_OPTIM_OPTIONS,
                                      kwargs...)
    Q_sum = _precompute_Q_sum(Q)
    x0 = pack_robit_nn(spec)
    obj = x -> transition_negll_robit_nn_fast(x, Q_sum, r_θ, r_I, spec.K, spec.ν; θ_grid, I_grid)
    result = optimize(obj, x0, LBFGS(), options; autodiff=:forward)
    new_spec = unpack_robit_nn(Optim.minimizer(result), r_θ, spec.K; ν=spec.ν)
    Π = _build_robit_nn_pi(Optim.minimizer(result), r_θ, r_I, spec.K, spec.ν; θ_grid, I_grid)
    return new_spec, Π
end

"""
    parametric_mstep_ccp(spec::OrderedProbitCCP, CCP_sum, r_θ, r_I; ...)

Optimize ordered probit CCP parameters. Returns (new_spec, CCP).
CCP_sum[I', I, θ'] are precomputed sufficient statistics.
"""
function parametric_mstep_ccp(spec::OrderedProbitCCP,
                               CCP_sum::Array{Float64,3},
                               r_θ::Int, r_I::Int;
                               θ_grid::AbstractVector=collect(1.0:r_θ),
                               I_grid::AbstractVector=collect(1.0:r_I),
                               options=DEFAULT_OPTIM_OPTIONS)
    x0 = pack_ordered_probit_ccp(spec)
    obj = x -> ccp_negll_ordered_probit_fast(x, CCP_sum, r_θ, r_I; θ_grid, I_grid)
    result = optimize(obj, x0, Newton(), options; autodiff=:forward)
    new_spec = unpack_ordered_probit_ccp(Optim.minimizer(result), r_I)
    CCP = ccp_from_ordered_probit(new_spec, r_θ, r_I; θ_grid, I_grid)
    return new_spec, CCP
end

"""
    parametric_mstep_ccp(spec::OrderedProbitCCPSimple, CCP_sum, r_θ, r_I; ...)

Optimize simple ordered probit CCP parameters (I' depends only on θ').
Returns (new_spec, CCP).
"""
function parametric_mstep_ccp(spec::OrderedProbitCCPSimple,
                               CCP_sum::Array{Float64,3},
                               r_θ::Int, r_I::Int;
                               θ_grid::AbstractVector=collect(1.0:r_θ),
                               options=DEFAULT_OPTIM_OPTIONS,
                               kwargs...)
    x0 = pack_ordered_probit_ccp_simple(spec)
    obj = x -> ccp_negll_ordered_probit_simple_fast(x, CCP_sum, r_θ, r_I; θ_grid)
    result = optimize(obj, x0, Newton(), options; autodiff=:forward)
    new_spec = unpack_ordered_probit_ccp_simple(Optim.minimizer(result), r_I)
    CCP = ccp_from_ordered_probit_simple(new_spec, r_θ, r_I; θ_grid)
    return new_spec, CCP
end

"""
    parametric_mstep_initial_I(spec::OrderedProbitInitialI, π0_marginals, r_θ, r_I; ...)

Optimize ordered probit initial I parameters. Returns (new_spec, P_I_θ).
π0_marginals[I, θ] = Σ_n γ_n(θ, I, t=1).
"""
function parametric_mstep_initial_I(spec::OrderedProbitInitialI,
                                     π0_marginals::Matrix{Float64},
                                     r_θ::Int, r_I::Int;
                                     θ_grid::AbstractVector=collect(1.0:r_θ),
                                     options=DEFAULT_OPTIM_OPTIONS)
    x0 = pack_ordered_probit_initial_I(spec)
    obj = x -> initial_I_negll_ordered_probit_fast(x, π0_marginals, r_θ, r_I; θ_grid)
    result = optimize(obj, x0, Newton(), options; autodiff=:forward)
    new_spec = unpack_ordered_probit_initial_I(Optim.minimizer(result), r_I)
    P_I_θ = initial_I_from_ordered_probit(new_spec, r_θ, r_I; θ_grid)
    return new_spec, P_I_θ
end

# ═════════════════════════════════════════════════════════════════
# 6. Initialization
# ═════════════════════════════════════════════════════════════════

"""
    initialize_ordered_probit(κ_list, r; rng) -> OrderedProbitEmission

Equally spaced cutpoints in [-1, 1], alpha ≈ 1 (positive for monotonicity).
"""
function initialize_ordered_probit(κ_list::Vector{Int}, r::Int;
                                    rng::AbstractRNG=Random.default_rng())
    M = length(κ_list)
    tau = Vector{Vector{Float64}}(undef, M)
    alpha = Vector{Float64}(undef, M)
    for m in 1:M
        n_cut = κ_list[m] - 1
        if n_cut == 1
            tau[m] = [0.0 + 0.1 * randn(rng)]
        else
            tau[m] = collect(range(-1.5, 1.5; length=n_cut)) .+ 0.05 .* randn(rng, n_cut)
        end
        alpha[m] = 0.5 + 0.1 * abs(randn(rng))
    end
    return OrderedProbitEmission(tau, alpha)
end

"""
    initialize_flexible_lpm(r_θ; rng) -> FlexibleLPMTransition
"""
function initialize_flexible_lpm(r_θ::Int;
                                  rng::AbstractRNG=Random.default_rng())
    n_cut = r_θ - 1
    # Decreasing order for survivor function cutoffs: c₁ > c₂ > ...
    c = collect(range(0.7, 0.3; length=n_cut)) .+ 0.02 .* randn(rng, n_cut)
    sort!(c; rev=true)  # ensure decreasing after noise
    gamma0 = vcat(0.0, 0.05 .* randn(rng, r_θ - 1))
    gamma1 = 0.05 .* ones(r_θ) .+ 0.01 .* randn(rng, r_θ)
    return FlexibleLPMTransition(c, gamma0, gamma1)
end

"""
    initialize_parsimonious_lpm(r_θ; rng) -> ParsimoniousLPMTransition
"""
function initialize_parsimonious_lpm(r_θ::Int;
                                      rng::AbstractRNG=Random.default_rng())
    n_cut = r_θ - 1
    # Decreasing order for survivor function cutoffs: c₁ > c₂ > ...
    c = collect(range(0.7, 0.3; length=n_cut)) .+ 0.02 .* randn(rng, n_cut)
    sort!(c; rev=true)  # ensure decreasing after noise
    return ParsimoniousLPMTransition(c,
        0.05 + 0.01 * randn(rng),
        0.05 + 0.01 * randn(rng),
        0.0)
end

"""
    initialize_ordered_probit_transition(r_θ; rng) -> OrderedProbitTransition

Equally spaced increasing cutpoints in [-1, 1], small positive β values.
"""
function initialize_ordered_probit_transition(r_θ::Int;
                                               rng::AbstractRNG=Random.default_rng())
    n_cut = r_θ - 1
    # Increasing cutpoints (CDF convention): c₁ < c₂ < ...
    c = collect(range(-1.0, 1.0; length=n_cut)) .+ 0.02 .* randn(rng, n_cut)
    sort!(c)  # ensure increasing after noise
    return OrderedProbitTransition(c,
        0.1 + 0.02 * randn(rng),
        0.1 + 0.02 * randn(rng),
        0.0)
end

"""
    initialize_mixture_probit(r_θ, K; rng, r_I=3) -> MixtureProbitTransition

Cutpoints centered at a_mean * I_mid for near-uniform starting PMF, spread-out log-slopes, near-uniform mixing.
"""
function initialize_mixture_probit(r_θ::Int, K::Int;
                                    rng::AbstractRNG=Random.default_rng(),
                                    r_I::Int=3)
    n_cut = r_θ - 1

    # Log-slopes: spread them out so components have different sensitivities
    a_raw = collect(range(-1.5, -0.5; length=K)) .+ 0.1 .* randn(rng, K)
    a = exp.(a_raw)

    # Cutpoints centered at a_mean * I_mid so that Φ(c_k - a·I) ≈ 0.5
    # at the midpoint of the I grid → near-uniform starting PMF
    I_mid = (1.0 + r_I) / 2.0
    center = sum(a) / K * I_mid
    c = collect(range(center - 0.8, center + 0.8; length=n_cut)) .+ 0.02 .* randn(rng, n_cut)
    sort!(c)
    c_raw = _pack_cutpoints(c)

    # Softmax logits: near-uniform initial mixing
    lambda = 0.05 .* randn(rng, K - 1, r_θ)

    return MixtureProbitTransition(c_raw, a_raw, lambda, K)
end

"""
    initialize_monotone_nn(r_θ, K; rng, r_I=3) -> MonotoneNNTransition

Decreasing cutpoints (survivor convention), spread-out hidden units with
varying θ and I sensitivities, near-zero output weights for gentle start.
"""
function initialize_monotone_nn(r_θ::Int, K::Int;
                                  rng::AbstractRNG=Random.default_rng(),
                                  r_I::Int=3)
    n_cut = r_θ - 1

    # Decreasing cutpoints (survivor convention): c₁ > c₂ > ...
    c = collect(range(0.5, -0.5; length=n_cut)) .+ 0.02 .* randn(rng, n_cut)
    sort!(c; rev=true)
    c_raw = _pack_survivor_cutoffs(c)

    # θ-weights: spread out so units respond differently to θ
    alpha1 = collect(range(-0.5, 0.5; length=K)) .+ 0.1 .* randn(rng, K)

    # I log-slopes: spread out, all enforce α₂ > 0 via exp
    alpha2_raw = collect(range(-1.5, -0.5; length=K)) .+ 0.1 .* randn(rng, K)

    # Biases: spread out so units activate at different (θ, I) combinations
    b = collect(range(-1.0, 1.0; length=K)) .+ 0.1 .* randn(rng, K)

    # Log-weights: small positive to start gently
    log_w = fill(-1.0, K) .+ 0.1 .* randn(rng, K)

    return MonotoneNNTransition(c_raw, alpha1, alpha2_raw, b, log_w, K)
end

"""
    initialize_robit_nn(r_θ, K; rng, r_I=3, ν=DEFAULT_ROBIT_DF) -> RobitNNTransition

Same initialization as monotone NN but returns a RobitNNTransition with
the specified degrees of freedom ν for the t-CDF outer link.
"""
function initialize_robit_nn(r_θ::Int, K::Int;
                               rng::AbstractRNG=Random.default_rng(),
                               r_I::Int=3,
                               ν::Float64=DEFAULT_ROBIT_DF)
    n_cut = r_θ - 1

    # Decreasing cutpoints (survivor convention): c₁ > c₂ > ...
    c = collect(range(0.5, -0.5; length=n_cut)) .+ 0.02 .* randn(rng, n_cut)
    sort!(c; rev=true)
    c_raw = _pack_survivor_cutoffs(c)

    # θ-weights: spread out so units respond differently to θ
    alpha1 = collect(range(-0.5, 0.5; length=K)) .+ 0.1 .* randn(rng, K)

    # I log-slopes: spread out, all enforce α₂ > 0 via exp
    alpha2_raw = collect(range(-1.5, -0.5; length=K)) .+ 0.1 .* randn(rng, K)

    # Biases: spread out so units activate at different (θ, I) combinations
    b = collect(range(-1.0, 1.0; length=K)) .+ 0.1 .* randn(rng, K)

    # Log-weights: small positive to start gently
    log_w = fill(-1.0, K) .+ 0.1 .* randn(rng, K)

    return RobitNNTransition(c_raw, alpha1, alpha2_raw, b, log_w, K, ν)
end

"""
    initialize_ordered_probit_ccp(r_I; rng) -> OrderedProbitCCP

Equally spaced increasing cutpoints in [-1, 1], small positive β values.
"""
function initialize_ordered_probit_ccp(r_I::Int;
                                        rng::AbstractRNG=Random.default_rng())
    n_cut = r_I - 1
    # Increasing cutpoints (CDF convention): c₁ < c₂ < ...
    c = collect(range(-1.0, 1.0; length=n_cut)) .+ 0.02 .* randn(rng, n_cut)
    sort!(c)  # ensure increasing after noise
    return OrderedProbitCCP(c,
        0.1 + 0.02 * randn(rng),
        0.1 + 0.02 * randn(rng),
        0.0)
end

"""
    initialize_ordered_probit_ccp_simple(r_I; rng) -> OrderedProbitCCPSimple

Equally spaced increasing cutpoints in [-1, 1], small positive β₁.
"""
function initialize_ordered_probit_ccp_simple(r_I::Int;
                                               rng::AbstractRNG=Random.default_rng())
    n_cut = r_I - 1
    c = collect(range(-1.0, 1.0; length=n_cut)) .+ 0.02 .* randn(rng, n_cut)
    sort!(c)
    return OrderedProbitCCPSimple(c, 0.1 + 0.02 * randn(rng))
end

"""
    initialize_ordered_probit_initial_I(r_I; rng) -> OrderedProbitInitialI

Equally spaced increasing cutpoints in [-1, 1], small positive β₁.
"""
function initialize_ordered_probit_initial_I(r_I::Int;
                                              rng::AbstractRNG=Random.default_rng())
    n_cut = r_I - 1
    c = collect(range(-1.0, 1.0; length=n_cut)) .+ 0.02 .* randn(rng, n_cut)
    sort!(c)
    return OrderedProbitInitialI(c, 0.1 + 0.02 * randn(rng))
end

