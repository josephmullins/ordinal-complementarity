using LinearAlgebra, Distributions, Random

# Full pairwise supermodularity-in-distribution constraint matrix (all (θ,I) and
# cutpoint pairs).  Retained as a utility / for full-vs-minimal comparisons; the
# tests use the minimal `get_S_adjacent` (same null, full rank).
function get_S(r_θ, r_I)
    p_idx = LinearIndices((r_θ-1, r_θ, r_I))
    np = prod(size(p_idx))
    num_rows = Int(r_θ * r_I * (r_θ-1) * (r_I-1) * (r_θ-1) / 4)
    S = zeros(num_rows, np)
    r_index = 1
    for kθ in 1:r_θ, kI in 1:r_I
        for kθ2 in kθ+1:r_θ, kI2 in kI+1:r_I
            for kθ3 in 1:r_θ-1
                S[r_index, p_idx[1,kθ,kI]:p_idx[kθ3,kθ,kI]] .= 1.
                S[r_index, p_idx[1,kθ2,kI2]:p_idx[kθ3,kθ2,kI2]] .= 1.
                S[r_index, p_idx[1,kθ,kI2]:p_idx[kθ3,kθ,kI2]] .= -1.
                S[r_index, p_idx[1,kθ2,kI]:p_idx[kθ3,kθ2,kI]] .= -1.
                r_index += 1
            end
        end
    end
    return S
end

"""
    get_test_stat(Q, p_est, data) -> (; T_n, Ω, Σ_score)

Compute the supermodularity test statistic, normalized covariance Ω,
and score covariance Σ_score. Regularizes Ω if it has non-positive eigenvalues.
"""
function get_test_stat(Q, p_est, data; α::Float64=0.05, rng::AbstractRNG=Random.default_rng())
    r_θ, _, r_I, N = size(Q)
    S = get_S_adjacent(r_θ, r_I)

    expectation_step!(Q, p_est, data)
    dL = get_score(Q, p_est.Π)
    Σ_score = cov(dL')
    Σ_inv = inv(Σ_score)
    V = S * Σ_inv * S'                                   # ∝ Var(δ̂); score path ⇒ scale=√N
    Π_vec = p_est.Π[1:end-1, :, :][:]
    r = mmm_test(S * Π_vec, V; scale = sqrt(N), α, rng)
    return (; r.T_n, r.Ω, Σ_score)
end

function crit_value(Ω, α; nsim = 100_000, rng::AbstractRNG=Random.default_rng())
    L = cholesky(Hermitian(Ω)).L
    d = size(Ω, 1)

    T = zeros(nsim)
    z = zeros(d)
    Lz = zeros(d)

    for r in 1:nsim
        randn!(rng, z)
        mul!(Lz, L, z)
        T[r] = sum(x -> max(0.0, x)^2, Lz)   # least-favorable null of the MMM statistic
    end

    return quantile(T, 1 - α)
end

"""
    crit_value_sup(Ω, α; nsim, rng) -> (c_super, c_sub)

Least-favourable critical values for the DIRECTED (sup-type) twin of the MMM
statistic, simulated from ζ ~ N(0,Ω) using the SAME Cholesky-of-Ω draws as
`crit_value`, but aggregating by the single worst coordinate instead of a
sum-of-squares:

  • `c_super` = (1-α) quantile of  (min(0, minimum(Lζ)))²   [complementarity side]
  • `c_sub`   = (1-α) quantile of  (max(0, maximum(Lζ)))²   [substitutability side]

Reuse `rng` for reproducibility. All moments treated as binding (no GMS).
"""
function crit_value_sup(Ω, α; nsim = 100_000, rng::AbstractRNG=Random.default_rng())
    L = cholesky(Hermitian(Ω)).L
    d = size(Ω, 1)

    Tsuper = zeros(nsim)   # (min(0, min_j Lζ_j))²
    Tsub   = zeros(nsim)   # (max(0, max_j Lζ_j))²
    z  = zeros(d)
    Lz = zeros(d)

    for r in 1:nsim
        randn!(rng, z)
        mul!(Lz, L, z)
        mn = minimum(Lz); mx = maximum(Lz)
        Tsuper[r] = min(0.0, mn)^2
        Tsub[r]   = max(0.0, mx)^2
    end

    return (c_super = quantile(Tsuper, 1 - α), c_sub = quantile(Tsub, 1 - α))
end

# ─────────────────────────────────────────────────────────────────
# Romano–Shaikh–Wolf (2014) two-step moment-selection critical value.
# Survival convention: for :super the null is m ≥ 0 (penalize the negative
# part), for :sub the null is m ≤ 0 (penalize the positive part).
# `z` is the √N-standardized moment vector; Ω its correlation matrix.
#   κ_β  = (1-β)-quantile of  max_j Z_j,  Z ~ N(0,Ω)   [correlation-aware slack]
#   μ̂_β = (z - κ_β) ∨ 0   (:super)   /   (z + κ_β) ∧ 0   (:sub)
#   crit = (1-(α-β))-quantile of  ‖W ∧ 0‖²  (:super) with W ~ N(μ̂_β, Ω)
# i.e. the centre is each moment's (1-β) simultaneous lower (:super) / upper
# (:sub) confidence bound truncated at 0: moments whose bound lies strictly
# inside the null are "slack" and are centred there, all others at 0 as in the
# least-favourable null.  Shifting a centre into the null can only shrink the
# simulated statistic, so crit ≤ c_lf, the LF crit at the same level 1-(α-β).
# If every moment is slack (μ̂_β strictly interior) c = Inf
# (auto fail-to-reject), matching RSW's (1 - 1{μ̂_β interior}) factor.
#
# Returns (; c, c_lf, κβ, n_slack).  The κ_β step, the RSW simulation and c_lf
# all use the same draws (common random numbers), so c ≤ c_lf holds exactly in
# simulation and any gap to the 1-α LF crit is the β level adjustment.
# ─────────────────────────────────────────────────────────────────
function crit_value_rsw(Ω, z::AbstractVector, α; direction::Symbol = :super,
                        β::Real = α / 10, nsim = 200_000,
                        rng::AbstractRNG = Random.default_rng())
    d = size(Ω, 1)
    L = cholesky(Hermitian(Ω)).L
    ζ = zeros(d); Lζ = zeros(d)
    # κ_β : simulated (1-β)-quantile of max_j Z_j
    rng0 = copy(rng)
    mx = zeros(nsim)
    for r in 1:nsim
        randn!(rng, ζ); mul!(Lζ, L, ζ)
        mx[r] = maximum(Lζ)
    end
    κβ = quantile(mx, 1 - β)
    if direction === :super
        μ = max.(z .- κβ, 0.0)
        n_slack = count(>(0.0), μ)
    else
        μ = min.(z .+ κβ, 0.0)
        n_slack = count(<(0.0), μ)
    end
    T = zeros(nsim); Tlf = zeros(nsim)
    for r in 1:nsim
        randn!(rng0, ζ); mul!(Lζ, L, ζ)
        s = 0.0; s0 = 0.0
        @inbounds for j in 1:d
            w = μ[j] + Lζ[j]
            if direction === :super
                s += min(0.0, w)^2; s0 += min(0.0, Lζ[j])^2
            else
                s += max(0.0, w)^2; s0 += max(0.0, Lζ[j])^2
            end
        end
        T[r] = s; Tlf[r] = s0
    end
    c_lf = quantile(Tlf, 1 - (α - β))
    c = n_slack == d ? Inf : quantile(T, 1 - (α - β))
    return (; c, c_lf, κβ, n_slack)
end

# ─────────────────────────────────────────────────────────────────
# Supermodularity-in-distribution (ordinal-complementarity) test.
#
# OC is  F(y|x∨y) + F(y|x∧y) ≤ F(y|x) + F(y|x'),  i.e.  δ = S·Π ≤ 0, where S·Π is
# the CDF diff-in-difference built by `get_S_adjacent`.  A VIOLATION is δ > 0, so
# the modified-method-of-moments statistic penalizes the POSITIVE part of δ̂,
# T = Σ max(0, δ̂/se)²  (the draft's ‖Δ ∨ 0‖²); rejection ⇒ OC violated.
#
# `get_S_adjacent` is the minimal generating set of inequalities — the adjacent
# unit-cell cross-differences — which defines the same null as the full pairwise
# `get_S` but is full rank (no rank-deficient Ω) and more powerful.  All
# `get_test_stat*` use it.  `mmm_test` drops degenerate (≈constant) inequalities.
# ─────────────────────────────────────────────────────────────────

"""
    get_S_adjacent(r_θ, r_I) -> S

Minimal supermodularity-in-distribution constraint matrix: one row per
(adjacent skill increment θ→θ+1, adjacent investment increment I→I+1, cutpoint
k).  Same CDF diff-in-difference sign convention as `get_S` (so `S·Π` is the CDF
DID and OC ⟺ `S·Π ≤ 0`), restricted to consecutive cells.  `(r_θ-1)(r_I-1)(r_θ-1)`
rows vs `r_θ(r_θ-1)²r_I(r_I-1)/4` for `get_S`.
"""
function get_S_adjacent(r_θ, r_I)
    p_idx = LinearIndices((r_θ-1, r_θ, r_I))
    ncut  = r_θ - 1
    nrow  = (r_θ-1) * (r_I-1) * ncut
    S = zeros(nrow, prod(size(p_idx)))
    r_index = 1
    for kθ in 1:(r_θ-1), kI in 1:(r_I-1)
        for k in 1:ncut
            S[r_index, p_idx[1, kθ,   kI  ]:p_idx[k, kθ,   kI  ]] .=  1.0
            S[r_index, p_idx[1, kθ+1, kI+1]:p_idx[k, kθ+1, kI+1]] .=  1.0
            S[r_index, p_idx[1, kθ,   kI+1]:p_idx[k, kθ,   kI+1]] .= -1.0
            S[r_index, p_idx[1, kθ+1, kI  ]:p_idx[k, kθ+1, kI  ]] .= -1.0
            r_index += 1
        end
    end
    return S
end

"""
    mmm_test(δ̂, V; scale=1.0, α=0.05, rng) -> (; T_n, c_α, reject, Ω, n_used)

Modified-method-of-moments test of `H₀: δ ≤ 0` for the inequality estimates
`δ̂ = S·Π̂`.  `T = Σ max(0, stat)²` with `stat = scale · diag(V)^{-1/2} · δ̂`;
rejection ⇒ OC violated.

`V` is proportional to `Var(δ̂)`; `scale` makes `stat` have unit variance — pass
`scale = √N` when `V = S Σ̂⁻¹ S'` (per-observation score covariance) and
`scale = 1` when `V = S Var(Π̂_free) S'` (delta-method covariance, already O(1/N)).
Degenerate (≈constant) inequalities are dropped before standardizing.
"""
function mmm_test(δ̂::AbstractVector, V::AbstractMatrix;
                  scale::Real=1.0, α::Real=0.05, direction::Symbol=:upper,
                  rng::AbstractRNG=Random.default_rng())
    dV   = diag(V)
    tol  = 1e-10 * max(maximum(dV), 1.0)
    keep = findall(>(tol), dV)
    isempty(keep) && return (; T_n=0.0, c_α=0.0, reject=false, Ω=zeros(0, 0), n_used=0)

    δk = δ̂[keep]; Vk = V[keep, keep]; dk = dV[keep]
    D_inv = diagm(1.0 ./ sqrt.(dk))
    Ω = Matrix(Symmetric(D_inv * Vk * D_inv))
    mn = minimum(eigvals(Symmetric(Ω)))
    mn ≤ 0 && (Ω += (abs(mn) + 1e-6) * I)

    stat = (scale .* D_inv) * δk
    # :upper tests H0: δ ≤ 0 (penalize positive part — the OC/complementarity test);
    # :lower tests H0: δ ≥ 0 (penalize negative part — the substitutability test).
    # The least-favorable null is symmetric, so the same `crit_value` serves both.
    T_n  = direction === :lower ? sum(min.(0.0, stat) .^ 2) : sum(max.(0.0, stat) .^ 2)
    c_α  = crit_value(Hermitian(Ω), α; rng = rng)
    return (; T_n, c_α, reject = T_n > c_α, Ω, n_used = length(keep))
end

function monte_carlo_simulation(p; N = 1_000, nboot = 500, seed0 = 102025)
    rng = MersenneTwister(seed0)
    r_θ = p.r_θ
    r_I = p.r_I
    np = (r_θ - 1) * r_θ * r_I
    Q = zeros(r_θ, r_θ, r_I, N)
    Πboot = zeros(np, nboot)
    Tboot = zeros(nboot)
    Reject = zeros(nboot)
    cboot = zeros(nboot)
    for b in 1:nboot
        data_sim = simulate_data(N, p, rng)
        p_boot, _ = expectation_maximization(data_sim, p;
            verbose=false, shared_skill_emission=true)
        Πboot[:,b] = p_boot.Π[1:end-1,:,:][:]
        result = get_test_stat(Q, p_boot, data_sim)
        Tboot[b] = result.T_n
        c_α = crit_value(Hermitian(result.Ω), 0.05; rng=rng)
        Reject[b] = Tboot[b] > c_α
        cboot[b] = c_α
    end
    return mean(Reject), Tboot, cboot, Πboot
end

function monte_carlo_simulation_threaded(p; N = 1_000, nboot = 500, seed0 = 102025)
    r_θ = p.r_θ
    r_I = p.r_I
    np = (r_θ - 1) * r_θ * r_I
    Πboot = zeros(np, nboot)
    Tboot = zeros(nboot)
    Reject = zeros(nboot)
    cboot = zeros(nboot)
    chunks = Iterators.partition(1:nboot, cld(nboot, Threads.nthreads()))
    tasks = map(enumerate(chunks)) do (ch_id, chunk)
        seedb = seed0 + ch_id
        Threads.@spawn begin
            try
                monte_carlo_simulation_chunk(chunk, p, N, seedb)
            catch e
                @error "Task failed" exception=(e, catch_backtrace())
                rethrow(e)
            end
        end
    end
    results = fetch.(tasks)

    for (chunk, result) in zip(chunks, results)
        Πboot_local, Tboot_local, cboot_local, Reject_local = result
        for (idx, b) in enumerate(chunk)
            Πboot[:,b] = Πboot_local[:,idx]
            Tboot[b] = Tboot_local[idx]
            cboot[b] = cboot_local[idx]
            Reject[b] = Reject_local[idx]
        end
    end

    return mean(Reject), Tboot, cboot, Πboot
end

function monte_carlo_simulation_chunk(chunk, p, N, seed0)
    rng = MersenneTwister(seed0)
    r_θ = p.r_θ
    r_I = p.r_I
    np = (r_θ - 1) * r_θ * r_I
    chunk_size = length(chunk)

    Πboot_local = zeros(np, chunk_size)
    Tboot_local = zeros(chunk_size)
    cboot_local = zeros(chunk_size)
    Reject_local = zeros(chunk_size)
    Q = zeros(r_θ, r_θ, r_I, N)

    for (idx, b) in enumerate(chunk)
        data_sim = simulate_data(N, p, rng)
        p_boot, _ = expectation_maximization(data_sim, p;
            verbose=false, shared_skill_emission=true)
        Πboot_local[:,idx] = p_boot.Π[1:end-1,:,:][:]
        result = get_test_stat(Q, p_boot, data_sim)
        Tboot_local[idx] = result.T_n
        c_α = crit_value(Hermitian(result.Ω), 0.05; rng=rng)
        Reject_local[idx] = Tboot_local[idx] > c_α
        cboot_local[idx] = c_α
    end

    return Πboot_local, Tboot_local, cboot_local, Reject_local
end

# ─────────────────────────────────────────────────────────────────
# Parametric inference
# ─────────────────────────────────────────────────────────────────
#
# For parametric transitions, the score is computed via chain rule:
#   dL/dφ = J' · dL/dΠ_vec
# where J = dΠ_vec/dφ is the Jacobian of the parametric mapping.
#
# Requires: include("parametric_models.jl") before this file.
# ─────────────────────────────────────────────────────────────────

"""
    get_score_parametric(Q, Π, spec, config) -> Matrix{Float64}

Score matrix (n_params × N) w.r.t. the transition parameters.
For nonparametric: delegates to `get_score(Q, Π)`.
For parametric: applies chain rule through the Jacobian.
"""
function get_score_parametric(Q, Π, specs::Dict{Symbol,Any}, config::NamedTuple)
    if config.transition == :nonparametric
        return get_score(Q, Π)
    end

    r_θ, _, r_I = size(Π)
    N = size(Q, 4)

    # Score w.r.t. ALL Π elements: dL/dΠ[k,θ,I] = Q[k,θ,I,n] / Π[k,θ,I]
    n_pi = r_θ * r_θ * r_I
    dL_dΠ = zeros(n_pi, N)
    pi_idx = LinearIndices((r_θ, r_θ, r_I))
    for n in 1:N, kI in 1:r_I, kθ in 1:r_θ, k in 1:r_θ
        ip = pi_idx[k, kθ, kI]
        dL_dΠ[ip, n] = Q[k, kθ, kI, n] / max(Π[k, kθ, kI], 1e-15)
    end

    # Jacobian of φ → Π_vec via ForwardDiff
    # Use _build_*_pi functions (not struct-based unpack) for ForwardDiff compatibility
    spec = specs[:transition]
    if config.transition == :flexible_lpm
        φ = pack_flexible_lpm(spec)
        pi_vec_flex(φ_vec) = vec(_build_flexible_lpm_pi(φ_vec, r_θ, r_I))
        J = ForwardDiff.jacobian(pi_vec_flex, φ)
    elseif config.transition == :parsimonious_lpm
        φ = pack_parsimonious_lpm(spec)
        pi_vec_pars(φ_vec) = vec(_build_parsimonious_lpm_pi(φ_vec, r_θ, r_I))
        J = ForwardDiff.jacobian(pi_vec_pars, φ)
    elseif config.transition == :ordered_probit
        φ = pack_ordered_probit_transition(spec)
        pi_vec_op(φ_vec) = vec(_build_ordered_probit_pi(φ_vec, r_θ, r_I))
        J = ForwardDiff.jacobian(pi_vec_op, φ)
    elseif config.transition == :mixture_probit
        φ = pack_mixture_probit(spec)
        K = spec.K
        pi_vec_mp(φ_vec) = vec(_build_mixture_probit_pi(φ_vec, r_θ, r_I, K))
        J = ForwardDiff.jacobian(pi_vec_mp, φ)
    end

    # Chain rule: dL/dφ = J' · dL/dΠ_vec
    return J' * dL_dΠ  # (n_φ × N)
end

"""
    get_test_stat_parametric(Q, p_est, data, specs, config) -> NamedTuple

Compute the supermodularity test statistic using parametric score covariance.
The S matrix operates on Π-space regardless of parameterization.
"""
function get_test_stat_parametric(Q, p_est, data, specs::Dict{Symbol,Any},
                                   config::NamedTuple; α::Float64=0.05, rng::AbstractRNG=Random.default_rng())
    r_θ, _, r_I, N = size(Q)
    S = get_S_adjacent(r_θ, r_I)

    expectation_step!(Q, p_est, data)

    # Score w.r.t. the free parameters of the nonparametric Π (S is in this space)
    dL_np = get_score(Q, p_est.Π)
    Σ_score_np = cov(dL_np')
    Σ_inv_np = inv(Σ_score_np)
    V = S * Σ_inv_np * S'                                # score path ⇒ scale=√N
    Π_vec = p_est.Π[1:end-1, :, :][:]
    r = mmm_test(S * Π_vec, V; scale = sqrt(N), α, rng)
    T_n = r.T_n; Ω = r.Ω

    # Also compute parametric score if applicable
    if config.transition != :nonparametric
        dL_param = get_score_parametric(Q, p_est.Π, specs, config)
        Σ_score_param = cov(dL_param')
        return (; T_n, Ω, Σ_score=Σ_score_np, Σ_score_param)
    else
        return (; T_n, Ω, Σ_score=Σ_score_np)
    end
end

"""
    pi_free_var_nonparametric(Σ_score, N; boundary_tol=1e-6) -> Matrix{Float64}

Compute Var(Π̂_free) from the nonparametric score covariance via the
information matrix equality: Var(Π̂_free) = Σ_score⁻¹ / N.

Drops parameters whose score variance is near zero (boundary parameters)
before inverting, then maps back to the full (n_free × n_free) matrix
with zeros for dropped parameters.
"""
function pi_free_var_nonparametric(Σ_score::AbstractMatrix, N::Int; boundary_tol=1e-6)
    np = size(Σ_score, 1)

    # Identify parameters with meaningful score variance
    good = [diag(Σ_score)[i] > boundary_tol for i in 1:np]
    idx_good = findall(good)

    Var_free = zeros(np, np)
    if isempty(idx_good)
        @warn "pi_free_var_nonparametric: all score variances near zero; returning zero Var"
        return Var_free
    end

    Σ_red = Σ_score[idx_good, idx_good]
    r = rank(Σ_red)
    if r < length(idx_good)
        @warn "pi_free_var_nonparametric: reduced score covariance has rank $r / $(length(idx_good)); using pinv"
        Σ_red_inv = pinv(Σ_red)
    else
        Σ_red_inv = inv(Σ_red)
    end
    Var_red = Σ_red_inv / N

    for (i, gi) in enumerate(idx_good), (j, gj) in enumerate(idx_good)
        Var_free[gi, gj] = Var_red[i, j]
    end

    return Var_free
end

"""
    pi_free_var_parametric(spec, config_transition, Σ_score_param, r_θ, r_I, N) -> Matrix{Float64}

Compute Var(Π̂_free) by the delta method through the parametric mapping φ → Π:
    Var(Π̂_free) = J · Var(φ̂) · J'

where J = ∂Π_free/∂φ and Var(φ̂) = Σ_score_param⁻¹ / N.

`spec` is the transition specification (e.g., OrderedProbitTransition).
`config_transition` is a Symbol (:ordered_probit, :parsimonious_lpm, :flexible_lpm).

Returns: (n_free × n_free) matrix where n_free = (r_θ-1) × r_θ × r_I.
"""
function pi_free_var_parametric(spec, config_transition::Symbol,
                                 Σ_score_param::AbstractMatrix,
                                 r_θ::Int, r_I::Int, N::Int)
    # Helper: extract free parameters of Π (first r_θ-1 rows) as flat vector
    function _pi_free_vec(Π::AbstractArray{T,3}) where T
        v = Vector{T}(undef, (r_θ - 1) * r_θ * r_I)
        idx = 0
        for kI in 1:r_I, kθ in 1:r_θ, j in 1:(r_θ - 1)
            idx += 1
            v[idx] = Π[j, kθ, kI]
        end
        return v
    end

    # Build φ → Π_free mapping
    if config_transition == :parsimonious_lpm
        φ = pack_parsimonious_lpm(spec::ParsimoniousLPMTransition)
        f = φ_vec -> _pi_free_vec(_build_parsimonious_lpm_pi(φ_vec, r_θ, r_I))
    elseif config_transition == :flexible_lpm
        φ = pack_flexible_lpm(spec::FlexibleLPMTransition)
        f = φ_vec -> _pi_free_vec(_build_flexible_lpm_pi(φ_vec, r_θ, r_I))
    elseif config_transition == :ordered_probit
        φ = pack_ordered_probit_transition(spec::OrderedProbitTransition)
        f = φ_vec -> _pi_free_vec(_build_ordered_probit_pi(φ_vec, r_θ, r_I))
    elseif config_transition == :mixture_probit
        spec_mp = spec::MixtureProbitTransition
        φ = pack_mixture_probit(spec_mp)
        K = spec_mp.K
        f = φ_vec -> _pi_free_vec(_build_mixture_probit_pi(φ_vec, r_θ, r_I, K))
    else
        error("pi_free_var_parametric requires :parsimonious_lpm, :flexible_lpm, :ordered_probit, or :mixture_probit")
    end

    J = ForwardDiff.jacobian(f, φ)  # (n_free × n_φ)

    # Invert parametric score covariance
    Var_φ = inv(Σ_score_param) / N

    return J * Var_φ * J'
end

"""
    survivor_se_parametric(specs, config, Σ_score_param, r_θ, r_I, N) -> Array{Float64,3}

Compute standard errors for survivor function values S̄(k|θ,I) = P(θ'>k|θ,I)
using the delta method through the parametric transition mapping φ → Π → S̄.

Returns: SE array of size (r_θ-1, r_θ, r_I) indexed by (cutoff k, θ, I).
"""
function survivor_se_parametric(specs::Dict{Symbol,Any}, config::NamedTuple,
                                 Σ_score_param::AbstractMatrix,
                                 r_θ::Int, r_I::Int, N::Int)
    Var_free = pi_free_var_parametric(specs[:transition], config.transition,
                                       Σ_score_param, r_θ, r_I, N)
    # Build Π from specs to pass to survivor_se_all
    spec = specs[:transition]
    if config.transition == :parsimonious_lpm
        φ = pack_parsimonious_lpm(spec::ParsimoniousLPMTransition)
        Π = _build_parsimonious_lpm_pi(φ, r_θ, r_I)
    elseif config.transition == :flexible_lpm
        φ = pack_flexible_lpm(spec::FlexibleLPMTransition)
        Π = _build_flexible_lpm_pi(φ, r_θ, r_I)
    elseif config.transition == :ordered_probit
        φ = pack_ordered_probit_transition(spec::OrderedProbitTransition)
        Π = _build_ordered_probit_pi(φ, r_θ, r_I)
    elseif config.transition == :mixture_probit
        spec_mp = spec::MixtureProbitTransition
        φ = pack_mixture_probit(spec_mp)
        Π = _build_mixture_probit_pi(φ, r_θ, r_I, spec_mp.K)
    end
    return survivor_se_all(Var_free, Π)
end

"""
    survivor_se_nonparametric(Q, Π, N) -> Array{Float64,3}

Compute standard errors for survivor function values S̄(k|θ,I) = P(θ'>k|θ,I)
directly from the nonparametric score covariance of the free parameters of Π.

Returns: SE array of size (r_θ-1, r_θ, r_I) indexed by (cutoff k, θ, I).
"""
function survivor_se_nonparametric(Q::Array{Float64,4}, Π::Array{Float64,3}, N::Int)
    dL = get_score(Q, Π)
    Σ_score = cov(dL')
    Var_free = pi_free_var_nonparametric(Σ_score, N)
    return survivor_se_all(Var_free, Π)
end

"""
    test_complementarity_parsimonious(specs, Σ_score_param, N) -> NamedTuple

One-sided Wald test for complementarity under parsimonious LPM (Model B).
H₀: γ₃ ≥ 0 (complementarity holds).
H₁: γ₃ < 0 (complementarity violated).

Returns: (; gamma3, se, z_stat, p_value, reject_05)
"""
function test_complementarity_parsimonious(specs::Dict{Symbol,Any},
                                            Σ_score_param::AbstractMatrix,
                                            N::Int)
    spec = specs[:transition]::ParsimoniousLPMTransition
    r_θ = length(spec.c) + 1

    # γ₃ is the last parameter in the packed vector
    φ = pack_parsimonious_lpm(spec)
    n_params = length(φ)
    idx_gamma3 = n_params  # last element

    # Asymptotic variance of φ̂: (1/N) · H⁻¹ · Σ_score · H⁻¹'
    # Under correct specification, Σ_score ≈ H, so Var(φ̂) ≈ Σ_score⁻¹ / N
    r = rank(Σ_score_param)
    if r < n_params
        @warn "Parametric score covariance has rank $r / $n_params; Wald test SEs may be unreliable"
        Σ_inv = pinv(Σ_score_param)
    else
        Σ_inv = inv(Σ_score_param)
    end
    se_gamma3 = sqrt(max(Σ_inv[idx_gamma3, idx_gamma3] / N, 1e-15))

    z_stat = spec.gamma3 / se_gamma3
    # One-sided: reject H₀ (complementarity) if z_stat << 0
    p_value = cdf(Normal(), z_stat)

    return (; gamma3=spec.gamma3, se=se_gamma3, z_stat, p_value,
              reject_05 = p_value < 0.05)
end

"""
    test_complementarity_flexible(specs, Σ_score_param, N) -> NamedTuple

Test complementarity under flexible LPM (Model A).
Complementarity ⟺ γ₁(θ) non-decreasing in θ.
Tests each consecutive difference γ₁(θ+1) - γ₁(θ) ≥ 0.

Returns: (; gamma1, diffs, se_diffs, z_stats, p_values, reject_any_05)
"""
function test_complementarity_flexible(specs::Dict{Symbol,Any},
                                        Σ_score_param::AbstractMatrix,
                                        N::Int)
    spec = specs[:transition]::FlexibleLPMTransition
    r_θ = length(spec.gamma1)

    φ = pack_flexible_lpm(spec)
    n_params = length(φ)

    # γ₁ are the last r_θ elements of the packed vector
    idx_gamma1 = (n_params - r_θ + 1):n_params

    r = rank(Σ_score_param)
    if r < n_params
        @warn "Parametric score covariance has rank $r / $n_params; flexible LPM test SEs may be unreliable"
        Σ_inv = pinv(Σ_score_param)
    else
        Σ_inv = inv(Σ_score_param)
    end
    Var_φ = Σ_inv / N

    # Test γ₁(θ+1) - γ₁(θ) ≥ 0 for θ = 1,...,r_θ-1
    n_diffs = r_θ - 1
    diffs = [spec.gamma1[θ+1] - spec.gamma1[θ] for θ in 1:n_diffs]
    se_diffs = Float64[]
    z_stats = Float64[]
    p_values = Float64[]

    for θ in 1:n_diffs
        i1 = idx_gamma1[θ]
        i2 = idx_gamma1[θ+1]
        var_diff = Var_φ[i2, i2] + Var_φ[i1, i1] - 2 * Var_φ[i1, i2]
        se = sqrt(max(var_diff, 1e-15))
        z = diffs[θ] / se
        push!(se_diffs, se)
        push!(z_stats, z)
        push!(p_values, cdf(Normal(), z))
    end

    return (; gamma1=spec.gamma1, diffs, se_diffs, z_stats, p_values,
              reject_any_05 = any(p_values .< 0.05))
end

# ─────────────────────────────────────────────────────────────────
# Multi-period inference
# ─────────────────────────────────────────────────────────────────

"""
    get_test_stat_mp_parametric(Var_free_list, params, data; α=0.05) -> Vector{NamedTuple}

Per-period supermodularity test using the parametric variance of Π̂_free.

Instead of inverting the nonparametric score covariance (which can be rank-deficient
when the transition model is parametric), this uses `Var_free = J Var(φ̂) J'` from
`pi_free_var_parametric` to form the variance of the inequality vector directly:
    V = S · Var_free · S'

This correctly propagates estimation uncertainty through the parametric mapping
and avoids the power loss caused by rank-deficient nonparametric score inversion.

Arguments:
  - Var_free_list: Vector of Var(Π̂_free) matrices, one per transition (from pi_free_var_parametric)
  - params: estimated parameters (for Π values)
  - data: data NamedTuple with r_θ, r_I, T, N

Returns a vector of T-1 NamedTuples, each containing:
  - t: transition index
  - T_n: test statistic
  - c_α: critical value at level α
  - reject: whether we reject H₀ (supermodularity)
  - Ω: normalized covariance matrix
"""
function get_test_stat_mp_parametric(Var_free_list::Vector{Matrix{Float64}},
                                     params, data; α::Float64=0.05, rng::AbstractRNG=Random.default_rng())
    (; r_θ, r_I, T, N) = data
    S = get_S_adjacent(r_θ, r_I)

    results = Vector{NamedTuple}(undef, T-1)

    for t in 1:T-1
        Π_t = copy(params.Π[t])
        Π_t .= max.(Π_t, 1e-15)
        for kI in 1:r_I, kθ in 1:r_θ
            Π_t[:, kθ, kI] ./= sum(Π_t[:, kθ, kI])
        end

        Var_free = Var_free_list[t]
        Π_vec = Π_t[1:end-1, :, :][:]

        V = S * Var_free * S'                            # delta-method cov ⇒ scale=1
        r = mmm_test(S * Π_vec, V; scale = 1.0, α, rng)
        results[t] = (; t, r.T_n, r.c_α, r.reject, r.Ω)
    end

    return results
end

"""
    get_test_stat_mp(ψ_all, params, data; α=0.05) -> Vector{NamedTuple}

Per-period supermodularity test for the multi-period model (nonparametric variance).

For each transition t = 1:T-1, constructs Q_t from ψ_all[t,:,:,:,:]
(the per-observation pairwise posteriors marginalized over I'),
computes the score of the observed-data log-likelihood w.r.t. Π_t,
then applies the same test statistic as `get_test_stat`.

NOTE: When the transition model is parametric, prefer `get_test_stat_mp_parametric`
which uses the correct parametric variance and avoids rank-deficiency issues.

Returns a vector of T-1 NamedTuples, each containing:
  - t: transition index
  - T_n: test statistic
  - c_α: critical value at level α
  - reject: whether we reject H₀ (supermodularity)
  - Ω: normalized covariance matrix
  - Σ_score: score covariance matrix
"""
function get_test_stat_mp(ψ_all::Array{Float64,5}, params, data; α::Float64=0.05,
                          rng::AbstractRNG=Random.default_rng())
    (; r_θ, r_I, T, N) = data
    S = get_S_adjacent(r_θ, r_I)

    results = Vector{NamedTuple}(undef, T-1)

    for t in 1:T-1
        # Q_t[kθ1, kθ, kI, n] = ψ_all[t, kθ1, kθ, kI, n]
        Q_t = ψ_all[t, :, :, :, :]

        # Floor Π to avoid 0/0 or x/0 in score computation
        Π_t = copy(params.Π[t])
        Π_t .= max.(Π_t, 1e-15)
        for kI in 1:r_I, kθ in 1:r_θ
            Π_t[:, kθ, kI] ./= sum(Π_t[:, kθ, kI])
        end

        # Score w.r.t. free parameters of Π_t
        dL = get_score(Q_t, Π_t)
        Σ_score = cov(dL')

        # Check for NaN/Inf in score covariance (degenerate transitions)
        if any(!isfinite, Σ_score)
            @warn "Period $t: score covariance contains NaN/Inf; skipping test"
            Ω_dummy = Matrix{Float64}(I, size(Σ_score))
            results[t] = (; t, T_n=NaN, c_α=NaN, reject=false, Ω=Ω_dummy, Σ_score)
            continue
        end

        # Check rank and invert
        r = rank(Σ_score)
        n_free = size(Σ_score, 1)
        if r < n_free
            @warn "Period $t: score covariance has rank $r / $n_free; using pinv"
            Σ_inv = pinv(Σ_score)
        else
            Σ_inv = inv(Σ_score)
        end

        V = S * Σ_inv * S'                               # score path ⇒ scale=√N
        Π_vec = Π_t[1:end-1, :, :][:]
        mt = mmm_test(S * Π_vec, V; scale = sqrt(N), α, rng)
        results[t] = (; t, mt.T_n, mt.c_α, mt.reject, mt.Ω, Σ_score)
    end

    return results
end

# ─────────────────────────────────────────────────────────────────
# Grouped tests: pool scores across transitions that share the same Π
# ─────────────────────────────────────────────────────────────────

"""
    get_test_stat_mp_grouped(ψ_all, params, data, transition_groups; α=0.05)

Supermodularity test with scores pooled across transitions within each group.
When Π is shared within a group, the total score is the sum of per-transition
scores. Returns one test result per group.
"""
function get_test_stat_mp_grouped(ψ_all::Array{Float64,5}, params, data,
                                  transition_groups::Vector{Vector{Int}};
                                  α::Float64=0.05, rng::AbstractRNG=Random.default_rng())
    (; r_θ, r_I, T, N) = data
    S = get_S_adjacent(r_θ, r_I)

    results = Vector{NamedTuple}(undef, length(transition_groups))

    for (g, group) in enumerate(transition_groups)
        t_rep = group[1]

        # Floor Π (shared within group, so use representative)
        Π_g = copy(params.Π[t_rep])
        Π_g .= max.(Π_g, 1e-15)
        for kI in 1:r_I, kθ in 1:r_θ
            Π_g[:, kθ, kI] ./= sum(Π_g[:, kθ, kI])
        end

        # Pool scores: sum across transitions in group
        dL_pooled = zeros(size(get_score(ψ_all[group[1], :, :, :, :], Π_g)))
        for t in group
            Q_t = ψ_all[t, :, :, :, :]
            dL_pooled .+= get_score(Q_t, Π_g)
        end

        Σ_score = cov(dL_pooled')

        if any(!isfinite, Σ_score)
            @warn "Group $g: score covariance contains NaN/Inf; skipping test"
            Ω_dummy = Matrix{Float64}(I, size(Σ_score))
            results[g] = (; group, T_n=NaN, c_α=NaN, reject=false, Ω=Ω_dummy, Σ_score)
            continue
        end

        r = rank(Σ_score)
        n_free = size(Σ_score, 1)
        if r < n_free
            @warn "Group $g: score covariance has rank $r / $n_free; using pinv"
            Σ_inv = pinv(Σ_score)
        else
            Σ_inv = inv(Σ_score)
        end

        V = S * Σ_inv * S'                               # score path ⇒ scale=√N
        Π_vec = Π_g[1:end-1, :, :][:]
        mt = mmm_test(S * Π_vec, V; scale = sqrt(N), α, rng)
        results[g] = (; group, mt.T_n, mt.c_α, mt.reject, mt.Ω, Σ_score)
    end

    return results
end

"""
    get_test_stat_mp_parametric_grouped(Var_free_list, params, data, transition_groups; α=0.05)

Parametric supermodularity test with variance pooled across transitions within each group.
When Π is shared within a group, the per-transition Var_free matrices are summed
(independent contributions to Fisher information). Returns one test result per group.
"""
function get_test_stat_mp_parametric_grouped(Var_free_list::Vector{Matrix{Float64}},
                                              params, data,
                                              transition_groups::Vector{Vector{Int}};
                                              α::Float64=0.05, rng::AbstractRNG=Random.default_rng())
    (; r_θ, r_I, T, N) = data
    S = get_S_adjacent(r_θ, r_I)

    results = Vector{NamedTuple}(undef, length(transition_groups))

    for (g, group) in enumerate(transition_groups)
        t_rep = group[1]

        Π_g = copy(params.Π[t_rep])
        Π_g .= max.(Π_g, 1e-15)
        for kI in 1:r_I, kθ in 1:r_θ
            Π_g[:, kθ, kI] ./= sum(Π_g[:, kθ, kI])
        end

        # Pool Var_free: sum across transitions (additive Fisher information)
        Var_free_pooled = sum(Var_free_list[t] for t in group)

        # Pool Var_free across the group (additive Fisher information)
        Π_vec = Π_g[1:end-1, :, :][:]
        V = S * Var_free_pooled * S'                     # delta-method cov ⇒ scale=1
        mt = mmm_test(S * Π_vec, V; scale = 1.0, α, rng)
        results[g] = (; group, mt.T_n, mt.c_α, mt.reject, mt.Ω)
    end

    return results
end
