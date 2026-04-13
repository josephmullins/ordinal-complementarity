# ─────────────────────────────────────────────────────────────────
# Multi-period EM estimation with Forward-Backward algorithm
#
# Extends the two-period model to T≥2 periods using the Baum-Welch
# algorithm. The latent state at each period is (θ_t, I_t) with:
#   - Factored transition: K_t(θ',I'|θ,I) = Π_t(θ'|θ,I) · CCP_t(I'|θ',I)
#   - Period-specific emissions: P_θ[t][m], P_I[t][m]
#   - Period-specific transitions: Π[t], CCP[t]
#
# Data convention: 0 = missing (same as estimation.jl)
# ─────────────────────────────────────────────────────────────────

using Random
using LinearAlgebra

# ─────────────────────────────────────────────────────────────────
# Parameter initialization
# ─────────────────────────────────────────────────────────────────

"""
    _init_emission_matrix(κ, r; rng) -> Matrix{Float64}

Initialize a single emission matrix of size (κ, r).
Binary measures (κ=2): monotone P(pass|state) increasing in state.
Other: near-uniform with random perturbation.
"""
function _init_emission_matrix(κ::Int, r::Int; rng::AbstractRNG)
    if κ == 2
        p_pass = [0.2 + 0.6 * (k-1) / max(r-1, 1) + 0.05 * randn(rng) for k in 1:r]
        p_pass = clamp.(p_pass, 0.05, 0.95)
        return [1.0 .- p_pass'; p_pass']
    else
        P = 0.5 * ones(κ, r) / κ .+ 0.5 * rand(rng, κ, r)
        P ./= sum(P; dims=1)
        return P
    end
end

"""
    initialize_params_mp(data; rng) -> NamedTuple

Create initial parameter guesses for multi-period EM.
"""
function initialize_params_mp(data; rng::AbstractRNG=Random.default_rng())
    (; r_θ, r_I, T, κ_θ, κ_I) = data

    # Period-specific skill emission matrices
    P_θ = [
        [_init_emission_matrix(κ_θ[t][m], r_θ; rng) for m in eachindex(κ_θ[t])]
        for t in 1:T
    ]

    # Period-specific investment emission matrices
    P_I = [
        [_init_emission_matrix(κ_I[t][m], r_I; rng) for m in eachindex(κ_I[t])]
        for t in 1:T
    ]

    # Transition matrices: slight diagonal dominance
    Π = [begin
        Π_t = ones(r_θ, r_θ, r_I) / r_θ
        for kI in 1:r_I, kθ in 1:r_θ
            Π_t[kθ, kθ, kI] += 0.3
        end
        for kI in 1:r_I, kθ in 1:r_θ
            Π_t[:, kθ, kI] ./= sum(Π_t[:, kθ, kI])
        end
        Π_t
    end for _ in 1:T-1]

    # Investment CCPs: slight diagonal dominance
    CCP = [begin
        C = ones(r_I, r_I, r_θ) / r_I
        for kθ1 in 1:r_θ, kI in 1:r_I
            C[kI, kI, kθ1] += 0.3
        end
        for kθ1 in 1:r_θ, kI in 1:r_I
            C[:, kI, kθ1] ./= sum(C[:, kI, kθ1])
        end
        C
    end for _ in 1:T-1]

    # Initial distribution: uniform
    π0 = ones(r_θ, r_I) / (r_θ * r_I)

    return (; Π, CCP, π0, P_θ, P_I, r_θ, r_I, T)
end

# ─────────────────────────────────────────────────────────────────
# E-step: Forward-Backward algorithm
# ─────────────────────────────────────────────────────────────────

"""
    compute_emission_likelihood_t(t, n, params, data) -> Matrix{Float64}

Compute joint emission likelihood L[kθ, kI] at period t for child n.
Missing measures (0) are skipped (contribute factor 1).
"""
function compute_emission_likelihood_t(t::Int, n::Int, params, data)
    (; r_θ, r_I) = params
    L = ones(r_θ, r_I)

    # Skill emissions: multiply along θ axis
    for m in eachindex(data.X_θ[t])
        x = data.X_θ[t][m][n]
        x == 0 && continue
        for kθ in 1:r_θ
            L[kθ, :] .*= params.P_θ[t][m][x, kθ]
        end
    end

    # Investment emissions: multiply along I axis
    for m in eachindex(data.X_I[t])
        x = data.X_I[t][m][n]
        x == 0 && continue
        for kI in 1:r_I
            L[:, kI] .*= params.P_I[t][m][x, kI]
        end
    end

    return L
end

"""
    compute_emission_likelihood_t!(L, t, n, params, data)

In-place version: fill preallocated L (r_θ × r_I).
"""
function compute_emission_likelihood_t!(L::Matrix{Float64}, t::Int, n::Int, params, data)
    (; r_θ, r_I) = params
    fill!(L, 1.0)

    for m in eachindex(data.X_θ[t])
        x = data.X_θ[t][m][n]
        x == 0 && continue
        for kθ in 1:r_θ
            @inbounds p = max(params.P_θ[t][m][x, kθ], 1e-300)
            for kI in 1:r_I
                @inbounds L[kθ, kI] *= p
            end
        end
    end

    for m in eachindex(data.X_I[t])
        x = data.X_I[t][m][n]
        x == 0 && continue
        for kI in 1:r_I
            @inbounds p = max(params.P_I[t][m][x, kI], 1e-300)
            for kθ in 1:r_θ
                @inbounds L[kθ, kI] *= p
            end
        end
    end

    return nothing
end

"""
    forward_backward_n!(α, β, scale, γ_slices, ξ_sum, n, params, data, L_scratch)
        -> log_likelihood

Run scaled forward-backward for one child n.

Arguments:
- α:         (T, r_θ, r_I) scratch — forward variables
- β:         (T, r_θ, r_I) scratch — backward variables
- scale:     (T,) scratch — scaling factors
- γ_slices:  writes γ[:,:,:,n] in-place for each t
- ξ_sum:     Vector of T-1 arrays (r_θ, r_I, r_θ, r_I), accumulated in-place
- L_scratch: Vector of T preallocated (r_θ, r_I) matrices
"""
function forward_backward_n!(
    α::Array{Float64,3}, β::Array{Float64,3}, scale::Vector{Float64},
    γ::Array{Float64,4}, ξ_sum::Vector{Array{Float64,4}},
    n::Int, params, data,
    L_scratch::Vector{Matrix{Float64}}
)
    (; Π, CCP, π0, r_θ, r_I, T) = params

    # ── Compute emission likelihoods at all periods ──
    for t in 1:T
        compute_emission_likelihood_t!(L_scratch[t], t, n, params, data)
    end

    # ── Forward pass ──
    for kI in 1:r_I, kθ in 1:r_θ
        @inbounds α[1, kθ, kI] = π0[kθ, kI] * L_scratch[1][kθ, kI]
    end
    s = 0.0
    for kI in 1:r_I, kθ in 1:r_θ
        @inbounds s += α[1, kθ, kI]
    end
    scale[1] = s
    if s > 0
        for kI in 1:r_I, kθ in 1:r_θ
            @inbounds α[1, kθ, kI] /= s
        end
    end

    for t in 1:T-1
        Π_t = Π[t]
        CCP_t = CCP[t]
        L_next = L_scratch[t+1]

        # Stage 1: A[kθ1, kI] = Σ_kθ α[t, kθ, kI] * Π_t[kθ1, kθ, kI]
        for kI in 1:r_I, kθ1 in 1:r_θ
            @inbounds α[t+1, kθ1, kI] = 0.0  # reuse α[t+1] as scratch A
        end
        for kI in 1:r_I, kθ in 1:r_θ, kθ1 in 1:r_θ
            @inbounds α[t+1, kθ1, kI] += α[t, kθ, kI] * Π_t[kθ1, kθ, kI]
        end

        # Stage 2: α_raw[kθ1, kI1] = Σ_kI A[kθ1, kI] * CCP_t[kI1, kI, kθ1]
        # We need a temporary since α[t+1] holds A
        # Use β[t+1] as temporary scratch (will be overwritten in backward pass)
        for kI1 in 1:r_I, kθ1 in 1:r_θ
            @inbounds β[t+1, kθ1, kI1] = 0.0
        end
        for kI in 1:r_I, kθ1 in 1:r_θ, kI1 in 1:r_I
            @inbounds β[t+1, kθ1, kI1] += α[t+1, kθ1, kI] * CCP_t[kI1, kI, kθ1]
        end

        # Multiply by emission and scale
        s = 0.0
        for kI1 in 1:r_I, kθ1 in 1:r_θ
            @inbounds α[t+1, kθ1, kI1] = β[t+1, kθ1, kI1] * L_next[kθ1, kI1]
            @inbounds s += α[t+1, kθ1, kI1]
        end
        scale[t+1] = s
        if s > 0
            for kI1 in 1:r_I, kθ1 in 1:r_θ
                @inbounds α[t+1, kθ1, kI1] /= s
            end
        end
    end

    # ── Backward pass ──
    for kI in 1:r_I, kθ in 1:r_θ
        @inbounds β[T, kθ, kI] = 1.0
    end

    for t in T-1:-1:1
        Π_t = Π[t]
        CCP_t = CCP[t]
        L_next = L_scratch[t+1]
        sc = scale[t+1]

        # Lb[kθ1, kI1] = L[t+1][kθ1, kI1] * β[t+1, kθ1, kI1]
        # Stage 1: b1[kθ1, kI] = Σ_{kI1} CCP_t[kI1, kI, kθ1] * Lb[kθ1, kI1]
        # Use α[1] as scratch (already used, will not be needed again in backward)
        # Actually, let's use a dedicated scratch approach — reuse L_scratch[1] as temp
        # Better: just compute inline with the loops since r is small

        # β_raw[kθ, kI] = Σ_{kθ1} Π_t[kθ1, kθ, kI] * Σ_{kI1} CCP_t[kI1, kI, kθ1] * Lb[kθ1, kI1]
        for kI in 1:r_I, kθ in 1:r_θ
            val = 0.0
            for kθ1 in 1:r_θ
                # Inner sum: Σ_{kI1} CCP_t[kI1, kI, kθ1] * L[t+1][kθ1,kI1] * β[t+1,kθ1,kI1]
                inner = 0.0
                for kI1 in 1:r_I
                    @inbounds inner += CCP_t[kI1, kI, kθ1] * L_next[kθ1, kI1] * β[t+1, kθ1, kI1]
                end
                @inbounds val += Π_t[kθ1, kθ, kI] * inner
            end
            @inbounds β[t, kθ, kI] = val / sc
        end
    end

    # ── Marginal posteriors γ ──
    for t in 1:T
        s = 0.0
        for kI in 1:r_I, kθ in 1:r_θ
            @inbounds v = α[t, kθ, kI] * β[t, kθ, kI]
            @inbounds γ[t, kθ, kI, n] = v
            s += v
        end
        if s > 0
            for kI in 1:r_I, kθ in 1:r_θ
                @inbounds γ[t, kθ, kI, n] /= s
            end
        end
    end

    # ── Pairwise posteriors: accumulate into ξ_sum ──
    for t in 1:T-1
        Π_t = Π[t]
        CCP_t = CCP[t]
        L_next = L_scratch[t+1]
        sc = scale[t+1]
        ξ_t = ξ_sum[t]

        for kI in 1:r_I, kθ in 1:r_θ
            @inbounds a_val = α[t, kθ, kI]
            a_val == 0.0 && continue
            for kθ1 in 1:r_θ
                @inbounds pi_val = a_val * Π_t[kθ1, kθ, kI]
                pi_val == 0.0 && continue
                for kI1 in 1:r_I
                    @inbounds xi_raw = pi_val * CCP_t[kI1, kI, kθ1] *
                                       L_next[kθ1, kI1] * β[t+1, kθ1, kI1] / sc
                    @inbounds ξ_t[kθ1, kI1, kθ, kI] += xi_raw
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

"""
    expectation_step_mp!(γ, ξ_sum, params, data) -> total_log_likelihood

Full E-step: run forward-backward for all N observations.
γ is (T, r_θ, r_I, N), ξ_sum is a Vector of T-1 arrays (r_θ, r_I, r_θ, r_I).
"""
function expectation_step_mp!(γ::Array{Float64,4},
                               ξ_sum::Vector{Array{Float64,4}},
                               params, data)
    (; r_θ, r_I, T, N) = data

    # Reset ξ_sum
    for t in 1:T-1
        fill!(ξ_sum[t], 0.0)
    end

    # Preallocate scratch
    α = zeros(T, r_θ, r_I)
    β = zeros(T, r_θ, r_I)
    scale = zeros(T)
    L_scratch = [zeros(r_θ, r_I) for _ in 1:T]

    ll_total = 0.0

    for n in 1:N
        ll_n = forward_backward_n!(α, β, scale, γ, ξ_sum, n, params, data, L_scratch)
        ll_total += ll_n
    end

    return ll_total
end

# ─────────────────────────────────────────────────────────────────
# M-step: Update parameters from sufficient statistics
# ─────────────────────────────────────────────────────────────────

"""
    update_π0_mp(γ) -> Matrix{Float64}

Update initial distribution π0[θ, I] = (1/N) Σ_n γ[1, θ, I, n].
"""
function update_π0_mp(γ::Array{Float64,4})
    r_θ = size(γ, 2)
    r_I = size(γ, 3)
    N = size(γ, 4)
    π0 = zeros(r_θ, r_I)
    for n in 1:N, kI in 1:r_I, kθ in 1:r_θ
        @inbounds π0[kθ, kI] += γ[1, kθ, kI, n]
    end
    π0 ./= N
    return π0
end

"""
    update_Π_mp(γ, ξ_sum) -> Vector{Array{Float64,3}}

Update period-specific skill transition matrices.
Π_t[θ', θ, I] = Σ_{I'} ξ_sum_t[θ', I', θ, I] / Σ_n γ[t, θ, I, n]
"""
function update_Π_mp(γ::Array{Float64,4}, ξ_sum::Vector{Array{Float64,4}})
    T_minus_1 = length(ξ_sum)
    r_θ = size(γ, 2)
    r_I = size(γ, 3)
    N = size(γ, 4)

    Π_new = Vector{Array{Float64,3}}(undef, T_minus_1)

    for t in 1:T_minus_1
        Π_t = zeros(r_θ, r_θ, r_I)
        ξ_t = ξ_sum[t]

        # Denominator: Σ_n γ[t, kθ, kI, n] (already summed in ξ via forward-backward)
        γ_sum_t = zeros(r_θ, r_I)
        for n in 1:N, kI in 1:r_I, kθ in 1:r_θ
            @inbounds γ_sum_t[kθ, kI] += γ[t, kθ, kI, n]
        end

        for kI in 1:r_I, kθ in 1:r_θ
            denom = γ_sum_t[kθ, kI]
            for kθ1 in 1:r_θ
                # Numerator: Σ_{kI1} ξ_t[kθ1, kI1, kθ, kI]
                numer = 0.0
                for kI1 in 1:r_I
                    @inbounds numer += ξ_t[kθ1, kI1, kθ, kI]
                end
                Π_t[kθ1, kθ, kI] = denom > 0 ? numer / denom : 1.0 / r_θ
            end
        end

        Π_new[t] = Π_t
    end

    return Π_new
end

"""
    update_CCP_mp(ξ_sum) -> Vector{Array{Float64,3}}

Update period-specific investment CCPs.
CCP_t[I', I, θ'] = Σ_θ ξ_sum_t[θ', I', θ, I] / Σ_θ Σ_{I''} ξ_sum_t[θ', I'', θ, I]
"""
function update_CCP_mp(ξ_sum::Vector{Array{Float64,4}})
    T_minus_1 = length(ξ_sum)
    r_θ = size(ξ_sum[1], 1)
    r_I = size(ξ_sum[1], 2)

    CCP_new = Vector{Array{Float64,3}}(undef, T_minus_1)

    for t in 1:T_minus_1
        CCP_t = zeros(r_I, r_I, r_θ)
        ξ_t = ξ_sum[t]

        for kθ1 in 1:r_θ, kI in 1:r_I
            # Denominator: Σ_kθ Σ_{kI1} ξ_t[kθ1, kI1, kθ, kI]
            denom = 0.0
            for kθ in 1:r_θ, kI1 in 1:r_I
                @inbounds denom += ξ_t[kθ1, kI1, kθ, kI]
            end
            for kI1 in 1:r_I
                # Numerator: Σ_kθ ξ_t[kθ1, kI1, kθ, kI]
                numer = 0.0
                for kθ in 1:r_θ
                    @inbounds numer += ξ_t[kθ1, kI1, kθ, kI]
                end
                CCP_t[kI1, kI, kθ1] = denom > 0 ? numer / denom : 1.0 / r_I
            end
        end

        CCP_new[t] = CCP_t
    end

    return CCP_new
end

"""
    update_emission_θ_mp(γ, data) -> Vector{Vector{Matrix{Float64}}}

Update period-specific skill emission matrices.
Weight for (n, kθ) at period t: w = Σ_{kI} γ[t, kθ, kI, n].
"""
function update_emission_θ_mp(γ::Array{Float64,4}, data)
    T = size(γ, 1)
    r_θ = size(γ, 2)
    r_I = size(γ, 3)
    N = size(γ, 4)

    P_θ_all = Vector{Vector{Matrix{Float64}}}(undef, T)

    for t in 1:T
        M = length(data.X_θ[t])
        P_θ_t = Vector{Matrix{Float64}}(undef, M)

        for m in 1:M
            κ = data.κ_θ[t][m]
            P = zeros(κ, r_θ)
            denom = zeros(r_θ)

            for n in 1:N
                @inbounds x = data.X_θ[t][m][n]
                x == 0 && continue
                for kθ in 1:r_θ
                    w = 0.0
                    for kI in 1:r_I
                        @inbounds w += γ[t, kθ, kI, n]
                    end
                    @inbounds P[x, kθ] += w
                    @inbounds denom[kθ] += w
                end
            end

            for kθ in 1:r_θ
                if denom[kθ] > 0
                    P[:, kθ] ./= denom[kθ]
                else
                    P[:, kθ] .= 1.0 / κ
                end
            end

            P_θ_t[m] = P
        end

        P_θ_all[t] = P_θ_t
    end

    return P_θ_all
end

"""
    update_emission_I_mp(γ, data) -> Vector{Vector{Matrix{Float64}}}

Update period-specific investment emission matrices.
Weight for (n, kI) at period t: w = Σ_{kθ} γ[t, kθ, kI, n].
"""
function update_emission_I_mp(γ::Array{Float64,4}, data)
    T = size(γ, 1)
    r_θ = size(γ, 2)
    r_I = size(γ, 3)
    N = size(γ, 4)

    P_I_all = Vector{Vector{Matrix{Float64}}}(undef, T)

    for t in 1:T
        M = length(data.X_I[t])
        P_I_t = Vector{Matrix{Float64}}(undef, M)

        for m in 1:M
            κ = data.κ_I[t][m]
            P = zeros(κ, r_I)
            denom = zeros(r_I)

            for n in 1:N
                @inbounds x = data.X_I[t][m][n]
                x == 0 && continue
                for kI in 1:r_I
                    w = 0.0
                    for kθ in 1:r_θ
                        @inbounds w += γ[t, kθ, kI, n]
                    end
                    @inbounds P[x, kI] += w
                    @inbounds denom[kI] += w
                end
            end

            for kI in 1:r_I
                if denom[kI] > 0
                    P[:, kI] ./= denom[kI]
                else
                    P[:, kI] .= 1.0 / κ
                end
            end

            P_I_t[m] = P
        end

        P_I_all[t] = P_I_t
    end

    return P_I_all
end

"""
    maximization_step_mp(γ, ξ_sum, data) -> NamedTuple

Full M-step: update all parameters from posteriors.
"""
function maximization_step_mp(γ::Array{Float64,4},
                               ξ_sum::Vector{Array{Float64,4}},
                               data)
    π0 = update_π0_mp(γ)
    Π = update_Π_mp(γ, ξ_sum)
    CCP = update_CCP_mp(ξ_sum)
    P_θ = update_emission_θ_mp(γ, data)
    P_I = update_emission_I_mp(γ, data)
    return (; Π, CCP, π0, P_θ, P_I)
end

# ─────────────────────────────────────────────────────────────────
# Main EM loop
# ─────────────────────────────────────────────────────────────────

"""
    expectation_maximization_mp(data, params_init; tol, maxiter, verbose)

Run multi-period EM algorithm. Returns (params_est, γ, ξ_sum).
Convergence criterion: max over all periods of max|ΔΠ_t|.
"""
function expectation_maximization_mp(data, params_init;
    tol::Float64=1e-5, maxiter::Int=2000, verbose::Bool=true)

    (; r_θ, r_I, T, N) = data

    # Preallocate posteriors
    γ = zeros(T, r_θ, r_I, N)
    ξ_sum = [zeros(r_θ, r_I, r_θ, r_I) for _ in 1:T-1]

    params = params_init
    err = Inf
    iter = 0

    while err > tol && iter < maxiter
        ll = expectation_step_mp!(γ, ξ_sum, params, data)

        new_params = maximization_step_mp(γ, ξ_sum, data)

        # Convergence: max change in any Π_t
        err = 0.0
        for t in 1:T-1
            err = max(err, maximum(abs.(new_params.Π[t] .- params.Π[t])))
        end
        iter += 1

        if verbose && (iter ≤ 5 || iter % 50 == 0)
            println("  iter $iter: max|ΔΠ| = $(round(err; sigdigits=4)), ll = $(round(ll; digits=2))")
        end

        params = (; new_params..., r_θ, r_I, T)
    end

    if verbose
        println("  Converged after $iter iterations (err = $(round(err; sigdigits=4)))")
    end

    # Final E-step at estimated parameters
    ll_final = expectation_step_mp!(γ, ξ_sum, params, data)
    if verbose
        println("  Final log-likelihood: $(round(ll_final; digits=2))")
    end

    return params, γ, ξ_sum
end

# ─────────────────────────────────────────────────────────────────
# Normalize latent factor ordering
# ─────────────────────────────────────────────────────────────────

"""
    normalize_params_mp(params, γ, data) -> (params_normalized, γ_normalized)

Relabel latent categories at each period independently using period-specific
emission matrices. With age-specific emissions, the label "θ=1" at period t is
identified by P_θ[t] independently from period t+1, so each period gets its
own permutation — exactly as the two-period case uses separate θ_order and θ'_order.

- θ_order_t from skill emissions at period t (via skill_index)
- I_order_t from investment emissions at period t (via investment_index)
- Π_t is permuted by (θ_order_{t+1}, θ_order_t, I_order_t)
- CCP_t is permuted by (I_order_{t+1}, I_order_t, θ_order_{t+1})

Requires: skill_index, investment_index from estimation.jl
"""
function normalize_params_mp(params, γ, data)
    (; Π, CCP, π0, P_θ, P_I, r_θ, r_I, T) = params

    # Period-specific orderings from emissions
    θ_orders = [sortperm(skill_index(P_θ[t], r_θ)) for t in 1:T]
    I_orders = [sortperm(investment_index(P_I[t], r_I)) for t in 1:T]

    for t in 1:T
        println("  Period $t: θ ordering = $(θ_orders[t]), I ordering = $(I_orders[t])")
    end

    # Permute Π_t[θ', θ, I]: axis 1 = θ' (period t+1), axis 2 = θ (period t), axis 3 = I (period t)
    Π_new = [Π[t][θ_orders[t+1], θ_orders[t], I_orders[t]] for t in 1:T-1]

    # Permute CCP_t[I', I, θ']: axis 1 = I' (period t+1), axis 2 = I (period t), axis 3 = θ' (period t+1)
    CCP_new = [CCP[t][I_orders[t+1], I_orders[t], θ_orders[t+1]] for t in 1:T-1]

    # Permute π0[θ_1, I_1]
    π0_new = π0[θ_orders[1], I_orders[1]]

    # Permute emission matrices: each period uses its own ordering
    P_θ_new = [[P_θ[t][m][:, θ_orders[t]] for m in eachindex(P_θ[t])] for t in 1:T]
    P_I_new = [[P_I[t][m][:, I_orders[t]] for m in eachindex(P_I[t])] for t in 1:T]

    # Permute γ[t, θ, I, n] with period-specific orderings
    γ_new = similar(γ)
    for t in 1:T
        γ_new[t, :, :, :] = γ[t, θ_orders[t], I_orders[t], :]
    end

    params_new = (; Π=Π_new, CCP=CCP_new, π0=π0_new, P_θ=P_θ_new, P_I=P_I_new, r_θ, r_I, T)
    return params_new, γ_new
end

# ─────────────────────────────────────────────────────────────────
# Display estimated parameters
# ─────────────────────────────────────────────────────────────────

function display_results_mp(params, data; max_emission_display::Int=10)
    (; Π, CCP, π0, P_θ, P_I, r_θ, r_I, T) = params

    println("\n" * "="^60)
    println("  MULTI-PERIOD ESTIMATED PARAMETERS (T=$T)")
    println("="^60)

    println("\n  Initial distribution π0(θ, I):")
    for kθ in 1:r_θ
        print("    θ=$kθ: ")
        for kI in 1:r_I
            print("  I=$kI: $(round(π0[kθ,kI]; digits=4))")
        end
        println()
    end

    for t in 1:T-1
        println("\n  ── Transition $t (period $t → $(t+1)) ──")

        println("  Skill transition Π_$t(θ'|θ, I):")
        for kI in 1:r_I
            println("    I = $kI:")
            for kθ1 in 1:r_θ
                print("      θ'=$kθ1: ")
                for kθ in 1:r_θ
                    print("  θ=$kθ: $(round(Π[t][kθ1,kθ,kI]; digits=4))")
                end
                println()
            end
        end

        println("  Investment CCP_$t(I'|I, θ'):")
        for kθ1 in 1:r_θ
            println("    θ' = $kθ1:")
            for kI1 in 1:r_I
                print("      I'=$kI1: ")
                for kI in 1:r_I
                    print("  I=$kI: $(round(CCP[t][kI1,kI,kθ1]; digits=4))")
                end
                println()
            end
        end
    end

    # Emission summaries per period
    for t in 1:T
        has_names = !isempty(data.measure_names_θ[t])
        n_skill = length(P_θ[t])
        n_show = min(n_skill, max_emission_display)

        println("\n  Period $t skill emissions ($n_skill measures, showing $n_show):")
        for m in 1:n_show
            name = has_names ? data.measure_names_θ[t][m] : "measure_$m"
            print("    $name:  P(high|θ)=")
            for kθ in 1:r_θ
                print("  $(round(P_θ[t][m][end, kθ]; digits=3))")
            end
            println()
        end
        if n_skill > max_emission_display
            n_monotone = sum(1:n_skill) do m
                p_high = [P_θ[t][m][end, k] for k in 1:r_θ]
                issorted(p_high)
            end
            println("    ... ($n_monotone / $n_skill monotone)")
        end

        n_inv = length(P_I[t])
        has_inv_names = !isempty(data.measure_names_I[t])
        if n_inv > 0
            n_show_inv = min(n_inv, max_emission_display)
            println("  Period $t investment emissions ($n_inv measures, showing $n_show_inv):")
            for m in 1:n_show_inv
                name = has_inv_names ? data.measure_names_I[t][m] : "invest_$m"
                print("    $name:  P(high|I)=")
                for kI in 1:r_I
                    print("  $(round(P_I[t][m][end, kI]; digits=3))")
                end
                println()
            end
        end
    end
end

# ─────────────────────────────────────────────────────────────────
# Score computation for inference (per-period)
# ─────────────────────────────────────────────────────────────────

"""
    compute_psi_all(params, data) -> Array{Float64, 5}

Run forward-backward storing per-observation pairwise posteriors
marginalized over I': ψ_t^n(θ', θ, I) = Σ_{I'} ξ_t^n(θ', I', θ, I).

Returns ψ_all of shape (T-1, r_θ, r_θ, r_I, N).
Used for per-period score computation in inference.
"""
function compute_psi_all(params, data)
    (; r_θ, r_I, T, N) = data

    ψ_all = zeros(T-1, r_θ, r_θ, r_I, N)

    # Scratch arrays
    α = zeros(T, r_θ, r_I)
    β = zeros(T, r_θ, r_I)
    scale = zeros(T)
    L_scratch = [zeros(r_θ, r_I) for _ in 1:T]
    γ_dummy = zeros(T, r_θ, r_I, 1)  # dummy, not needed

    for n in 1:N
        # Compute emissions
        for t in 1:T
            compute_emission_likelihood_t!(L_scratch[t], t, n, params, data)
        end

        # Forward pass (same as in forward_backward_n!)
        (; Π, CCP, π0) = params

        for kI in 1:r_I, kθ in 1:r_θ
            @inbounds α[1, kθ, kI] = π0[kθ, kI] * L_scratch[1][kθ, kI]
        end
        s = sum(@view α[1, :, :])
        scale[1] = s
        if s > 0; @view(α[1, :, :]) ./= s; end

        # We need a temp array for the CCP integration step
        temp = zeros(r_θ, r_I)

        for t in 1:T-1
            Π_t = Π[t]; CCP_t = CCP[t]; L_next = L_scratch[t+1]

            # Stage 1: A[kθ1, kI] = Σ_kθ α[t, kθ, kI] * Π_t[kθ1, kθ, kI]
            fill!(temp, 0.0)
            for kI in 1:r_I, kθ in 1:r_θ, kθ1 in 1:r_θ
                @inbounds temp[kθ1, kI] += α[t, kθ, kI] * Π_t[kθ1, kθ, kI]
            end

            # Stage 2: α_raw[kθ1, kI1] = Σ_kI temp[kθ1, kI] * CCP_t[kI1, kI, kθ1]
            for kI1 in 1:r_I, kθ1 in 1:r_θ
                @inbounds α[t+1, kθ1, kI1] = 0.0
            end
            for kI in 1:r_I, kθ1 in 1:r_θ, kI1 in 1:r_I
                @inbounds α[t+1, kθ1, kI1] += temp[kθ1, kI] * CCP_t[kI1, kI, kθ1]
            end

            s = 0.0
            for kI1 in 1:r_I, kθ1 in 1:r_θ
                @inbounds α[t+1, kθ1, kI1] *= L_next[kθ1, kI1]
                @inbounds s += α[t+1, kθ1, kI1]
            end
            scale[t+1] = s
            if s > 0
                for kI1 in 1:r_I, kθ1 in 1:r_θ
                    @inbounds α[t+1, kθ1, kI1] /= s
                end
            end
        end

        # Backward pass
        for kI in 1:r_I, kθ in 1:r_θ
            @inbounds β[T, kθ, kI] = 1.0
        end
        for t in T-1:-1:1
            Π_t = Π[t]; CCP_t = CCP[t]; L_next = L_scratch[t+1]
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

        # Compute per-observation ψ_t^n(kθ1, kθ, kI) = Σ_{kI1} ξ_t^n(kθ1, kI1, kθ, kI)
        for t in 1:T-1
            Π_t = Π[t]; CCP_t = CCP[t]; L_next = L_scratch[t+1]
            sc = scale[t+1]

            for kI in 1:r_I, kθ in 1:r_θ
                @inbounds a_val = α[t, kθ, kI]
                a_val == 0.0 && continue
                for kθ1 in 1:r_θ
                    @inbounds pi_val = a_val * Π_t[kθ1, kθ, kI]
                    pi_val == 0.0 && continue
                    psi = 0.0
                    for kI1 in 1:r_I
                        @inbounds psi += CCP_t[kI1, kI, kθ1] * L_next[kθ1, kI1] * β[t+1, kθ1, kI1]
                    end
                    @inbounds ψ_all[t, kθ1, kθ, kI, n] = pi_val * psi / sc
                end
            end
        end
    end

    return ψ_all
end

# ═════════════════════════════════════════════════════════════════
# Parametric emission & transition support
#
# These functions adapt the parametric M-step machinery from
# parametric_models.jl to work with the multi-period posteriors
# (γ, ξ_sum) instead of the two-period Q array.
# ═════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────
# Adapter: marginal weights from γ for emission M-step
# ─────────────────────────────────────────────────────────────────

"""
    _marginal_weights_from_gamma(γ_t, axis, r) → Matrix{Float64}

Compute marginal posterior weights W[k, n] from period-t posteriors γ_t (r_θ × r_I × N).
- axis = :theta → W[kθ, n] = Σ_{kI} γ_t[kθ, kI, n]  (for skill emissions)
- axis = :I     → W[kI, n] = Σ_{kθ} γ_t[kθ, kI, n]  (for investment emissions)
"""
function _marginal_weights_from_gamma(γ_t::AbstractArray{Float64,3},
                                       axis::Symbol, r::Int)
    r_θ, r_I, N = size(γ_t)
    W = zeros(r, N)
    if axis == :theta
        for n in 1:N, kθ in 1:r_θ
            @inbounds W[kθ, n] = sum(γ_t[kθ, kI, n] for kI in 1:r_I)
        end
    elseif axis == :I
        for n in 1:N, kI in 1:r_I
            @inbounds W[kI, n] = sum(γ_t[kθ, kI, n] for kθ in 1:r_θ)
        end
    else
        error("Unknown axis: $axis (expected :theta or :I)")
    end
    return W
end

# ─────────────────────────────────────────────────────────────────
# Adapter: Q_sum from ξ_sum for transition M-step
# ─────────────────────────────────────────────────────────────────

"""
    _qsum_from_xi(ξ_sum_t) → Array{Float64,3}

Compute transition sufficient statistics from pairwise posteriors.
ξ_sum_t has shape (r_θ, r_I, r_θ, r_I) = (θ', I', θ, I).
Returns Q_sum_t[kθ', kθ, kI] = Σ_{kI'} ξ_sum_t[kθ', kI', kθ, kI].
"""
function _qsum_from_xi(ξ_sum_t::Array{Float64,4})
    return dropdims(sum(ξ_sum_t; dims=2); dims=2)
end

"""
    _ccpsum_from_xi(ξ_sum_t) → Array{Float64,3}

Compute CCP sufficient statistics from pairwise posteriors.
ξ_sum_t has shape (r_θ, r_I, r_θ, r_I) = (θ', I', θ, I).
Returns CCP_sum_t[I', I, θ'] = Σ_θ ξ_sum_t[θ', I', θ, I].
"""
function _ccpsum_from_xi(ξ_sum_t::Array{Float64,4})
    r_θ, r_I = size(ξ_sum_t, 1), size(ξ_sum_t, 2)
    CCP_sum = zeros(r_I, r_I, r_θ)
    for kθ1 in 1:r_θ, kI in 1:r_I, kI1 in 1:r_I
        for kθ in 1:r_θ
            @inbounds CCP_sum[kI1, kI, kθ1] += ξ_sum_t[kθ1, kI1, kθ, kI]
        end
    end
    return CCP_sum
end

# ─────────────────────────────────────────────────────────────────
# Parametric emission M-step (multi-period)
# ─────────────────────────────────────────────────────────────────

"""
    parametric_mstep_emission_mp(specs_list, γ, data, field, r; options)
    → (new_specs_list, P_per_period)

Optimize ordered probit emission parameters for all periods.

- specs_list: Vector of T OrderedProbitEmission specs
- γ: (T, r_θ, r_I, N) marginal posteriors
- field: :theta or :I (which latent variable the emissions measure)
- r: cardinality of the latent variable (r_θ or r_I)

For each period t, extracts marginal weights from γ[t,:,:,:] and optimizes
each measure independently via _single_measure_negll (from parametric_models.jl).
"""
function parametric_mstep_emission_mp(specs_list::Vector{OrderedProbitEmission},
                                       γ::Array{Float64,4},
                                       data, field::Symbol, r::Int;
                                       options=DEFAULT_OPTIM_OPTIONS)
    T = size(γ, 1)

    # Select data arrays based on field
    if field == :theta
        X_all = data.X_θ    # X_all[t][m][n]
        κ_all = data.κ_θ    # κ_all[t][m]
        axis = :theta
    elseif field == :I
        X_all = data.X_I
        κ_all = data.κ_I
        axis = :I
    else
        error("Unknown field: $field")
    end

    new_specs = Vector{OrderedProbitEmission}(undef, T)
    P_per_period = Vector{Vector{Matrix{Float64}}}(undef, T)

    for t in 1:T
        γ_t = @view γ[t, :, :, :]
        W = _marginal_weights_from_gamma(γ_t, axis, r)

        X_measures = X_all[t]
        κ_list = κ_all[t]
        spec = specs_list[t]
        M = length(κ_list)

        new_tau = Vector{Vector{Float64}}(undef, M)
        new_alpha = Vector{Float64}(undef, M)
        P_list = Vector{Matrix{Float64}}(undef, M)

        for m in 1:M
            n_cut = κ_list[m] - 1
            x0 = vcat(_pack_cutpoints(spec.tau[m]), spec.alpha[m])
            obj = x -> _single_measure_negll(x, W, X_measures[m], κ_list[m], r)
            xopt = try
                result = optimize(obj, x0, BFGS(), options; autodiff=:forward)
                Optim.minimizer(result)
            catch
                # BFGS line search can fail with degenerate posteriors;
                # fall back to NelderMead which doesn't need gradients
                result = optimize(obj, x0, NelderMead(), options)
                Optim.minimizer(result)
            end
            new_tau[m] = _unpack_cutpoints(xopt[1:n_cut])
            new_alpha[m] = xopt[n_cut + 1]
            P_list[m] = ordered_probit_matrix(new_tau[m], new_alpha[m], r)
        end

        new_specs[t] = OrderedProbitEmission(new_tau, new_alpha)
        P_per_period[t] = P_list
    end

    return new_specs, P_per_period
end

# ─────────────────────────────────────────────────────────────────
# Parametric transition M-step (multi-period)
# ─────────────────────────────────────────────────────────────────

"""
    parametric_mstep_transition_mp(specs_list, ξ_sum, r_θ, r_I; ...) → (new_specs, Π_list)

Optimize parametric transition parameters for all T-1 transitions.
One overload per transition type (dispatch on spec element type).
"""
function parametric_mstep_transition_mp(specs_list::Vector{FlexibleLPMTransition},
                                         ξ_sum::Vector{Array{Float64,4}},
                                         r_θ::Int, r_I::Int;
                                         I_grid::AbstractVector=collect(1.0:r_I),
                                         options=DEFAULT_OPTIM_OPTIONS)
    T_minus_1 = length(ξ_sum)
    new_specs = Vector{FlexibleLPMTransition}(undef, T_minus_1)
    Π_list = Vector{Array{Float64,3}}(undef, T_minus_1)

    for t in 1:T_minus_1
        Q_sum_t = _qsum_from_xi(ξ_sum[t])
        x0 = pack_flexible_lpm(specs_list[t])
        obj = x -> transition_negll_flexible_fast(x, Q_sum_t, r_θ, r_I; I_grid)
        result = optimize(obj, x0, LBFGS(), options; autodiff=:forward)
        new_specs[t] = unpack_flexible_lpm(Optim.minimizer(result), r_θ)
        Π_list[t] = transition_from_flexible_lpm(new_specs[t], r_θ, r_I; I_grid)
    end

    return new_specs, Π_list
end

function parametric_mstep_transition_mp(specs_list::Vector{ParsimoniousLPMTransition},
                                         ξ_sum::Vector{Array{Float64,4}},
                                         r_θ::Int, r_I::Int;
                                         θ_grid::AbstractVector=collect(1.0:r_θ),
                                         I_grid::AbstractVector=collect(1.0:r_I),
                                         options=DEFAULT_OPTIM_OPTIONS)
    T_minus_1 = length(ξ_sum)
    new_specs = Vector{ParsimoniousLPMTransition}(undef, T_minus_1)
    Π_list = Vector{Array{Float64,3}}(undef, T_minus_1)

    for t in 1:T_minus_1
        Q_sum_t = _qsum_from_xi(ξ_sum[t])
        x0 = pack_parsimonious_lpm(specs_list[t])
        obj = x -> transition_negll_parsimonious_fast(x, Q_sum_t, r_θ, r_I; θ_grid, I_grid)
        result = optimize(obj, x0, LBFGS(), options; autodiff=:forward)
        new_specs[t] = unpack_parsimonious_lpm(Optim.minimizer(result), r_θ)
        Π_list[t] = transition_from_parsimonious_lpm(new_specs[t], r_θ, r_I; θ_grid, I_grid)
    end

    return new_specs, Π_list
end

function parametric_mstep_transition_mp(specs_list::Vector{OrderedProbitTransition},
                                         ξ_sum::Vector{Array{Float64,4}},
                                         r_θ::Int, r_I::Int;
                                         θ_grid::AbstractVector=collect(1.0:r_θ),
                                         I_grid::AbstractVector=collect(1.0:r_I),
                                         monotone_I::Bool=false,
                                         options=DEFAULT_OPTIM_OPTIONS)
    T_minus_1 = length(ξ_sum)
    new_specs = Vector{OrderedProbitTransition}(undef, T_minus_1)
    Π_list = Vector{Array{Float64,3}}(undef, T_minus_1)

    for t in 1:T_minus_1
        Q_sum_t = _qsum_from_xi(ξ_sum[t])
        x0 = pack_ordered_probit_transition(specs_list[t]; monotone_I)
        obj = x -> transition_negll_ordered_probit_fast(x, Q_sum_t, r_θ, r_I; θ_grid, I_grid, monotone_I)
        result = optimize(obj, x0, Newton(), options; autodiff=:forward)
        new_specs[t] = unpack_ordered_probit_transition(Optim.minimizer(result), r_θ; monotone_I)
        Π_list[t] = transition_from_ordered_probit(new_specs[t], r_θ, r_I; θ_grid, I_grid)
    end

    return new_specs, Π_list
end

function parametric_mstep_transition_mp(specs_list::Vector{MixtureProbitTransition},
                                         ξ_sum::Vector{Array{Float64,4}},
                                         r_θ::Int, r_I::Int;
                                         I_grid::AbstractVector=collect(1.0:r_I),
                                         options=DEFAULT_OPTIM_OPTIONS,
                                         kwargs...)
    T_minus_1 = length(ξ_sum)
    K = specs_list[1].K
    new_specs = Vector{MixtureProbitTransition}(undef, T_minus_1)
    Π_list = Vector{Array{Float64,3}}(undef, T_minus_1)

    for t in 1:T_minus_1
        Q_sum_t = _qsum_from_xi(ξ_sum[t])
        x0 = pack_mixture_probit(specs_list[t])
        obj = x -> transition_negll_mixture_probit_fast(x, Q_sum_t, r_θ, r_I, K; I_grid)
        result = optimize(obj, x0, LBFGS(), options; autodiff=:forward)
        new_specs[t] = unpack_mixture_probit(Optim.minimizer(result), r_θ, K)
        Π_list[t] = _build_mixture_probit_pi(Optim.minimizer(result), r_θ, r_I, K; I_grid)
    end

    return new_specs, Π_list
end

function parametric_mstep_transition_mp(specs_list::Vector{MonotoneNNTransition},
                                         ξ_sum::Vector{Array{Float64,4}},
                                         r_θ::Int, r_I::Int;
                                         θ_grid::AbstractVector=collect(1.0:r_θ),
                                         I_grid::AbstractVector=collect(1.0:r_I),
                                         options=DEFAULT_OPTIM_OPTIONS,
                                         kwargs...)
    T_minus_1 = length(ξ_sum)
    K = specs_list[1].K
    new_specs = Vector{MonotoneNNTransition}(undef, T_minus_1)
    Π_list = Vector{Array{Float64,3}}(undef, T_minus_1)

    for t in 1:T_minus_1
        Q_sum_t = _qsum_from_xi(ξ_sum[t])
        x0 = pack_monotone_nn(specs_list[t])
        obj = x -> transition_negll_monotone_nn_fast(x, Q_sum_t, r_θ, r_I, K; θ_grid, I_grid)
        result = optimize(obj, x0, LBFGS(), options; autodiff=:forward)
        new_specs[t] = unpack_monotone_nn(Optim.minimizer(result), r_θ, K)
        Π_list[t] = _build_monotone_nn_pi(Optim.minimizer(result), r_θ, r_I, K; θ_grid, I_grid)
    end

    return new_specs, Π_list
end

function parametric_mstep_transition_mp(specs_list::Vector{RobitNNTransition},
                                         ξ_sum::Vector{Array{Float64,4}},
                                         r_θ::Int, r_I::Int;
                                         θ_grid::AbstractVector=collect(1.0:r_θ),
                                         I_grid::AbstractVector=collect(1.0:r_I),
                                         options=DEFAULT_OPTIM_OPTIONS,
                                         kwargs...)
    T_minus_1 = length(ξ_sum)
    K = specs_list[1].K
    ν = specs_list[1].ν
    new_specs = Vector{RobitNNTransition}(undef, T_minus_1)
    Π_list = Vector{Array{Float64,3}}(undef, T_minus_1)

    for t in 1:T_minus_1
        Q_sum_t = _qsum_from_xi(ξ_sum[t])
        x0 = pack_robit_nn(specs_list[t])
        obj = x -> transition_negll_robit_nn_fast(x, Q_sum_t, r_θ, r_I, K, ν; θ_grid, I_grid)
        result = optimize(obj, x0, LBFGS(), options; autodiff=:forward)
        new_specs[t] = unpack_robit_nn(Optim.minimizer(result), r_θ, K; ν)
        Π_list[t] = _build_robit_nn_pi(Optim.minimizer(result), r_θ, r_I, K, ν; θ_grid, I_grid)
    end

    return new_specs, Π_list
end

# ─────────────────────────────────────────────────────────────────
# Pooled transition M-steps (shared Π across grouped transitions)
#
# transition_groups: Vector of vectors, e.g. [[1,2,3], [4,5,6]]
# means transitions 1-3 share one Π, transitions 4-6 share another.
# ─────────────────────────────────────────────────────────────────

"""
    update_Π_mp_grouped(γ, ξ_sum, transition_groups) -> Vector{Array{Float64,3}}

Nonparametric M-step that pools sufficient statistics across transitions
in each group, producing a single shared Π per group.
"""
function update_Π_mp_grouped(γ::Array{Float64,4}, ξ_sum::Vector{Array{Float64,4}},
                              transition_groups::Vector{Vector{Int}})
    T_minus_1 = length(ξ_sum)
    r_θ = size(γ, 2)
    r_I = size(γ, 3)
    N = size(γ, 4)

    Π_new = Vector{Array{Float64,3}}(undef, T_minus_1)

    for group in transition_groups
        # Pool sufficient statistics across transitions in this group
        Π_g = zeros(r_θ, r_θ, r_I)
        γ_sum_g = zeros(r_θ, r_I)

        for t in group
            for n in 1:N, kI in 1:r_I, kθ in 1:r_θ
                @inbounds γ_sum_g[kθ, kI] += γ[t, kθ, kI, n]
            end
            ξ_t = ξ_sum[t]
            for kI in 1:r_I, kθ in 1:r_θ, kθ1 in 1:r_θ
                for kI1 in 1:r_I
                    @inbounds Π_g[kθ1, kθ, kI] += ξ_t[kθ1, kI1, kθ, kI]
                end
            end
        end

        # Normalize
        for kI in 1:r_I, kθ in 1:r_θ
            denom = γ_sum_g[kθ, kI]
            for kθ1 in 1:r_θ
                Π_g[kθ1, kθ, kI] = denom > 0 ? Π_g[kθ1, kθ, kI] / denom : 1.0 / r_θ
            end
        end

        # Assign same Π to all transitions in the group
        for t in group
            Π_new[t] = Π_g
        end
    end

    return Π_new
end

"""
    parametric_mstep_transition_mp_grouped(specs_list, ξ_sum, r_θ, r_I,
                                            transition_groups; ...) -> (new_specs, Π_list)

Parametric transition M-step that pools sufficient statistics across
transitions in each group, optimizing a single set of parameters per group.
"""
function parametric_mstep_transition_mp_grouped(specs_list::Vector{OrderedProbitTransition},
                                                 ξ_sum::Vector{Array{Float64,4}},
                                                 r_θ::Int, r_I::Int,
                                                 transition_groups::Vector{Vector{Int}};
                                                 θ_grid::AbstractVector=collect(1.0:r_θ),
                                                 I_grid::AbstractVector=collect(1.0:r_I),
                                                 monotone_I::Bool=false,
                                                 options=DEFAULT_OPTIM_OPTIONS)
    T_minus_1 = length(ξ_sum)
    new_specs = Vector{OrderedProbitTransition}(undef, T_minus_1)
    Π_list = Vector{Array{Float64,3}}(undef, T_minus_1)

    for group in transition_groups
        # Pool Q_sum across transitions in this group
        Q_sum_pooled = _qsum_from_xi(ξ_sum[group[1]])
        for t in group[2:end]
            Q_sum_pooled = Q_sum_pooled .+ _qsum_from_xi(ξ_sum[t])
        end

        # Use the first transition's spec as starting point
        x0 = pack_ordered_probit_transition(specs_list[group[1]]; monotone_I)
        obj = x -> transition_negll_ordered_probit_fast(x, Q_sum_pooled, r_θ, r_I; θ_grid, I_grid, monotone_I)
        result = optimize(obj, x0, Newton(), options; autodiff=:forward)
        spec_g = unpack_ordered_probit_transition(Optim.minimizer(result), r_θ; monotone_I)
        Π_g = transition_from_ordered_probit(spec_g, r_θ, r_I; θ_grid, I_grid)

        # Assign to all transitions in the group
        for t in group
            new_specs[t] = spec_g
            Π_list[t] = Π_g
        end
    end

    return new_specs, Π_list
end

function parametric_mstep_transition_mp_grouped(specs_list::Vector{MixtureProbitTransition},
                                                 ξ_sum::Vector{Array{Float64,4}},
                                                 r_θ::Int, r_I::Int,
                                                 transition_groups::Vector{Vector{Int}};
                                                 I_grid::AbstractVector=collect(1.0:r_I),
                                                 options=DEFAULT_OPTIM_OPTIONS,
                                                 kwargs...)
    T_minus_1 = length(ξ_sum)
    K = specs_list[1].K
    new_specs = Vector{MixtureProbitTransition}(undef, T_minus_1)
    Π_list = Vector{Array{Float64,3}}(undef, T_minus_1)

    for group in transition_groups
        Q_sum_pooled = _qsum_from_xi(ξ_sum[group[1]])
        for t in group[2:end]
            Q_sum_pooled = Q_sum_pooled .+ _qsum_from_xi(ξ_sum[t])
        end

        x0 = pack_mixture_probit(specs_list[group[1]])
        obj = x -> transition_negll_mixture_probit_fast(x, Q_sum_pooled, r_θ, r_I, K; I_grid)
        result = optimize(obj, x0, LBFGS(), options; autodiff=:forward)
        spec_g = unpack_mixture_probit(Optim.minimizer(result), r_θ, K)
        Π_g = _build_mixture_probit_pi(Optim.minimizer(result), r_θ, r_I, K; I_grid)

        for t in group
            new_specs[t] = spec_g
            Π_list[t] = Π_g
        end
    end

    return new_specs, Π_list
end

function parametric_mstep_transition_mp_grouped(specs_list::Vector{MonotoneNNTransition},
                                                 ξ_sum::Vector{Array{Float64,4}},
                                                 r_θ::Int, r_I::Int,
                                                 transition_groups::Vector{Vector{Int}};
                                                 θ_grid::AbstractVector=collect(1.0:r_θ),
                                                 I_grid::AbstractVector=collect(1.0:r_I),
                                                 options=DEFAULT_OPTIM_OPTIONS,
                                                 kwargs...)
    T_minus_1 = length(ξ_sum)
    K = specs_list[1].K
    new_specs = Vector{MonotoneNNTransition}(undef, T_minus_1)
    Π_list = Vector{Array{Float64,3}}(undef, T_minus_1)

    for group in transition_groups
        Q_sum_pooled = _qsum_from_xi(ξ_sum[group[1]])
        for t in group[2:end]
            Q_sum_pooled = Q_sum_pooled .+ _qsum_from_xi(ξ_sum[t])
        end

        x0 = pack_monotone_nn(specs_list[group[1]])
        obj = x -> transition_negll_monotone_nn_fast(x, Q_sum_pooled, r_θ, r_I, K; θ_grid, I_grid)
        result = optimize(obj, x0, LBFGS(), options; autodiff=:forward)
        spec_g = unpack_monotone_nn(Optim.minimizer(result), r_θ, K)
        Π_g = _build_monotone_nn_pi(Optim.minimizer(result), r_θ, r_I, K; θ_grid, I_grid)

        for t in group
            new_specs[t] = spec_g
            Π_list[t] = Π_g
        end
    end

    return new_specs, Π_list
end

function parametric_mstep_transition_mp_grouped(specs_list::Vector{RobitNNTransition},
                                                 ξ_sum::Vector{Array{Float64,4}},
                                                 r_θ::Int, r_I::Int,
                                                 transition_groups::Vector{Vector{Int}};
                                                 θ_grid::AbstractVector=collect(1.0:r_θ),
                                                 I_grid::AbstractVector=collect(1.0:r_I),
                                                 options=DEFAULT_OPTIM_OPTIONS,
                                                 kwargs...)
    T_minus_1 = length(ξ_sum)
    K = specs_list[1].K
    ν = specs_list[1].ν
    new_specs = Vector{RobitNNTransition}(undef, T_minus_1)
    Π_list = Vector{Array{Float64,3}}(undef, T_minus_1)

    for group in transition_groups
        Q_sum_pooled = _qsum_from_xi(ξ_sum[group[1]])
        for t in group[2:end]
            Q_sum_pooled = Q_sum_pooled .+ _qsum_from_xi(ξ_sum[t])
        end

        x0 = pack_robit_nn(specs_list[group[1]])
        obj = x -> transition_negll_robit_nn_fast(x, Q_sum_pooled, r_θ, r_I, K, ν; θ_grid, I_grid)
        result = optimize(obj, x0, LBFGS(), options; autodiff=:forward)
        spec_g = unpack_robit_nn(Optim.minimizer(result), r_θ, K; ν)
        Π_g = _build_robit_nn_pi(Optim.minimizer(result), r_θ, r_I, K, ν; θ_grid, I_grid)

        for t in group
            new_specs[t] = spec_g
            Π_list[t] = Π_g
        end
    end

    return new_specs, Π_list
end

# ─────────────────────────────────────────────────────────────────
# Parametric CCP M-step (multi-period)
# ─────────────────────────────────────────────────────────────────

"""
    parametric_mstep_ccp_mp(specs_list, ξ_sum, r_θ, r_I; ...) → (new_specs, CCP_list)

Optimize ordered probit CCP parameters for all T-1 transitions.
"""
function parametric_mstep_ccp_mp(specs_list::Vector{OrderedProbitCCP},
                                  ξ_sum::Vector{Array{Float64,4}},
                                  r_θ::Int, r_I::Int;
                                  θ_grid::AbstractVector=collect(1.0:r_θ),
                                  I_grid::AbstractVector=collect(1.0:r_I),
                                  options=DEFAULT_OPTIM_OPTIONS)
    T_minus_1 = length(ξ_sum)
    new_specs = Vector{OrderedProbitCCP}(undef, T_minus_1)
    CCP_list = Vector{Array{Float64,3}}(undef, T_minus_1)

    for t in 1:T_minus_1
        CCP_sum_t = _ccpsum_from_xi(ξ_sum[t])
        new_specs[t], CCP_list[t] = parametric_mstep_ccp(
            specs_list[t], CCP_sum_t, r_θ, r_I; θ_grid, I_grid, options)
    end

    return new_specs, CCP_list
end

"""
    parametric_mstep_ccp_mp(specs_list::Vector{OrderedProbitCCPSimple}, ξ_sum, r_θ, r_I; ...)

Optimize simple ordered probit CCP parameters (I' depends only on θ') for all T-1 transitions.
"""
function parametric_mstep_ccp_mp(specs_list::Vector{OrderedProbitCCPSimple},
                                  ξ_sum::Vector{Array{Float64,4}},
                                  r_θ::Int, r_I::Int;
                                  θ_grid::AbstractVector=collect(1.0:r_θ),
                                  options=DEFAULT_OPTIM_OPTIONS)
    T_minus_1 = length(ξ_sum)
    new_specs = Vector{OrderedProbitCCPSimple}(undef, T_minus_1)
    CCP_list = Vector{Array{Float64,3}}(undef, T_minus_1)

    for t in 1:T_minus_1
        CCP_sum_t = _ccpsum_from_xi(ξ_sum[t])
        new_specs[t], CCP_list[t] = parametric_mstep_ccp(
            specs_list[t], CCP_sum_t, r_θ, r_I; θ_grid, options)
    end

    return new_specs, CCP_list
end

# ─────────────────────────────────────────────────────────────────
# Parametric initialization (multi-period)
# ─────────────────────────────────────────────────────────────────

"""
    initialize_params_mp_parametric(data, config; rng) → (params, specs)

Initialize multi-period parameters with parametric components.
Starts from nonparametric initialization, then replaces parametric components
with ordered probit emissions and/or parametric transitions.

Config fields: emission_θ, emission_I, transition (each :ordered_probit, :nonparametric, etc.)
"""
function initialize_params_mp_parametric(data, config::NamedTuple;
                                          rng::AbstractRNG=Random.default_rng())
    # Start with nonparametric base
    params = initialize_params_mp(data; rng)
    (; r_θ, r_I, T) = data

    specs = Dict{Symbol, Any}()

    # ── Skill emissions ──
    if get(config, :emission_θ, :nonparametric) == :ordered_probit
        probit_θ = [initialize_ordered_probit(data.κ_θ[t], r_θ; rng) for t in 1:T]
        P_θ = [emission_matrices_from_probit(probit_θ[t], r_θ) for t in 1:T]
        specs[:probit_θ] = probit_θ
        params = merge(params, (; P_θ))
    end

    # ── Investment emissions ──
    if get(config, :emission_I, :nonparametric) == :ordered_probit
        probit_I = [initialize_ordered_probit(data.κ_I[t], r_I; rng) for t in 1:T]
        P_I = [emission_matrices_from_probit(probit_I[t], r_I) for t in 1:T]
        specs[:probit_I] = probit_I
        params = merge(params, (; P_I))
    end

    # ── Transitions ──
    trans = get(config, :transition, :nonparametric)
    if trans == :flexible_lpm
        trans_specs = [initialize_flexible_lpm(r_θ; rng) for _ in 1:T-1]
        Π = [transition_from_flexible_lpm(trans_specs[t], r_θ, r_I) for t in 1:T-1]
        specs[:transition] = trans_specs
        params = merge(params, (; Π))
    elseif trans == :parsimonious_lpm
        trans_specs = [initialize_parsimonious_lpm(r_θ; rng) for _ in 1:T-1]
        Π = [transition_from_parsimonious_lpm(trans_specs[t], r_θ, r_I) for t in 1:T-1]
        specs[:transition] = trans_specs
        params = merge(params, (; Π))
    elseif trans == :ordered_probit
        trans_specs = [initialize_ordered_probit_transition(r_θ; rng) for _ in 1:T-1]
        Π = [transition_from_ordered_probit(trans_specs[t], r_θ, r_I) for t in 1:T-1]
        specs[:transition] = trans_specs
        params = merge(params, (; Π))
    elseif trans == :mixture_probit
        K_mix = get(config, :K, 2)
        trans_specs = [initialize_mixture_probit(r_θ, K_mix; rng, r_I) for _ in 1:T-1]
        Π = [transition_from_mixture_probit(trans_specs[t], r_θ, r_I) for t in 1:T-1]
        specs[:transition] = trans_specs
        params = merge(params, (; Π))
    elseif trans == :monotone_nn
        K_nn = get(config, :K, 3)
        trans_specs = [initialize_monotone_nn(r_θ, K_nn; rng, r_I) for _ in 1:T-1]
        Π = [transition_from_monotone_nn(trans_specs[t], r_θ, r_I) for t in 1:T-1]
        specs[:transition] = trans_specs
        params = merge(params, (; Π))
    elseif trans == :robit_nn
        K_nn = get(config, :K, 3)
        ν = get(config, :ν, DEFAULT_ROBIT_DF)
        trans_specs = [initialize_robit_nn(r_θ, K_nn; rng, r_I, ν) for _ in 1:T-1]
        Π = [transition_from_robit_nn(trans_specs[t], r_θ, r_I) for t in 1:T-1]
        specs[:transition] = trans_specs
        params = merge(params, (; Π))
    end

    # ── CCP ──
    ccp_type = get(config, :ccp, :nonparametric)
    if ccp_type == :ordered_probit
        ccp_specs = [initialize_ordered_probit_ccp(r_I; rng) for _ in 1:T-1]
        CCP = [ccp_from_ordered_probit(ccp_specs[t], r_θ, r_I) for t in 1:T-1]
        specs[:ccp] = ccp_specs
        params = merge(params, (; CCP))
    elseif ccp_type == :ordered_probit_simple
        ccp_specs = [initialize_ordered_probit_ccp_simple(r_I; rng) for _ in 1:T-1]
        CCP = [ccp_from_ordered_probit_simple(ccp_specs[t], r_θ, r_I) for t in 1:T-1]
        specs[:ccp] = ccp_specs
        params = merge(params, (; CCP))
    end

    # ── Initial distribution ──
    init_type = get(config, :initial, :nonparametric)
    if init_type == :ordered_probit
        init_I_spec = initialize_ordered_probit_initial_I(r_I; rng)
        P_I_θ = initial_I_from_ordered_probit(init_I_spec, r_θ, r_I)
        # Marginal p(θ): uniform initialization
        p_θ = ones(r_θ) / r_θ
        π0 = π0_from_initial_I(p_θ, P_I_θ)
        specs[:initial_I] = init_I_spec
        params = merge(params, (; π0))
    end

    return params, specs
end

# ─────────────────────────────────────────────────────────────────
# Parametric M-step dispatcher (multi-period)
# ─────────────────────────────────────────────────────────────────

"""
    maximization_step_mp_parametric(γ, ξ_sum, data, specs, config;
                                     transition_groups=nothing) → (params_new, specs_new)

M-step that dispatches each component to parametric or nonparametric update.
Returns updated params (dense matrices) and updated specs.

If `transition_groups` is provided (e.g. [[1,2,3],[4,5,6]]), transitions in
each group share a single production function Π, estimated by pooling
sufficient statistics across the group.
"""
function maximization_step_mp_parametric(γ::Array{Float64,4},
                                          ξ_sum::Vector{Array{Float64,4}},
                                          data, specs::Dict{Symbol,Any},
                                          config::NamedTuple;
                                          transition_groups::Union{Nothing,Vector{Vector{Int}}}=nothing,
                                          mstep_options=DEFAULT_OPTIM_OPTIONS)
    (; r_θ, r_I) = data
    new_specs = copy(specs)

    # ── Skill emissions ──
    if get(config, :emission_θ, :nonparametric) == :ordered_probit
        new_specs[:probit_θ], P_θ = parametric_mstep_emission_mp(
            specs[:probit_θ], γ, data, :theta, r_θ; options=mstep_options)
    else
        P_θ = update_emission_θ_mp(γ, data)
    end

    # ── Investment emissions ──
    if get(config, :emission_I, :nonparametric) == :ordered_probit
        new_specs[:probit_I], P_I = parametric_mstep_emission_mp(
            specs[:probit_I], γ, data, :I, r_I; options=mstep_options)
    else
        P_I = update_emission_I_mp(γ, data)
    end

    # ── Transitions ──
    trans = get(config, :transition, :nonparametric)
    mono_I = get(config, :monotone_I, false)
    if trans != :nonparametric && haskey(specs, :transition)
        if transition_groups !== nothing
            new_specs[:transition], Π = parametric_mstep_transition_mp_grouped(
                specs[:transition], ξ_sum, r_θ, r_I, transition_groups;
                monotone_I=mono_I, options=mstep_options)
        else
            new_specs[:transition], Π = parametric_mstep_transition_mp(
                specs[:transition], ξ_sum, r_θ, r_I;
                monotone_I=mono_I, options=mstep_options)
        end
    else
        if transition_groups !== nothing
            Π = update_Π_mp_grouped(γ, ξ_sum, transition_groups)
        else
            Π = update_Π_mp(γ, ξ_sum)
        end
    end

    # ── CCP ──
    ccp_type = get(config, :ccp, :nonparametric)
    if ccp_type in (:ordered_probit, :ordered_probit_simple) && haskey(specs, :ccp)
        new_specs[:ccp], CCP = parametric_mstep_ccp_mp(
            specs[:ccp], ξ_sum, r_θ, r_I; options=mstep_options)
    else
        CCP = update_CCP_mp(ξ_sum)
    end

    # ── Initial distribution ──
    init_type = get(config, :initial, :nonparametric)
    if init_type == :ordered_probit && haskey(specs, :initial_I)
        # Factor π0[θ, I] = p(θ) · P(I|θ)
        # p(θ): nonparametric marginal from γ at t=1
        N = size(γ, 4)
        p_θ = zeros(r_θ)
        for n in 1:N, kI in 1:r_I, kθ in 1:r_θ
            p_θ[kθ] += γ[1, kθ, kI, n]
        end
        p_θ ./= sum(p_θ)

        # Sufficient statistics for P(I|θ): π0_marginals[I, θ] = Σ_n γ_n(θ, I, t=1)
        π0_marginals = zeros(r_I, r_θ)
        for n in 1:N, kI in 1:r_I, kθ in 1:r_θ
            π0_marginals[kI, kθ] += γ[1, kθ, kI, n]
        end

        new_specs[:initial_I], P_I_θ = parametric_mstep_initial_I(
            specs[:initial_I], π0_marginals, r_θ, r_I; options=mstep_options)
        π0 = π0_from_initial_I(p_θ, P_I_θ)
    else
        π0 = update_π0_mp(γ)
    end

    params_new = (; Π, CCP, π0, P_θ, P_I)
    return params_new, new_specs
end

# ─────────────────────────────────────────────────────────────────
# Parametric EM loop (multi-period)
# ─────────────────────────────────────────────────────────────────

"""
    expectation_maximization_mp_parametric(data, params_init, specs, config; ...)
    → (params, γ, ξ_sum, specs)

EM algorithm with parametric model support for multi-period estimation.
The E-step is unchanged (forward-backward on dense matrices).
The M-step dispatches to parametric or nonparametric updates based on config.

If `transition_groups` is provided (e.g. [[1,2,3],[4,5,6]]), transitions in
each group share a single production function Π.
"""
function expectation_maximization_mp_parametric(data, params_init,
                                                 specs::Dict{Symbol,Any},
                                                 config::NamedTuple;
                                                 tol::Float64=1e-5,
                                                 maxiter::Int=2000,
                                                 verbose::Bool=true,
                                                 transition_groups::Union{Nothing,Vector{Vector{Int}}}=nothing,
                                                 mstep_options=DEFAULT_OPTIM_OPTIONS)
    (; r_θ, r_I, T, N) = data

    γ = zeros(T, r_θ, r_I, N)
    ξ_sum = [zeros(r_θ, r_I, r_θ, r_I) for _ in 1:T-1]

    params = params_init
    current_specs = specs
    err = Inf
    iter = 0

    while err > tol && iter < maxiter
        ll = expectation_step_mp!(γ, ξ_sum, params, data)

        params_new, current_specs = maximization_step_mp_parametric(
            γ, ξ_sum, data, current_specs, config;
            transition_groups=transition_groups,
            mstep_options=mstep_options)

        iter += 1

        # Convergence: max change in any Π_t
        err = maximum(
            maximum(abs.(params_new.Π[t] .- params.Π[t]))
            for t in 1:T-1
        )

        if verbose && (iter ≤ 5 || iter % 50 == 0)
            println("  iter $iter: max|ΔΠ| = $(round(err; sigdigits=4)), ll = $(round(ll; digits=2))")
        end

        params = (; params_new..., r_θ, r_I, T)
    end

    if verbose
        println("  Converged after $iter iterations (err = $(round(err; sigdigits=4)))")
    end

    # Final E-step at converged parameters
    ll_final = expectation_step_mp!(γ, ξ_sum, params, data)
    if verbose
        println("  Final log-likelihood: $(round(ll_final; digits=2))")
    end

    return params, γ, ξ_sum, current_specs
end

# ─────────────────────────────────────────────────────────────────
# Funnel restart: multi-period EM
# ─────────────────────────────────────────────────────────────────

"""
    run_em_restarts_mp(data; n_initial=20, n_semifinal=5,
                       phase1_iters=10, phase2_iters=100,
                       maxiter=2000, tol=1e-5, verbose=true,
                       config=nothing,
                       transition_groups=nothing,
                       mstep_options=DEFAULT_OPTIM_OPTIONS)

Run multi-period EM with funnel restart strategy.
If `config` is `nothing`, uses nonparametric EM; otherwise parametric.

Returns (best_params, best_γ, best_ξ_sum, best_ll, best_specs).
`best_specs` is `nothing` for nonparametric.
"""
function run_em_restarts_mp(data; n_initial::Int=20, n_semifinal::Int=5,
                            phase1_iters::Int=10, phase2_iters::Int=100,
                            maxiter::Int=2000, tol::Float64=1e-5, verbose::Bool=true,
                            config=nothing,
                            transition_groups::Union{Nothing,Vector{Vector{Int}}}=nothing,
                            mstep_options=DEFAULT_OPTIM_OPTIONS)
    (; r_θ, r_I, T, N) = data
    use_parametric = config !== nothing

    γ = zeros(T, r_θ, r_I, N)
    ξ_sum = [zeros(r_θ, r_I, r_θ, r_I) for _ in 1:T-1]

    # Phase 1: screen
    println("  Phase 1: screening $n_initial candidates × $phase1_iters iterations")
    candidates = []
    for i in 1:n_initial
        rng = MersenneTwister(42 + i)
        if use_parametric
            p_init, specs_init = initialize_params_mp_parametric(data, config; rng)
            p_est, γ_out, ξ_out, specs = expectation_maximization_mp_parametric(
                data, p_init, specs_init, config;
                maxiter=phase1_iters, verbose=false,
                transition_groups=transition_groups, mstep_options=mstep_options)
        else
            p_init = initialize_params_mp(data; rng)
            p_est, γ_out, ξ_out = expectation_maximization_mp(data, p_init;
                maxiter=phase1_iters, verbose=false)
            specs = nothing
        end
        ll = expectation_step_mp!(γ, ξ_sum, p_est, data)
        push!(candidates, (params=p_est, specs=specs, ll=ll))
        println("    Candidate $i/$n_initial: ll = $(round(ll; digits=2))")
    end
    sort!(candidates, by=c -> c.ll, rev=true)
    resize!(candidates, n_semifinal)
    println("  Phase 1 complete — keeping top $n_semifinal (ll range: $(round(candidates[end].ll; digits=2)) to $(round(candidates[1].ll; digits=2)))")

    # Phase 2: refine
    println("  Phase 2: refining $n_semifinal candidates × $phase2_iters iterations")
    for j in 1:length(candidates)
        if use_parametric
            p_est, γ_out, ξ_out, specs = expectation_maximization_mp_parametric(
                data, candidates[j].params, candidates[j].specs, config;
                maxiter=phase2_iters, verbose=false,
                transition_groups=transition_groups, mstep_options=mstep_options)
        else
            p_est, γ_out, ξ_out = expectation_maximization_mp(data, candidates[j].params;
                maxiter=phase2_iters, verbose=false)
            specs = nothing
        end
        ll = expectation_step_mp!(γ, ξ_sum, p_est, data)
        candidates[j] = (params=p_est, specs=specs, ll=ll)
        println("    Candidate $j/$n_semifinal: ll = $(round(ll; digits=2))")
    end
    sort!(candidates, by=c -> c.ll, rev=true)
    println("  Phase 2 complete — best ll = $(round(candidates[1].ll; digits=2))")

    # Phase 3: converge the winner
    println("  Phase 3: converging best candidate to tol=$tol")
    if use_parametric
        p_est, γ_final, ξ_final, specs_final = expectation_maximization_mp_parametric(
            data, candidates[1].params, candidates[1].specs, config;
            maxiter=maxiter, tol=tol, verbose=verbose,
            transition_groups=transition_groups, mstep_options=mstep_options)
    else
        p_est, γ_final, ξ_final = expectation_maximization_mp(data, candidates[1].params;
            maxiter=maxiter, tol=tol, verbose=verbose)
        specs_final = nothing
    end
    ll = expectation_step_mp!(γ_final, ξ_final, p_est, data)

    return p_est, γ_final, ξ_final, ll, specs_final
end
