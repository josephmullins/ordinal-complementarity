# ─────────────────────────────────────────────────────────────────
# EM estimation for the skill formation model
# Handles multiple measures per type, missing values (encoded as 0),
# and flexible numbers of categories per measure.
#
# Two-period model:
#   Period 1: observe multiple noisy measures of θ and I
#   Period 2: observe multiple noisy measures of θ'
#   Latent: (θ', θ, I) with transition Π(θ'|θ,I) and initial δ(θ,I)
# ─────────────────────────────────────────────────────────────────

using Random
using LinearAlgebra

# ─────────────────────────────────────────────────────────────────
# Parameter initialization
# ─────────────────────────────────────────────────────────────────

"""
    initialize_params(data; rng)

Create initial parameter guesses for EM.
- Binary skill measures (κ=2): monotone P(pass|θ) increasing in θ
- Other emission matrices: close to uniform with random perturbation
- Π: uniform transitions with slight diagonal bias
- π0: uniform initial distribution
- Shared skill emissions: P_Y = P_θ when data.shared_skill_emission is true
"""
function initialize_params(data; rng::AbstractRNG=Random.default_rng())
    (; r_θ, r_I, κ_θ, κ_I, κ_Y) = data
    shared = hasproperty(data, :shared_skill_emission) && data.shared_skill_emission

    # Emission matrices for skill measures
    P_θ = [begin
        if κ_θ[m] == 2
            # Binary: monotone initialization P(pass|θ) increasing in θ
            p_pass = [0.2 + 0.6 * (k-1) / max(r_θ-1, 1) + 0.05 * randn(rng) for k in 1:r_θ]
            p_pass = clamp.(p_pass, 0.05, 0.95)
            P = [1.0 .- p_pass'; p_pass']
        else
            P = 0.5 * ones(κ_θ[m], r_θ) / κ_θ[m] .+ 0.5 * rand(rng, κ_θ[m], r_θ)
            P ./= sum(P; dims=1)
            P
        end
    end for m in eachindex(κ_θ)]

    # Emission matrices for period 1 investment
    P_I = [begin
        P = 0.5 * ones(κ_I[m], r_I) / κ_I[m] .+ 0.5 * rand(rng, κ_I[m], r_I)
        P ./= sum(P; dims=1)
        P
    end for m in eachindex(κ_I)]

    # Emission matrices for period 2 skills
    if shared
        P_Y = P_θ  # shared with period 1
    else
        P_Y = [begin
            if κ_Y[m] == 2
                p_pass = [0.2 + 0.6 * (k-1) / max(r_θ-1, 1) + 0.05 * randn(rng) for k in 1:r_θ]
                p_pass = clamp.(p_pass, 0.05, 0.95)
                P = [1.0 .- p_pass'; p_pass']
            else
                P = 0.5 * ones(κ_Y[m], r_θ) / κ_Y[m] .+ 0.5 * rand(rng, κ_Y[m], r_θ)
                P ./= sum(P; dims=1)
                P
            end
        end for m in eachindex(κ_Y)]
    end

    # Transition matrix: slight diagonal dominance
    Π = ones(r_θ, r_θ, r_I) / r_θ
    for kI in 1:r_I, kθ in 1:r_θ
        Π[kθ, kθ, kI] += 0.3
    end
    for kI in 1:r_I, kθ in 1:r_θ
        Π[:, kθ, kI] ./= sum(Π[:, kθ, kI])
    end

    # Initial distribution: uniform
    π0 = ones(r_θ, r_I) / (r_θ * r_I)

    return (; Π, π0, P_θ, P_I, P_Y, r_θ, r_I)
end

# ─────────────────────────────────────────────────────────────────
# E-step: Compute posteriors
# ─────────────────────────────────────────────────────────────────

"""
    expectation_step_n!(Q, n, params, data) -> log-likelihood

Compute the posterior Q[θ', θ, I] for observation n.
Missing measures (encoded as 0) are skipped in the likelihood.
"""
function expectation_step_n!(Q, n::Int, params, data)
    (; Π, π0, P_θ, P_I, P_Y) = params
    (; X_θ, X_I, Y) = data

    for kI in axes(Q,3), kθ in axes(Q,2), kθ1 in axes(Q,1)
        q = π0[kθ, kI] * Π[kθ1, kθ, kI]

        # Period 1 skill measures
        for m in eachindex(P_θ)
            x = X_θ[m][n]
            if x > 0  # not missing
                q *= P_θ[m][x, kθ]
            end
        end

        # Period 1 investment measures
        for m in eachindex(P_I)
            x = X_I[m][n]
            if x > 0
                q *= P_I[m][x, kI]
            end
        end

        # Period 2 skill measures
        for m in eachindex(P_Y)
            y = Y[m][n]
            if y > 0
                q *= P_Y[m][y, kθ1]
            end
        end

        Q[kθ1, kθ, kI] = q
    end

    ll = sum(Q)
    if ll > 0
        Q ./= ll
    end
    return ll
end

"""
    expectation_step!(Q, params, data) -> total log-likelihood

Compute posteriors for all N observations.
Q is a 4D array: Q[θ', θ, I, n].
"""
function expectation_step!(Q, params, data)
    ll = 0.0
    for n in axes(Q, 4)
        @views lln = expectation_step_n!(Q[:,:,:,n], n, params, data)
        if lln > 0
            ll += log(lln)
        end
    end
    return ll
end

# ─────────────────────────────────────────────────────────────────
# M-step: Update parameters
# ─────────────────────────────────────────────────────────────────

"""
    update_emission_θ(Q, X_θ, κ_θ) -> Vector{Matrix}

Update emission matrices for period 1 skill measures.
The latent dimension is axis 2 of Q (marginalized over θ' and I).
"""
function update_emission_θ(Q, X_θ, κ_θ)
    r_θ = size(Q, 2)
    N = size(Q, 4)
    M = length(X_θ)

    P_θ = Vector{Matrix{Float64}}(undef, M)

    for m in 1:M
        P = zeros(κ_θ[m], r_θ)
        denom = zeros(r_θ)

        for n in 1:N
            x = X_θ[m][n]
            x == 0 && continue  # skip missing
            for kθ in 1:r_θ
                @views w = sum(Q[:, kθ, :, n])
                P[x, kθ] += w
                denom[kθ] += w
            end
        end

        for kθ in 1:r_θ
            if denom[kθ] > 0
                P[:, kθ] ./= denom[kθ]
            else
                P[:, kθ] .= 1.0 / κ_θ[m]
            end
        end

        P_θ[m] = P
    end

    return P_θ
end

"""
    update_emission_Y(Q, Y, κ_Y) -> Vector{Matrix}

Update emission matrices for period 2 skill measures.
The latent dimension is axis 1 of Q (θ', marginalized over θ and I).
"""
function update_emission_Y(Q, Y, κ_Y)
    r_θ = size(Q, 1)
    N = size(Q, 4)
    M = length(Y)

    P_Y = Vector{Matrix{Float64}}(undef, M)

    for m in 1:M
        P = zeros(κ_Y[m], r_θ)
        denom = zeros(r_θ)

        for n in 1:N
            y = Y[m][n]
            y == 0 && continue
            for kθ1 in 1:r_θ
                @views w = sum(Q[kθ1, :, :, n])
                P[y, kθ1] += w
                denom[kθ1] += w
            end
        end

        for kθ1 in 1:r_θ
            if denom[kθ1] > 0
                P[:, kθ1] ./= denom[kθ1]
            else
                P[:, kθ1] .= 1.0 / κ_Y[m]
            end
        end

        P_Y[m] = P
    end

    return P_Y
end

"""
    update_emission_I(Q, X_I, κ_I) -> Vector{Matrix}

Update emission matrices for investment measures.
The latent dimension is axis 3 of Q (I, marginalized over θ and θ').
"""
function update_emission_I(Q, X_I, κ_I)
    r_I = size(Q, 3)
    N = size(Q, 4)
    M = length(X_I)

    P_I = Vector{Matrix{Float64}}(undef, M)

    for m in 1:M
        P = zeros(κ_I[m], r_I)
        denom = zeros(r_I)

        for n in 1:N
            x = X_I[m][n]
            x == 0 && continue
            for kI in 1:r_I
                @views w = sum(Q[:, :, kI, n])
                P[x, kI] += w
                denom[kI] += w
            end
        end

        for kI in 1:r_I
            if denom[kI] > 0
                P[:, kI] ./= denom[kI]
            else
                P[:, kI] .= 1.0 / κ_I[m]
            end
        end

        P_I[m] = P
    end

    return P_I
end

"""
    update_Π(Q) -> Array{Float64, 3}

Update transition matrix Π[θ', θ, I].
"""
function update_Π(Q)
    r1, r2, r3, _ = size(Q)
    Π_numer = sum(Q; dims=4)
    Π_denom = sum(Q; dims=(1,4))
    Π = reshape(Π_numer ./ Π_denom, r1, r2, r3)
    # Handle division by zero
    for kI in 1:r3, kθ in 1:r2
        s = sum(Π[:, kθ, kI])
        if !isfinite(s) || s ≤ 0
            Π[:, kθ, kI] .= 1.0 / r1
        end
    end
    return Π
end

"""
    update_π0(Q) -> Matrix{Float64}

Update initial joint distribution π0[θ, I].
"""
function update_π0(Q)
    _, r_θ, r_I, N = size(Q)
    # Marginal over θ' (axis 1), sum over observations
    π0 = reshape(sum(Q; dims=(1,4)), r_θ, r_I) ./ N
    return π0
end

"""
    update_emission_skill_shared(Q, X_θ, Y, κ) -> Vector{Matrix}

Update emission matrices for skill measures, pooling evidence from
period 1 (latent θ on axis 2) and period 2 (latent θ' on axis 1).
Used when P_θ = P_Y (shared emission assumption).
"""
function update_emission_skill_shared(Q, X_θ, Y, κ)
    r_θ = size(Q, 2)
    N = size(Q, 4)
    M = length(X_θ)

    P = Vector{Matrix{Float64}}(undef, M)

    for m in 1:M
        Pm = zeros(κ[m], r_θ)
        denom = zeros(r_θ)

        for n in 1:N
            # Period 1: X_θ[m][n] weighted by marginal over (θ', I) → axis 2
            x = X_θ[m][n]
            if x > 0
                for k in 1:r_θ
                    @views w = sum(Q[:, k, :, n])
                    Pm[x, k] += w
                    denom[k] += w
                end
            end

            # Period 2: Y[m][n] weighted by marginal over (θ, I) → axis 1
            y = Y[m][n]
            if y > 0
                for k in 1:r_θ
                    @views w = sum(Q[k, :, :, n])
                    Pm[y, k] += w
                    denom[k] += w
                end
            end
        end

        for k in 1:r_θ
            if denom[k] > 0
                Pm[:, k] ./= denom[k]
            else
                Pm[:, k] .= 1.0 / κ[m]
            end
        end

        P[m] = Pm
    end

    return P
end

function maximization_step(Q, data; shared_skill_emission::Bool=false)
    (; X_θ, X_I, Y, κ_θ, κ_I, κ_Y) = data
    if shared_skill_emission
        P_shared = update_emission_skill_shared(Q, X_θ, Y, κ_θ)
        P_θ = P_shared
        P_Y = P_shared
    else
        P_θ = update_emission_θ(Q, X_θ, κ_θ)
        P_Y = update_emission_Y(Q, Y, κ_Y)
    end
    P_I = update_emission_I(Q, X_I, κ_I)
    Π = update_Π(Q)
    π0 = update_π0(Q)
    return (; P_θ, P_Y, P_I, Π, π0)
end

# ─────────────────────────────────────────────────────────────────
# Main EM loop
# ─────────────────────────────────────────────────────────────────

"""
    expectation_maximization(data, params_init; tol, maxiter, verbose, shared_skill_emission)

Run EM algorithm. Returns estimated parameters and posteriors Q.
Set shared_skill_emission=true to constrain P_Y = P_θ (same emission for both periods).
"""
function expectation_maximization(data, params_init;
    tol::Float64=1e-5, maxiter::Int=1000, verbose::Bool=true,
    shared_skill_emission::Bool=false)

    (; r_θ, r_I, N) = data
    Q = zeros(r_θ, r_θ, r_I, N)

    params = params_init
    err = Inf
    iter = 0

    while err > tol && iter < maxiter
        ll = expectation_step!(Q, params, data)
        params_new = maximization_step(Q, data; shared_skill_emission)
        params_new = (; params_new..., r_θ=params_init.r_θ, r_I=params_init.r_I)
        iter += 1

        err = maximum(abs.(params_new.Π .- params.Π))

        if verbose && (iter ≤ 5 || iter % 50 == 0)
            println("  iter $iter: max|ΔΠ| = $(round(err; sigdigits=4)), ll = $(round(ll; digits=2))")
        end

        params = params_new
    end

    if verbose
        println("  Converged after $iter iterations (err = $(round(err; sigdigits=4)))")
    end

    # Final E-step to get posteriors at estimated parameters
    expectation_step!(Q, params, data)

    return params, Q
end

# ─────────────────────────────────────────────────────────────────
# Score and Hessian for inference
# ─────────────────────────────────────────────────────────────────

"""
    get_score(Q, Π) -> Matrix{Float64}

Compute the n_p × N score matrix for the free parameters of Π.
Free parameters: Π[1:r_θ-1, :, :] (since the last row is determined by normalization).
"""
function get_score(Q, Π)
    r_θ, _, r_I = size(Π)
    N = size(Q, 4)
    p_idx = LinearIndices((r_θ-1, r_θ, r_I))
    np = prod(size(p_idx))
    dL = zeros(np, N)

    for n in 1:N
        for kI in 1:r_I, kθ in 1:r_θ, kθ1 in 1:r_θ-1
            ip = p_idx[kθ1, kθ, kI]
            dL[ip, n] = Q[kθ1, kθ, kI, n] / Π[kθ1, kθ, kI] -
                        Q[r_θ, kθ, kI, n] / Π[r_θ, kθ, kI]
        end
    end
    return dL
end

"""
    get_hessian_Π(Q, Π) -> Matrix{Float64}

Compute the Hessian of the log-likelihood w.r.t. the free parameters of Π.
"""
function get_hessian_Π(Q, Π)
    r_θ, _, r_I = size(Π)
    N = size(Q, 4)
    Qsum = reshape(sum(Q; dims=4) / N, r_θ, r_θ, r_I)

    p_idx = LinearIndices((r_θ-1, r_θ, r_I))
    np = prod(size(p_idx))
    H = zeros(np, np)
    for kI in 1:r_I, kθ in 1:r_θ, kθ1 in 1:r_θ-1
        i_p = p_idx[kθ1, kθ, kI]
        H[i_p, i_p] = -Qsum[kθ1, kθ, kI] / Π[kθ1, kθ, kI]^2
        for kθ2 in 1:r_θ-1
            i_p2 = p_idx[kθ2, kθ, kI]
            H[i_p, i_p2] += Qsum[r_θ, kθ, kI] / Π[r_θ, kθ, kI]^2
        end
    end
    return H
end

# ─────────────────────────────────────────────────────────────────
# Survivor function standard errors (delta method)
# ─────────────────────────────────────────────────────────────────

"""
    survivor_se_all(Var_free, Π) -> Array{Float64, 3}

Compute standard errors for all survivor function values F̄(k|θ,I) = P(θ'>k|θ,I)
using the delta method, given the variance-covariance matrix of the free
parameters of Π.

`Var_free` is (n_free × n_free) where n_free = (r_θ-1) × r_θ × r_I,
indexed by `LinearIndices((r_θ-1, r_θ, r_I))`.

The survivor function F̄(k|θ,I) = 1 - Σ_{j=1}^{k} Π[j,θ,I] is linear in
these parameters, so:

    Var(F̄(k|θ,I)) = Σ_{j≤k, j'≤k} Var_free[idx(j,θ,I), idx(j',θ,I)]

Returns: SE array of size (r_θ-1, r_θ, r_I) indexed by (cutoff k, θ, I).
"""
function survivor_se_all(Var_free::AbstractMatrix, Π::Array{Float64,3})
    r_θ, _, r_I = size(Π)
    p_idx = LinearIndices((r_θ-1, r_θ, r_I))

    SE = zeros(r_θ-1, r_θ, r_I)

    for kI in 1:r_I, kθ in 1:r_θ, k in 1:r_θ-1
        var_s = 0.0
        for j in 1:k, j2 in 1:k
            var_s += Var_free[p_idx[j, kθ, kI], p_idx[j2, kθ, kI]]
        end
        SE[k, kθ, kI] = sqrt(max(0.0, var_s))
    end

    return SE
end

"""
    survivor_se_all(Σ_score, Π, N) -> Array{Float64, 3}

Convenience method: computes Var(Π̂_free) from the nonparametric score covariance
via `pi_free_var_nonparametric`, then calls `survivor_se_all(Var_free, Π)`.

For parametric models, use `pi_free_var_parametric` to compute Var_free instead.
"""
function survivor_se_all(Σ_score::AbstractMatrix, Π::Array{Float64,3}, N::Int)
    Var_free = pi_free_var_nonparametric(Σ_score, N)
    return survivor_se_all(Var_free, Π)
end

# ─────────────────────────────────────────────────────────────────
# Normalize latent factor ordering
# ─────────────────────────────────────────────────────────────────

"""
    skill_index(P_θ, r_θ) -> Vector{Float64}

Compute an average "skill score" for each latent θ category.
For each measure m, compute E[measure | θ=k] normalized to [0,1],
then average across measures. For binary items (κ=2), this is just P(pass|θ).
"""
function skill_index(P_θ, r_θ)
    M = length(P_θ)
    idx = zeros(r_θ)
    for k in 1:r_θ
        for m in 1:M
            κ_m = size(P_θ[m], 1)
            # E[measure | θ=k], normalized to [0,1]
            e_val = sum(j * P_θ[m][j, k] for j in 1:κ_m) / κ_m
            idx[k] += e_val
        end
        idx[k] /= M
    end
    return idx
end

"""
    investment_index(P_I, r_I) -> Vector{Float64}

Compute an average "investment score" for each latent I category.
For each measure m, compute E[measure | I=k] normalized to [0,1],
then average across measures.
"""
function investment_index(P_I, r_I)
    M = length(P_I)
    idx = zeros(r_I)
    for k in 1:r_I
        for m in 1:M
            κ_m = size(P_I[m], 1)
            e_val = sum(j * P_I[m][j, k] for j in 1:κ_m) / κ_m
            idx[k] += e_val
        end
        idx[k] /= M
    end
    return idx
end

"""
    investment_index_pi(Π, r_θ, r_I) -> Vector{Float64}

Compute an investment quality index from the transition matrix Π.
For each I, compute E[θ' | I] averaged over θ (uniform weights).
Higher values = investment type shifts skill distribution upward.
"""
function investment_index_pi(Π, r_θ, r_I)
    idx = zeros(r_I)
    for kI in 1:r_I
        for kθ in 1:r_θ
            # E[θ' | θ=kθ, I=kI]
            e_theta_prime = sum(kθ1 * Π[kθ1, kθ, kI] for kθ1 in 1:r_θ)
            idx[kI] += e_theta_prime
        end
        idx[kI] /= r_θ  # average over θ
    end
    return idx
end

"""
    normalize_params(params, Q, data) -> (params_normalized, Q_normalized)

Relabel latent categories so that:
- θ=1 < θ=2 < ... by average P_θ emission (period 1 skill ordering)
- θ'=1 < θ'=2 < ... by average P_Y emission (period 2 skill ordering)
- I=1 < I=2 < ... by average E[θ'|I] from Π (investment quality)

θ and θ' are permuted INDEPENDENTLY because the model does not constrain
P_θ = P_Y. The likelihood is invariant to separate relabelings of θ and θ'.
"""
function normalize_params(params, Q, data)
    (; Π, π0, P_θ, P_I, P_Y, r_θ, r_I) = params

    # Separate orderings for θ (period 1) and θ' (period 2)
    θ_idx  = skill_index(P_θ, r_θ)
    θ′_idx = skill_index(P_Y, r_θ)

    θ_order  = sortperm(θ_idx)
    θ′_order = sortperm(θ′_idx)

    # Investment ordering: compute E[θ'|I] on the θ'-reordered Π so that
    # the cardinal labels 1,...,r_θ on the θ' axis are correctly ordered.
    Π_θ′sorted = Π[θ′_order, :, :]
    I_idx  = investment_index_pi(Π_θ′sorted, r_θ, r_I)
    I_order  = sortperm(I_idx)

    println("  Skill index (θ, period 1):   ", [round(θ_idx[k]; digits=4) for k in 1:r_θ])
    println("  θ ordering:                  ", θ_order, " (1=lowest, $(r_θ)=highest)")
    println("  Skill index (θ', period 2):  ", [round(θ′_idx[k]; digits=4) for k in 1:r_θ])
    println("  θ' ordering:                 ", θ′_order, " (1=lowest, $(r_θ)=highest)")
    println("  Investment index (E[θ'|I]):  ", [round(I_idx[k]; digits=4) for k in 1:r_I])
    println("  Investment ordering:         ", I_order, " (1=lowest, $(r_I)=highest)")

    # Permute Π[θ', θ, I]: axis 1 = θ', axis 2 = θ, axis 3 = I
    Π_new = Π[θ′_order, θ_order, I_order]

    # Permute π0[θ, I]
    π0_new = π0[θ_order, I_order]

    # Permute emission matrices: P_θ columns indexed by θ, P_Y columns by θ'
    P_θ_new = [P_θ[m][:, θ_order] for m in eachindex(P_θ)]
    P_Y_new = [P_Y[m][:, θ′_order] for m in eachindex(P_Y)]

    # Permute investment emission matrices: columns indexed by I
    P_I_new = [P_I[m][:, I_order] for m in eachindex(P_I)]

    # Permute Q[θ', θ, I, n]
    Q_new = Q[θ′_order, θ_order, I_order, :]

    params_new = (; Π=Π_new, π0=π0_new, P_θ=P_θ_new, P_I=P_I_new, P_Y=P_Y_new, r_θ, r_I)
    return params_new, Q_new
end

# ─────────────────────────────────────────────────────────────────
# Display estimated parameters
# ─────────────────────────────────────────────────────────────────

function display_results(params, data; max_emission_display::Int=10)
    (; Π, π0, P_θ, P_I, P_Y, r_θ, r_I) = params

    println("\n" * "="^60)
    println("  ESTIMATED PARAMETERS")
    println("="^60)

    println("\n  Initial distribution π0(θ, I):")
    for kθ in 1:r_θ
        print("    θ=$kθ: ")
        for kI in 1:r_I
            print("  I=$kI: $(round(π0[kθ,kI]; digits=4))")
        end
        println()
    end

    println("\n  Transition matrix Π(θ'|θ, I):")
    for kI in 1:r_I
        println("    I = $kI:")
        for kθ1 in 1:r_θ
            print("      θ'=$kθ1: ")
            for kθ in 1:r_θ
                print("  θ=$kθ: $(round(Π[kθ1,kθ,kI]; digits=4))")
            end
            println()
        end
    end

    # Skill emissions: show summary for many measures
    n_skill = length(P_θ)
    has_names = hasproperty(data, :measure_names_θ)

    if n_skill > max_emission_display
        println("\n  Skill emission matrices ($(n_skill) measures, showing first $max_emission_display):")
        show_idx = 1:max_emission_display
    else
        println("\n  Skill emission matrices ($(n_skill) measures):")
        show_idx = 1:n_skill
    end
    for m in show_idx
        name = has_names ? data.measure_names_θ[m] : "measure_$m"
        print("    $name:  P(pass|θ)=")
        for kθ in 1:r_θ
            print("  $(round(P_θ[m][end, kθ]; digits=3))")
        end
        println()
    end
    if n_skill > max_emission_display && has_names
        # Show monotonicity summary
        n_monotone = sum(1:n_skill) do m
            p_pass = [P_θ[m][end, k] for k in 1:r_θ]
            issorted(p_pass)
        end
        println("    ... ($n_monotone / $n_skill have monotone P(pass|θ))")
        # Print all measure names
        println("\n  All skill measure names:")
        for (m, name) in enumerate(data.measure_names_θ)
            p_pass = [P_θ[m][end, k] for k in 1:r_θ]
            mono = issorted(p_pass) ? "✓" : " "
            println("    [$mono] $m. $name:  P(pass|θ)= $(join([round(p; digits=3) for p in p_pass], "  "))")
        end
    end

    # Investment emissions
    println("\n  Investment emission matrices:")
    n_inv = length(P_I)
    for m in 1:n_inv
        name = has_names && hasproperty(data, :measure_names_I) ? data.measure_names_I[m] : "investment_$m"
        println("    $name (κ=$(size(P_I[m],1))):")
        for k in 1:size(P_I[m], 1)
            print("      cat=$k: ")
            for kI in 1:r_I
                print("  I=$kI: $(round(P_I[m][k,kI]; digits=4))")
            end
            println()
        end
    end
end

# ─────────────────────────────────────────────────────────────────
# Parametric EM estimation
# ─────────────────────────────────────────────────────────────────
#
# These functions extend the EM to support parametric emission and
# transition models. The E-step is unchanged — parametric parameters
# are converted to dense matrices each iteration. Only the M-step
# differs: Optim.jl replaces closed-form updates for parametric
# components.
#
# Requires: include("parametric_models.jl") before this file.
# ─────────────────────────────────────────────────────────────────

"""
    ParametricConfig

Configuration for which model components use parametric vs nonparametric forms.

Fields (all Symbol):
- `emission_θ`:  `:nonparametric` or `:ordered_probit`
- `emission_Y`:  `:nonparametric` or `:ordered_probit`
- `emission_I`:  `:nonparametric` or `:ordered_probit`
- `transition`:  `:nonparametric`, `:flexible_lpm`, or `:parsimonious_lpm`
"""
const ParametricConfig = @NamedTuple{
    emission_θ::Symbol,
    emission_Y::Symbol,
    emission_I::Symbol,
    transition::Symbol,
}

const DEFAULT_PARAMETRIC_CONFIG = (
    emission_θ  = :nonparametric,
    emission_Y  = :nonparametric,
    emission_I  = :nonparametric,
    transition  = :nonparametric,
)

"""
    initialize_params_parametric(data, config; rng) -> (params, specs)

Initialize parameters, creating parametric specs for configured components
and converting them to dense matrices.
"""
function initialize_params_parametric(data, config::NamedTuple;
                                       rng::AbstractRNG=Random.default_rng())
    # Start with nonparametric initialization for all components
    params = initialize_params(data; rng)
    (; r_θ, r_I, κ_θ, κ_I, κ_Y) = data

    specs = Dict{Symbol, Any}()

    # Emission θ (period 1 skills)
    if config.emission_θ == :ordered_probit
        specs[:probit_θ] = initialize_ordered_probit(κ_θ, r_θ; rng)
        P_θ = emission_matrices_from_probit(specs[:probit_θ], r_θ)
        params = merge(params, (; P_θ))
    end

    # Emission Y (period 2 skills)
    if config.emission_Y == :ordered_probit
        specs[:probit_Y] = initialize_ordered_probit(κ_Y, r_θ; rng)
        P_Y = emission_matrices_from_probit(specs[:probit_Y], r_θ)
        params = merge(params, (; P_Y))
    end

    # Emission I (investment)
    if config.emission_I == :ordered_probit
        specs[:probit_I] = initialize_ordered_probit(κ_I, r_I; rng)
        P_I = emission_matrices_from_probit(specs[:probit_I], r_I)
        params = merge(params, (; P_I))
    end

    # Transition
    if config.transition == :flexible_lpm
        specs[:transition] = initialize_flexible_lpm(r_θ; rng)
        Π = transition_from_flexible_lpm(specs[:transition], r_θ, r_I)
        params = merge(params, (; Π))
    elseif config.transition == :parsimonious_lpm
        specs[:transition] = initialize_parsimonious_lpm(r_θ; rng)
        Π = transition_from_parsimonious_lpm(specs[:transition], r_θ, r_I)
        params = merge(params, (; Π))
    elseif config.transition == :ordered_probit
        specs[:transition] = initialize_ordered_probit_transition(r_θ; rng)
        Π = transition_from_ordered_probit(specs[:transition], r_θ, r_I)
        params = merge(params, (; Π))
    elseif config.transition == :mixture_probit
        K_mix = get(config, :K, 2)
        specs[:transition] = initialize_mixture_probit(r_θ, K_mix; rng, r_I)
        Π = transition_from_mixture_probit(specs[:transition], r_θ, r_I)
        params = merge(params, (; Π))
    end

    return params, specs
end

"""
    maximization_step_parametric(Q, data, specs, config) -> (params_new, specs_new)

M-step that dispatches each component to nonparametric or parametric update.
Returns updated params (always containing full dense matrices) and updated specs.
"""
function maximization_step_parametric(Q::Array{Float64,4}, data, specs::Dict{Symbol,Any},
                                       config::NamedTuple;
                                       shared_skill_emission::Bool=false,
                                       mstep_options=DEFAULT_OPTIM_OPTIONS)
    (; X_θ, X_I, Y, κ_θ, κ_I, κ_Y) = data
    r_θ = size(Q, 2)
    r_I = size(Q, 3)

    new_specs = copy(specs)

    # ── Period 1 skill emissions ──
    if config.emission_θ == :ordered_probit
        new_specs[:probit_θ], P_θ = parametric_mstep_emission(
            specs[:probit_θ], Q, X_θ, κ_θ, :theta, r_θ; options=mstep_options)
    else
        P_θ = update_emission_θ(Q, X_θ, κ_θ)
    end

    # ── Period 2 skill emissions ──
    if config.emission_Y == :ordered_probit
        new_specs[:probit_Y], P_Y = parametric_mstep_emission(
            specs[:probit_Y], Q, Y, κ_Y, :theta_prime, r_θ; options=mstep_options)
    else
        P_Y = update_emission_Y(Q, Y, κ_Y)
    end

    # ── Investment emissions ──
    if config.emission_I == :ordered_probit
        new_specs[:probit_I], P_I = parametric_mstep_emission(
            specs[:probit_I], Q, X_I, κ_I, :I, r_I; options=mstep_options)
    else
        P_I = update_emission_I(Q, X_I, κ_I)
    end

    # ── Transition ──
    if config.transition in (:flexible_lpm, :parsimonious_lpm, :ordered_probit, :mixture_probit)
        new_specs[:transition], Π = parametric_mstep_transition(
            specs[:transition], Q, r_θ, r_I; options=mstep_options)
    else
        Π = update_Π(Q)
    end

    # ── Initial distribution (always nonparametric) ──
    π0 = update_π0(Q)

    params_new = (; Π, π0, P_θ, P_I, P_Y, r_θ, r_I)
    return params_new, new_specs
end

"""
    expectation_maximization_parametric(data, params_init, specs, config; ...) -> (params, Q, specs)

EM algorithm with parametric model support. The E-step is unchanged (uses
dense matrices). The M-step dispatches to parametric or nonparametric updates.
"""
function expectation_maximization_parametric(data, params_init, specs::Dict{Symbol,Any},
                                              config::NamedTuple;
                                              tol::Float64=1e-5, maxiter::Int=1000,
                                              verbose::Bool=true,
                                              shared_skill_emission::Bool=false,
                                              mstep_options=DEFAULT_OPTIM_OPTIONS)
    (; r_θ, r_I, N) = data
    Q = zeros(r_θ, r_θ, r_I, N)

    params = params_init
    current_specs = specs
    err = Inf
    iter = 0

    while err > tol && iter < maxiter
        ll = expectation_step!(Q, params, data)
        params_new, current_specs = maximization_step_parametric(
            Q, data, current_specs, config; shared_skill_emission, mstep_options)
        params_new = (; params_new..., r_θ=params_init.r_θ, r_I=params_init.r_I)
        iter += 1

        err = maximum(abs.(params_new.Π .- params.Π))

        if verbose && (iter ≤ 5 || iter % 50 == 0)
            println("  iter $iter: max|ΔΠ| = $(round(err; sigdigits=4)), ll = $(round(ll; digits=2))")
        end

        params = params_new
    end

    if verbose
        println("  Converged after $iter iterations (err = $(round(err; sigdigits=4)))")
    end

    # Final E-step at estimated parameters
    expectation_step!(Q, params, data)

    return params, Q, current_specs
end

# ─────────────────────────────────────────────────────────────────
# Funnel restart: nonparametric two-period EM
# ─────────────────────────────────────────────────────────────────

"""
    run_em_restarts(data; n_initial=20, n_semifinal=5,
                    phase1_iters=10, phase2_iters=100,
                    maxiter=2000, tol=1e-5, verbose=true,
                    shared_skill_emission=false)

Run EM with funnel restart strategy:
  Phase 1: `n_initial` candidates × `phase1_iters` iterations → keep top `n_semifinal`
  Phase 2: `n_semifinal` candidates × `phase2_iters` iterations → keep top 1
  Phase 3: 1 candidate → run to convergence (maxiter/tol)

Returns (best_params, best_Q, best_ll).
"""
function run_em_restarts(data; n_initial::Int=20, n_semifinal::Int=5,
                         phase1_iters::Int=10, phase2_iters::Int=100,
                         maxiter::Int=2000, tol::Float64=1e-5, verbose::Bool=true,
                         shared_skill_emission::Bool=false)
    (; r_θ, r_I, N) = data
    Q = zeros(r_θ, r_θ, r_I, N)

    # Phase 1: screen
    println("  Phase 1: screening $n_initial candidates × $phase1_iters iterations")
    candidates = Vector{NamedTuple{(:params, :ll), Tuple{Any, Float64}}}(undef, n_initial)
    for i in 1:n_initial
        rng = MersenneTwister(42 + i)
        p_init = initialize_params(data; rng)
        p_est, _ = expectation_maximization(data, p_init;
            maxiter=phase1_iters, verbose=false, shared_skill_emission=shared_skill_emission)
        ll = expectation_step!(Q, p_est, data)
        candidates[i] = (params=p_est, ll=ll)
        println("    Candidate $i/$n_initial: ll = $(round(ll; digits=2))")
    end
    sort!(candidates, by=c -> c.ll, rev=true)
    resize!(candidates, n_semifinal)
    println("  Phase 1 complete — keeping top $n_semifinal (ll range: $(round(candidates[end].ll; digits=2)) to $(round(candidates[1].ll; digits=2)))")

    # Phase 2: refine
    println("  Phase 2: refining $n_semifinal candidates × $phase2_iters iterations")
    for j in 1:length(candidates)
        p_est, _ = expectation_maximization(data, candidates[j].params;
            maxiter=phase2_iters, verbose=false, shared_skill_emission=shared_skill_emission)
        ll = expectation_step!(Q, p_est, data)
        candidates[j] = (params=p_est, ll=ll)
        println("    Candidate $j/$n_semifinal: ll = $(round(ll; digits=2))")
    end
    sort!(candidates, by=c -> c.ll, rev=true)
    println("  Phase 2 complete — best ll = $(round(candidates[1].ll; digits=2))")

    # Phase 3: converge the winner
    println("  Phase 3: converging best candidate to tol=$tol")
    p_est, Q_final = expectation_maximization(data, candidates[1].params;
        maxiter=maxiter, tol=tol, verbose=verbose, shared_skill_emission=shared_skill_emission)
    ll = expectation_step!(Q_final, p_est, data)

    return p_est, Q_final, ll
end

# ─────────────────────────────────────────────────────────────────
# Funnel restart: parametric two-period EM
# ─────────────────────────────────────────────────────────────────

"""
    run_em_restarts_parametric(data, config; n_initial=20, n_semifinal=5,
                               phase1_iters=10, phase2_iters=100,
                               maxiter=2000, tol=1e-5, verbose=true,
                               shared_skill_emission=false,
                               mstep_options=DEFAULT_OPTIM_OPTIONS)

Funnel restart strategy for parametric EM.
Returns (best_params, best_Q, best_ll, best_specs).
"""
function run_em_restarts_parametric(data, config::NamedTuple;
                                     n_initial::Int=20, n_semifinal::Int=5,
                                     phase1_iters::Int=10, phase2_iters::Int=100,
                                     maxiter::Int=2000, tol::Float64=1e-5, verbose::Bool=true,
                                     shared_skill_emission::Bool=false,
                                     mstep_options=DEFAULT_OPTIM_OPTIONS)
    (; r_θ, r_I, N) = data
    Q = zeros(r_θ, r_θ, r_I, N)

    # Phase 1: screen
    println("  Phase 1: screening $n_initial candidates × $phase1_iters iterations")
    candidates = Vector{NamedTuple{(:params, :specs, :ll), Tuple{Any, Dict{Symbol,Any}, Float64}}}(undef, n_initial)
    for i in 1:n_initial
        rng = MersenneTwister(42 + i)
        p_init, specs_init = initialize_params_parametric(data, config; rng)
        p_est, _, specs = expectation_maximization_parametric(
            data, p_init, specs_init, config;
            maxiter=phase1_iters, verbose=false, shared_skill_emission=shared_skill_emission,
            mstep_options=mstep_options)
        ll = expectation_step!(Q, p_est, data)
        candidates[i] = (params=p_est, specs=specs, ll=ll)
        println("    Candidate $i/$n_initial: ll = $(round(ll; digits=2))")
    end
    sort!(candidates, by=c -> c.ll, rev=true)
    resize!(candidates, n_semifinal)
    println("  Phase 1 complete — keeping top $n_semifinal (ll range: $(round(candidates[end].ll; digits=2)) to $(round(candidates[1].ll; digits=2)))")

    # Phase 2: refine
    println("  Phase 2: refining $n_semifinal candidates × $phase2_iters iterations")
    for j in 1:length(candidates)
        p_est, _, specs = expectation_maximization_parametric(
            data, candidates[j].params, candidates[j].specs, config;
            maxiter=phase2_iters, verbose=false, shared_skill_emission=shared_skill_emission,
            mstep_options=mstep_options)
        ll = expectation_step!(Q, p_est, data)
        candidates[j] = (params=p_est, specs=specs, ll=ll)
        println("    Candidate $j/$n_semifinal: ll = $(round(ll; digits=2))")
    end
    sort!(candidates, by=c -> c.ll, rev=true)
    println("  Phase 2 complete — best ll = $(round(candidates[1].ll; digits=2))")

    # Phase 3: converge the winner
    println("  Phase 3: converging best candidate to tol=$tol")
    p_est, Q_final, specs_final = expectation_maximization_parametric(
        data, candidates[1].params, candidates[1].specs, config;
        maxiter=maxiter, tol=tol, verbose=verbose, shared_skill_emission=shared_skill_emission,
        mstep_options=mstep_options)
    ll = expectation_step!(Q_final, p_est, data)

    return p_est, Q_final, ll, specs_final
end
