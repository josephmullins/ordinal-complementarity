using JuMP, HiGHS

"""
    expected_Q(Q::AbstractVector, Π::AbstractArray{<:Real,3}) -> Matrix

Reward matrix `c[θ, I] = E[Q(θ') | θ_t=θ, I_t=I] = Σ_θ' Π[θ', θ, I] * Q[θ']`.
`Π` has shape `(r_θ, r_θ, r_I)` with the first index running over θ'.
"""
function expected_Q(Q::AbstractVector, Π::AbstractArray{<:Real,3})
    r_θ_out, r_θ, r_I = size(Π)
    length(Q) == r_θ_out || throw(DimensionMismatch("Q must match size(Π,1)"))
    c = zeros(eltype(Π), r_θ, r_I)
    @inbounds for k in 1:r_I, j in 1:r_θ, i in 1:r_θ_out
        c[j, k] += Π[i, j, k] * Q[i]
    end
    return c
end

"""
    solve_ot(p_θ, p_I, c; maximize=true) -> (value, γ)

Solve the discrete optimal-transport LP with marginals `p_θ`, `p_I` and reward
matrix `c[θ, I]`. Returns the optimal value and the joint `γ[θ, I]`.
"""
function solve_ot(p_θ::AbstractVector, p_I::AbstractVector, c::AbstractMatrix;
                   maximize::Bool=true)
    r_θ, r_I = length(p_θ), length(p_I)
    size(c) == (r_θ, r_I) || throw(DimensionMismatch("c must be $r_θ×$r_I"))
    isapprox(sum(p_θ), sum(p_I); atol=1e-10) ||
        throw(ArgumentError("marginals must have equal total mass"))

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, γ[1:r_θ, 1:r_I] >= 0)
    @constraint(model, [i=1:r_θ], sum(γ[i, :]) == p_θ[i])
    @constraint(model, [j=1:r_I], sum(γ[:, j]) == p_I[j])
    obj = sum(c[i, j] * γ[i, j] for i in 1:r_θ, j in 1:r_I)
    maximize ? @objective(model, Max, obj) : @objective(model, Min, obj)
    optimize!(model)
    is_solved_and_feasible(model) ||
        error("OT LP did not solve: $(termination_status(model))")
    return objective_value(model), value.(γ)
end

"""
    comonotone_coupling(p_θ, p_I) -> γ

Hoeffding–Fréchet upper coupling: the joint supported on the comonotone rank
curve (rank(θ) = rank(I)). Supports both marginals strictly. This is the OT
optimum when the reward is supermodular in (θ, I).
"""
function comonotone_coupling(p_θ::AbstractVector, p_I::AbstractVector;
                               tol::Real=1e-12)
    r_θ, r_I = length(p_θ), length(p_I)
    isapprox(sum(p_θ), sum(p_I); atol=1e-10) ||
        throw(ArgumentError("marginals must have equal total mass"))
    γ = zeros(promote_type(eltype(p_θ), eltype(p_I)), r_θ, r_I)
    rθ, rI = collect(float.(p_θ)), collect(float.(p_I))
    i, j = 1, 1
    while i <= r_θ && j <= r_I
        m = min(rθ[i], rI[j])
        γ[i, j] += m
        rθ[i] -= m; rI[j] -= m
        if rθ[i] <= tol && rI[j] <= tol
            i += 1; j += 1
        elseif rθ[i] <= tol
            i += 1
        else
            j += 1
        end
    end
    return γ
end

welfare(γ::AbstractMatrix, c::AbstractMatrix) = sum(γ .* c)

"""
    solve_sequential_ot(p_θ_1, p_I_ts, Π̄_ts, Q; maximize=true)
        -> (value, γs, F_θs)

Sequential (multi-period) planner reallocation LP.

Inputs:
- `p_θ_1` : initial θ-marginal (length r_θ)
- `p_I_ts` : vector of length `T-1` giving the fixed I-marginals per period
- `Π̄_ts`  : vector of length `T-1` of regime-marginalized transitions,
             each shaped `(r_θ', r_θ, r_I)` with the first index running over θ'
- `Q`     : terminal reward vector of length r_θ (evaluated on θ_T)

Decision variables are joints `γ_t[θ, I]` for `t = 1..T-1`.
Constraints: γ_1's θ-marginal is `p_θ_1`; for t ≥ 2, γ_t's θ-marginal equals
the pushforward of γ_{t-1} through Π̄_{t-1}; γ_t's I-marginal equals `p_I_ts[t]`.
Objective: terminal welfare `Σ_θ' Q[θ'] · F_θ_T[θ']` = `Σ_{θ,I} γ_{T-1}[θ, I] · c_{T-1}[θ, I]`
with `c_{T-1}[θ, I] = Σ_θ' Π̄_{T-1}[θ', θ, I] · Q[θ']`.
"""
function solve_sequential_ot(p_θ_1::AbstractVector,
                              p_I_ts::AbstractVector{<:AbstractVector},
                              Π̄_ts::AbstractVector{<:AbstractArray{<:Real,3}},
                              Q::AbstractVector;
                              maximize::Bool=true)
    Tm1 = length(Π̄_ts)
    length(p_I_ts) == Tm1 || throw(ArgumentError("need T-1 I-marginals"))
    r_θ = length(p_θ_1); r_I = length(p_I_ts[1])
    for t in 1:Tm1
        size(Π̄_ts[t]) == (r_θ, r_θ, r_I) ||
            throw(DimensionMismatch("Π̄_ts[$t] must be ($r_θ, $r_θ, $r_I)"))
        length(p_I_ts[t]) == r_I ||
            throw(DimensionMismatch("p_I_ts[$t] must have length $r_I"))
        isapprox(sum(p_I_ts[t]), 1.0; atol=1e-8) ||
            @warn "p_I_ts[$t] does not sum to 1" sum(p_I_ts[t])
    end
    isapprox(sum(p_θ_1), 1.0; atol=1e-8) || @warn "p_θ_1 doesn't sum to 1"

    model = Model(HiGHS.Optimizer); set_silent(model)
    @variable(model, γ[1:Tm1, 1:r_θ, 1:r_I] >= 0)

    # Initial θ-marginal on γ_1
    @constraint(model, [i=1:r_θ], sum(γ[1, i, :]) == p_θ_1[i])

    # I-marginals fixed at empirical per period
    @constraint(model, [t=1:Tm1, j=1:r_I], sum(γ[t, :, j]) == p_I_ts[t][j])

    # Transition consistency: θ-marginal of γ_{t+1} = pushforward of γ_t
    #   Σ_I γ_{t+1}[θ'', I] = Σ_{θ, I'} γ_t[θ, I'] · Π̄_t[θ'', θ, I']
    @constraint(model, [t=1:Tm1-1, θpp=1:r_θ],
        sum(γ[t+1, θpp, j] for j in 1:r_I) ==
        sum(Π̄_ts[t][θpp, i, j] * γ[t, i, j] for i in 1:r_θ, j in 1:r_I))

    # Objective: Σ_{θ,I} γ_{T-1}[θ, I] · c_{T-1}[θ, I]
    c_T = zeros(r_θ, r_I)
    @inbounds for j in 1:r_I, i in 1:r_θ, k in 1:r_θ
        c_T[i, j] += Π̄_ts[Tm1][k, i, j] * Q[k]
    end
    obj = sum(c_T[i, j] * γ[Tm1, i, j] for i in 1:r_θ, j in 1:r_I)
    maximize ? @objective(model, Max, obj) : @objective(model, Min, obj)
    optimize!(model)
    is_solved_and_feasible(model) ||
        error("sequential LP failed: $(termination_status(model))")

    γs = [Float64[value(γ[t, i, j]) for i in 1:r_θ, j in 1:r_I] for t in 1:Tm1]
    # Reconstruct F_θ_t for t=1..T
    F_θs = Vector{Vector{Float64}}(undef, Tm1+1)
    F_θs[1] = collect(p_θ_1)
    for t in 1:Tm1
        F_θs[t+1] = [sum(Π̄_ts[t][k, i, j] * γs[t][i, j] for i in 1:r_θ, j in 1:r_I)
                      for k in 1:r_θ]
    end
    return objective_value(model), γs, F_θs
end

"""
    marginals(γ) -> (p_θ, p_I)

Row/column sums of a joint.
"""
marginals(γ::AbstractMatrix) = (vec(sum(γ, dims=2)), vec(sum(γ, dims=1)))
