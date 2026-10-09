# ============================================================================
# Induced reduced-form skill transition under EXOGENOUS income.
#
#   Π_ind(θ' | θ, y) = Σ_{kP,kω} π_Pω[kP,kω]
#                        · Σ_I  CCP(I | θ, kP, kω, ε_eff(y,kP,kω))
#                              · Π(θ' | θ, I, kP, kω)
#
# Composes the technology Π (RobitNNTransitionV3) with the investment policy
# (OrderedProbitCCPV3Eps).  Income is the SOLE channel for the income type ε_Y
# (β₂ on log y was ~0; the settled spec replaced it with the discrete type), so
# "set income to y" means "set ε_Y to the value that produces y in this regime":
#
#   ε_eff(y, kP, kω) = (log y − β₀ − β_P·kP − β_ω·kω) / δ                (income model)
#
# which then enters the CCP index via β₅·ε_eff.  This mirrors the income-transfer
# counterfactual (scripts/counterfactual_income_transfers.jl).
#
# The regime z=(θ^P,ω) is marginalized with the UNCONDITIONAL prior π_Pω, NOT a
# posterior given (θ,y): income and current skill are set EXOGENOUSLY, so the
# regime is drawn from its population marginal and is independent of the
# manipulated (θ,y).  (Conditioning the regime on y would re-introduce the
# selection channel we are deliberately shutting down.)
#
# Output for a transition group: an r_θ × r_θ × n_y array indexed [θ', θ, y-bin].
# This is structurally identical to a technology array with the second
# conditioning dimension being an income grid instead of investment, so every
# downstream test (get_S_adjacent, mmm_test, the t± LP) applies with n_y in
# place of r_I.  Because the CCP weight depends on the regime AND on ε_eff(y,z),
# the regime sum cannot be factored out — this is NOT marginalize_Pi_v3 ∘ CCP.
# ============================================================================

using Distributions

"""
    model_income_grid(pv; n_y=5, qs=nothing) -> Vector{Float64}

Income support points (levels, dollars) at quantiles `qs` (default the n_y
midpoint quantiles (i-0.5)/n_y) of the model-implied income distribution

    log y ~ Σ_{kP,kω,ε} π_Pω[kP,kω]·π_ε[ε] · N(β₀+β_P kP+β_ω kω+δ ε, σ²).
"""
function model_income_grid(pv; n_y::Int = 5, qs = nothing)
    inc = pv.income_model
    σ   = exp(inc.log_sigma)
    π_Pω = Array{Float64}(pv.π_Pω)
    π_ε  = Array{Float64}(pv.π_ε)
    r_P, r_ω = size(π_Pω)
    r_Y = length(π_ε)
    means = Float64[]
    weights = Float64[]
    for kP in 1:r_P, kω in 1:r_ω, kε in 1:r_Y
        push!(means, inc.beta0 + inc.betaP * kP + inc.betaω * kω + inc.delta * kε)
        push!(weights, π_Pω[kP, kω] * π_ε[kε])
    end
    weights ./= sum(weights)
    mix = MixtureModel([Normal(m, σ) for m in means], weights)
    probs = qs === nothing ? [(i - 0.5) / n_y for i in 1:n_y] : collect(qs)
    return [exp(quantile(mix, p)) for p in probs]   # back to levels
end

"""
    induced_transition_income(trans_spec, ccp_spec, income_model, π_Pω, y_levels;
                              r_θ, r_I) -> Array{Float64,3}   # [θ', θ, y-bin]

Build the induced reduced-form transition Π_ind(θ'|θ,y) for ONE transition group
(its shared `trans_spec::RobitNNTransitionV3`), evaluated at the income levels
`y_levels` (dollars).  Columns are normalized to sum to one over θ'.
"""
function induced_transition_income(trans_spec, ccp_spec, income_model,
                                   π_Pω::AbstractMatrix, y_levels::AbstractVector;
                                   r_θ::Int, r_I::Int)
    r_P, r_ω = size(π_Pω)
    n_y = length(y_levels)
    β0, βP, βω, δ = income_model.beta0, income_model.betaP,
                    income_model.betaω, income_model.delta
    c_ccp = Vector{Float64}(ccp_spec.c)
    β1, β3, β4, β5 = ccp_spec.beta1, ccp_spec.beta3, ccp_spec.beta4, ccp_spec.beta5

    v_pi = pack_robit_nn_v3(trans_spec)
    K, ν = trans_spec.K, trans_spec.ν

    Π_ind = zeros(r_θ, r_θ, n_y)
    for kP in 1:r_P, kω in 1:r_ω
        w_z = π_Pω[kP, kω]
        w_z < 1e-15 && continue
        Π_z = _build_robit_nn_v3_pi(v_pi, r_θ, r_I, K, ν, float(kP), float(kω))  # [θ',θ,I]
        for (iy, y) in enumerate(y_levels)
            ε_eff = (log(y) - β0 - βP * kP - βω * kω) / δ
            regime_part = β3 * kP + β4 * kω + β5 * ε_eff
            for kθ in 1:r_θ
                η = β1 * kθ + regime_part
                ccp = _ordered_probit_pmf(c_ccp, η, r_I)        # P(I | θ, z, y)  length r_I
                for kI in 1:r_I
                    wkI = w_z * ccp[kI]
                    wkI < 1e-18 && continue
                    @views Π_ind[:, kθ, iy] .+= wkI .* Π_z[:, kθ, kI]
                end
            end
        end
    end
    # normalize columns over θ' (π_Pω need not sum to exactly 1 numerically)
    for iy in 1:n_y, kθ in 1:r_θ
        s = sum(@view Π_ind[:, kθ, iy])
        s > 0 && (@views Π_ind[:, kθ, iy] ./= s)
    end
    return Π_ind
end

"""
    induced_transitions_by_group(pv, transition_groups, y_levels; r_θ, r_I)
        -> Vector{Array{Float64,3}}

One induced transition per group (indexed by the group's representative period
position in `transition_groups`), matching the layout used for Π̄ elsewhere.
The CCP is shared across periods in the ε_Y spec, so groups differ only through
their shared technology `trans_spec`.
"""
function induced_transitions_by_group(pv, transition_groups, y_levels; r_θ::Int, r_I::Int)
    π_Pω = Array{Float64}(pv.π_Pω)
    return [induced_transition_income(pv.trans_specs[g], pv.ccp_spec, pv.income_model,
                                      π_Pω, y_levels; r_θ = r_θ, r_I = r_I)
            for g in 1:length(transition_groups)]
end
