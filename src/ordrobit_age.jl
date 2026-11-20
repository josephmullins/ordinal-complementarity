# One technology kernel for all transitions, with age as a state (2026-11-19, exploratory).
#
# Per-skill ordered robit in which, for transition t → t+1 (t = 1..T−1),
#   F̄_t(j | θ, I, θ^P, ω) = F_ν( c_{j,t} + μ_θ(t) + β_θ(t)·I + γ_P θ^P + γ_ω ω ),
#   μ_θ(t)       = Σ_d M[θ−1, d] · x_t^d      (μ_1(t) = 0),
#   log β_θ(t)   = Σ_d B[θ, d]   · x_t^d,     x_t = (t − (T)/2) / ((T−2)/2) ∈ [−1, 1],
# with survivor cutoffs c_{·,t} free in every period (each period's θ′ scale is set by its own
# emissions) and γ_P, γ_ω common. Each period's technology is exactly an OrderedRobitPerThetaV3,
# so φ_age ↦ φ_period(t) is linear and every period-level tool works on `age_period_specs(s)`.
# Use with transition_groups = [[1, …, T−1]] (one group, one spec).
abstract type AgeSpecT end
struct OrderedRobitAgeV3 <: AgeSpecT
    c_raw::Matrix{Float64}   # (r_θ−1) × (T−1), packed survivor cutoffs per period
    M::Matrix{Float64}       # (r_θ−1) × (deg+1)
    B::Matrix{Float64}       # r_θ × (deg+1), coefficients of β_raw
    γP::Float64
    γω::Float64
    ν::Float64
end
age_deg(s::OrderedRobitAgeV3) = size(s.M, 2) - 1
age_x(t, Tm1) = (t - (Tm1 + 1) / 2) / ((Tm1 - 1) / 2)
age_basis(t, Tm1, deg) = [age_x(t, Tm1)^d for d in 0:deg]

pack_ordrobit(s::OrderedRobitAgeV3) = vcat(vec(s.c_raw), vec(s.M), vec(s.B), s.γP, s.γω)
function unpack_age(v, r_θ, Tm1, deg, ν)
    nc = r_θ - 1; i = 0
    c = reshape(collect(v[i+1:i+nc*Tm1]), nc, Tm1); i += nc * Tm1
    M = reshape(collect(v[i+1:i+nc*(deg+1)]), nc, deg + 1); i += nc * (deg + 1)
    B = reshape(collect(v[i+1:i+r_θ*(deg+1)]), r_θ, deg + 1); i += r_θ * (deg + 1)
    OrderedRobitAgeV3(c, M, B, v[i+1], v[i+2], ν)
end
unpack_age(v, s::OrderedRobitAgeV3) = unpack_age(v, size(s.B, 1), size(s.c_raw, 2), age_deg(s), s.ν)

"""Period-t OrderedRobitPerThetaV3 parameter vector from the packed age vector (linear in v)."""
function age_period_vec(v::AbstractVector{T}, t, r_θ, Tm1, deg) where T
    nc = r_θ - 1; i = 0
    c = v[i+(t-1)*nc+1:i+t*nc]; i += nc * Tm1
    M = reshape(v[i+1:i+nc*(deg+1)], nc, deg + 1); i += nc * (deg + 1)
    B = reshape(v[i+1:i+r_θ*(deg+1)], r_θ, deg + 1); i += r_θ * (deg + 1)
    b = age_basis(t, Tm1, deg)
    vcat(c, M * b, B * b, v[i+1], v[i+2])
end
age_period_specs(s::OrderedRobitAgeV3) =
    [unpack_ordrobit(OrderedRobitPerThetaV3, age_period_vec(pack_ordrobit(s), t, size(s.B, 1), size(s.c_raw, 2), age_deg(s)),
                     size(s.B, 1), s.ν) for t in 1:size(s.c_raw, 2)]

"""Least-squares projection of period-specific per-skill specs onto the age polynomial."""
function age_from_period_specs(specs::Vector{OrderedRobitPerThetaV3}, deg; β_floor = log(0.02))
    Tm1 = length(specs); r_θ = length(specs[1].β_raw)
    X = reduce(vcat, [age_basis(t, Tm1, deg)' for t in 1:Tm1])
    Mt = reduce(hcat, [s.μ for s in specs])                         # (r_θ−1) × Tm1
    Bt = reduce(hcat, [max.(s.β_raw, β_floor) for s in specs])       # r_θ × Tm1
    OrderedRobitAgeV3(reduce(hcat, [s.c_raw for s in specs]), (X \ Mt')', (X \ Bt')',
                      mean(s.γP for s in specs), mean(s.γω for s in specs), specs[1].ν)
end

_pack_spec(s::OrderedRobitAgeV3) = pack_ordrobit(s)
n_trans_params(s::OrderedRobitAgeV3) = length(pack_ordrobit(s))

function precompute_Pi_z(trans_specs::Vector{S}, transition_groups::Vector{Vector{Int}},
                         r_θ::Int, r_I::Int, r_P::Int, r_ω::Int;
                         θ_grid::AbstractVector = 1.0:r_θ, I_grid::AbstractVector = 1.0:r_I,
                         P_grid::AbstractVector = 1.0:r_P, ω_grid::AbstractVector = 1.0:r_ω) where {S<:AgeSpecT}
    @assert length(trans_specs) == 1 && length(transition_groups) == 1
    s = trans_specs[1]; Tm1 = size(s.c_raw, 2); deg = age_deg(s); v = pack_ordrobit(s)
    @assert transition_groups[1] == collect(1:Tm1)
    Pi_z = [Vector{Array{Float64,3}}(undef, Tm1) for _ in 1:r_P*r_ω]
    for t in 1:Tm1
        vt = _pvec(S)(v, t, r_θ, Tm1, deg)
        for kω in 1:r_ω, kP in 1:r_P
            Pi_z[(kP - 1) * r_ω + kω][t] = _build_ordrobit_pi(OrderedRobitPerThetaV3, vt, r_θ, r_I, s.ν,
                                                               P_grid[kP], ω_grid[kω]; θ_grid, I_grid)
        end
    end
    Pi_z
end

function age_transition_negll(pvec, v::AbstractVector{T}, Q_t, r_θ, r_I, ν, r_P, r_ω, Tm1, deg;
                              θ_grid, I_grid, P_grid, ω_grid) where T
    sum(transition_negll_ordrobit(OrderedRobitPerThetaV3, pvec(v, t, r_θ, Tm1, deg), Q_t[t],
                                  r_θ, r_I, ν, r_P, r_ω; θ_grid, I_grid, P_grid, ω_grid) for t in 1:Tm1)
end

function mstep_transition_v3_grouped(trans_specs::Vector{S}, xi_sum_z::Vector{Vector{Array{Float64,4}}},
                                     transition_groups::Vector{Vector{Int}},
                                     r_θ::Int, r_I::Int, r_P::Int, r_ω::Int;
                                     θ_grid::AbstractVector = collect(1.0:r_θ), I_grid::AbstractVector = collect(1.0:r_I),
                                     P_grid::AbstractVector = collect(1.0:r_P), ω_grid::AbstractVector = collect(1.0:r_ω),
                                     options = DEFAULT_OPTIM_OPTIONS, λ_logw::Float64 = 0.0) where {S<:AgeSpecT}
    s = trans_specs[1]; Tm1 = size(s.c_raw, 2); deg = age_deg(s)
    Q_t = [[_qsum_from_xi(xi_sum_z[z][t]) for z in 1:r_P*r_ω] for t in 1:Tm1]
    obj = x -> age_transition_negll(_pvec(S), x, Q_t, r_θ, r_I, s.ν, r_P, r_ω, Tm1, deg; θ_grid, I_grid, P_grid, ω_grid)
    x0 = pack_ordrobit(s)
    xopt = try
        Optim.minimizer(optimize(obj, x0, LBFGS(), options; autodiff = :forward))
    catch err
        @warn "age ordered-robit M-step: HagerZhang failed ($(typeof(err))), retrying with BackTracking"
        try
            Optim.minimizer(optimize(obj, x0, LBFGS(linesearch = Optim.LineSearches.BackTracking()), options; autodiff = :forward))
        catch
            x0
        end
    end
    [obj(xopt) <= obj(x0) ? unpack_age(xopt, s) : s]
end

# ── Variant: skill intercepts μ_θ(t) FREE in every period; only log β_θ(t) is polynomial ──
struct OrderedRobitAgeBV3 <: AgeSpecT
    c_raw::Matrix{Float64}   # (r_θ−1) × (T−1)
    M::Matrix{Float64}       # (r_θ−1) × (T−1), free intercepts
    B::Matrix{Float64}       # r_θ × (deg+1)
    γP::Float64
    γω::Float64
    ν::Float64
end
age_deg(s::OrderedRobitAgeBV3) = size(s.B, 2) - 1
pack_ordrobit(s::OrderedRobitAgeBV3) = vcat(vec(s.c_raw), vec(s.M), vec(s.B), s.γP, s.γω)
function unpack_age(v, s::OrderedRobitAgeBV3)
    r_θ = size(s.B, 1); nc = r_θ - 1; Tm1 = size(s.c_raw, 2); deg = age_deg(s); i = 0
    c = reshape(collect(v[i+1:i+nc*Tm1]), nc, Tm1); i += nc * Tm1
    M = reshape(collect(v[i+1:i+nc*Tm1]), nc, Tm1); i += nc * Tm1
    B = reshape(collect(v[i+1:i+r_θ*(deg+1)]), r_θ, deg + 1); i += r_θ * (deg + 1)
    OrderedRobitAgeBV3(c, M, B, v[i+1], v[i+2], s.ν)
end
function age_period_vec_B(v::AbstractVector{T}, t, r_θ, Tm1, deg) where T
    nc = r_θ - 1; i = 0
    c = v[(t-1)*nc+1:t*nc]; i += nc * Tm1
    μ = v[i+(t-1)*nc+1:i+t*nc]; i += nc * Tm1
    B = reshape(v[i+1:i+r_θ*(deg+1)], r_θ, deg + 1); i += r_θ * (deg + 1)
    vcat(c, μ, B * age_basis(t, Tm1, deg), v[i+1], v[i+2])
end
_pvec(::Type{OrderedRobitAgeV3}) = age_period_vec
_pvec(::Type{OrderedRobitAgeBV3}) = age_period_vec_B
age_period_specs(s::OrderedRobitAgeBV3) =
    [unpack_ordrobit(OrderedRobitPerThetaV3, age_period_vec_B(pack_ordrobit(s), t, size(s.B, 1), size(s.c_raw, 2), age_deg(s)),
                     size(s.B, 1), s.ν) for t in 1:size(s.c_raw, 2)]
function ageB_from_period_specs(specs::Vector{OrderedRobitPerThetaV3}, deg; β_floor = log(0.02))
    a = age_from_period_specs(specs, deg; β_floor)
    OrderedRobitAgeBV3(a.c_raw, reduce(hcat, [s.μ for s in specs]), a.B, a.γP, a.γω, a.ν)
end
ageB_from_age(a::OrderedRobitAgeV3) =     # exact embedding (same Π) of the all-polynomial spec
    OrderedRobitAgeBV3(a.c_raw, reduce(hcat, [p.μ for p in age_period_specs(a)]), a.B, a.γP, a.γω, a.ν)
_pack_spec(s::OrderedRobitAgeBV3) = pack_ordrobit(s)
n_trans_params(s::OrderedRobitAgeBV3) = length(pack_ordrobit(s))

# ── Inference hooks (src/inference_v3_eps.jl): one Jacobian block per period, all feeding the
#    single parameter group; J_list[t][z] = ∂vec(Π_{z,t})/∂φ_age ──
_trans_score_blocks(trans_specs::Vector{S}, transition_groups) where {S<:AgeSpecT} =
    [(1, t, [t]) for t in transition_groups[1]]

function precompute_J_Pi_z_eps(trans_specs::Vector{S}, transition_groups::Vector{Vector{Int}},
                                r_θ::Int, r_I::Int, r_P::Int, r_ω::Int;
                                θ_grid::AbstractVector = 1.0:r_θ, I_grid::AbstractVector = 1.0:r_I,
                                P_grid::AbstractVector = 1.0:r_P, ω_grid::AbstractVector = 1.0:r_ω) where {S<:AgeSpecT}
    s = trans_specs[1]; Tm1 = size(s.c_raw, 2); deg = age_deg(s); φ = pack_ordrobit(s); pv = _pvec(S)
    J_list = [Vector{Matrix{Float64}}(undef, r_P * r_ω) for _ in 1:Tm1]
    Π_list = [Vector{Array{Float64,3}}(undef, r_P * r_ω) for _ in 1:Tm1]
    for t in 1:Tm1, kω in 1:r_ω, kP in 1:r_P
        z = regime_z(kP, kω, r_ω)
        f = x -> _build_ordrobit_pi(OrderedRobitPerThetaV3, pv(x, t, r_θ, Tm1, deg), r_θ, r_I, s.ν, P_grid[kP], ω_grid[kω]; θ_grid, I_grid)
        Π_list[t][z] = f(φ)
        J_list[t][z] = ForwardDiff.jacobian(x -> vec(f(x)), φ)
    end
    J_list, Π_list
end

"""Delta-method covariance of each period's regime-marginalized Π̄_t free entries
(ordering of Pi_to_free: kI outer, kθ, j = 1..r_θ−1 inner), using Var of (φ_age, α_Pω)."""
function delta_cov_Pibar_free_age(inf_result, params_v3_eps, params_skill)
    (; r_P, r_ω, π_Pω, trans_specs) = params_v3_eps; (; r_θ, r_I) = params_skill
    s = trans_specs[1]; S = typeof(s); Tm1 = size(s.c_raw, 2); deg = age_deg(s); pv = _pvec(S)
    π_flat = Float64[π_Pω[kP, kω] for kP in 1:r_P for kω in 1:r_ω]
    θc = vcat(pack_ordrobit(s), probs_to_softmax(π_flat; ref = 1)); nφ = length(pack_ordrobit(s))
    idx = vcat(collect(inf_result.ranges.trans_g[1]), collect(inf_result.ranges.π_Pω))
    V = inf_result.Var_φ[idx, idx]
    map(1:Tm1) do t
        fmap = function (θv)
            φ = θv[1:nφ]; π = softmax_to_probs(θv[nφ+1:end]); vt = pv(φ, t, r_θ, Tm1, deg)
            Πb = sum(π[(kP-1)*r_ω+kω] .* _build_ordrobit_pi(OrderedRobitPerThetaV3, vt, r_θ, r_I, s.ν, Float64(kP), Float64(kω))
                     for kP in 1:r_P, kω in 1:r_ω)
            [Πb[j, kθ, kI] for kI in 1:r_I for kθ in 1:r_θ for j in 1:r_θ-1]
        end
        J = ForwardDiff.jacobian(fmap, θc)
        (; t, free_hat = fmap(θc), Var = Symmetric(J * V * J'))
    end
end
