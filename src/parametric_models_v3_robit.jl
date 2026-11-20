# Ordered-robit technologies (no hidden layer), as alternatives to RobitNNTransitionV3.
#
#   F̄(j | θ, I, θ^P, ω) = F_ν( c_j + η(θ, I) + γ_P·θ^P + γ_ω·ω ),   c_1 > c_2 > … (survivor cutoffs)
#
#   OrderedRobitAddV3      η = a·θ + b·I                       (additively separable; b = exp(b_raw) > 0)
#   OrderedRobitPerThetaV3 η = μ_θ + β_θ·I,  μ_1 = 0           (skill-specific intercept and investment
#                                                                slope; β_θ = exp(β_raw_θ) > 0)
#
# PerTheta nests Add: μ_θ = a(θ−1), β_θ = b, c_j → c_j + a  (see per_theta_from_add).
# Requires parametric_models.jl / parametric_models_v3.jl / estimation_multiperiod_v3.jl loaded first.

struct OrderedRobitAddV3
    c_raw::Vector{Float64}
    a::Float64
    b_raw::Float64
    γP::Float64
    γω::Float64
    ν::Float64
end

struct OrderedRobitPerThetaV3
    c_raw::Vector{Float64}
    μ::Vector{Float64}        # r_θ − 1 intercepts for θ = 2..r_θ (μ_1 = 0)
    β_raw::Vector{Float64}    # r_θ investment slopes (log scale)
    γP::Float64
    γω::Float64
    ν::Float64
end

pack_ordrobit(s::OrderedRobitAddV3) = vcat(s.c_raw, s.a, s.b_raw, s.γP, s.γω)
pack_ordrobit(s::OrderedRobitPerThetaV3) = vcat(s.c_raw, s.μ, s.β_raw, s.γP, s.γω)

function unpack_ordrobit(::Type{OrderedRobitAddV3}, v, r_θ, ν)
    nc = r_θ - 1
    OrderedRobitAddV3(collect(v[1:nc]), v[nc+1], v[nc+2], v[nc+3], v[nc+4], ν)
end
function unpack_ordrobit(::Type{OrderedRobitPerThetaV3}, v, r_θ, ν)
    nc = r_θ - 1; i = nc
    μ = collect(v[i+1:i+r_θ-1]); i += r_θ - 1
    β = collect(v[i+1:i+r_θ]);   i += r_θ
    OrderedRobitPerThetaV3(collect(v[1:nc]), μ, β, v[i+1], v[i+2], ν)
end

# index η[θ, I] for one regime (ForwardDiff-compatible)
function _ordrobit_index(::Type{OrderedRobitAddV3}, v::AbstractVector{T}, r_θ, r_I, kP, kω, θ_grid, I_grid) where T
    nc = r_θ - 1
    a, b, gP, gω = v[nc+1], exp(v[nc+2]), v[nc+3], v[nc+4]
    [a * θ_grid[kθ] + b * I_grid[kI] + gP * kP + gω * kω for kθ in 1:r_θ, kI in 1:r_I]
end
function _ordrobit_index(::Type{OrderedRobitPerThetaV3}, v::AbstractVector{T}, r_θ, r_I, kP, kω, θ_grid, I_grid) where T
    nc = r_θ - 1; i = nc
    μ = vcat(zero(T), v[i+1:i+r_θ-1]); i += r_θ - 1
    β = exp.(v[i+1:i+r_θ]);            i += r_θ
    gP, gω = v[i+1], v[i+2]
    [μ[kθ] + β[kθ] * I_grid[kI] + gP * kP + gω * kω for kθ in 1:r_θ, kI in 1:r_I]
end

function _build_ordrobit_pi(S::Type, v::AbstractVector{T}, r_θ, r_I, ν, kP, kω;
                            θ_grid = 1.0:r_θ, I_grid = 1.0:r_I) where T
    nc = r_θ - 1
    c = _unpack_survivor_cutoffs(v[1:nc])
    η = _ordrobit_index(S, v, r_θ, r_I, kP, kω, θ_grid, I_grid)
    Π = zeros(T, r_θ, r_θ, r_I)
    for kI in 1:r_I, kθ in 1:r_θ
        Sb = [Φ_t(c[k] + η[kθ, kI], ν) for k in 1:nc]
        pmf = Vector{T}(undef, r_θ)
        pmf[1] = one(T) - Sb[1]
        for k in 2:nc; pmf[k] = Sb[k-1] - Sb[k]; end
        pmf[r_θ] = Sb[nc]
        pmf .= max.(pmf, T(1e-12)); pmf ./= sum(pmf)
        Π[:, kθ, kI] = pmf
    end
    Π
end

function transition_negll_ordrobit(S::Type, v::AbstractVector{T}, Q_sum_z, r_θ, r_I, ν, r_P, r_ω;
                                   θ_grid, I_grid, P_grid, ω_grid) where T
    negll = zero(T)
    for kω in 1:r_ω, kP in 1:r_P
        z = (kP - 1) * r_ω + kω
        negll += _transition_negll_precomputed(
            _build_ordrobit_pi(S, v, r_θ, r_I, ν, P_grid[kP], ω_grid[kω]; θ_grid, I_grid), Q_sum_z[z])
    end
    negll
end

function initialize_ordrobit_add_v3(r_θ::Int; rng = Random.default_rng(), ν = DEFAULT_ROBIT_DF)
    c = collect(range(0.0, -2.0; length = r_θ - 1)) .+ 0.05 .* randn(rng, r_θ - 1)
    sort!(c; rev = true)
    OrderedRobitAddV3(_pack_survivor_cutoffs(c), 0.5 + 0.1 * randn(rng), log(0.15), 0.05 * randn(rng), 0.05 * randn(rng), ν)
end

"""Exact embedding of an additive spec into the per-θ family (same Π)."""
function per_theta_from_add(s::OrderedRobitAddV3, r_θ::Int)
    c = _unpack_survivor_cutoffs(s.c_raw) .+ s.a          # absorb a·θ at θ = 1
    OrderedRobitPerThetaV3(_pack_survivor_cutoffs(c), [s.a * (θ - 1) for θ in 2:r_θ],
                           fill(s.b_raw, r_θ), s.γP, s.γω, s.ν)
end


# ── Per-skill ordered robit with a skill-specific squared investment term ──
#   η = μ_θ + β_θ·I + δ_θ·I²   (μ_1 = 0; β_θ = exp(β_raw) > 0; δ_θ unrestricted)
struct OrderedRobitPerThetaQuadV3
    c_raw::Vector{Float64}
    μ::Vector{Float64}
    β_raw::Vector{Float64}
    δ::Vector{Float64}
    γP::Float64
    γω::Float64
    ν::Float64
end
pack_ordrobit(s::OrderedRobitPerThetaQuadV3) = vcat(s.c_raw, s.μ, s.β_raw, s.δ, s.γP, s.γω)
function unpack_ordrobit(::Type{OrderedRobitPerThetaQuadV3}, v, r_θ, ν)
    nc = r_θ - 1; i = nc
    μ = collect(v[i+1:i+r_θ-1]); i += r_θ - 1
    β = collect(v[i+1:i+r_θ]);   i += r_θ
    δ = collect(v[i+1:i+r_θ]);   i += r_θ
    OrderedRobitPerThetaQuadV3(collect(v[1:nc]), μ, β, δ, v[i+1], v[i+2], ν)
end
function _ordrobit_index(::Type{OrderedRobitPerThetaQuadV3}, v::AbstractVector{T}, r_θ, r_I, kP, kω, θ_grid, I_grid) where T
    nc = r_θ - 1; i = nc
    μ = vcat(zero(T), v[i+1:i+r_θ-1]); i += r_θ - 1
    β = exp.(v[i+1:i+r_θ]);            i += r_θ
    δ = v[i+1:i+r_θ];                  i += r_θ
    gP, gω = v[i+1], v[i+2]
    [μ[kθ] + β[kθ] * I_grid[kI] + δ[kθ] * I_grid[kI]^2 + gP * kP + gω * kω for kθ in 1:r_θ, kI in 1:r_I]
end
"""Exact embedding of the linear per-skill spec (δ = 0)."""
per_theta_quad_from_per_theta(s::OrderedRobitPerThetaV3) =
    OrderedRobitPerThetaQuadV3(copy(s.c_raw), copy(s.μ), copy(s.β_raw), zeros(length(s.β_raw)), s.γP, s.γω, s.ν)


const OrdRobitSpec = Union{OrderedRobitAddV3, OrderedRobitPerThetaV3, OrderedRobitPerThetaQuadV3}

# ── Dispatches for the V3 EM machinery ──
_pack_spec(s::OrdRobitSpec) = pack_ordrobit(s)

function _make_build_fn(spec::S, r_θ::Int, r_I::Int; θ_grid::AbstractVector = 1.0:r_θ,
                        I_grid::AbstractVector = 1.0:r_I) where {S<:OrdRobitSpec}
    ν = spec.ν
    (φ, kP, kω) -> _build_ordrobit_pi(S, φ, r_θ, r_I, ν, kP, kω; θ_grid, I_grid)
end

function precompute_Pi_z(trans_specs::Vector{S}, transition_groups::Vector{Vector{Int}},
                         r_θ::Int, r_I::Int, r_P::Int, r_ω::Int;
                         θ_grid::AbstractVector = 1.0:r_θ, I_grid::AbstractVector = 1.0:r_I,
                         P_grid::AbstractVector = 1.0:r_P, ω_grid::AbstractVector = 1.0:r_ω) where {S<:OrdRobitSpec}
    Pi_z = [Vector{Array{Float64,3}}(undef, sum(length, transition_groups)) for _ in 1:r_P*r_ω]
    for (g, group) in enumerate(transition_groups)
        spec = trans_specs[g]; v = pack_ordrobit(spec)
        for kω in 1:r_ω, kP in 1:r_P
            Π = _build_ordrobit_pi(S, v, r_θ, r_I, spec.ν, P_grid[kP], ω_grid[kω]; θ_grid, I_grid)
            for t in group; Pi_z[(kP - 1) * r_ω + kω][t] = Π; end
        end
    end
    Pi_z
end

function mstep_transition_v3_grouped(trans_specs::Vector{S}, xi_sum_z::Vector{Vector{Array{Float64,4}}},
                                     transition_groups::Vector{Vector{Int}},
                                     r_θ::Int, r_I::Int, r_P::Int, r_ω::Int;
                                     θ_grid::AbstractVector = collect(1.0:r_θ), I_grid::AbstractVector = collect(1.0:r_I),
                                     P_grid::AbstractVector = collect(1.0:r_P), ω_grid::AbstractVector = collect(1.0:r_ω),
                                     options = DEFAULT_OPTIM_OPTIONS, λ_logw::Float64 = 0.0) where {S<:OrdRobitSpec}
    new_specs = Vector{S}(undef, length(transition_groups))
    for (g, group) in enumerate(transition_groups)
        Q_sum_z = [reduce(.+, [_qsum_from_xi(xi_sum_z[z][t]) for t in group]) for z in 1:r_P*r_ω]
        ν = trans_specs[g].ν
        obj = x -> transition_negll_ordrobit(S, x, Q_sum_z, r_θ, r_I, ν, r_P, r_ω; θ_grid, I_grid, P_grid, ω_grid)
        x0 = pack_ordrobit(trans_specs[g])
        # HagerZhang can hit a non-finite trial point (AssertionError) far out in the
        # tails; fall back to a backtracking line search, else keep the current value
        xopt = try
            Optim.minimizer(optimize(obj, x0, LBFGS(), options; autodiff = :forward))
        catch err
            @warn "ordered-robit transition M-step: HagerZhang failed ($(typeof(err))), retrying with BackTracking"
            try
                Optim.minimizer(optimize(obj, x0, LBFGS(linesearch = Optim.LineSearches.BackTracking()), options; autodiff = :forward))
            catch
                x0
            end
        end
        new_specs[g] = obj(xopt) <= obj(x0) ? unpack_ordrobit(S, xopt, r_θ, ν) : trans_specs[g]
    end
    new_specs
end

n_trans_params(s::OrdRobitSpec) = length(pack_ordrobit(s))
