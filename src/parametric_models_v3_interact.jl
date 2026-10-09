# RobitNNTransitionV3 with θ·I interaction term (alpha5)
# Pack: [c_raw(n_cut), α1(K), α2_raw(K), α3(K), α4(K), α5(K), b(K), log_w(K)]
#       = n_cut + 7K params

struct RobitNNTransitionV3Interact
    c_raw::Vector{Float64}
    alpha1::Vector{Float64}
    alpha2_raw::Vector{Float64}
    alpha3::Vector{Float64}
    alpha4::Vector{Float64}
    alpha5::Vector{Float64}      # θ·I interaction
    b::Vector{Float64}
    log_w::Vector{Float64}
    K::Int
    ν::Float64
end

function pack_robit_nn_v3_interact(s::RobitNNTransitionV3Interact)
    vcat(s.c_raw, s.alpha1, s.alpha2_raw, s.alpha3, s.alpha4, s.alpha5, s.b, s.log_w)
end

function unpack_robit_nn_v3_interact(v::AbstractVector, r_θ::Int, K::Int; ν::Float64=DEFAULT_ROBIT_DF)
    nc = r_θ - 1; i = 0
    c_raw  = v[i+1:i+nc]; i += nc
    a1     = v[i+1:i+K]; i += K
    a2_raw = v[i+1:i+K]; i += K
    a3     = v[i+1:i+K]; i += K
    a4     = v[i+1:i+K]; i += K
    a5     = v[i+1:i+K]; i += K
    b      = v[i+1:i+K]; i += K
    lw     = v[i+1:i+K]
    RobitNNTransitionV3Interact(collect(c_raw), collect(a1), collect(a2_raw),
                                 collect(a3), collect(a4), collect(a5),
                                 collect(b), collect(lw), K, ν)
end

function _build_robit_nn_v3_interact_pi(vec::AbstractVector{T}, r_θ::Int, r_I::Int,
                                         K::Int, ν::Float64, kP_val::Real, kω_val::Real;
                                         θ_grid::AbstractVector=1.0:r_θ,
                                         I_grid::AbstractVector=1.0:r_I) where T
    nc = r_θ - 1; i = 0
    c  = _unpack_survivor_cutoffs(vec[i+1:i+nc]); i += nc
    a1 = vec[i+1:i+K]; i+=K; a2 = exp.(vec[i+1:i+K]); i+=K
    a3 = vec[i+1:i+K]; i+=K; a4 = vec[i+1:i+K]; i+=K
    a5 = vec[i+1:i+K]; i+=K; b  = vec[i+1:i+K]; i+=K
    w  = exp.(vec[i+1:i+K])
    Π = zeros(T, r_θ, r_θ, r_I)
    for kI in 1:r_I, kθ in 1:r_θ
        mu = zero(T)
        for h in 1:K
            mu += w[h] * Φ(a1[h]*θ_grid[kθ] + a2[h]*I_grid[kI] +
                           a3[h]*kP_val + a4[h]*kω_val +
                           a5[h]*θ_grid[kθ]*I_grid[kI] + b[h])
        end
        S_bar = Vector{T}(undef, nc)
        for k in 1:nc; S_bar[k] = Φ_t(c[k] + mu, ν); end
        pmf = Vector{T}(undef, r_θ)
        pmf[1] = one(T) - S_bar[1]
        for k in 2:nc; pmf[k] = S_bar[k-1] - S_bar[k]; end
        pmf[r_θ] = S_bar[nc]
        pmf .= max.(pmf, T(1e-12)); pmf ./= sum(pmf)
        Π[:, kθ, kI] = pmf
    end
    return Π
end

function transition_negll_robit_nn_v3_interact(vec::AbstractVector{T}, Q_sum_z,
                                                r_θ, r_I, K, ν, r_P, r_ω;
                                                θ_grid=1.0:r_θ, I_grid=1.0:r_I,
                                                P_grid=1.0:r_P, ω_grid=1.0:r_ω) where T
    negll = zero(T)
    for kω in 1:r_ω, kP in 1:r_P
        z = (kP - 1) * r_ω + kω
        Π = _build_robit_nn_v3_interact_pi(vec, r_θ, r_I, K, ν, P_grid[kP], ω_grid[kω];
                                             θ_grid, I_grid)
        negll += _transition_negll_precomputed(Π, Q_sum_z[z])
    end
    return negll
end

function initialize_robit_nn_v3_interact(r_θ::Int, K::Int; rng=Random.default_rng(), ν=DEFAULT_ROBIT_DF)
    nc = r_θ - 1
    c = collect(range(-1.0, 1.0; length=nc)) .+ 0.05 .* randn(rng, nc); sort!(c)
    RobitNNTransitionV3Interact(
        _pack_survivor_cutoffs(c),
        0.3 .+ 0.2 .* randn(rng, K),
        0.1 .* randn(rng, K),
        0.05 .* randn(rng, K),
        0.05 .* randn(rng, K),
        0.05 .* randn(rng, K),
        randn(rng, K),
        0.5 .* randn(rng, K),
        K, ν)
end

# ── Dispatches for the EM machinery ──

_pack_spec(s::RobitNNTransitionV3Interact) = pack_robit_nn_v3_interact(s)

function _make_build_fn(spec::RobitNNTransitionV3Interact, r_θ::Int, r_I::Int;
                         θ_grid::AbstractVector=1.0:r_θ, I_grid::AbstractVector=1.0:r_I)
    K = spec.K; ν = spec.ν
    return (φ, kP_val, kω_val) -> _build_robit_nn_v3_interact_pi(φ, r_θ, r_I, K, ν,
                                                                     kP_val, kω_val; θ_grid, I_grid)
end

function precompute_Pi_z(trans_specs::Vector{RobitNNTransitionV3Interact},
                           transition_groups::Vector{Vector{Int}},
                           r_θ::Int, r_I::Int, r_P::Int, r_ω::Int;
                           θ_grid::AbstractVector=1.0:r_θ, I_grid::AbstractVector=1.0:r_I,
                           P_grid::AbstractVector=1.0:r_P, ω_grid::AbstractVector=1.0:r_ω)
    r_Z = r_P * r_ω
    T_minus_1 = sum(length, transition_groups)
    Pi_z = [Vector{Array{Float64,3}}(undef, T_minus_1) for _ in 1:r_Z]
    for (g, group) in enumerate(transition_groups)
        spec = trans_specs[g]
        vec_g = pack_robit_nn_v3_interact(spec)
        K = spec.K; ν = spec.ν
        for kω in 1:r_ω, kP in 1:r_P
            z = (kP - 1) * r_ω + kω
            Π = _build_robit_nn_v3_interact_pi(vec_g, r_θ, r_I, K, ν,
                                                 P_grid[kP], ω_grid[kω]; θ_grid, I_grid)
            for t in group; Pi_z[z][t] = Π; end
        end
    end
    return Pi_z
end

function mstep_transition_v3_grouped(
    trans_specs::Vector{RobitNNTransitionV3Interact},
    xi_sum_z::Vector{Vector{Array{Float64,4}}},
    transition_groups::Vector{Vector{Int}},
    r_θ::Int, r_I::Int, r_P::Int, r_ω::Int;
    θ_grid::AbstractVector=collect(1.0:r_θ), I_grid::AbstractVector=collect(1.0:r_I),
    P_grid::AbstractVector=collect(1.0:r_P), ω_grid::AbstractVector=collect(1.0:r_ω),
    options=DEFAULT_OPTIM_OPTIONS)

    r_Z = r_P * r_ω
    new_specs = Vector{RobitNNTransitionV3Interact}(undef, length(transition_groups))
    for (g, group) in enumerate(transition_groups)
        Q_sum_z = Vector{Array{Float64,3}}(undef, r_Z)
        for z in 1:r_Z
            Q_sum_z[z] = _qsum_from_xi(xi_sum_z[z][group[1]])
            for t in group[2:end]
                Q_sum_z[z] = Q_sum_z[z] .+ _qsum_from_xi(xi_sum_z[z][t])
            end
        end
        K = trans_specs[g].K; ν = trans_specs[g].ν
        x0 = pack_robit_nn_v3_interact(trans_specs[g])
        obj = x -> transition_negll_robit_nn_v3_interact(x, Q_sum_z, r_θ, r_I, K, ν,
                                                           r_P, r_ω; θ_grid, I_grid, P_grid, ω_grid)
        result = optimize(obj, x0, LBFGS(), options; autodiff=:forward)
        new_specs[g] = unpack_robit_nn_v3_interact(Optim.minimizer(result), r_θ, K; ν)
    end
    return new_specs
end
