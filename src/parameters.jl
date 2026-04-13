using Distributions, Random

function default_model()
    r_θ = 3
    r_I = 3
    P_I_mat = zeros(3,r_I)
    P_θ_mat = zeros(3,r_θ)
    P_I_mat[:,1]= [0.6,0.2,0.2]
    P_I_mat[:,2]= [0.2,0.6,0.2]
    P_I_mat[:,3]= [0.1,0.2,0.7]

    P_θ_mat[:,1]= [0.6,0.2,0.2]
    P_θ_mat[:,2]= [0.1,0.7,0.2]
    P_θ_mat[:,3]= [0.15,0.15,0.7]

    # Wrap in vectors for unified format
    P_θ = [P_θ_mat]
    P_I = [P_I_mat]
    P_Y = [P_θ_mat]  # shared emission with P_θ

    γ = [0.1,0.1,0.]
    c = [0.25,0.5]
    Igrid = [-1.,0.,1.]
    θgrid = [-1.,0.,1.]
    Π = threshold_transitions(c,γ,Igrid,θgrid)
    π0 = ones(r_θ,r_I) / (r_θ*r_I) #<- uniform distribution here
    return (; Π, π0, P_θ, P_I, P_Y, r_θ, r_I)
end

function threshold_transitions(c,γ,Igrid,θgrid)
    r_θ = length(θgrid)
    r_I = length(Igrid)
    F = [ck - (γ[1]*θ + γ[2]*I + γ[3]*θ*I) for ck in c, θ in θgrid, I in Igrid]
    Π = zeros(r_θ, r_θ, r_I)
    Π[1,:,:] .= F[1,:,:]
    for k in 2:r_θ-1
        Π[k,:,:] .= F[k,:,:] .- F[k-1,:,:]
    end
    Π[r_θ,:,:] = 1. .- F[r_θ-1,:,:]
    return Π
end

function simulate_data(N, p, rng::AbstractRNG=Random.default_rng())
    (; π0, Π, r_θ, r_I, P_I, P_θ) = p
    # Extract single emission matrices from vector wrappers
    P_θ_mat = P_θ[1]
    P_I_mat = P_I[1]
    κ_θ = size(P_θ_mat, 1)
    κ_I = size(P_I_mat, 1)

    F0 = Categorical(π0[:])
    Π_dist = [Categorical(Π[:,kθ,kI]) for kθ in axes(Π,2), kI in axes(Π,3)]
    P_θ_dist = [Categorical(P_θ_mat[:,kθ]) for kθ in axes(Π,2)]
    P_I_dist = [Categorical(P_I_mat[:,kI]) for kI in axes(Π,3)]
    idx_inv = CartesianIndices((r_θ,r_I))

    X_I_vec = zeros(Int64, N)
    X_θ_vec = zeros(Int64, N)
    Y_vec = zeros(Int64, N)
    for n in 1:N
        kθ, kI = Tuple(idx_inv[rand(rng, F0)])
        kθ_next = rand(rng, Π_dist[kθ,kI])
        X_I_vec[n] = rand(rng, P_I_dist[kI])
        X_θ_vec[n] = rand(rng, P_θ_dist[kθ])
        Y_vec[n] = rand(rng, P_θ_dist[kθ_next])
    end

    # Return in unified format: Vector{Vector{Int}} for measures
    return (; X_θ = [X_θ_vec], X_I = [X_I_vec], Y = [Y_vec],
              κ_θ = [κ_θ], κ_I = [κ_I], κ_Y = [κ_θ],
              N = N, r_θ = r_θ, r_I = r_I)
end
