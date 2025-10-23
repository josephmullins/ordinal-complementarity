using Distributions, Random

function default_model()
    r_θ = 3
    r_I = 3
    P_I = zeros(3,r_I)
    P_θ = zeros(3,r_θ)
    P_I[:,1]= [0.6,0.2,0.2]
    P_I[:,2]= [0.2,0.6,0.2]
    P_I[:,3]= [0.1,0.2,0.7]

    P_θ[:,1]= [0.6,0.2,0.2]
    P_θ[:,2]= [0.1,0.7,0.2]
    P_θ[:,3]= [0.15,0.15,0.7]

    γ = [0.1,0.1,0.]
    c = [0.25,0.5]
    Igrid = [-1.,0.,1.]
    θgrid = [-1.,0.,1.]
    Π = threshold_transitions(c,γ,Igrid,θgrid)
    π0 = ones(r_θ,r_I) / (r_θ*r_I) #<- uniform distribution here
    return (;Π,π0,P_I,P_θ,r_θ,r_I)
end

Φ(x) = cdf(Normal(),x)

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

function simulate_data(N,p,rng::AbstractRNG=Random.default_rng())
    (;π0,Π,r_θ,r_I,P_I,P_θ) = p
    F0 = Categorical(π0[:])
    Π_dist = [Categorical(Π[:,kθ,kI]) for kθ in axes(Π,2), kI in axes(Π,3)]
    P_θ_dist = [Categorical(P_θ[:,kθ]) for kθ in axes(Π,2)]
    P_I_dist = [Categorical(P_I[:,kI]) for kI in axes(Π,3)]
    idx_inv = CartesianIndices((r_θ,r_I))
    X_I = zeros(Int64,N)
    X_θ = zeros(Int64,N)
    Y = zeros(Int64,N)
    for n in 1:N
        kθ,kI = Tuple(idx_inv[rand(rng, F0)])
        kθ_next = rand(rng, Π_dist[kθ,kI])
        X_I[n] = rand(rng, P_I_dist[kI])
        X_θ[n] = rand(rng, P_θ_dist[kθ])
        Y[n] = rand(rng, P_θ_dist[kθ_next])
    end
    return (;X_I,X_θ,Y)
end

