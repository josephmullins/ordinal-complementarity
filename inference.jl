using LinearAlgebra, Distributions, Random

function get_S(r_θ,r_I)
    p_idx = LinearIndices((r_θ-1,r_θ,r_I))
    np = prod(size(p_idx))
    R = r_θ * r_I
    num_rows = Int(r_θ*r_I*(r_θ-1)*(r_I-1)*(r_θ-1)/4)
    S = zeros(num_rows,np)
    r_index = 1
    for kθ in 1:r_θ, kI in 1:r_I
        for kθ2 in kθ+1:r_θ, kI2 in kI+1:r_I
            for kθ3 in 1:r_θ-1
                S[r_index,p_idx[1,kθ,kI]:p_idx[kθ3,kθ,kI]] .= 1.
                S[r_index,p_idx[1,kθ2,kI2]:p_idx[kθ3,kθ2,kI2]] .= 1.
                S[r_index,p_idx[1,kθ,kI2]:p_idx[kθ3,kθ,kI2]] .= -1.
                S[r_index,p_idx[1,kθ2,kI]:p_idx[kθ3,kθ2,kI]] .= -1.
                r_index += 1
            end
        end
    end
    return S
end

function get_test_stat(Q,p_est,data)
    r_θ,_,r_I,N = size(Q)
    S = get_S(r_θ,r_I)
    S = S[1:8,:] #<- temporary fix
    _ = expectation_step(Q,p_est,data)
    dL = get_score(Q, p_est.Π)
    Σ = inv(cov(dL'))
    V = S * Σ * S'
    Dinv = diagm(sqrt.(1 ./ diag(V)))
    Πvec = p_est.Π[1:end-1,:,:][:]
    Ω = Dinv * V * Dinv
    Tvec = sum(min.(0., sqrt(N) * Dinv * S*Πvec) .^2)
    return norm(Tvec), Ω, Σ
end

function crit_value(Ω, α; nsim = 100_000, rng::AbstractRNG=Random.default_rng())
    # Pre-compute Cholesky factorization once
    L = cholesky(Hermitian(Ω)).L
    d = size(Ω, 1)

    T = zeros(nsim)
    z = zeros(d)      # Standard normals
    Tvec = zeros(d)   # Transformed result

    for r in 1:nsim
        randn!(rng, z)      # Fill with standard normals - truly allocation-free
        mul!(Tvec, L, z)    # Tvec = L * z (in-place matrix-vector multiply)
        T[r] = sum(x -> min(0.0, x)^2, Tvec)
    end

    return quantile(T, 1 - α)
end

function monte_carlo_simulation(p;N = 1_000, nboot = 500, seed0 = 102025)
    rng = MersenneTwister(seed0)
    # p containts "true" parameters
    np = 3 * 3 * 2 #<- need to update this whatever is in p
    Q = zeros(3,3,3,N) #<- need to update this as well!
    Πboot = zeros(np,nboot)
    Tboot = zeros(nboot)
    Reject = zeros(nboot)
    cboot = zeros(nboot)
    for b in 1:nboot
        #println(" Doing trial $b")
        data_sim = simulate_data(N,p,rng)
        p_boot = expectation_maximization(data_sim, p)
        Πboot[:,b] = p_boot.Π[1:2,:,:][:]
        Tboot[b],Ω,_ = get_test_stat(Q, p_boot, data_sim)
        c_α = crit_value(Hermitian(Ω),0.05; rng=rng)
        Reject[b] = Tboot[b] > c_α
        cboot[b] = c_α
    end
    return mean(Reject), Tboot, cboot, Πboot
end

function monte_carlo_simulation_threaded(p;N = 1_000, nboot = 500, seed0 = 102025)
    # p containts "true" parameters
    np = 3 * 3 * 2 #<- need to update this whatever is in p
    Πboot = zeros(np,nboot)
    Tboot = zeros(nboot)
    Reject = zeros(nboot)
    cboot = zeros(nboot)
    chunks = Iterators.partition(1:nboot,cld(nboot,Threads.nthreads()))
    tasks = map(enumerate(chunks)) do (ch_id, chunk)
        seedb = seed0 + ch_id
        Threads.@spawn begin
            try
                monte_carlo_simulation_chunk!(chunk,Πboot,Tboot,cboot,Reject,p,N,seedb)
            catch e
                @error "Task failed" exception=(e, catch_backtrace())
                rethrow(e)
            end
        end
    end
    fetch.(tasks)
    return mean(Reject), Tboot, cboot, Πboot
end

function monte_carlo_simulation_chunk!(chunk,Πboot,Tboot,cboot,Reject,p,N,seed0)
    rng = MersenneTwister(seed0)
    Q = zeros(3,3,3,N) #<- need to update this as well!
    for b in chunk
        data_sim = simulate_data(N,p,rng)
        p_boot = expectation_maximization(data_sim, p)
        Πboot[:,b] = p_boot.Π[1:2,:,:][:]
        Tboot[b],Ω,_ = get_test_stat(Q, p_boot, data_sim)
        c_α = crit_value(Hermitian(Ω),0.05; rng=rng)
        Reject[b] = Tboot[b] > c_α
        cboot[b] = c_α
    end
end