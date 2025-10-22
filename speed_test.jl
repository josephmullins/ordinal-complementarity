include("parameters.jl")
include("estimation.jl")
include("inference.jl")
using LinearAlgebra, Distributions, DelimitedFiles

p = default_model()

N = 1_000
monte_carlo_simulation_threaded(p; N, nboot = Threads.nthreads())
@time monte_carlo_simulation_threaded(p; N, nboot = 1_000)


nboot = 100
chunk = 1:10
np = 3 * 3 * 2 #<- need to update this whatever is in p
Πboot = zeros(np,nboot)
Tboot = zeros(nboot)
Reject = zeros(nboot)
cboot = zeros(nboot)
N = 1_000
@time monte_carlo_simulation_chunk!(chunk,Πboot,Tboot,cboot,Reject,p,1_000,1010)
@time data_sim = simulate_data(N,p);
@time p_boot = expectation_maximization(data_sim, p);
b = 1
@time Πboot[:,b] = p_boot.Π[1:2,:,:][:];
Tboot[b],Ω,_ = get_test_stat(Q, p_boot, data_sim);
@time c_α = crit_value(Hermitian(Ω),0.05);

Tvec = zeros(8)
F = MvNormal(Hermitian(Ω))
@time rand!(F,Tvec);
rand!(F,Tvec)

function crit_value(Ω, α; nsim = 100_000)
    # Pre-compute Cholesky factorization once
    L = cholesky(Hermitian(Ω)).L
    d = size(Ω, 1)
    
    T = zeros(nsim)
    z = zeros(d)      # Standard normals
    Tvec = zeros(d)   # Transformed result
    
    for r in 1:nsim
        randn!(z)           # Fill with standard normals - truly allocation-free
        mul!(Tvec, L, z)    # Tvec = L * z (in-place matrix-vector multiply)
        T[r] = sum(x -> min(0.0, x)^2, Tvec)
    end
    
    return quantile(T, 1 - α)
end