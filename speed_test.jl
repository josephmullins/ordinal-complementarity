include("parameters.jl")
include("estimation.jl")
include("inference.jl")
using LinearAlgebra, Distributions, DelimitedFiles

p = default_model()

N = 1_000
monte_carlo_simulation_threaded(p; N, nboot = Threads.nthreads())
@time monte_carlo_simulation_threaded(p; N, nboot = 100)


# nboot = 100
# chunk = 1:10
# np = 3 * 3 * 2 #<- need to update this whatever is in p
# Πboot = zeros(np,nboot)
# Tboot = zeros(nboot)
# Reject = zeros(nboot)
# cboot = zeros(nboot)
# N = 1_000

# @time monte_carlo_simulation_chunk!(chunk,Πboot,Tboot,cboot,Reject,p,1_000,1010)
# @time data_sim = simulate_data(N,p);
# @time p_boot = expectation_maximization(data_sim, p);
# b = 1
# @time Πboot[:,b] = p_boot.Π[1:2,:,:][:];
# Tboot[b],Ω,_ = get_test_stat(Q, p_boot, data_sim);
# @time c_α = crit_value(Hermitian(Ω),0.05);


# function time_one_bootstrap(p; N=1_000)
#     rng = MersenneTwister(102025)
#     Q = zeros(3,3,3,N)

#     t1 = @elapsed data_sim = simulate_data(N,p,rng)
#     t2 = @elapsed p_boot = expectation_maximization(data_sim, p)
#     t3 = @elapsed Tval,Ω,_ = get_test_stat(Q, p_boot, data_sim)
#     t4 = @elapsed c_α = crit_value(Hermitian(Ω),0.05; rng=rng)

#     println("simulate: $(round(t1*1000))ms, EM: $(round(t2*1000))ms, stat: $(round(t3*1000))ms, crit: $(round(t4*1000))ms")
#     println("Total: $(round((t1+t2+t3+t4)*1000))ms")
# end

# time_one_bootstrap(p)