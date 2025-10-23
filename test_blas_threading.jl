include("parameters.jl")
include("estimation.jl")
using LinearAlgebra, Distributions, Random

# Test impact of BLAS threading on single iteration
p = default_model()
N = 1_000

println("Testing BLAS threading impact on single bootstrap iteration")
println("=" ^ 60)

# Test with BLAS threads = 10
LinearAlgebra.BLAS.set_num_threads(10)
println("\nWith BLAS threads = 10:")
rng = MersenneTwister(12345)
Q = zeros(3,3,3,N)
data_sim = simulate_data(N,p,rng)
p_boot = expectation_maximization(data_sim, p)
t_multi = @elapsed begin
    for i in 1:50
        include("inference.jl")  # need get_test_stat
        Tval,Ω,Σ = get_test_stat(Q, p_boot, data_sim)
    end
end
println("  50 get_test_stat calls: $(round(t_multi*1000, digits=1)) ms ($(round(t_multi*1000/50, digits=2)) ms each)")

# Test with BLAS threads = 1
LinearAlgebra.BLAS.set_num_threads(1)
println("\nWith BLAS threads = 1:")
rng = MersenneTwister(12345)
Q = zeros(3,3,3,N)
data_sim = simulate_data(N,p,rng)
p_boot = expectation_maximization(data_sim, p)
t_single = @elapsed begin
    for i in 1:50
        include("inference.jl")
        Tval,Ω,Σ = get_test_stat(Q, p_boot, data_sim)
    end
end
println("  50 get_test_stat calls: $(round(t_single*1000, digits=1)) ms ($(round(t_single*1000/50, digits=2)) ms each)")

println("\nSingle-threaded BLAS is $(round(100*(1 - t_single/t_multi), digits=1))% faster for your problem size")
