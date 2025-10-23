include("parameters.jl")
include("estimation.jl")
include("inference.jl")
using LinearAlgebra, Distributions

p = default_model()
N = 1_000
nboot = 100

println("\nThreading Diagnostic")
println("===================")
println("Available threads: ", Threads.nthreads())
println("Number of bootstraps: ", nboot)
println("Bootstraps per thread: ", cld(nboot, Threads.nthreads()))
println()

# Profile where time is spent in a single iteration
println("Profiling single bootstrap iteration:")
rng = MersenneTwister(12345)
Q = zeros(3,3,3,N)

t1 = @elapsed data_sim = simulate_data(N,p,rng)
t2 = @elapsed p_boot = expectation_maximization(data_sim, p)
t3 = @elapsed Tval,Ω,_ = get_test_stat(Q, p_boot, data_sim)
t4 = @elapsed c_α = crit_value(Hermitian(Ω),0.05; rng=rng)

println("  simulate_data:     $(round(t1*1000, digits=1)) ms ($(round(100*t1/(t1+t2+t3+t4), digits=1))%)")
println("  EM:                $(round(t2*1000, digits=1)) ms ($(round(100*t2/(t1+t2+t3+t4), digits=1))%)")
println("  get_test_stat:     $(round(t3*1000, digits=1)) ms ($(round(100*t3/(t1+t2+t3+t4), digits=1))%)")
println("  crit_value:        $(round(t4*1000, digits=1)) ms ($(round(100*t4/(t1+t2+t3+t4), digits=1))%)")
println("  Total per iter:    $(round((t1+t2+t3+t4)*1000, digits=1)) ms")
println()

# Check for BLAS threading interaction
println("BLAS threads: ", LinearAlgebra.BLAS.get_num_threads())
println("BLAS library: ", LinearAlgebra.BLAS.get_config())
println()

# Time threaded vs serial
println("Running serial version...")
@time r_serial = monte_carlo_simulation(p; N=N, nboot=nboot, seed0=102025)

println("\nRunning threaded version...")
@time r_threaded = monte_carlo_simulation_threaded(p; N=N, nboot=nboot, seed0=102025)

println("\nResults match: ", isapprox(r_serial[1], r_threaded[1], atol=1e-10))
