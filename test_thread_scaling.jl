include("parameters.jl")
include("estimation.jl")
include("inference.jl")
using LinearAlgebra, Distributions

p = default_model()
N = 1_000
nboot = 400  # Larger workload to see scaling

println("\nThread Scaling Test (with BLAS threads = 1)")
println("=" ^ 70)
println("Julia threads: $(Threads.nthreads())")
println("BLAS threads: $(LinearAlgebra.BLAS.get_num_threads())")
println("Testing with nboot=$nboot\n")

# Warm-up run
println("Warming up...")
monte_carlo_simulation_threaded(p; N=N, nboot=10, seed0=102025)

# Actual test
println("\nRunning benchmark...")
GC.gc()
t = @elapsed result = monte_carlo_simulation_threaded(p; N=N, nboot=nboot, seed0=102025)

println("\nResults:")
println("-" ^ 70)
println("Time: $(round(t, digits=3)) seconds")
println("Time per bootstrap: $(round(1000*t/nboot, digits=2)) ms")
println("Rejection rate: $(round(result[1], digits=4))")
println("\nExpected time with different thread counts:")
println("  1 thread:  ~$(round(t * Threads.nthreads(), digits=1))s (estimated)")
println("  $(Threads.nthreads()) threads: $(round(t, digits=1))s (actual)")
println("  Speedup: $(round(Threads.nthreads() * t / (t * Threads.nthreads()), digits=2))x (vs 1 thread estimate)")
