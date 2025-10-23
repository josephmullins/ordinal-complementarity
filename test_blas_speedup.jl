include("parameters.jl")
include("estimation.jl")
include("inference.jl")
using LinearAlgebra, Distributions

p = default_model()
N = 1_000
nboot = 100

println("\nTesting BLAS threading impact on monte_carlo_simulation_threaded")
println("=" ^ 70)

# Store original BLAS thread count
original_blas_threads = LinearAlgebra.BLAS.get_num_threads()
println("Original BLAS threads: $original_blas_threads")
println("Julia threads: $(Threads.nthreads())")
println("Testing with nboot=$nboot\n")

# Test with original BLAS settings
println("Test 1: BLAS threads = $original_blas_threads (current setting)")
println("-" ^ 70)
LinearAlgebra.BLAS.set_num_threads(original_blas_threads)
GC.gc()  # Clear garbage before timing
t1 = @elapsed r1 = monte_carlo_simulation_threaded(p; N=N, nboot=nboot, seed0=102025)
println("Time: $(round(t1, digits=3)) seconds")
println("Rejection rate: $(round(r1[1], digits=4))\n")

# Test with BLAS threads = 1
println("Test 2: BLAS threads = 1")
println("-" ^ 70)
LinearAlgebra.BLAS.set_num_threads(1)
GC.gc()  # Clear garbage before timing
t2 = @elapsed r2 = monte_carlo_simulation_threaded(p; N=N, nboot=nboot, seed0=102025)
println("Time: $(round(t2, digits=3)) seconds")
println("Rejection rate: $(round(r2[1], digits=4))\n")

# Restore original
LinearAlgebra.BLAS.set_num_threads(original_blas_threads)

# Summary
println("=" ^ 70)
println("SUMMARY")
println("=" ^ 70)
if t2 < t1
    speedup = (t1 - t2) / t1 * 100
    println("✓ BLAS threads=1 is FASTER by $(round(speedup, digits=1))%")
    println("  ($(round(t1, digits=3))s → $(round(t2, digits=3))s)")
    println("\nRecommendation: Add this line to the top of inference.jl:")
    println("  LinearAlgebra.BLAS.set_num_threads(1)")
else
    slowdown = (t2 - t1) / t1 * 100
    println("✗ BLAS threads=1 is SLOWER by $(round(slowdown, digits=1))%")
    println("  ($(round(t1, digits=3))s → $(round(t2, digits=3))s)")
    println("\nRecommendation: Keep current BLAS settings")
end
println("\nNote: Results may vary between runs due to system load and thermal throttling.")
