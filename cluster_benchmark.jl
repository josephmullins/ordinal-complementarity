#!/usr/bin/env julia

# Cluster benchmark script to test thread scaling
# Usage: julia -t N cluster_benchmark.jl

include("parameters.jl")
include("estimation.jl")
include("inference.jl")
using LinearAlgebra, Distributions

# Print system info
println("\n" * "=" ^ 80)
println("CLUSTER BENCHMARK - Thread Scaling Test")
println("=" ^ 80)
println("\nSystem Information:")
println("-" ^ 80)
println("Julia version: ", VERSION)
println("Julia threads: ", Threads.nthreads())
println("BLAS threads:  ", LinearAlgebra.BLAS.get_num_threads())
println("BLAS library:  ", LinearAlgebra.BLAS.vendor())
println("Hostname:      ", gethostname())

# CPU info (Linux)
if Sys.islinux()
    try
        cpu_info = read(`lscpu`, String)
        for line in split(cpu_info, '\n')
            if occursin(r"^Model name:|^CPU\(s\):|^Thread\(s\) per core:|^Core\(s\) per socket:", line)
                println(strip(line))
            end
        end
    catch
        println("CPU info: Unable to detect")
    end
end

println("-" ^ 80)

# Test parameters
p = default_model()
N = 1_000
nboot_small = 500
nboot_large = 2_000

# Warm-up
println("\nWarming up Julia (compiling functions)...")
monte_carlo_simulation_threaded(p; N=N, nboot=10, seed0=102025)
println("✓ Warm-up complete\n")

# Small workload test
println("=" ^ 80)
println("TEST 1: Small workload (nboot=$nboot_small)")
println("=" ^ 80)
GC.gc()
t_small = @elapsed r_small = monte_carlo_simulation_threaded(p; N=N, nboot=nboot_small, seed0=102025)
println("Time:               $(round(t_small, digits=3)) seconds")
println("Time per bootstrap: $(round(1000*t_small/nboot_small, digits=2)) ms")
println("Rejection rate:     $(round(r_small[1], digits=4))")
speedup_small = nboot_small / (t_small * Threads.nthreads() / nboot_small)
println("Efficiency:         $(round(100 * t_small / (t_small * Threads.nthreads()) * Threads.nthreads(), digits=1))% (ideal: 100%)")

# Large workload test
println("\n" * "=" ^ 80)
println("TEST 2: Large workload (nboot=$nboot_large)")
println("=" ^ 80)
GC.gc()
t_large = @elapsed r_large = monte_carlo_simulation_threaded(p; N=N, nboot=nboot_large, seed0=102025)
println("Time:               $(round(t_large, digits=3)) seconds")
println("Time per bootstrap: $(round(1000*t_large/nboot_large, digits=2)) ms")
println("Rejection rate:     $(round(r_large[1], digits=4))")
speedup_large = nboot_large / (t_large * Threads.nthreads() / nboot_large)
println("Efficiency:         $(round(100 * t_large / (t_large * Threads.nthreads()) * Threads.nthreads(), digits=1))% (ideal: 100%)")

# Summary
println("\n" * "=" ^ 80)
println("SUMMARY")
println("=" ^ 80)
println("Configuration: $(Threads.nthreads()) Julia threads, $(LinearAlgebra.BLAS.get_num_threads()) BLAS threads")
println("Small workload: $(round(t_small, digits=3))s for $nboot_small bootstraps")
println("Large workload: $(round(t_large, digits=3))s for $nboot_large bootstraps")
println("\nRecommendations:")
if t_small / nboot_small > t_large / nboot_large * 1.1
    println("⚠ Small workload has higher overhead - use larger nboot for better efficiency")
end
efficiency = 100 * (1 / (t_small * Threads.nthreads() / nboot_small))
if efficiency < 50
    println("⚠ Threading efficiency < 50% - try fewer threads or larger workload")
elseif efficiency > 80
    println("✓ Good threading efficiency ($(round(efficiency, digits=1))%)")
else
    println("○ Moderate threading efficiency ($(round(efficiency, digits=1))%)")
end
println("\nTo test different thread counts, run:")
println("  julia -t 1 cluster_benchmark.jl")
println("  julia -t 4 cluster_benchmark.jl")
println("  julia -t 8 cluster_benchmark.jl")
println("  julia -t 16 cluster_benchmark.jl")
println("=" ^ 80)
