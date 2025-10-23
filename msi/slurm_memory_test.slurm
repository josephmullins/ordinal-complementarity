#!/bin/bash
#SBATCH --job-name=julia_mem_test
#SBATCH --output=mem_test_%j_%a.log
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=8,32,64,128

# Test if memory allocation affects performance
# Array tests: 8G, 32G, 64G, 128G

MEMORY=${SLURM_ARRAY_TASK_ID}G

echo "=========================================="
echo "Testing with memory=${MEMORY}"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: 4"
echo "=========================================="

# Change to the project directory
cd ~/ordinal-complementarity

# Run benchmark with 4 threads (optimal from previous tests)
export JULIA_NUM_THREADS=4

cat > temp_mem_test.jl <<'EOF'
include("parameters.jl")
include("estimation.jl")
include("inference.jl")
using LinearAlgebra, Distributions

p = default_model()
N = 1_000
nboot = 1000

println("Memory info:")
println("  Allocated memory: $MEMORY")
println("  Julia threads: $(Threads.nthreads())")
println("  BLAS threads: $(LinearAlgebra.BLAS.get_num_threads())")
println()

# Warm up
monte_carlo_simulation_threaded(p; N=N, nboot=10, seed0=102025)

# Test 3 times for consistency
times = Float64[]
for i in 1:3
    GC.gc()
    t = @elapsed result = monte_carlo_simulation_threaded(p; N=N, nboot=nboot, seed0=102025+i)
    push!(times, t)
    println("Run $i: $(round(t, digits=3))s ($(round(1000*t/nboot, digits=2))ms per bootstrap)")
end

println()
println("Average: $(round(sum(times)/3, digits=3))s")
println("Std dev: $(round(std(times), digits=3))s")
EOF

$OLDHOME/julia-1.9.3/bin/julia -t 4 temp_mem_test.jl
rm temp_mem_test.jl

echo ""
echo "Completed at $(date)"
