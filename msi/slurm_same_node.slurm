#!/bin/bash
#SBATCH --job-name=julia_same_node
#SBATCH --output=same_node_%j.log
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32

# This script runs ALL thread counts on the SAME NODE
# to eliminate node-to-node performance variation

echo "=========================================="
echo "Running all thread counts on the same node"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Memory: 64G"
echo "=========================================="

# Change to the project directory
cd ~/ordinal-complementarity

# Test with nboot=1000 for better statistics
NBOOT=1000

for THREADS in 1 4 8 16 32; do
    echo ""
    echo "=========================================="
    echo "Testing with $THREADS threads"
    echo "=========================================="

    # Create a temporary test script with the right nboot
    cat > temp_test_${THREADS}.jl <<EOF
include("parameters.jl")
include("estimation.jl")
include("inference.jl")
using LinearAlgebra, Distributions

p = default_model()
N = 1_000
nboot = $NBOOT

println("Node: \$(gethostname())")
println("Julia threads: \$(Threads.nthreads())")
println("BLAS threads: \$(LinearAlgebra.BLAS.get_num_threads())")
println("Testing nboot=$nboot")
println()

# Warm up
monte_carlo_simulation_threaded(p; N=N, nboot=10, seed0=102025)

# Actual test
GC.gc()
t = @elapsed result = monte_carlo_simulation_threaded(p; N=N, nboot=nboot, seed0=102025)

println("Time: \$(round(t, digits=3))s")
println("Time per bootstrap: \$(round(1000*t/nboot, digits=2))ms")
println("Rejection rate: \$(round(result[1], digits=4))")
EOF

    $OLDHOME/julia-1.9.3/bin/julia -t $THREADS temp_test_${THREADS}.jl
    rm temp_test_${THREADS}.jl

done

echo ""
echo "=========================================="
echo "All tests completed at $(date)"
echo "=========================================="
