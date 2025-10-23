#!/bin/bash
#SBATCH --job-name=julia_benchmark
#SBATCH --output=benchmark_%j_%a.log
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32
#SBATCH --array=1,4,8,16,32

# This script tests different thread counts
# It will submit 5 jobs with 1, 4, 8, 16, and 32 threads
# Note: We request 32 CPUs for all jobs (the maximum we'll test)
#       Julia will only use JULIA_NUM_THREADS of them

# Set the number of threads based on the array task ID
export JULIA_NUM_THREADS=${SLURM_ARRAY_TASK_ID}

echo "=========================================="
echo "Starting benchmark with $JULIA_NUM_THREADS threads"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="

# Change to the project directory
cd ~/ordinal-complementarity

# Run the benchmark
$OLDHOME/julia-1.9.3/bin/julia -t ${JULIA_NUM_THREADS} cluster_benchmark.jl

echo ""
echo "Benchmark completed at $(date)"
