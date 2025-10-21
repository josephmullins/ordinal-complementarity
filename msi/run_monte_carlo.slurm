#!/bin/bash -l        
#SBATCH --time=2:00:00
#SBATCH --ntasks=50
#SBATCH --mem=100g
#SBATCH --tmp=100g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=mullinsj@umn.edu 
#SBATCH --output=output/run_monte_carlo.out

cd ~/ordinal-complementarity
$OLDHOME/julia-1.9.3/bin/julia -t 50 run_monte_carlo_threaded.jl