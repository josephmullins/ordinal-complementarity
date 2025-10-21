#!/bin/bash -l        
#SBATCH --time=1:00:00
#SBATCH --ntasks=100
#SBATCH --mem=100g
#SBATCH --tmp=100g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=mullinsj@umn.edu 
#SBATCH --output=output/run_speed_test.out

cd ~/ordinal-complementarity
$OLDHOME/julia-1.9.3/bin/julia -t 20 speed_test.jl
$OLDHOME/julia-1.9.3/bin/julia -t 50 speed_test.jl
$OLDHOME/julia-1.9.3/bin/julia -t 100 speed_test.jl