include("parameters.jl")
include("estimation.jl")
include("inference.jl")
using LinearAlgebra, Distributions, DelimitedFiles

p = default_model()

sample_sizes = (500,1_000,5_000,10_000)

results = map(sample_sizes) do N
    seed0 = 1010 + N
    r = monte_carlo_simulation_threaded(p; N, nboot = 10_000,seed0)
    println("Finished doing sample size $N")
    @show length(r[2])
    @show sum(r[2])
    return r
end

size_mc = [r[1] for r in results]
writedlm("output/results/size_boot.csv",size_mc)
