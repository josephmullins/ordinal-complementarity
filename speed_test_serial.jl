include("parameters.jl")
include("estimation.jl")
include("inference.jl")
using LinearAlgebra, Distributions, DelimitedFiles

p = default_model()

N = 500
monte_carlo_simulation(p; N, nboot = Threads.nthreads())
@time monte_carlo_simulation(p; N, nboot = 50)
