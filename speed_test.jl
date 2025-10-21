include("parameters.jl")
include("estimation.jl")
include("inference.jl")
using LinearAlgebra, Distributions, DelimitedFiles

p = default_model()

N = 1_000
monte_carlo_simulation_threaded(p; N, nboot = Threads.nthreads())
@time monte_carlo_simulation_threaded(p; N, nboot = 1_000)
