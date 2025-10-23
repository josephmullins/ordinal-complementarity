include("parameters.jl")
include("estimation.jl")
include("inference.jl")
using LinearAlgebra, Distributions

p = default_model()
N = 1_000
nboot = 20

println("Running with nboot=$nboot")

# Run serial
r_serial = monte_carlo_simulation(p; N=N, nboot=nboot, seed0=102025)

# Run threaded
r_threaded = monte_carlo_simulation_threaded(p; N=N, nboot=nboot, seed0=102025)

println("\nSerial rejection rate: ", r_serial[1])
println("Threaded rejection rate: ", r_threaded[1])
println("Difference: ", abs(r_serial[1] - r_threaded[1]))

println("\nFirst 10 Tboot values:")
println("Serial:   ", r_serial[2][1:10])
println("Threaded: ", r_threaded[2][1:10])

println("\nSorted Tboot values (should be similar distributions):")
println("Serial:   ", sort(r_serial[2])[1:10])
println("Threaded: ", sort(r_threaded[2])[1:10])
