include("estimation.jl")
include("parameters.jl")
include("inference.jl")
using LinearAlgebra

# simulate some data from default parameters
p = default_model()
N = 500

data = simulate_data(N,p)
@time data = simulate_data(N,p)

# jointly test the posterior calculation and the max step on π0
# write as a function, looks like it works.
Q = zeros(3,3,3,N)
ll = expectation_step(Q,p,data)
@time expectation_step(Q,p,data)

# test the max step on P_I:
P_I = update_P_I(Q,data.X_I)
P_θ = update_P_θ(Q,data.Y,data.X_θ)
@time P_I = update_P_I(Q,data.X_I)
@time P_θ = update_P_θ(Q,data.Y,data.X_θ)

Π = update_Π(Q);
@time update_Π(Q);

# test the max step on Π

p_est = expectation_maximization(data, p)
@code_warntype expectation_maximization(data, p);

#expectation_step!(Q,p,data)
expectation_step!(Q,p_est,data)

@time dL = get_score(Q, p_est.Π);


# Monte-Carlo Simulation
# 1. forecast the variance of the estimates:



@time get_test_stat(Q,p_est,data);
crit_value(Hermitian(Ω),0.05)

S = get_S(3,3)

expectation_step!(Q,p_est,data)
dL = get_score(Q, p_est.Π)
V_est = 1/N * inv(cov(dL'))


# the variance here doesn't look too different!
mean(Reject)
histogram(Tboot)


# NEXT: work on convergence issue? (options to speed it up?)
# NEXT: estimate standard errors for Π
# AND: get the test statistic