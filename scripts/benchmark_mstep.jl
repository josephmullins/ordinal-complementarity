#!/usr/bin/env julia
#
# Benchmark the parametric M-step to identify performance bottlenecks.
#
# Usage: julia scripts/benchmark_mstep.jl
#

using LinearAlgebra, Random, Statistics, BenchmarkTools

include(joinpath(@__DIR__, "..", "src", "parameters.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation.jl"))

# ─────────────────────────────────────────────────────────────────
# Setup: simulate data and run E-step to get Q
# ─────────────────────────────────────────────────────────────────

const RNG = MersenneTwister(42)
const N_OBS = 2000

println("="^60)
println("  M-STEP BENCHMARK")
println("="^60)

# Default model: r_θ=3, r_I=3, 1 measure each (κ=3)
p_true = default_model()
data_small = simulate_data(N_OBS, p_true, RNG)

println("\nSmall problem: r_θ=$(data_small.r_θ), r_I=$(data_small.r_I), N=$(N_OBS)")
println("  Skill measures: $(length(data_small.κ_θ)) (κ=$(data_small.κ_θ))")
println("  Investment measures: $(length(data_small.κ_I)) (κ=$(data_small.κ_I))")

# Realistic problem: many binary measures
function make_multimeasure_data(N, n_skill, n_inv, n_outcome; rng=MersenneTwister(42))
    p = default_model()
    base = simulate_data(N, p, rng)
    X_θ = [copy(base.X_θ[1]) for _ in 1:n_skill]
    X_I = [copy(base.X_I[1]) for _ in 1:n_inv]
    Y   = [copy(base.Y[1])   for _ in 1:n_outcome]
    for m in 1:n_skill; X_θ[m] = [x >= 2 ? 2 : 1 for x in X_θ[m]] end
    for m in 1:n_inv;   X_I[m] = [x >= 2 ? 2 : 1 for x in X_I[m]] end
    for m in 1:n_outcome; Y[m] = [y >= 2 ? 2 : 1 for y in Y[m]] end
    return (; X_θ, X_I, Y,
              κ_θ=fill(2, n_skill), κ_I=fill(2, n_inv), κ_Y=fill(2, n_outcome),
              N, r_θ=3, r_I=3)
end

data_large = make_multimeasure_data(N_OBS, 10, 20, 10)
println("\nLarge problem: $(length(data_large.κ_θ)) skill, $(length(data_large.κ_I)) inv, $(length(data_large.κ_Y)) outcome measures")

# ─────────────────────────────────────────────────────────────────
# Initialize and run E-step
# ─────────────────────────────────────────────────────────────────

config_pars = (emission_θ=:ordered_probit, emission_Y=:ordered_probit,
               emission_I=:ordered_probit, transition=:parsimonious_lpm)
config_flex = (emission_θ=:ordered_probit, emission_Y=:ordered_probit,
               emission_I=:ordered_probit, transition=:flexible_lpm)

p_small, specs_small = initialize_params_parametric(data_small, config_pars; rng=MersenneTwister(1))
Q_small = zeros(3, 3, 3, N_OBS)
expectation_step!(Q_small, p_small, data_small)

p_large, specs_large = initialize_params_parametric(data_large, config_pars; rng=MersenneTwister(1))
Q_large = zeros(3, 3, 3, N_OBS)
expectation_step!(Q_large, p_large, data_large)

_, specs_flex_small = initialize_params_parametric(data_small, config_flex; rng=MersenneTwister(1))
_, specs_flex_large = initialize_params_parametric(data_large, config_flex; rng=MersenneTwister(1))

# ─────────────────────────────────────────────────────────────────
# Correctness check: verify optimized path matches original
# ─────────────────────────────────────────────────────────────────

println("\n" * "─"^60)
println("  CORRECTNESS CHECK")
println("─"^60)

# Test precomputed weights match on-the-fly computation
W_theta = _precompute_marginal_weights(Q_small, :theta, 3)
for n in [1, 100, N_OBS], k in 1:3
    w_old = _marginal_weight(Q_small, n, k, :theta)
    w_new = W_theta[k, n]
    @assert abs(w_old - w_new) < 1e-14 "Weight mismatch at n=$n, k=$k"
end
println("  Marginal weights: OK")

# Test single-measure negll matches multi-measure negll (for 1 measure)
x0 = pack_probit(specs_small[:probit_θ])
W_s = _precompute_marginal_weights(Q_small, :theta, 3)
val_joint = emission_negll(x0, Q_small, data_small.X_θ, data_small.κ_θ, :theta, 3)
val_single = _single_measure_negll(x0, W_s, data_small.X_θ[1], data_small.κ_θ[1], 3)
@assert abs(val_joint - val_single) < 1e-10 "Single-measure negll doesn't match joint: $val_joint vs $val_single"
println("  Single-measure negll: OK (diff=$(abs(val_joint - val_single)))")

# Test Q_sum precomputation
Q_sum = _precompute_Q_sum(Q_small)
Pi_test = transition_from_parsimonious_lpm(specs_small[:transition]::ParsimoniousLPMTransition, 3, 3)
val_full = _transition_negll(Pi_test, Q_small)
val_pre = _transition_negll_precomputed(Pi_test, Q_sum)
@assert abs(val_full - val_pre) < 1e-10 "Q_sum precomputed negll mismatch: $val_full vs $val_pre"
println("  Q_sum precomputed negll: OK (diff=$(abs(val_full - val_pre)))")

# Test that optimized M-step gives same results
spec_new1, P1 = parametric_mstep_emission(specs_small[:probit_θ], Q_small,
    data_small.X_θ, data_small.κ_θ, :theta, 3)
spec_new2, Pi2 = parametric_mstep_transition(specs_small[:transition]::ParsimoniousLPMTransition,
    Q_small, 3, 3)
println("  Optimized emission M-step: converged (α=$(round.(spec_new1.alpha; digits=3)))")
println("  Optimized transition M-step: converged (γ₃=$(round(spec_new2.gamma3; digits=4)))")

# ─────────────────────────────────────────────────────────────────
# BENCHMARKS
# ─────────────────────────────────────────────────────────────────

println("\n" * "─"^60)
println("  1. NONPARAMETRIC M-STEP (baseline)")
println("─"^60)

print("  Small: ")
@btime maximization_step($Q_small, $data_small)
print("  Large: ")
@btime maximization_step($Q_large, $data_large)

println("\n" * "─"^60)
println("  2. FULL PARAMETRIC M-STEP (optimized)")
println("─"^60)

print("  Small (pars):  ")
@btime maximization_step_parametric($Q_small, $data_small, $specs_small, $config_pars)
print("  Large (pars):  ")
@btime maximization_step_parametric($Q_large, $data_large, $specs_large, $config_pars)

println("\n" * "─"^60)
println("  3. INDIVIDUAL COMPONENTS")
println("─"^60)

println("\n  Emission θ (ordered probit):")
print("    Small (1 meas, κ=3):   ")
@btime parametric_mstep_emission($(specs_small[:probit_θ]), $Q_small,
    $(data_small.X_θ), $(data_small.κ_θ), :theta, 3)
print("    Large (10 meas, κ=2):  ")
@btime parametric_mstep_emission($(specs_large[:probit_θ]), $Q_large,
    $(data_large.X_θ), $(data_large.κ_θ), :theta, 3)

println("\n  Emission Y (ordered probit):")
print("    Small (1 meas, κ=3):   ")
@btime parametric_mstep_emission($(specs_small[:probit_Y]), $Q_small,
    $(data_small.Y), $(data_small.κ_Y), :theta_prime, 3)
print("    Large (10 meas, κ=2):  ")
@btime parametric_mstep_emission($(specs_large[:probit_Y]), $Q_large,
    $(data_large.Y), $(data_large.κ_Y), :theta_prime, 3)

println("\n  Emission I (ordered probit):")
print("    Small (1 meas, κ=3):   ")
@btime parametric_mstep_emission($(specs_small[:probit_I]), $Q_small,
    $(data_small.X_I), $(data_small.κ_I), :I, 3)
print("    Large (20 meas, κ=2):  ")
@btime parametric_mstep_emission($(specs_large[:probit_I]), $Q_large,
    $(data_large.X_I), $(data_large.κ_I), :I, 3)

println("\n  Transition (parsimonious LPM):")
print("    Small:  ")
@btime parametric_mstep_transition($(specs_small[:transition]::ParsimoniousLPMTransition),
    $Q_small, 3, 3)
print("    Large:  ")
@btime parametric_mstep_transition($(specs_large[:transition]::ParsimoniousLPMTransition),
    $Q_large, 3, 3)

println("\n  Transition (flexible LPM):")
print("    Small:  ")
@btime parametric_mstep_transition($(specs_flex_small[:transition]::FlexibleLPMTransition),
    $Q_small, 3, 3)
print("    Large:  ")
@btime parametric_mstep_transition($(specs_flex_large[:transition]::FlexibleLPMTransition),
    $Q_large, 3, 3)

# ─────────────────────────────────────────────────────────────────
# 4. Inner function benchmarks
# ─────────────────────────────────────────────────────────────────

println("\n" * "─"^60)
println("  4. OBJECTIVE + GRADIENT EVALUATIONS")
println("─"^60)

using ForwardDiff

# Precomputed weights
W_theta_s = _precompute_marginal_weights(Q_small, :theta, 3)
W_theta_l = _precompute_marginal_weights(Q_large, :theta, 3)
W_I_l = _precompute_marginal_weights(Q_large, :I, 3)
Q_sum_s = _precompute_Q_sum(Q_small)
Q_sum_l = _precompute_Q_sum(Q_large)

# Single-measure objective (κ=3 has 3 params: 2 cutpoints + alpha; κ=2 has 2 params)
x_single_3 = vcat(_pack_cutpoints([0.0, 1.0]), 0.5)  # κ=3: 3 params
x_single_2 = vcat(_pack_cutpoints([0.0]), 0.5)        # κ=2: 2 params
f_single_s(x) = _single_measure_negll(x, W_theta_s, data_small.X_θ[1], 3, 3)
f_single_l(x) = _single_measure_negll(x, W_theta_l, data_large.X_θ[1], 2, 3)

println("\n  Single-measure negll (objective):")
print("    Small (κ=3, 3 params):  ")
@btime _single_measure_negll($x_single_3, $W_theta_s, $(data_small.X_θ[1]), 3, 3)
print("    Large (κ=2, 2 params):  ")
@btime _single_measure_negll($x_single_2, $W_theta_l, $(data_large.X_θ[1]), 2, 3)

println("\n  Single-measure gradient (ForwardDiff):")
print("    Small (κ=3, 3 params):  ")
@btime ForwardDiff.gradient($f_single_s, $x_single_3)
print("    Large (κ=2, 2 params):  ")
@btime ForwardDiff.gradient($f_single_l, $x_single_2)

# Transition objectives with precomputed Q_sum
x_pars = pack_parsimonious_lpm(specs_small[:transition]::ParsimoniousLPMTransition)
x_flex = pack_flexible_lpm(specs_flex_small[:transition]::FlexibleLPMTransition)

f_pars_fast(x) = transition_negll_parsimonious_fast(x, Q_sum_s, 3, 3)
f_pars_old(x)  = transition_negll_parsimonious(x, Q_small, 3, 3)
f_flex_fast(x) = transition_negll_flexible_fast(x, Q_sum_s, 3, 3)
f_flex_old(x)  = transition_negll_flexible(x, Q_small, 3, 3)

println("\n  Transition negll (objective):")
print("    Parsimonious (old, N loop):  ")
@btime transition_negll_parsimonious($x_pars, $Q_small, 3, 3)
print("    Parsimonious (new, Q_sum):   ")
@btime transition_negll_parsimonious_fast($x_pars, $Q_sum_s, 3, 3)

println("\n  Transition gradient:")
print("    Parsimonious (old):          ")
@btime ForwardDiff.gradient($f_pars_old, $x_pars)
print("    Parsimonious (new, Q_sum):   ")
@btime ForwardDiff.gradient($f_pars_fast, $x_pars)
print("    Flexible (old):              ")
@btime ForwardDiff.gradient($f_flex_old, $x_flex)
print("    Flexible (new, Q_sum):       ")
@btime ForwardDiff.gradient($f_flex_fast, $x_flex)

# ─────────────────────────────────────────────────────────────────
# 5. E-step for reference
# ─────────────────────────────────────────────────────────────────

println("\n" * "─"^60)
println("  5. E-STEP (reference)")
println("─"^60)
print("  Small: ")
@btime expectation_step!($Q_small, $p_small, $data_small)
print("  Large: ")
@btime expectation_step!($Q_large, $p_large, $data_large)

# ─────────────────────────────────────────────────────────────────
# 6. Weight precomputation cost
# ─────────────────────────────────────────────────────────────────

println("\n" * "─"^60)
println("  6. PRECOMPUTATION COSTS")
println("─"^60)
print("  _precompute_marginal_weights (small): ")
@btime _precompute_marginal_weights($Q_small, :theta, 3)
print("  _precompute_marginal_weights (large): ")
@btime _precompute_marginal_weights($Q_large, :theta, 3)
print("  _precompute_Q_sum (small):            ")
@btime _precompute_Q_sum($Q_small)
print("  _precompute_Q_sum (large):            ")
@btime _precompute_Q_sum($Q_large)

println("\n" * "="^60)
println("  DONE")
println("="^60)
