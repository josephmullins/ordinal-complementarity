# β5/δ: log-income semi-elasticity of the investment CCP index (via the ε_Y
# channel) — point estimate from the fit, SE from the child bootstrap.
# Writes output/paper_tables_boot/ccp_income_semielasticity.csv.
# (Previously written as a side effect of counterfactual_income_piecewise_10k.jl.)
#
# ENV: FIT_OVERRIDE (default output/estimation_v3_eps_reg.jld2)
#      BOOT_COV     (default output/bootstrap_reg.jld2)
using JLD2, Statistics, Printf

const REPO     = joinpath(@__DIR__, "..")
const FIT_PATH = get(ENV, "FIT_OVERRIDE", joinpath(REPO, "output", "estimation_v3_eps_reg.jld2"))
const BOOT     = get(ENV, "BOOT_COV",     joinpath(REPO, "output", "bootstrap_reg.jld2"))
const OUT_DIR  = joinpath(REPO, "output", "paper_tables_boot")

params = load(FIT_PATH, "params_v3_eps")
β5, δ  = params.ccp_spec.beta5, params.income_model.delta

bc = load(BOOT)
ratio = bc["boot_ratio"][findall(bc["ok"])]
se = std(ratio)

isdir(OUT_DIR) || mkpath(OUT_DIR)
open(joinpath(OUT_DIR, "ccp_income_semielasticity.csv"), "w") do io
    println(io, "quantity,value")
    println(io, "beta5,", β5)
    println(io, "delta,", δ)
    println(io, "beta5_over_delta,", β5 / δ)
    println(io, "beta5_over_delta_boot_se,", se)
    println(io, "boot_B_ok,", length(ratio))
end
@printf "β5/δ = %.4f  (bootstrap SE %.4f, B=%d)\n" β5/δ se length(ratio)
