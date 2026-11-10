using LinearAlgebra, Statistics, Printf, JLD2, CSV, DataFrames, Plots, Optim, ForwardDiff

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3_eps.jl"))
include(joinpath(@__DIR__, "..", "src", "counterfactuals.jl"))

# ─────────────────────────────────────────────────────────────────
# Income-transfer counterfactual, LINEAR COST variant.
#
# Policy: y*(regime z) = y*(kP, kω, kε), applied to every child in that
# regime regardless of θ_1. Because income is a function of the regime
# (via the income model), this is equivalently written as y*(y_baseline).
#
# With linear cost, the planner's problem decouples across regimes. For
# each z, she solves (per outcome Q, per price λ):
#
#     max_{y*(z)}  Σ_{θ_1} π_0(θ_1|kP,kω) · W(z, θ_1, y*(z))
#                 − λ · (y*(z) − y_baseline(z))
#
# where W(z, θ_1, y*) = Σ_{θ_T} Q(θ_T) · F_θ_T(z, θ_1, ε_eff(y*)).
# ε_eff = (log y* − β₀ − β_P kP − β_ω kω) / δ.
# ─────────────────────────────────────────────────────────────────

const FIT_PATH = get(ENV, "FIT_OVERRIDE", joinpath(@__DIR__, "..", "output", "estimation_v3_eps_reg.jld2"))
const TRANSITION_GROUPS = [[1,2,3], [4,5,6]]
const OUT_DIR = joinpath(@__DIR__, "..", "output", "counterfactuals")
isdir(OUT_DIR) || mkpath(OUT_DIR)

@info "Loading fit…"
fit       = load(FIT_PATH)
params    = fit["params_v3_eps"]
gamma_agg = fit["gamma_agg"]
dims      = fit["data_dims"]
r_θ, r_I, r_P, r_ω, r_Y = dims.r_θ, dims.r_I, dims.r_P, dims.r_ω, dims.r_Y
T = dims.T

π_Pω = Array{Float64}(params.π_Pω)
π_ε  = Array{Float64}(params.π_ε)
π0   = Array{Float64}(params.π0)                      # (r_θ, r_P, r_ω)
π_ω_marg = vec(sum(π_Pω, dims=1))

ccp    = params.ccp_spec
income = params.income_model
β0, βP, βω, δ = income.beta0, income.betaP, income.betaω, income.delta
β1, β2, β3, β4, β5 = ccp.beta1, ccp.beta2, ccp.beta3, ccp.beta4, ccp.beta5
c_ccp  = Vector{Float64}(ccp.c)
@assert isapprox(β2, 0.0; atol=1e-12)

# Regime-specific transitions (one per group)
function build_Pi_z(spec::RobitNNTransitionV3, r_θ::Int, r_I::Int,
                    kP::Real, kω::Real)
    v = pack_robit_nn_v3(spec); K, ν = spec.K, spec.ν
    _build_robit_nn_v3_pi(v, r_θ, r_I, K, ν, float(kP), float(kω))
end
Π_grp_z = [Dict{Tuple{Int,Int},Array{Float64,3}}() for _ in 1:2]
for g in 1:2, kP in 1:r_P, kω in 1:r_ω
    Π_grp_z[g][(kP, kω)] = build_Pi_z(params.trans_specs[g], r_θ, r_I, kP, kω)
end
group_of(t) = t <= 3 ? 1 : 2

# Earnings / Q (same as prior scripts)
function earnings_bin_means()
    ci  = CSV.read(joinpath(@__DIR__, "..", "NLSY", "ChildIncome", "ChildIncome_Clean.csv"), DataFrame)
    cpi = CSV.read(joinpath(@__DIR__, "..", "output", "cpi_u_annual.csv"), DataFrame)
    cpi_tbl = Dict{Int,Float64}(Int(cpi[i, :year]) => Float64(cpi[i, :cpi]) for i in 1:nrow(cpi))
    base = cpi_tbl[2010]
    ci.WageInc_num = [ismissing(x) ? missing : tryparse(Float64, string(x)) for x in ci.WageInc]
    ci.YEAR_int    = [tryparse(Int, string(y)) for y in ci.YEAR]
    ci.WageInc_real = [
        (w === missing || w === nothing || yr === missing || yr === nothing) ? missing :
            w / (get(cpi_tbl, yr, base) / base)
        for (w, yr) in zip(ci.WageInc_num, ci.YEAR_int)]
    ci_adult = ci[(ci.Age .>= 25) .& (ci.Age .<= 30), :]
    earn = combine(groupby(ci_adult, :KID)) do sub
        w = filter(!isnothing, collect(skipmissing(sub.WageInc_real)))
        isempty(w) ? (; AvgEarnings=missing, N_obs=0) : (; AvgEarnings=mean(w), N_obs=length(w))
    end
    earn = earn[earn.N_obs .>= 2, :]
    v = collect(skipmissing(earn.AvgEarnings))
    cuts = quantile(v, [0.2, 0.4, 0.6, 0.8])
    bin_of(x) = x <= cuts[1] ? 1 : x <= cuts[2] ? 2 : x <= cuts[3] ? 3 : x <= cuts[4] ? 4 : 5
    bins = bin_of.(v)
    [mean(v[bins .== q]) for q in 1:5]
end
earn_means = earnings_bin_means()

function Q_from_adult(A, ys, π_ω_vec)
    κ, r_θ, r_ω = size(A)
    Q = zeros(r_θ)
    for j in 1:r_θ, v in 1:r_ω, k in 1:κ
        Q[j] += π_ω_vec[v] * ys[k] * A[k, j, v]
    end
    Q
end
adult_mats  = Vector{Array{Float64,3}}(params.adult_matrices)
adult_names = ["HS", "College", "EarnQ", "AlwaysEmp", "logEarn"]
values      = [[0.0, 1.0], [0.0, 1.0], earn_means, [0.0, 1.0]]
Qs          = [Q_from_adult(adult_mats[m], values[m], π_ω_marg) for m in 1:4]
push!(Qs, log.(Qs[3]))

# ─────────────────────────────────────────────────────────────────
# Forward sim (same as before; β₂ = 0 allows propagating F_θ directly)
# ─────────────────────────────────────────────────────────────────

function simulate_FθT(kP::Int, kω::Int, θ_1::Int, ε_Y_value::Tε,
                      Π_per_t::Vector{Array{Float64,3}}) where Tε
    PT = promote_type(Tε, Float64)
    CCP = zeros(PT, r_I, r_θ)
    regime_part = β3 * kP + β4 * kω + β5 * ε_Y_value
    for kθ in 1:r_θ
        η = β1 * kθ + regime_part
        CCP[:, kθ] = _ordered_probit_pmf(c_ccp, η, r_I)
    end
    F_θ = zeros(PT, r_θ); F_θ[θ_1] = one(PT)
    for t in 1:T-1
        Π_t = Π_per_t[t]
        F_new = zeros(PT, r_θ)
        for kθ_prev in 1:r_θ
            F_θ[kθ_prev] < 1e-15 && continue
            for kI in 1:r_I
                w = F_θ[kθ_prev] * CCP[kI, kθ_prev]
                for kθ_next in 1:r_θ
                    F_new[kθ_next] += w * Π_t[kθ_next, kθ_prev, kI]
                end
            end
        end
        F_θ = F_new
    end
    F_θ
end

# For a regime, π_0-weighted welfare contribution at a given ε_eff
function welfare_regime(kP::Int, kω::Int, ε_eff::Tε, Q::AbstractVector) where Tε
    Π_t_z = [Π_grp_z[group_of(t)][(kP, kω)] for t in 1:T-1]
    π_0_cond = π0[:, kP, kω] ./ sum(π0[:, kP, kω])
    W = zero(promote_type(Tε, Float64))
    for θ_1 in 1:r_θ
        F = simulate_FθT(kP, kω, θ_1, ε_eff, Π_t_z)
        W += π_0_cond[θ_1] * sum(F .* Q)
    end
    W
end

# ─────────────────────────────────────────────────────────────────
# Per-regime optimization (linear cost, 1-D)
# ─────────────────────────────────────────────────────────────────

function regime_baseline_income(kP, kω, kε)
    exp(β0 + βP * kP + βω * kω + δ * kε)
end

function optimize_regime_transfer(kP::Int, kω::Int, kε::Int,
                                   Q::AbstractVector, λ::Real;
                                   log_bound::Real=3.0)
    y_base = regime_baseline_income(kP, kω, kε)
    log_y_base = log(y_base)
    function neg_obj(log_y)
        ε_eff = (log_y - β0 - βP * kP - βω * kω) / δ
        W = welfare_regime(kP, kω, ε_eff, Q)
        -(W - λ * (exp(log_y) - y_base))
    end
    # One-sided: transfers only (no taxing), so y* ≥ y_base.
    res = optimize(neg_obj, log_y_base, log_y_base + log_bound,
                    Brent(); rel_tol=1e-9, abs_tol=1e-11)
    log_y_opt = Optim.minimizer(res)
    ε_opt = (log_y_opt - β0 - βP*kP - βω*kω) / δ
    (; y_opt=exp(log_y_opt), log_y_opt, log_y_base, y_base,
       W=welfare_regime(kP, kω, ε_opt, Q),
       W_base=welfare_regime(kP, kω, float(kε), Q),
       converged=Optim.converged(res))
end

# ─────────────────────────────────────────────────────────────────
# Sweep regimes for each outcome and each λ
# ─────────────────────────────────────────────────────────────────

# Per-outcome λ grids. For EarnQ both sides in $, so dimensionless.
# For probability outcomes, λ is in 1/$, so scale by ~1/(baseline Q) to get
# a comparable "% welfare per dollar" range.
function lambda_grid(outcome_idx)
    Q = Qs[outcome_idx]
    Q_scale = maximum(Q) - minimum(Q)
    y_scale = 50_000.0             # typical baseline income
    λ_nat   = Q_scale / y_scale    # marginal-welfare-per-$ at rough scale
    # Span a wide multiplicative range around this natural scale.
    [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0] .* λ_nat
end

struct RegimeResult
    kP::Int; kω::Int; kε::Int
    y_base::Float64
    y_opt::Float64
    W::Float64
    W_base::Float64
    weight::Float64        # π_Pω · π_ε
end

sweep = Dict{String,Dict{Float64,Vector{RegimeResult}}}()
for m in eachindex(Qs)
    name = adult_names[m]
    sweep[name] = Dict{Float64,Vector{RegimeResult}}()
    for λ in lambda_grid(m)
        rs = RegimeResult[]
        for kP in 1:r_P, kω in 1:r_ω, kε in 1:r_Y
            o = optimize_regime_transfer(kP, kω, kε, Qs[m], λ)
            push!(rs, RegimeResult(kP, kω, kε, o.y_base, o.y_opt,
                                    o.W, o.W_base,
                                    π_Pω[kP, kω] * π_ε[kε]))
        end
        sweep[name][λ] = rs
    end
    @info "  done $name"
end

# ─────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────

println("\n" * "="^86)
println("Linear-cost income-transfer counterfactual — policy y*(income), per-regime")
println("="^86)

for m in eachindex(Qs)
    name = adult_names[m]
    println("\n[$(name)]")
    @printf "%10s %15s %15s %15s %15s\n" "λ" "W_base" "W_opt" "Δ%" "total_cost(\$)"
    for λ in lambda_grid(m)
        rs = sweep[name][λ]
        Wb = sum(r.weight * r.W_base for r in rs)
        Wo = sum(r.weight * r.W      for r in rs)
        C  = sum(r.weight * (r.y_opt - r.y_base) for r in rs)
        @printf "%10.4g %15.5g %15.5g %+14.3f%% %+15.1f\n" λ Wb Wo (100*(Wo-Wb)/abs(Wb)) C
    end
end

# ─────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────

out_colors = Dict("HS"=>:orange, "College"=>:red, "EarnQ"=>:purple,
                   "AlwaysEmp"=>:brown, "logEarn"=>:teal)

# Figure A: policy function y*(y_baseline) for each outcome at a middle λ value.
# Pick the λ index that yields a nontrivial interior optimum.
policy_panels = []
for m in eachindex(Qs)
    name = adult_names[m]
    λgrid = lambda_grid(m)
    λ_mid = λgrid[2]                 # 0.1 × natural scale — large but interior transfers
    rs = sweep[name][λ_mid]
    xs = [r.y_base for r in rs]
    ts = [r.y_opt - r.y_base for r in rs]   # transfer amount in $
    order = sortperm(xs)
    xs, ts = xs[order], ts[order]
    p = scatter(xs, ts; xlabel="baseline income \$", ylabel="transfer y* − y_base  \$",
                 title="Q = $name   (λ = $(round(λ_mid; sigdigits=3)))", titlefontsize=9,
                 color=out_colors[name], label=false, markersize=5,
                 tickfontsize=7, xlabelfontsize=8, ylabelfontsize=8,
                 framestyle=:box, xscale=:log10)
    hline!(p, [0]; label=false, ls=:dot, color=:gray60, lw=1)
    push!(policy_panels, p)
end
fig_policy = plot(policy_panels...; layout=(2, 3), size=(1500, 820),
                  plot_title="Optimal y*(y_baseline) — per-regime, linear cost, interior λ",
                  plot_titlefontsize=11, margin=3Plots.mm)
savefig(fig_policy, joinpath(OUT_DIR, "cf_income_linear_policy.png"))
println("\nsaved cf_income_linear_policy.png")

# Figure B: cost–welfare frontier — welfare gain vs total cost across λ
frontier_panels = []
for m in eachindex(Qs)
    name = adult_names[m]
    λgrid = lambda_grid(m)
    pts_C = Float64[]; pts_ΔW = Float64[]
    for λ in λgrid
        rs = sweep[name][λ]
        Wb = sum(r.weight * r.W_base for r in rs)
        Wo = sum(r.weight * r.W      for r in rs)
        C  = sum(r.weight * (r.y_opt - r.y_base) for r in rs)
        push!(pts_C, C); push!(pts_ΔW, (Wo - Wb)/abs(Wb)*100)
    end
    order = sortperm(pts_C)
    p = plot(pts_C[order], pts_ΔW[order]; xlabel="total cost (\$ per child)",
             ylabel="ΔW / W_base (%)", title="Q = $name", titlefontsize=10,
             color=out_colors[name], lw=2, marker=:circle, ms=5, label=false,
             framestyle=:box, tickfontsize=7, xlabelfontsize=8, ylabelfontsize=8)
    for (i, λ) in enumerate(λgrid[order])
        annotate!(p, [(pts_C[order][i], pts_ΔW[order][i] + 0.03,
                        text(@sprintf("λ=%.3g", λ), 6, :black))])
    end
    push!(frontier_panels, p)
end
fig_frontier = plot(frontier_panels...; layout=(2, 3), size=(1500, 820),
                    plot_title="Welfare–cost frontier as λ varies",
                    plot_titlefontsize=11, margin=3Plots.mm)
savefig(fig_frontier, joinpath(OUT_DIR, "cf_income_linear_frontier.png"))
println("saved cf_income_linear_frontier.png")

@info "Done."
