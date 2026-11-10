using LinearAlgebra, Statistics, Printf, JLD2, CSV, DataFrames, Plots, Optim, ForwardDiff, LaTeXStrings

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3_eps.jl"))
include(joinpath(@__DIR__, "..", "src", "counterfactuals.jl"))
include(joinpath(@__DIR__, "..", "src", "slides_palette.jl"))

# Paper-aware font sizes (slides values are the default). PAPER_FIGS from slides_palette.jl.
const FS_T  = PAPER_FIGS ? 18 : 10
const FS_G  = PAPER_FIGS ? 17 :  9
const FS_TK = PAPER_FIGS ? 15 :  8
const FS_L  = PAPER_FIGS ? 14 :  7
const FS_A  = PAPER_FIGS ? 12 :  7
const FS_PT = PAPER_FIGS ? 20 : 12

# ─────────────────────────────────────────────────────────────────
# Income-transfer counterfactual (resource-constrained version).
#
# Policy: y*(θ_1) is a 4-vector of *level* incomes, one per initial skill.
# Every child with θ_1 = k gets the same y*(k), regardless of her regime.
#
# Interpretation (per user): income is the SOLE channel for ε_Y. The policy
# replaces each child's ε_Y with ε_Y^eff(θ_1, kP, kω) such that her (permanent
# log) income under the new policy matches the planner's target. Specifically
#   ε_Y^eff = (log y*(θ_1) − β₀ − β_P·kP − β_ω·kω) / δ.
# The modified ε_Y^eff enters the CCP index via β₅·ε_Y^eff. Transitions and
# all other CCP channels are unchanged.
#
# Budget: total population level income is preserved:
#   Σ_{θ_1} π_0^marg(θ_1) · y*(θ_1) = E[exp(μ_n)] (baseline permanent level).
# Level-space budget (not log) per user.
#
# Optimization: smooth nonlinear on (r_θ − 1)-dim simplex via softmax. We
# parameterize shares s(θ_1) of total budget, so feasibility is automatic.
# ForwardDiff + L-BFGS.
# ─────────────────────────────────────────────────────────────────

const REG_RUN  = get(ENV, "REG_RUN", "0") == "1"
const FIT_PATH = get(ENV, "FIT_OVERRIDE", joinpath(@__DIR__, "..", "output", "estimation_v3_eps_reg.jld2"))
const TRANSITION_GROUPS = [[1,2,3], [4,5,6]]
const OUT_DIR = joinpath(@__DIR__, "..", "output", REG_RUN ? "counterfactuals_reg" : "counterfactuals")
isdir(OUT_DIR) || mkpath(OUT_DIR)

@info "Loading fit…"
fit       = load(FIT_PATH)
params    = fit["params_v3_eps"]
gamma_agg = fit["gamma_agg"]
dims      = fit["data_dims"]
r_θ, r_I, r_P, r_ω, r_Y = dims.r_θ, dims.r_I, dims.r_P, dims.r_ω, dims.r_Y
N, T = dims.N, dims.T

π_Pω = Array{Float64}(params.π_Pω)
π_ε  = Array{Float64}(params.π_ε)
π0   = Array{Float64}(params.π0)                      # (r_θ, r_P, r_ω)

ccp        = params.ccp_spec
income     = params.income_model
β0, βP, βω, δ = income.beta0, income.betaP, income.betaω, income.delta
β1, β2, β3, β4, β5 = ccp.beta1, ccp.beta2, ccp.beta3, ccp.beta4, ccp.beta5
c_ccp      = Vector{Float64}(ccp.c)
@assert isapprox(β2, 0.0; atol=1e-12) "β₂ = 0 assumed (V3+ε_Y). Need β₂ = 0 for the\nfast-simplified forward simulation below."

# Build regime-specific transitions per transition group (not marginalized over z).
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

# ─────────────────────────────────────────────────────────────────
# Earnings bin means + Q functions (same as before)
# ─────────────────────────────────────────────────────────────────

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
π_ω_marg = vec(sum(π_Pω, dims=1))
adult_mats  = Vector{Array{Float64,3}}(params.adult_matrices)
adult_names = ["HS", "College", "EarnQ", "AlwaysEmp", "logEarn"]
values      = [[0.0, 1.0], [0.0, 1.0], earn_means, [0.0, 1.0]]
Qs          = [Q_from_adult(adult_mats[m], values[m], π_ω_marg) for m in 1:4]
push!(Qs, log.(Qs[3]))

# ─────────────────────────────────────────────────────────────────
# Forward simulation given (kP, kω, θ_1, ε_Y_value). β₂ = 0 means
# CCP is independent of I_{t-1}, so we can propagate F_θ_t directly.
#
#   F_θ_{t+1}[θ'] = Σ_{θ, I} F_θ_t[θ] · CCP[I | θ, z, ε_Y] · Π_t[θ' | θ, I, z]
# ─────────────────────────────────────────────────────────────────

function simulate_FθT(kP::Int, kω::Int, θ_1::Int, ε_Y_value::Tε,
                      Π_per_t::Vector{Array{Float64,3}}) where Tε
    PT = promote_type(Tε, Float64)
    # CCP[I | θ]: r_I × r_θ matrix (same at every t because β₂=0, β₅·ε_Y scalar)
    CCP = zeros(PT, r_I, r_θ)
    regime_part = β3 * kP + β4 * kω + β5 * ε_Y_value
    for kθ in 1:r_θ
        η = β1 * kθ + regime_part
        CCP[:, kθ] = _ordered_probit_pmf(c_ccp, η, r_I)
    end
    F_θ = zeros(PT, r_θ); F_θ[θ_1] = one(PT)
    for t in 1:T-1
        Π_t = Π_per_t[t]       # (r_θ', r_θ, r_I)
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

# Per-period transitions (same group shared within)
Π_per_t = [Π_grp_z[group_of(t)] for t in 1:T-1]   # Dict per t — same ref

# ─────────────────────────────────────────────────────────────────
# Welfare evaluators
# ─────────────────────────────────────────────────────────────────

# Baseline: each child at her original kε, weighted by π_ε.
function welfare_baseline(Q::AbstractVector)
    W = 0.0; FθT_sum = zeros(r_θ)
    for kP in 1:r_P, kω in 1:r_ω
        Π_t_z = [Π_grp_z[group_of(t)][(kP, kω)] for t in 1:T-1]
        for kε in 1:r_Y
            for θ1 in 1:r_θ
                p = π_Pω[kP, kω] * π0[θ1, kP, kω] / sum(π0[:, kP, kω]) *
                    π_ε[kε]
                F = simulate_FθT(kP, kω, θ1, float(kε), Π_t_z)
                W       += p * sum(F .* Q)
                FθT_sum .+= p .* F
            end
        end
    end
    (W, FθT_sum)
end

# Counterfactual: y_log_policy is a r_θ vector of log incomes.
# Every (kP, kω, θ_1) child gets ε_Y^eff = (y_log(θ_1) − β0 − βP·kP − βω·kω) / δ.
function welfare_policy(y_log_policy::AbstractVector{Tp}, Q::AbstractVector) where Tp
    PT = promote_type(Tp, Float64)
    W  = zero(PT); FθT_sum = zeros(PT, r_θ)
    for kP in 1:r_P, kω in 1:r_ω
        Π_t_z = [Π_grp_z[group_of(t)][(kP, kω)] for t in 1:T-1]
        for θ1 in 1:r_θ
            ε_eff = (y_log_policy[θ1] - β0 - βP * kP - βω * kω) / δ
            p = π_Pω[kP, kω] * π0[θ1, kP, kω] / sum(π0[:, kP, kω])
            F = simulate_FθT(kP, kω, θ1, ε_eff, Π_t_z)
            W       += p * sum(F .* Q)
            FθT_sum .+= p .* F
        end
    end
    (W, FθT_sum)
end

# ─────────────────────────────────────────────────────────────────
# Budget: total *level* income preserved.
#   budget_rhs = Σ_{kP,kω,kε} π_Pω(kP,kω) π_ε(kε) · exp(β0 + βP·kP + βω·kω + δ·kε)
# Policy constraint: Σ_{θ1} π_0^marg(θ_1) · exp(y_log(θ_1)) = budget_rhs.
# Parameterize by shares s(θ_1) > 0, Σ_θ1 π_0^marg(θ_1) · s(θ_1) = 1, via softmax.
# ─────────────────────────────────────────────────────────────────

π_0_marg = [sum(π_Pω[kP, kω] * π0[θ1, kP, kω] / sum(π0[:, kP, kω])
                  for kP in 1:r_P, kω in 1:r_ω) for θ1 in 1:r_θ]
@printf "π_0^marg(θ_1) = %s\n" join([@sprintf("%.3f", x) for x in π_0_marg], "  ")

budget_rhs = sum(π_Pω[kP, kω] * π_ε[kε] * exp(β0 + βP*kP + βω*kω + δ*kε)
                  for kP in 1:r_P, kω in 1:r_ω, kε in 1:r_Y)
@printf "Budget (avg permanent level income per child): \$%.0f\n" budget_rhs

# Softmax parameterization of shares on the simplex
# share[θ1] = exp(z[θ1]) / Σ_j π_0_marg[j] · exp(z[j])
# → Σ_{θ1} π_0_marg[θ1] · share[θ1] = 1
# level[θ1] = budget_rhs · share[θ1]
function shares_from_z(z::AbstractVector{Tz}) where Tz
    PT = promote_type(Tz, Float64)
    ez = exp.(z)
    denom = sum(π_0_marg[j] * ez[j] for j in 1:r_θ)
    PT.(ez) ./ denom                           # shares, NOT normalized to 1
end
function policy_from_z(z::AbstractVector)
    shares = shares_from_z(z)
    [log(budget_rhs * shares[j]) for j in 1:r_θ]
end
function check_budget(y_log_policy::AbstractVector)
    sum(π_0_marg[j] * exp(y_log_policy[j]) for j in 1:r_θ)
end

# Baseline policy (z = 0 → uniform shares means level[j] = budget_rhs)
y_log_baseline_uniform = fill(log(budget_rhs), r_θ)
@printf "Budget check on uniform-level policy: %.5g vs target %.5g\n" check_budget(y_log_baseline_uniform) budget_rhs

# ─────────────────────────────────────────────────────────────────
# Optimize per outcome
# ─────────────────────────────────────────────────────────────────

results = Dict{String,Any}()
W_baseline_vec = Vector{Float64}(undef, length(Qs))
FT_baseline_vec = Vector{Vector{Float64}}(undef, length(Qs))
for m in eachindex(Qs)
    Wb, Fb = welfare_baseline(Qs[m])
    W_baseline_vec[m]  = Wb
    FT_baseline_vec[m] = Fb
end

for m in eachindex(Qs)
    name = adult_names[m]
    Q = Qs[m]
    obj_z(z) = -welfare_policy(policy_from_z(z), Q)[1]
    z0 = zeros(r_θ)
    @info "Optimizing income transfers for $name …"
    opt = optimize(obj_z, z0, LBFGS(),
                   Optim.Options(x_abstol=1e-8, f_abstol=1e-10, iterations=500);
                   autodiff=:forward)
    z_opt  = Optim.minimizer(opt)
    y_opt  = policy_from_z(z_opt)
    W_opt, F_opt = welfare_policy(y_opt, Q)
    results[name] = (; z_opt, y_log=y_opt,
                       level = exp.(y_opt),
                       W_opt, F_opt,
                       W_base = W_baseline_vec[m], F_base = FT_baseline_vec[m],
                       converged = Optim.converged(opt),
                       iters = Optim.iterations(opt))
end

# ─────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────

println("\n" * "="^86)
println("Income-transfer counterfactual — policy y*(θ_1), level-income budget preserved.")
println("="^86)
@printf "Baseline per-child permanent level income: \$%.0f\n" budget_rhs

for m in eachindex(Qs)
    name = adult_names[m]; r = results[name]
    Δpct = 100 * (r.W_opt - r.W_base) / abs(r.W_base)
    println("─"^86)
    @printf "[%s]\n" name
    @printf "  W_baseline = %12.5g\n" r.W_base
    @printf "  W_policy   = %12.5g    (+%.3f%%)\n" r.W_opt Δpct
    @printf "  Optimal y*(θ_1) level:  %s\n" join([@sprintf("\$%.0f", x) for x in r.level], "  ")
    @printf "  (log scale):            %s\n" join([@sprintf("%.3f", x) for x in r.y_log], "  ")
    @printf "  Budget check:           \$%.0f vs target \$%.0f\n" check_budget(r.y_log) budget_rhs
    @printf "  Converged: %s  (iters: %d)\n" r.converged r.iters
end

# ── Persist the income-reallocation welfare table for the paper ─────────────
# Source of record for the "income vs investment reallocation" comparison.
# The investment column / ratio is joined against welfare_investment.csv.
let tbl_dir = joinpath(@__DIR__, "..", "output", REG_RUN ? "paper_tables_reg" : "paper_tables")
    isdir(tbl_dir) || mkpath(tbl_dir)
    df = DataFrame(outcome = String[], W_base = Float64[], W_opt = Float64[], pct_gain = Float64[])
    for m in eachindex(Qs)
        name = adult_names[m]; r = results[name]
        push!(df, (name, r.W_base, r.W_opt, 100*(r.W_opt - r.W_base)/abs(r.W_base)))
    end
    CSV.write(joinpath(tbl_dir, "welfare_income.csv"), df)
    println("  saved output/paper_tables/welfare_income.csv")
end

# ─────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────

out_colors = OUTCOME_COLORS

# Figure A: Optimal income transfer schedule y*(θ_1) per outcome
p_trans = plot(xlabel=L"\theta_1", ylabel=L"\mathrm{Optimal\ income}\ y^*(\theta_1)",
               title="Optimal income transfers by initial skill",
               titlefontsize=11, xticks=1:r_θ,
               legend=:topright, legendfontsize=8,
               size=(820, 560), framestyle=:box)
hline!(p_trans, [budget_rhs]; label="baseline per-child avg (\$$(round(Int,budget_rhs)))",
       ls=:dash, color=PAL_GREY, lw=1.5)
for m in eachindex(Qs)
    name = adult_names[m]; r = results[name]
    plot!(p_trans, 1:r_θ, r.level; label=name, lw=2,
          color=out_colors[name], marker=:circle, ms=5)
end
save_slide_fig(p_trans, "cf_income_policy"; png_dir=OUT_DIR)
println("\nsaved cf_income_policy.{png,pdf}")

# Figure B: Terminal F_θ_T under baseline vs optimal income transfer, per outcome
term_panels = []
for m in eachindex(Qs)
    name = adult_names[m]; r = results[name]
    xs = 1:r_θ; w = 0.36
    p = bar(xs .- w/2, r.F_base; bar_width=w, color=PAL_TEAL, label="baseline",
             xlabel=L"\theta_T", ylabel=(m in (1,4) ? L"P(\theta_T = k)" : ""),
             title="Q = $name", titlefontsize=FS_T, ylims=(0, 0.65),
             xticks=(xs, string.(xs)), tickfontsize=FS_TK,
             xlabelfontsize=FS_G, ylabelfontsize=FS_G,
             legend=(m==1 ? :topleft : false), legendfontsize=FS_L, framestyle=:box)
    bar!(p, xs .+ w/2, r.F_opt; bar_width=w, color=PAL_RUST, label="income transfer")
    # Per-bar numeric annotations removed — they overlapped and cluttered the panels.
    push!(term_panels, p)
end
fig_term = plot(term_panels...; layout=(2, 3), size=(PAPER_FIGS ? (1180, 660) : (1500, 760)),
                plot_title=L"Terminal\ F_{\theta_T}\ \mathrm{(baseline\ vs\ income\ transfer)}",
                plot_titlefontsize=FS_PT, margin=3Plots.mm)
save_slide_fig(fig_term, "cf_income_terminal_Ftheta"; png_dir=OUT_DIR)
println("saved cf_income_terminal_Ftheta.{png,pdf}")

@info "Done."
