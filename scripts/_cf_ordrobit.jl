# Paper counterfactuals (first_draft.tex §6.5 + App. "myopic static planner") re-run on the grouped
# per-skill ordered-robit fit (exploratory, 2026-11-20). Same problems, same anchors, same code paths
# (src/counterfactuals.jl), with the technology built from OrderedRobitPerThetaV3 instead of the robit NN.
#
#  A. Investment reallocation, fixed marginals F_{I_t}:
#     dynamic  = stacked 6-period LP (solve_sequential_ot), F_θ1 empirical;
#     myopic   = one-shot OT per t on the EMPIRICAL (F_θt, F_It), reward E[Q(θ_{t+1})|θ,I] (paper App.);
#     myopic chained = the myopic rule applied along the propagated F_θt (feasible, so W comparable to dynamic).
#  B. Income reallocation y(θ_1), level budget Σ p(θ1) y(θ1) = E[Y] (paper §cf-income), CCP fixed:
#     dynamic  = y(θ_1) held for all six periods, value E[Q(θ_T)];
#     static counterpart (NOT in the paper) = at each t separately, y_t(θ_t) with budget Σ p_t(θ) y_t(θ) = E[Y],
#                value E[Q(θ_{t+1})] one step ahead, all other periods at baseline.
# ENV FIT (default q5a seed 1), TAG (q5a). Writes output/cf_ordrobit_{TAG}.txt (stdout) and figures/cf_{TAG}_*.png
using LinearAlgebra, Statistics, Printf, JLD2, CSV, DataFrames, Plots, Optim, ForwardDiff
const REPO = joinpath(@__DIR__, "..")
for f in ["parametric_models.jl", "parametric_models_v3.jl", "parametric_models_v3_eps.jl", "estimation_multiperiod.jl",
          "estimation_multiperiod_v3.jl", "parametric_models_v3_robit.jl", "counterfactuals.jl"]
    include(joinpath(REPO, "src", f))
end
const FIT = get(ENV, "FIT", "estimation_v3_eps_ordrobit_pertheta_rI4_noPIAT3_grp_q5a_tidy_seed1.jld2")
const TAG = get(ENV, "TAG", "q5a")
const TG = [[1, 2, 3], [4, 5, 6]]
F = load(joinpath(REPO, "output", FIT)); pv = F["params_v3_eps"]; γagg = F["gamma_agg"]
(; N, T, r_θ, r_I, r_P, r_ω, r_Y) = F["data_dims"]
π_Pω = Array{Float64}(pv.π_Pω); π_ε = Array{Float64}(pv.π_ε); π0 = Array{Float64}(pv.π0)
π_ω = vec(sum(π_Pω; dims = 1))
group_of(t) = t <= 3 ? 1 : 2
Π̄_ts = marginalize_Pi_v3(pv.trans_specs, TG, π_Pω, r_θ, r_I, r_P, r_ω)
Π_z = Dict((g, kP, kω) => _build_ordrobit_pi(OrderedRobitPerThetaV3, pack_ordrobit(pv.trans_specs[g]), r_θ, r_I,
                                              pv.trans_specs[g].ν, float(kP), float(kω))
           for g in 1:2, kP in 1:r_P, kω in 1:r_ω)
println("fit $FIT   LL $(round(F["best_ll"]; digits = 1))")

# ── anchors Q(θ_T) = E[outcome | θ_T] (same construction as the paper scripts) ──
function earnings_bin_means()
    ci = CSV.read(joinpath(REPO, "NLSY", "ChildIncome", "ChildIncome_Clean.csv"), DataFrame)
    cpi = CSV.read(joinpath(REPO, "output", "cpi_u_annual.csv"), DataFrame)
    tbl = Dict{Int,Float64}(Int(cpi[i, :year]) => Float64(cpi[i, :cpi]) for i in 1:nrow(cpi)); base = tbl[2010]
    ci.W = [ismissing(x) ? missing : tryparse(Float64, string(x)) for x in ci.WageInc]
    ci.Y = [tryparse(Int, string(y)) for y in ci.YEAR]
    ci.R = [(w === missing || w === nothing || y === nothing) ? missing : w / (get(tbl, y, base) / base) for (w, y) in zip(ci.W, ci.Y)]
    a = ci[(ci.Age .>= 25) .& (ci.Age .<= 30), :]
    e = combine(groupby(a, :KID)) do s
        w = filter(!isnothing, collect(skipmissing(s.R))); isempty(w) ? (; m = missing, n = 0) : (; m = mean(w), n = length(w))
    end
    v = collect(skipmissing(e[e.n .>= 2, :m])); cuts = quantile(v, [0.2, 0.4, 0.6, 0.8])
    b = [x <= cuts[1] ? 1 : x <= cuts[2] ? 2 : x <= cuts[3] ? 3 : x <= cuts[4] ? 4 : 5 for x in v]
    [mean(v[b .== q]) for q in 1:5]
end
Q_from_adult(A, ys) = [sum(π_ω[v] * ys[k] * A[k, j, v] for v in 1:size(A, 3), k in 1:size(A, 1)) for j in 1:size(A, 2)]
AM = Vector{Array{Float64,3}}(pv.adult_matrices)
NAMES = ["HS", "College", "EarnQ", "AlwaysEmp", "logEarn"]
Qs = [Q_from_adult(AM[m], ys) for (m, ys) in enumerate([[0.0, 1.0], [0.0, 1.0], earnings_bin_means(), [0.0, 1.0]])]
push!(Qs, log.(Qs[3]))
println("\nAnchors Q(θ_T):"); for m in eachindex(Qs); @printf("  %-9s %s\n", NAMES[m], join([@sprintf("%10.4g", q) for q in Qs[m]])); end

# ══════════════════ A. investment reallocation ══════════════════
γd = [dropdims(mean(@view(γagg[t, :, :, :]); dims = 3); dims = 3) for t in 1:T]
pθ = [vec(sum(γd[t]; dims = 2)) for t in 1:T]; pI = [vec(sum(γd[t]; dims = 1)) for t in 1:T-1]
EIθ(γ, p) = [p[j] < 1e-12 ? NaN : sum(i * γ[j, i] for i in 1:r_I) / p[j] for j in 1:r_θ]
push(γ, Π) = [sum(Π[k, i, j] * γ[i, j] for i in 1:r_θ, j in 1:r_I) for k in 1:r_θ]
println("\n", "="^100, "\nA. INVESTMENT REALLOCATION (F_It fixed at empirical; F_θ1 empirical)\n", "="^100)
@printf("  empirical F_θ1 = %s   F_θT = %s\n", round.(pθ[1]; digits = 3), round.(pθ[T]; digits = 3))
for t in 1:T-1; @printf("  F_I t=%d %s\n", t, round.(pI[t]; digits = 3)); end
dyn = []; myo = []; myoc = []
for (m, Q) in enumerate(Qs)
    v, γs, Fs = solve_sequential_ot(pθ[1], pI, Π̄_ts, Q); push!(dyn, (; v, γs, Fs))
    # myopic on empirical marginals (paper's static planner, allocation only)
    push!(myo, [solve_ot(pθ[t], pI[t], expected_Q(Q, Π̄_ts[t]); maximize = true)[2] for t in 1:T-1])
    # myopic rule chained along its own propagated F_θ (feasible ⇒ comparable welfare)
    Fc = copy(pθ[1]); γc = Matrix{Float64}[]
    for t in 1:T-1
        _, γ = solve_ot(Fc, pI[t], expected_Q(Q, Π̄_ts[t]); maximize = true); push!(γc, γ); Fc = push(γ, Π̄_ts[t])
    end
    # baseline that is feasible in the same sense: the data's coupling RULE P(I|θ) applied along the chain
    # (keeps E[I|θ] as in data; I-marginal drifts slightly) and the comonotone rule chained
    Fb = copy(pθ[1]); Fm = copy(pθ[1])
    for t in 1:T-1
        Pc = γd[t] ./ pθ[t]; Fb = push(Fb .* Pc, Π̄_ts[t])
        Fm = push(comonotone_coupling(Fm, pI[t]), Π̄_ts[t])
    end
    push!(myoc, (; W = sum(Fc .* Q), F = Fc, γ = γc, Wb = sum(Fb .* Q), Fb, Wc = sum(Fm .* Q), Fm))
end
println("\n  Welfare E[Q(θ_T)].  data-rule = data P(I|θ) chained;  comonotone = comonotone coupling chained;")
println("  myopic = one-shot OT per t chained;  dynamic = stacked LP.  % vs data-rule.")
@printf("  %-9s %12s %12s %12s %12s   %8s %8s %8s\n", "anchor", "data-rule", "comonotone", "myopic", "dynamic", "%como", "%myopic", "%dyn")
for m in eachindex(Qs)
    b = myoc[m].Wb; f(x) = 100 * (x - b) / abs(b)
    @printf("  %-9s %12.5g %12.5g %12.5g %12.5g   %+8.2f %+8.2f %+8.2f\n", NAMES[m], b, myoc[m].Wc, myoc[m].W, dyn[m].v, f(myoc[m].Wc), f(myoc[m].W), f(dyn[m].v))
end
println("\n  Terminal F_θT:")
@printf("  %-9s %-30s %-30s %-30s\n", "anchor", "data-rule", "myopic chained", "dynamic")
for m in eachindex(Qs)
    @printf("  %-9s %-30s %-30s %-30s\n", NAMES[m], string(round.(myoc[m].Fb; digits = 3)), string(round.(myoc[m].F; digits = 3)), string(round.(dyn[m].Fs[T]; digits = 3)))
end
println("\n  E[I | θ] by period (θ = 1..4):  data | comonotone | dynamic per anchor | myopic (empirical marginals) per anchor")
for t in 1:T-1
    @printf("  t=%d data %s  como %s\n", t, round.(EIθ(γd[t], pθ[t]); digits = 2), round.(EIθ(comonotone_coupling(pθ[t], pI[t]), pθ[t]); digits = 2))
    for m in eachindex(Qs)
        @printf("      %-9s dyn %-26s myopic %s\n", NAMES[m], string(round.(EIθ(dyn[m].γs[t], dyn[m].Fs[t]); digits = 2)), string(round.(EIθ(myo[m][t], pθ[t]); digits = 2)))
    end
end

# ══════════════════ B. income reallocation ══════════════════
ccp = pv.ccp_spec; inc = pv.income_model
β0, βP, βω, δ = inc.beta0, inc.betaP, inc.betaω, inc.delta
β1, β2, β3, β4, β5 = ccp.beta1, ccp.beta2, ccp.beta3, ccp.beta4, ccp.beta5; cc = Vector{Float64}(ccp.c)
@assert abs(β2) < 1e-12 "β₂ = 0 required"
CCPm(kP, kω, ε, ::Type{P} = Float64) where {P} = (M = zeros(P, r_I, r_θ);
    for kθ in 1:r_θ; M[:, kθ] = _ordered_probit_pmf(cc, β1 * kθ + β3 * kP + β4 * kω + β5 * ε, r_I); end; M)
step(Fθ, C, Π) = [sum(Fθ[a] * C[i, a] * Π[k, a, i] for a in 1:r_θ, i in 1:r_I) for k in 1:r_θ]
p0(θ1, kP, kω) = π_Pω[kP, kω] * π0[θ1, kP, kω] / sum(π0[:, kP, kω])
π0m = [sum(p0(θ1, kP, kω) for kP in 1:r_P, kω in 1:r_ω) for θ1 in 1:r_θ]
budget = sum(π_Pω[kP, kω] * π_ε[k] * exp(β0 + βP * kP + βω * kω + δ * k) for kP in 1:r_P, kω in 1:r_ω, k in 1:r_Y)
εeff(y, kP, kω) = (y - β0 - βP * kP - βω * kω) / δ
# baseline distribution of (θ_t, kP, kω, kε): model-implied, CCP at own ε
D = [zeros(r_θ, r_P, r_ω, r_Y) for _ in 1:T]
for kP in 1:r_P, kω in 1:r_ω, k in 1:r_Y
    Fθ = [p0(θ1, kP, kω) * π_ε[k] for θ1 in 1:r_θ]; C = CCPm(kP, kω, float(k))
    for t in 1:T
        D[t][:, kP, kω, k] = Fθ; t < T && (Fθ = step(Fθ, C, Π_z[(group_of(t), kP, kω)]))
    end
end
pθt = [vec(sum(D[t]; dims = (2, 3, 4))) for t in 1:T]
FT_base = pθt[T]
println("\n", "="^100, "\nB. INCOME REALLOCATION (CCP fixed; level budget E[Y] = \$$(round(Int, budget)) per child)\n", "="^100)
@printf("  β₅/δ = %.2f.  model-implied F_θT at baseline = %s  (empirical %s)\n", β5 / δ, round.(FT_base; digits = 3), round.(pθ[T]; digits = 3))
# dynamic: y(θ_1) for all periods
function W_dyn(y::AbstractVector{P}, Q) where {P}
    FT = zeros(P, r_θ)
    for kP in 1:r_P, kω in 1:r_ω, θ1 in 1:r_θ
        C = CCPm(kP, kω, εeff(y[θ1], kP, kω), P); Fθ = zeros(P, r_θ); Fθ[θ1] = p0(θ1, kP, kω)
        for t in 1:T-1; Fθ = step(Fθ, C, Π_z[(group_of(t), kP, kω)]); end
        FT .+= Fθ
    end
    sum(FT .* Q), FT
end
# static at t: y_t(θ_t) applied in period t only, value E[Q(θ_{t+1})]
function W_stat(y::AbstractVector{P}, Q, t) where {P}
    W = zero(P)
    for kP in 1:r_P, kω in 1:r_ω, k in 1:r_Y
        Fθ = D[t][:, kP, kω, k]
        for a in 1:r_θ
            pmf = _ordered_probit_pmf(cc, β1 * a + β3 * kP + β4 * kω + β5 * εeff(y[a], kP, kω), r_I)
            Π = Π_z[(group_of(t), kP, kω)]
            W += Fθ[a] * sum(pmf[i] * Π[kk, a, i] * Q[kk] for i in 1:r_I, kk in 1:r_θ)
        end
    end
    W
end
pol(z, w) = (e = exp.(z); log.(budget .* e ./ sum(w .* e)))   # Σ w·exp(y) = budget
function opt(obj, w)
    r = optimize(z -> -obj(pol(z, w)), zeros(r_θ), LBFGS(), Optim.Options(x_abstol = 1e-9, f_abstol = 1e-12, iterations = 1000); autodiff = :forward)
    pol(Optim.minimizer(r), w), Optim.converged(r)
end
inc_dyn = []; inc_stat = []
for (m, Q) in enumerate(Qs)
    yd, cv = opt(y -> W_dyn(y, Q)[1], π0m); Wd, FTd = W_dyn(yd, Q); Wb = sum(FT_base .* Q)
    push!(inc_dyn, (; y = exp.(yd), W = Wd, Wb, FT = FTd, cv))
    st = []
    for t in 1:T-1
        ys, cvs = opt(y -> W_stat(y, Q, t), pθt[t]); push!(st, (; y = exp.(ys), W = W_stat(ys, Q, t), Wb = sum(pθt[t+1] .* Q), cv = cvs))
    end
    push!(inc_stat, st)
end
println("\n  Dynamic income reallocation y(θ_1), held through all six periods (paper's experiment):")
@printf("  %-9s %12s %12s %8s   %-44s %s\n", "anchor", "W_base", "W_opt", "Δ%", "optimal y(θ_1) (\$)", "F_θT opt")
for m in eachindex(Qs)
    r = inc_dyn[m]
    @printf("  %-9s %12.5g %12.5g %+8.2f   %-44s %s%s\n", NAMES[m], r.Wb, r.W, 100 * (r.W - r.Wb) / abs(r.Wb),
            join([@sprintf("%9.0f", x) for x in r.y]), round.(r.FT; digits = 3), r.cv ? "" : "  (not converged)")
end
println("  Decomposition: uniform level income (everyone gets E[Y], no targeting by θ_1) vs optimum:")
for m in eachindex(Qs)
    Wu = W_dyn(fill(log(budget), r_θ), Qs[m])[1]; r = inc_dyn[m]
    @printf("    %-9s uniform Δ%% %+6.2f   optimal Δ%% %+6.2f   (targeting share of gain %.0f%%)\n", NAMES[m],
            100 * (Wu - r.Wb) / abs(r.Wb), 100 * (r.W - r.Wb) / abs(r.Wb), 100 * (r.W - Wu) / (r.W - r.Wb))
end
@printf("  population weights p(θ_1) = %s;  baseline F_θT = %s\n", round.(π0m; digits = 3), round.(FT_base; digits = 3))
println("\n  Static counterpart: income y_t(θ_t) in period t only, value E[Q(θ_{t+1})], budget weights p_t(θ):")
for t in 1:T-1
    @printf("  t=%d  p_t(θ) = %s\n", t, round.(pθt[t]; digits = 3))
    for m in eachindex(Qs)
        r = inc_stat[m][t]
        @printf("      %-9s Δ%% %+7.3f   y_t(θ) = %s%s\n", NAMES[m], 100 * (r.W - r.Wb) / abs(r.Wb), join([@sprintf("%9.0f", x) for x in r.y]), r.cv ? "" : "  (nc)")
    end
end

# ══════════════════ figures ══════════════════
ENV["PAPER_FIGS"] = "1"; include(joinpath(REPO, "src", "slides_palette.jl"))
COL = [OUTCOME_COLORS[n] for n in NAMES]
AG = ["0–1", "2–3", "4–5", "6–7", "8–9", "10–11"]
function legstrip(extra)
    l = plot(; framestyle = :none, grid = false, legend = :inside, legend_columns = 4 + length(extra), legendfontsize = 11)
    for (lab, c, ls, mk) in extra; plot!(l, [NaN], [NaN]; color = c, ls, marker = mk, lw = 2.5, label = lab); end
    for m in eachindex(Qs); plot!(l, [NaN], [NaN]; color = COL[m], lw = 2.5, label = NAMES[m]); end
    l
end
function EIfig(getE, title, stem)
    P = []
    for t in 1:T-1
        p = plot(1:r_θ, EIθ(γd[t], pθ[t]); color = :black, lw = 2.5, marker = :circle, legend = false, ylims = (0.9, r_I + 0.1),
                 xticks = 1:r_θ, yticks = 1:r_I, title = "t=$t (ages $(AG[t]))", xlabel = "θ", ylabel = t in (1, 4) ? "E[I | θ]" : "")
        plot!(p, 1:r_θ, EIθ(comonotone_coupling(pθ[t], pI[t]), pθ[t]); color = PAL_GREY, ls = :dash, marker = :diamond, lw = 2)
        for m in eachindex(Qs); plot!(p, (1:r_θ) .+ 0.03 * (m - 3), getE(m, t); color = COL[m], lw = 2, marker = :circle, ms = 3); end
        push!(P, p)
    end
    f = plot(P..., legstrip([("data", :black, :solid, :circle), ("comonotone", PAL_GREY, :dash, :diamond)]);
             layout = @layout([a b c; d e f; g{0.1h}]), size = (1300, 900), plot_title = title, plot_titlefontsize = 14)
    savefig(f, joinpath(REPO, "figures", "cf_$(TAG)_$stem.png")); println("saved figures/cf_$(TAG)_$stem.png")
end
EIfig((m, t) -> EIθ(dyn[m].γs[t], dyn[m].Fs[t]), "Dynamic planner: optimal E[I | θ] by period", "EI_dynamic")
EIfig((m, t) -> EIθ(myo[m][t], pθ[t]), "Myopic (static per-period) planner: optimal E[I | θ]", "EI_static")
function termfig(get, labs, cols, title, stem)
    P = []
    for m in eachindex(Qs)
        Fs = get(m); w = 0.8 / length(Fs)
        p = plot(; title = "Q = $(NAMES[m])", ylims = (0, 0.75), xticks = 1:r_θ, xlabel = "θ_T", legend = m == 1 ? :topleft : false)
        for (i, Fv) in enumerate(Fs); bar!(p, (1:r_θ) .+ (i - (length(Fs) + 1) / 2) * w, Fv; bar_width = w, color = cols[i], label = labs[i]); end
        push!(P, p)
    end
    f = plot(P...; layout = (2, 3), size = (1300, 760), plot_title = title, plot_titlefontsize = 14)
    savefig(f, joinpath(REPO, "figures", "cf_$(TAG)_$stem.png")); println("saved figures/cf_$(TAG)_$stem.png")
end
termfig(m -> [myoc[m].Fb, myoc[m].F, dyn[m].Fs[T]], ["data rule", "myopic", "dynamic"], [PAL_GREY, PAL_GOLD, PAL_RUST],
        "Investment reallocation: terminal skill distribution", "invest_terminal")
termfig(m -> [FT_base, inc_dyn[m].FT], ["baseline", "income optimum"], [PAL_TEAL, PAL_RUST],
        "Income reallocation: terminal skill distribution", "income_terminal")
let
    k = 1e-3
    p1 = plot(; title = "Dynamic: y(θ₁) held ages 0–13", xlabel = "initial skill θ₁", ylabel = "income (\$000)", xticks = 1:r_θ, legend = false)
    hline!(p1, [budget * k]; color = PAL_GREY, ls = :dash)
    for m in eachindex(Qs); plot!(p1, 1:r_θ, inc_dyn[m].y .* k; color = COL[m], lw = 2.5, marker = :circle); end
    P = [p1]
    for t in (1, 3, 4, 6)
        p = plot(; title = "Static, t=$t (ages $(AG[t]))", xlabel = "current skill θ_t", ylabel = t == 4 ? "income (\$000)" : "", xticks = 1:r_θ, legend = false)
        hline!(p, [budget * k]; color = PAL_GREY, ls = :dash)
        for m in eachindex(Qs); plot!(p, 1:r_θ, inc_stat[m][t].y .* k; color = COL[m], lw = 2.5, marker = :circle); end
        push!(P, p)
    end
    l = plot(; framestyle = :none, grid = false, legend = :left, legendfontsize = 12)
    plot!(l, [NaN], [NaN]; color = PAL_GREY, ls = :dash, lw = 2, label = "mean income (budget)")
    for m in eachindex(Qs); plot!(l, [NaN], [NaN]; color = COL[m], lw = 2.5, marker = :circle, label = NAMES[m]); end
    push!(P, l)
    f = plot(P...; layout = (2, 3), size = (1300, 800), plot_title = "Optimal income by skill", plot_titlefontsize = 14, left_margin = 5Plots.mm, bottom_margin = 4Plots.mm)
    savefig(f, joinpath(REPO, "figures", "cf_$(TAG)_income_policy.png")); println("saved figures/cf_$(TAG)_income_policy.png")
end
jldsave(joinpath(REPO, "output", "cf_ordrobit_$(TAG).jld2"); NAMES, Qs, dyn, myo, myoc, inc_dyn, inc_stat, budget, pθ, pI)
