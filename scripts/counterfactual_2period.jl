using LinearAlgebra, Statistics, Printf, JLD2, CSV, DataFrames, Plots, LaTeXStrings

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3_eps.jl"))
include(joinpath(@__DIR__, "..", "src", "counterfactuals.jl"))
include(joinpath(@__DIR__, "..", "src", "slides_palette.jl"))

# Paper-aware font sizes for multi-panel figures (slides values are the default).
# PAPER_FIGS is defined in slides_palette.jl.
const FS_T  = PAPER_FIGS ? 18 : 10   # panel title
const FS_G  = PAPER_FIGS ? 17 :  9   # axis guide (x/y labels)
const FS_TK = PAPER_FIGS ? 15 :  8   # tick labels
const FS_L  = PAPER_FIGS ? 14 :  7   # legend
const FS_A  = PAPER_FIGS ? 12 :  7   # in-bar annotations
const FS_PT = PAPER_FIGS ? 20 : 12   # overall plot_title

# ─────────────────────────────────────────────────────────────────
# Illustrative 2-period reallocation.
#
# Collapse the 7-period model into 2 "meta-periods":
#   step 1 = age group 1 (ages 0–5, t ∈ {1,2,3})  with transition Π̄_group1
#   step 2 = age group 2 (ages 6–11, t ∈ {4,5,6}) with transition Π̄_group2
#
# All marginals averaged across the t's in each age group.
# Produces: heatmaps, E[I|θ], F_θ trajectory, terminal F_θ bar chart.
# ─────────────────────────────────────────────────────────────────

const REG_RUN  = get(ENV, "REG_RUN", "0") == "1"
const FIT_PATH = get(ENV, "FIT_OVERRIDE", joinpath(@__DIR__, "..", "output", "estimation_v3_eps_reg.jld2"))
const TRANSITION_GROUPS = [[1,2,3], [4,5,6]]
const OUT_DIR = joinpath(@__DIR__, "..", "output", REG_RUN ? "counterfactuals_reg" : "counterfactuals")
isdir(OUT_DIR) || mkpath(OUT_DIR)

fit       = load(FIT_PATH)
params    = fit["params_v3_eps"]
gamma_agg = fit["gamma_agg"]
dims      = fit["data_dims"]
r_θ, r_I, r_P, r_ω = dims.r_θ, dims.r_I, dims.r_P, dims.r_ω
T = dims.T

π_Pω = Array{Float64}(params.π_Pω)
π_ω  = vec(sum(π_Pω, dims=1))

function build_Pi_bar(spec, π_Pω, r_θ, r_I, r_P, r_ω)
    v = pack_robit_nn_v3(spec); K, ν = spec.K, spec.ν
    Π̄ = zeros(r_θ, r_θ, r_I)
    for kP in 1:r_P, kω in 1:r_ω
        Π_z = _build_robit_nn_v3_pi(v, r_θ, r_I, K, ν, float(kP), float(kω))
        @. Π̄ += π_Pω[kP, kω] * Π_z
    end
    Π̄
end
Π̄_1step = [build_Pi_bar(params.trans_specs[g], π_Pω, r_θ, r_I, r_P, r_ω) for g in 1:2]

"""
Cumulative transition: apply the 1-step group transition `n_apply` times with
I held fixed. Required when collapsing multiple calendar periods into one
meta-period; here each age group contains 3 one-step transitions.
"""
function cumulative_transition(Π̄_1::AbstractArray{<:Real,3}, n_apply::Int)
    r_θ_out, r_θ, r_I = size(Π̄_1)
    @assert r_θ_out == r_θ
    Π̄_cum = zeros(r_θ, r_θ, r_I)
    for j in 1:r_I
        M = Π̄_1[:, :, j]            # M[i, k] = P(θ' = i | θ = k, I = j)
        Π̄_cum[:, :, j] = M^n_apply
    end
    Π̄_cum
end

const STEPS_PER_GROUP = 1   # 1-step transitions: treat each meta-period as a single period
Π̄_step = [cumulative_transition(Π̄_1step[g], STEPS_PER_GROUP) for g in 1:2]

# ─────────────────────────────────────────────────────────────────
# Averaged marginals per age group
# ─────────────────────────────────────────────────────────────────

γ_data_all = [dropdims(mean(@view(gamma_agg[t, :, :, :]), dims=3), dims=3) for t in 1:T]
avg(v)     = sum(v) / length(v)

# Representative (θ, I) marginals per meta-period
p_θ_steps = [avg([vec(sum(γ_data_all[t], dims=2)) for t in TRANSITION_GROUPS[g]])
             for g in 1:2]                                   # length 2
p_I_steps = [avg([vec(sum(γ_data_all[t], dims=1)) for t in TRANSITION_GROUPS[g]])
             for g in 1:2]                                   # length 2
γ_data_step = [avg([γ_data_all[t] for t in TRANSITION_GROUPS[g]]) for g in 1:2]  # averaged joints

# Terminal empirical θ-distribution (for the bar chart baseline)
p_θ_T_data = vec(sum(γ_data_all[T], dims=2))

# ─────────────────────────────────────────────────────────────────
# Earnings bin means and Q functions
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

function Q_from_adult(A, ys, π_ω)
    κ, r_θ, r_ω = size(A)
    Q = zeros(r_θ)
    for j in 1:r_θ, v in 1:r_ω, k in 1:κ
        Q[j] += π_ω[v] * ys[k] * A[k, j, v]
    end
    Q
end
adult_mats  = Vector{Array{Float64,3}}(params.adult_matrices)
adult_names = ["HS", "College", "EarnQ", "AlwaysEmp", "logEarn"]
values      = [[0.0, 1.0], [0.0, 1.0], earn_means, [0.0, 1.0]]
Qs          = [Q_from_adult(adult_mats[m], values[m], π_ω) for m in 1:4]
push!(Qs, log.(Qs[3]))

# ─────────────────────────────────────────────────────────────────
# Static-per-step (myopic) and sequential solutions
# ─────────────────────────────────────────────────────────────────

function E_I_given_θ(γ, p_θ)
    out = zeros(length(p_θ))
    for j in eachindex(p_θ)
        p_θ[j] < 1e-12 && (out[j] = NaN; continue)
        s = 0.0
        for i in 1:size(γ, 2); s += i * γ[j, i]; end
        out[j] = s / p_θ[j]
    end
    out
end

function solve_static_2step(Q)
    # Step 1: reward E[Q(θ_1) | θ_0, I_0] via Π̄_step[1]
    c1 = expected_Q(Q, Π̄_step[1])
    w1, γ1 = solve_ot(p_θ_steps[1], p_I_steps[1], c1; maximize=true)
    # Step 2: reward E[Q(θ_2) | θ_1, I_1] via Π̄_step[2]
    c2 = expected_Q(Q, Π̄_step[2])
    w2, γ2 = solve_ot(p_θ_steps[2], p_I_steps[2], c2; maximize=true)
    (; γ1, γ2, w_step1=w1, w_step2=w2)
end

results = Dict{String,Any}()
for m in eachindex(Qs)
    name = adult_names[m]; Q = Qs[m]
    stat = solve_static_2step(Q)
    Wseq, γs_seq, Fs_seq = solve_sequential_ot(p_θ_steps[1], p_I_steps, Π̄_step, Q)
    # Observed terminal-welfare baseline
    W_data_obs = sum(p_θ_T_data .* Q)
    results[name] = (; Q, static=stat, Wseq, γs_seq, Fs_seq, W_data_obs)
end

println("\n" * "="^86)
println("Two-period illustrative counterfactual   (group 1 = ages 0-5, group 2 = ages 6-11)")
println("="^86)
@printf "p_θ_step1 = %s\n" join([@sprintf("%.3f", x) for x in p_θ_steps[1]], "  ")
@printf "p_θ_step2 = %s\n" join([@sprintf("%.3f", x) for x in p_θ_steps[2]], "  ")
@printf "p_I_step1 = %s\n" join([@sprintf("%.3f", x) for x in p_I_steps[1]], "  ")
@printf "p_I_step2 = %s\n" join([@sprintf("%.3f", x) for x in p_I_steps[2]], "  ")
@printf "p_θ_T (data, t=7) = %s\n" join([@sprintf("%.3f", x) for x in p_θ_T_data], "  ")

for m in eachindex(Qs)
    name = adult_names[m]; r = results[name]
    println("─"^86)
    @printf "[%s]   Q(θ) = %s\n" name join([@sprintf("%.4g", q) for q in r.Q], "  ")
    @printf "  W_data (empirical p_θ_T · Q)      = %12.5g\n" r.W_data_obs
    @printf "  W_sequential (2-period LP)        = %12.5g    (+%.2f%% vs data)\n" r.Wseq (100*(r.Wseq - r.W_data_obs)/abs(r.W_data_obs))
end
println("\nNote: the 2-period model is an illustrative collapse. Π̄ is Π̄_group^3 (3 one-step transitions with I held fixed), and p_θ_mid under the LP differs from empirical p_θ_step2 by construction.")

# ── Persist the investment-reallocation welfare table for the paper ──────────
# Source of record for the "welfare gains from optimal investment transport"
# table. Replication: regenerate via scripts/generate_paper_figures.sh.
let tbl_dir = joinpath(@__DIR__, "..", "output", REG_RUN ? "paper_tables_reg" : "paper_tables")
    isdir(tbl_dir) || mkpath(tbl_dir)
    df = DataFrame(outcome = String[], W_data = Float64[], W_plan = Float64[], pct_gain = Float64[])
    for name in adult_names
        r = results[name]
        push!(df, (name, r.W_data_obs, r.Wseq, 100*(r.Wseq - r.W_data_obs)/abs(r.W_data_obs)))
    end
    CSV.write(joinpath(tbl_dir, "welfare_investment.csv"), df)
    println("  saved output/paper_tables/welfare_investment.csv")
end

# ═════════════════════════════════════════════════════════════════
# FIGURES
# ═════════════════════════════════════════════════════════════════

STEP_LABELS = ["step 1: ages 0–5", "step 2: ages 6–11"]

# ─── Figure 0: anchor outcomes E[Z | θ_T] as functions of terminal skill ─────
# Plots each anchor Q_m(θ_T) = E[outcome_m | θ_T] (marginalised over ω), both as
# raw levels (own y-scale, left) and normalised to [0,1] (right) so the *shapes*
# can be compared. Motivation: the optimal policies agree across anchors except
# College — this figure shows College is the one anchor that is sharply convex
# in θ_T (almost all of its value sits in the top skill step).
let
    norm01(x) = (x .- minimum(x)) ./ (maximum(x) - minimum(x) + eps())
    pretty = Dict("HS"=>"HS completion", "College"=>"College completion",
                  "EarnQ"=>"adult earnings", "AlwaysEmp"=>"employment",
                  "logEarn"=>"log earnings")
    # Single panel: normalised anchor curves (shape comparison). The marginal-
    # value (ΔQ) panel was dropped — the paper only needs the anchor levels.
    pL = plot(xlabel=L"\theta_T", ylabel=L"E[Q(\theta_T)]\ \mathrm{(normalised\ to\ [0,1])}",
              title="Adult outcomes as functions of terminal skill", xticks=1:r_θ,
              legend=:topleft, ylims=(-0.03, 1.08),
              titlefontsize=FS_T, legendfontsize=FS_L, tickfontsize=FS_TK,
              xlabelfontsize=FS_G, ylabelfontsize=FS_G)
    for m in eachindex(Qs)
        name = adult_names[m]
        plot!(pL, 1:r_θ, norm01(Qs[m]); label=get(pretty, name, name),
              color=OUTCOME_COLORS[name], lw=2.8, marker=:circle, ms=7)
    end
    fig_anchor = plot(pL; size=(PAPER_FIGS ? (760, 520) : (640, 460)),
                      margin=4Plots.mm, framestyle=:box)
    save_slide_fig(fig_anchor, "anchor_outcomes_by_theta"; png_dir=OUT_DIR)
    println("  saved anchor_outcomes_by_theta.{png,pdf}")
end

# ─── Actual conditional outcome distributions P(Z | θ_T) (raw levels) ────────
# Complements the (normalised) anchor figure with the actual probabilities:
#   (a) success probability for each binary outcome,
#   (b) the full earnings-quintile distribution P(q | θ_T).
let
    margω(A, k, j) = sum(π_ω[v] * A[k, j, v] for v in 1:size(A, 3))
    binaries = [(1, "HS completion"), (2, "College completion"), (4, "employment")]
    pA = plot(xlabel=L"\theta_T", ylabel=L"P(\mathrm{outcome}=1 \mid \theta_T)",
              title="Binary outcomes (success probability)", xticks=1:r_θ,
              ylims=(-0.03, 1.03), legend=:topleft,
              titlefontsize=FS_T, legendfontsize=FS_L, tickfontsize=FS_TK,
              xlabelfontsize=FS_G, ylabelfontsize=FS_G)
    for (m, lab) in binaries
        A = adult_mats[m]; κ = size(A, 1)
        p1 = [margω(A, κ, j) for j in 1:r_θ]      # highest category = "1"
        plot!(pA, 1:r_θ, p1; label=lab, color=OUTCOME_COLORS[adult_names[m]],
              lw=2.8, marker=:circle, ms=7)
    end
    Aearn = adult_mats[3]; nq = size(Aearn, 1)
    Pq = [margω(Aearn, q, j) for q in 1:nq, j in 1:r_θ]
    Fq = cumsum(Pq, dims=1)               # Fq[q, j] = P(quintile ≤ q | θ_T = j)
    qcols = [cgrad(:viridis)[t] for t in range(0.12, 0.85; length=max(nq-1, 1))]
    pB = plot(xlabel=L"\theta_T", ylabel=L"P(\mathrm{quintile} \leq q \mid \theta_T)",
              title="Earnings-quintile CDF", xticks=1:r_θ, ylims=(-0.03, 1.03),
              legend=:topright, titlefontsize=FS_T, legendfontsize=FS_L,
              tickfontsize=FS_TK, xlabelfontsize=FS_G, ylabelfontsize=FS_G)
    for q in 1:(nq-1)                     # q = nq is identically 1 (trivial)
        plot!(pB, 1:r_θ, Fq[q, :]; label="quintile ≤ $q", color=qcols[q],
              lw=2.8, marker=:circle, ms=6)
    end
    fig_dist = plot(pA, pB; layout=(1, 2), size=(PAPER_FIGS ? (1180, 500) : (1200, 500)),
                    margin=4Plots.mm,
                    plot_title="Conditional outcome distributions given terminal skill",
                    plot_titlefontsize=FS_PT)
    save_slide_fig(fig_dist, "outcome_distributions_by_theta"; png_dir=OUT_DIR)
    println("  saved outcome_distributions_by_theta.{png,pdf}")
end


# ─── Figure 1: γ heatmaps (data vs sequential) per outcome ─────
function heatpanel(γ, title_str; clim, show_ylabel)
    heatmap(1:r_I, 1:r_θ, γ; c=:viridis, clims=clim, colorbar=false,
            xticks=(1:r_I, string.(1:r_I)), yticks=(1:r_θ, string.(1:r_θ)),
            title=title_str, titlefontsize=9,
            xlabel="I", ylabel=(show_ylabel ? "θ" : ""),
            aspect_ratio=:equal, framestyle=:box, tickfontsize=7,
            xlabelfontsize=8, ylabelfontsize=8)
end

for m in eachindex(Qs)
    name = adult_names[m]; r = results[name]
    γs_data = γ_data_step
    γs_seq  = r.γs_seq
    mx = maximum(maximum.(vcat(γs_data, γs_seq)))
    clim = (0.0, mx)
    ps = [heatpanel(γs_data[1], "data, $(STEP_LABELS[1])";   clim, show_ylabel=true),
          heatpanel(γs_data[2], "data, $(STEP_LABELS[2])";   clim, show_ylabel=false),
          heatpanel(γs_seq[1],  "seq, $(STEP_LABELS[1])";    clim, show_ylabel=true),
          heatpanel(γs_seq[2],  "seq, $(STEP_LABELS[2])";    clim, show_ylabel=false)]
    fig = plot(ps...; layout=(2, 2), size=(780, 780),
               plot_title="Q = $name    γ[θ, I]   (max = $(round(mx; digits=3)))",
               plot_titlefontsize=11, margin=2Plots.mm)
    path = joinpath(OUT_DIR, "cf_2p_heatmaps_$(lowercase(name)).png")
    savefig(fig, path)
    println("  saved $path")
end

# ─── Figure 2: E[I | θ] per step, overlaying all outcomes ───────
out_colors = OUTCOME_COLORS
const BASELINE_COLOR = PAL_NAVY
const OT_COLOR       = PAL_RUST
EI_panels = []
for s in 1:2
    p_θ_d = p_θ_steps[s]
    γ_d   = γ_data_step[s]
    γ_cm  = comonotone_coupling(p_θ_d, p_I_steps[s])
    E_d   = E_I_given_θ(γ_d,  p_θ_d)
    E_cm  = E_I_given_θ(γ_cm, p_θ_d)
    p = plot(1:r_θ, E_d;
             label=(s==1 ? "data" : ""), lw=2.5, color=:black, marker=:circle, ms=5,
             xlabel="θ", ylabel=(s==1 ? "E[I | θ]" : ""),
             title=STEP_LABELS[s], titlefontsize=10,
             xticks=1:r_θ, yticks=1:r_I, ylims=(0.9, r_I+0.1),
             legend=(s==1 ? :topleft : false), legendfontsize=7,
             tickfontsize=8, xlabelfontsize=9, ylabelfontsize=9)
    plot!(p, 1:r_θ, E_cm; label=(s==1 ? "comonotone" : ""), lw=2, color=:gray,
          ls=:dash, marker=:diamond, ms=4)
    for m in eachindex(Qs)
        name = adult_names[m]; r = results[name]
        γ_stat = s == 1 ? r.static.γ1 : r.static.γ2
        # For sequential the θ-marginal differs from data at step 2
        p_θ_seq = s == 1 ? p_θ_d : r.Fs_seq[2]
        E_stat  = E_I_given_θ(γ_stat,    p_θ_d)
        E_seq   = E_I_given_θ(r.γs_seq[s], p_θ_seq)
        plot!(p, 1:r_θ, E_stat;
              label=(s==1 ? "static: $name" : ""), lw=1.4, ls=:dot,
              color=out_colors[name], marker=:circle, ms=3)
        plot!(p, 1:r_θ, E_seq;
              label=(s==1 ? "seq: $name" : ""), lw=1.8,
              color=out_colors[name], marker=:utriangle, ms=4)
    end
    push!(EI_panels, p)
end
fig_E = plot(EI_panels...; layout=(1, 2), size=(1200, 500),
             plot_title="E[I | θ]: data, comonotone, static OT, sequential OT",
             plot_titlefontsize=12, margin=3Plots.mm)
save_slide_fig(fig_E, "cf_2p_EI_given_theta"; png_dir=OUT_DIR)
println("  saved cf_2p_EI_given_theta.{png,pdf}")

# ─── Figures 2a/2b/2c: sequential reveal of the optimal policy ──
# Same axes across all three frames so the reveal is visually clean.
#   (a) data + comonotone benchmark only
#   (b) data + max-earnings (EarnQ) policy only
#   (c) data + all four sequential OT policies (HS, College, EarnQ, AlwaysEmp)
const REVEAL_OUTCOMES = ["HS", "College", "EarnQ", "AlwaysEmp"]
const Y_LIMS_EI       = (0.9, r_I + 0.1)

function build_EI_panel(step::Int; show_data=true, show_comono=true,
                         show_earn=false, show_logearn=false, show_all=false,
                         show_legend=false)
    p_θ_d = p_θ_steps[step]
    γ_d   = γ_data_step[step]
    γ_cm  = comonotone_coupling(p_θ_d, p_I_steps[step])
    E_d   = E_I_given_θ(γ_d,  p_θ_d)
    E_cm  = E_I_given_θ(γ_cm, p_θ_d)

    function E_seq_for(name)
        r_name  = results[name]
        p_θ_seq = step == 1 ? p_θ_d : r_name.Fs_seq[2]
        E_I_given_θ(r_name.γs_seq[step], p_θ_seq)
    end

    # Legends are drawn in a dedicated strip below the panels (see
    # build_EI_reveal_figure), so the data panels themselves carry no legend.
    p = plot(xlabel = L"\theta",
             ylabel = step == 1 ? L"E[I \mid \theta]" : "",
             title  = STEP_LABELS[step], titlefontsize = FS_T,
             xticks = 1:r_θ, yticks = 1:r_I, ylims = Y_LIMS_EI,
             legend = false,
             tickfontsize = FS_TK,
             xlabelfontsize = FS_G, ylabelfontsize = FS_G)
    if show_data
        plot!(p, 1:r_θ, E_d;
              lw = 3.2, color = PAL_TEXT, marker = :circle, ms = 6,
              label = show_legend ? "data" : "")
    end
    if show_comono
        plot!(p, 1:r_θ, E_cm;
              lw = 2.8, color = PAL_GREY, ls = :dash, marker = :diamond, ms = 5,
              label = show_legend ? "comonotone benchmark" : "")
    end
    if show_earn
        plot!(p, 1:r_θ, E_seq_for("EarnQ");
              lw = 3.0, color = OUTCOME_COLORS["EarnQ"],
              marker = :utriangle, ms = 6,
              label = show_legend ? "planner: adult earnings" : "")
    end
    if show_logearn
        plot!(p, 1:r_θ, E_seq_for("logEarn");
              lw = 3.0, color = OUTCOME_COLORS["logEarn"],
              marker = :dtriangle, ms = 6,
              label = show_legend ? "planner: log earnings" : "")
    end
    if show_all
        pretty_name = Dict("HS" => "HS completion",
                           "College" => "College completion",
                           "EarnQ" => "adult earnings",
                           "AlwaysEmp" => "employment")
        for name in REVEAL_OUTCOMES
            plot!(p, 1:r_θ, E_seq_for(name);
                  lw = 2.8, color = OUTCOME_COLORS[name],
                  marker = :utriangle, ms = 5,
                  label = show_legend ? "planner: $(pretty_name[name])" : "")
        end
    end
    p
end

# A dedicated legend strip below the two data panels: nothing is drawn in the
# data region, so no curve is ever obscured. Series are replicated as off-screen
# NaN points purely to populate the legend.
function build_EI_legend_strip(; show_comono, show_earn, show_logearn, show_all)
    # Short labels (the caption names the planners in full) so the legend box
    # fits within the figure width without clipping.
    pretty_name = Dict("HS" => "HS planner", "College" => "College planner",
                       "EarnQ" => "earnings planner", "AlwaysEmp" => "employment planner")
    leg = plot(framestyle = :none, grid = false, ticks = nothing,
               xlims = (0, 1), ylims = (0, 1), legend = :inside,
               legend_columns = 3, legendfontsize = FS_L)
    plot!(leg, [NaN], [NaN]; color = PAL_TEXT, marker = :circle, lw = 3.2, ms = 6,
          label = "data")
    show_comono && plot!(leg, [NaN], [NaN]; color = PAL_GREY, ls = :dash,
          marker = :diamond, lw = 2.8, ms = 5, label = "comonotone benchmark")
    show_earn && plot!(leg, [NaN], [NaN]; color = OUTCOME_COLORS["EarnQ"],
          marker = :utriangle, lw = 3.0, ms = 6, label = "earnings planner")
    show_logearn && plot!(leg, [NaN], [NaN]; color = OUTCOME_COLORS["logEarn"],
          marker = :dtriangle, lw = 3.0, ms = 6, label = "log-earnings planner")
    if show_all
        for name in REVEAL_OUTCOMES
            plot!(leg, [NaN], [NaN]; color = OUTCOME_COLORS[name],
                  marker = :utriangle, lw = 2.8, ms = 5,
                  label = pretty_name[name])
        end
    end
    leg
end

function build_EI_reveal_figure(stem::AbstractString, title_str::AbstractString;
                                 show_comono=true, show_earn=false,
                                 show_logearn=false, show_all=false)
    panels = [build_EI_panel(s;
                show_comono = show_comono,
                show_earn   = show_earn,
                show_logearn = show_logearn,
                show_all    = show_all,
                show_legend = false) for s in 1:2]
    leg = build_EI_legend_strip(; show_comono, show_earn, show_logearn, show_all)
    lay = @layout [grid(1, 2); b{0.18h}]
    fig = plot(panels[1], panels[2], leg; layout = lay,
               size = (PAPER_FIGS ? (980, 500) : (1200, 560)),
               margin = 3Plots.mm)
    save_slide_fig(fig, stem; png_dir = OUT_DIR)
    println("  saved $(stem).{png,pdf}")
end

build_EI_reveal_figure("cf_2p_EI_reveal1_data_comono",
    "E[I | θ]: data vs.\\ comonotone benchmark";
    show_comono = true)
build_EI_reveal_figure("cf_2p_EI_reveal2_data_earn",
    "E[I | θ]: adding the adult-earnings and log-earnings planners";
    show_comono = true, show_earn = true, show_logearn = true)
build_EI_reveal_figure("cf_2p_EI_reveal3_all",
    "E[I | θ]: data vs.\\ planner solutions across outcomes";
    show_comono = true, show_all = true)

# ─── Figure 3: F_θ_t trajectory (3 points: init, mid, terminal) ─
traj_panels = []
colors_θ = [:steelblue, :seagreen, :darkorange, :firebrick]
for m in eachindex(Qs)
    name = adult_names[m]; r = results[name]
    xs = [0, 1, 2]       # step index
    xlabels = ["init (pre-step 1)", "mid (between)", "terminal (post-step 2)"]
    p = plot(xlabel="step", ylabel="P(θ = k)", title="Q = $name", titlefontsize=10,
             xticks=(xs, xlabels), legend=:outerright, legendfontsize=7, ylims=(0, 0.65),
             framestyle=:box)
    data_curve = [p_θ_steps[1], p_θ_steps[2], p_θ_T_data]
    seq_curve  = r.Fs_seq      # length 3: init, mid, terminal
    for k in 1:r_θ
        plot!(p, xs, [data_curve[j][k] for j in 1:3];
              label="data  θ=$k", color=colors_θ[k], lw=1.8, ls=:solid, marker=:circle, ms=4)
        plot!(p, xs, [seq_curve[j][k]  for j in 1:3];
              label="seq  θ=$k",  color=colors_θ[k], lw=1.8, ls=:dash,  marker=:utriangle, ms=4)
    end
    push!(traj_panels, p)
end
fig_traj = plot(traj_panels...; layout=(2, 3), size=(1700, 820),
                plot_title="F_θ trajectory — data vs sequential planner (2-period)",
                plot_titlefontsize=12, margin=3Plots.mm)
save_slide_fig(fig_traj, "cf_2p_Ftheta_trajectory"; png_dir=OUT_DIR)
println("  saved cf_2p_Ftheta_trajectory.{png,pdf}")

# ─── Figure 4: Terminal F_θ bar chart, data vs sequential ──────
term_panels = []
for m in eachindex(Qs)
    name = adult_names[m]; r = results[name]
    F_data = p_θ_T_data
    F_seq  = r.Fs_seq[3]
    xs = 1:r_θ; w = 0.36
    p = bar(xs .- w/2, F_data; bar_width=w, color=PAL_TEAL, label="data",
             xlabel=L"\theta_T", ylabel=(m in (1,4) ? L"P(\theta_T = k)" : ""),
             title="Q = $name", titlefontsize=FS_T, ylims=(0, 0.65),
             xticks=(xs, string.(xs)), tickfontsize=FS_TK,
             xlabelfontsize=FS_G, ylabelfontsize=FS_G,
             legend=(m==1 ? :topleft : false), legendfontsize=FS_L, framestyle=:box)
    bar!(p, xs .+ w/2, F_seq; bar_width=w, color=PAL_RUST, label="planner")
    # Per-bar numeric annotations removed — they overlapped and cluttered the panels.
    push!(term_panels, p)
end
fig_term = plot(term_panels...; layout=(2, 3), size=(PAPER_FIGS ? (1180, 660) : (1500, 760)),
                plot_title=L"Terminal\ skill\ distribution\ F_{\theta_T}",
                plot_titlefontsize=FS_PT, margin=3Plots.mm)
save_slide_fig(fig_term, "cf_2p_terminal_Ftheta"; png_dir=OUT_DIR)
println("  saved cf_2p_terminal_Ftheta.{png,pdf}")

# ─────────────────────────────────────────────────────────────────
# Figure 5: E[I_2 | I_1] — investment persistence across steps
#
# Joint P(I_1, I_2) is induced by γ_1, the intervening transition, and γ_2:
#   P(I_1, I_2) = Σ_{θ_0, θ_1} γ_1[θ_0, I_1] · Π̄_step1[θ_1|θ_0, I_1]
#                              · γ_2[θ_1, I_2] / F_θ_1[θ_1]
# ─────────────────────────────────────────────────────────────────

function joint_I1_I2(γ1, γ2, Π̄1)
    r_θ_out, r_θ, r_I = size(Π̄1)
    # F_θ_1 implied by γ_1 via pushforward (matches γ_2's θ-marginal by LP construction)
    F_θ_1 = [sum(Π̄1[k, i, j] * γ1[i, j] for i in 1:r_θ, j in 1:r_I) for k in 1:r_θ_out]
    P = zeros(r_I, r_I)           # P[i1, i2]
    for i1 in 1:r_I, i2 in 1:r_I
        s = 0.0
        for θ0 in 1:r_θ, θ1 in 1:r_θ_out
            F_θ_1[θ1] < 1e-12 && continue
            s += γ1[θ0, i1] * Π̄1[θ1, θ0, i1] * γ2[θ1, i2] / F_θ_1[θ1]
        end
        P[i1, i2] = s
    end
    P
end

function E_I2_given_I1(P)
    r_I = size(P, 1); out = fill(NaN, r_I)
    for i1 in 1:r_I
        m = sum(P[i1, :])
        m < 1e-12 && continue
        out[i1] = sum(i2 * P[i1, i2] for i2 in 1:r_I) / m
    end
    out
end

# Data joint: γ_data_step1 and γ_data_step2, joined through Π̄_step[1]
P_data = joint_I1_I2(γ_data_step[1], γ_data_step[2], Π̄_step[1])
E_data_I2 = E_I2_given_I1(P_data)

# Comonotone benchmark at each step, joined through Π̄_step[1]
γ_cm_1 = comonotone_coupling(p_θ_steps[1], p_I_steps[1])
γ_cm_2 = comonotone_coupling(p_θ_steps[2], p_I_steps[2])
P_cm = joint_I1_I2(γ_cm_1, γ_cm_2, Π̄_step[1])
E_cm_I2 = E_I2_given_I1(P_cm)

# OT for each outcome
E_OT_I2 = Dict{String,Vector{Float64}}()
for m in eachindex(Qs)
    name = adult_names[m]; r = results[name]
    P_ot = joint_I1_I2(r.γs_seq[1], r.γs_seq[2], Π̄_step[1])
    E_OT_I2[name] = E_I2_given_I1(P_ot)
end

# Plot
const EI2_OUTCOME_LABEL = Dict("HS" => "planner: HS completion",
                                 "College" => "planner: College completion",
                                 "EarnQ" => "planner: adult earnings",
                                 "AlwaysEmp" => "planner: employment",
                                 "logEarn" => "planner: log earnings")

p = plot(xlabel="I₁", ylabel="E[I₂ | I₁]",
          title="Investment persistence across the two steps",
          titlefontsize=11, xticks=1:r_I, yticks=1:r_I,
          xlims=(0.7, r_I+0.3), ylims=(0.7, r_I+0.3),
          legend=:topright, legendfontsize=9,
          size=(780, 620), framestyle=:box, aspect_ratio=:equal)
plot!(p, 1:r_I, 1:r_I; label="45° reference (I₂ = I₁)",
      lw=1.5, color=PAL_GREY, ls=:dot)
plot!(p, 1:r_I, E_data_I2;
      label="data", lw=3.2, color=PAL_TEXT, marker=:circle, ms=6)
plot!(p, 1:r_I, E_cm_I2;
      label="comonotone benchmark", lw=2.8, color=PAL_GREY, ls=:dash,
      marker=:diamond, ms=5)
for name in REVEAL_OUTCOMES
    plot!(p, 1:r_I, E_OT_I2[name];
          label=EI2_OUTCOME_LABEL[name], lw=2.8,
          color=OUTCOME_COLORS[name], marker=:utriangle, ms=5)
end
save_slide_fig(p, "cf_2p_EI2_given_I1"; png_dir=OUT_DIR)
println("  saved cf_2p_EI2_given_I1.{png,pdf}")

# Print the numbers
println("\nE[I_2 | I_1]   (I_1 = 1..$r_I)")
@printf "  data       = %s\n" join([@sprintf("%.3f", x) for x in E_data_I2], "  ")
@printf "  comonotone = %s\n" join([@sprintf("%.3f", x) for x in E_cm_I2], "  ")
for m in eachindex(Qs)
    name = adult_names[m]
    @printf "  seq %-9s = %s\n" name join([@sprintf("%.3f", x) for x in E_OT_I2[name]], "  ")
end
@printf "  (P(I_1) marginal: %s)\n" join([@sprintf("%.3f", x) for x in p_I_steps[1]], "  ")

# ─────────────────────────────────────────────────────────────────
# Figure 6: E[I_2 | I_1, θ_0] — one panel per initial skill level
#
# E[I_2 | I_1, θ_0] = Σ_{θ_1, I_2} Π̄_step1[θ_1|θ_0, I_1] · γ_2[θ_1, I_2]/F_θ_1[θ_1] · I_2
# ─────────────────────────────────────────────────────────────────

function E_I2_given_I1_θ0(γ2, Π̄1)
    r_θ_out, r_θ, r_I = size(Π̄1)
    # F_θ_1: θ-marginal of γ2 (matches pushforward in LP / γ_data-chained in baseline)
    F_θ_1 = vec(sum(γ2, dims=2))
    E = zeros(r_θ, r_I)     # E[θ_0, I_1]
    for θ0 in 1:r_θ, i1 in 1:r_I
        s = 0.0
        for θ1 in 1:r_θ_out
            F_θ_1[θ1] < 1e-12 && continue
            for i2 in 1:r_I
                s += Π̄1[θ1, θ0, i1] * γ2[θ1, i2] / F_θ_1[θ1] * i2
            end
        end
        E[θ0, i1] = s
    end
    E
end

E_data_by_θ = E_I2_given_I1_θ0(γ_data_step[2], Π̄_step[1])
E_cm_by_θ   = E_I2_given_I1_θ0(γ_cm_2,          Π̄_step[1])
E_OT_by_θ   = Dict{String,Matrix{Float64}}()
for m in eachindex(Qs)
    name = adult_names[m]; r = results[name]
    E_OT_by_θ[name] = E_I2_given_I1_θ0(r.γs_seq[2], Π̄_step[1])
end

panels_byθ = []
for θ0 in 1:r_θ
    p_θ0_mass = p_θ_steps[1][θ0]
    title_str = @sprintf "θ_0 = %d   (P=%.3f)" θ0 p_θ0_mass
    p = plot(xlabel="I_1", ylabel=(θ0 in (1,3) ? "E[I_2 | I_1, θ_0]" : ""),
             title=title_str, titlefontsize=10, xticks=1:r_I, yticks=1:r_I,
             xlims=(0.7, r_I+0.3), ylims=(0.5, r_I+0.5),
             legend=(θ0 == 1 ? :topright : false), legendfontsize=7,
             framestyle=:box)
    plot!(p, 1:r_I, 1:r_I; label=(θ0==1 ? "I_2 = I_1" : ""),
          lw=1, color=:gray70, ls=:dot)
    plot!(p, 1:r_I, E_data_by_θ[θ0, :]; label=(θ0==1 ? "data" : ""),
          lw=2.5, color=:black, marker=:circle, ms=5)
    plot!(p, 1:r_I, E_cm_by_θ[θ0, :]; label=(θ0==1 ? "comonotone" : ""),
          lw=2, color=:gray40, ls=:dash, marker=:diamond, ms=4)
    for m in eachindex(Qs)
        name = adult_names[m]
        plot!(p, 1:r_I, E_OT_by_θ[name][θ0, :];
              label=(θ0==1 ? "seq: $name" : ""), lw=1.8,
              color=out_colors[name], marker=:utriangle, ms=4)
    end
    push!(panels_byθ, p)
end
fig_byθ = plot(panels_byθ...; layout=(2, 2), size=(1100, 820),
                plot_title="E[I_2 | I_1, θ_0] — investment persistence by initial skill",
                plot_titlefontsize=12, margin=3Plots.mm)
save_slide_fig(fig_byθ, "cf_2p_EI2_given_I1_theta0"; png_dir=OUT_DIR)
println("  saved cf_2p_EI2_given_I1_theta0.{png,pdf}")

println("\nE[I_2 | I_1, θ_0]   (rows = θ_0, cols = I_1 = 1..$r_I)")
println("  data:")
for θ0 in 1:r_θ
    @printf "    θ_0=%d  %s\n" θ0 join([@sprintf("%.3f", E_data_by_θ[θ0, i]) for i in 1:r_I], "  ")
end
println("  sequential (HS — others nearly identical):")
for θ0 in 1:r_θ
    @printf "    θ_0=%d  %s\n" θ0 join([@sprintf("%.3f", E_OT_by_θ["HS"][θ0, i]) for i in 1:r_I], "  ")
end

@info "Done."
