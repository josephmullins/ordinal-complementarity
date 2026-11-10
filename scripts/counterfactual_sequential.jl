using LinearAlgebra, Statistics, Printf, JLD2, CSV, DataFrames, Plots

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3_eps.jl"))
include(joinpath(@__DIR__, "..", "src", "counterfactuals.jl"))
include(joinpath(@__DIR__, "..", "src", "slides_palette.jl"))

# Paper-aware font sizes (slides defaults). PAPER_FIGS from slides_palette.jl.
const FS_T  = PAPER_FIGS ? 17 : 10   # panel title
const FS_G  = PAPER_FIGS ? 16 :  9   # axis guide
const FS_TK = PAPER_FIGS ? 14 :  8   # tick labels
const FS_L  = PAPER_FIGS ? 12 :  7   # legend / annotations
const FS_PT = PAPER_FIGS ? 20 : 12   # overall plot title

# ─────────────────────────────────────────────────────────────────
# Full sequential (multi-period) planner reallocation via one LP.
# Initial F_θ_1 fixed at empirical, I-marginal at each t fixed at empirical.
# Terminal reward is Q_m(θ_T) for each of the four adult outcomes.
# ─────────────────────────────────────────────────────────────────

const REG_RUN  = get(ENV, "REG_RUN", "0") == "1"
const FIT_PATH = get(ENV, "FIT_OVERRIDE", joinpath(@__DIR__, "..", "output", "estimation_v3_eps_reg.jld2"))
const TRANSITION_GROUPS = [[1,2,3], [4,5,6]]
const OUT_DIR = joinpath(@__DIR__, "..", "output", REG_RUN ? "counterfactuals_reg" : "counterfactuals")
isdir(OUT_DIR) || mkpath(OUT_DIR)

fit = load(FIT_PATH)
params    = fit["params_v3_eps"]
gamma_agg = fit["gamma_agg"]                # (T, r_θ, r_I, N)
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
Π̄_group = [build_Pi_bar(params.trans_specs[g], π_Pω, r_θ, r_I, r_P, r_ω)
             for g in 1:length(TRANSITION_GROUPS)]
group_of(t) = t <= 3 ? 1 : 2
Π̄_ts = [Π̄_group[group_of(t)] for t in 1:T-1]    # length T-1 = 6

# Earnings bin means
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

# Per-period data marginals
γ_data  = [dropdims(mean(@view(gamma_agg[t, :, :, :]), dims=3), dims=3) for t in 1:T]
p_θ_emp = [vec(sum(γ_data[t], dims=2)) for t in 1:T]
p_I_emp = [vec(sum(γ_data[t], dims=1)) for t in 1:T-1]   # I-marginals for t=1..T-1

# ─────────────────────────────────────────────────────────────────
# Solve sequential LP per outcome
# ─────────────────────────────────────────────────────────────────

seq_results = Vector{NamedTuple}(undef, length(Qs))
for m in eachindex(Qs)
    v, γs, Fs = solve_sequential_ot(p_θ_emp[1], p_I_emp, Π̄_ts, Qs[m])
    seq_results[m] = (; v, γs, Fs, Q=Qs[m])
end

# ─────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────

println("\n" * "="^86)
println("Sequential reallocation  (planner reallocates at every t, budgets = empirical p_I_t)")
println("Initial F_θ_1 fixed at empirical.   Terminal reward Q_m(θ_T) for each outcome.")
println("="^86)

for m in eachindex(Qs)
    name = adult_names[m]; Q = Qs[m]
    W_seq   = seq_results[m].v
    # Observed terminal welfare under data
    W_data  = sum(p_θ_emp[T] .* Q)
    # Static reallocation only at t=6 (reuse our earlier computation)
    c6      = expected_Q(Q, Π̄_group[2])
    w_stat6, _ = solve_ot(p_θ_emp[T-1], p_I_emp[T-1], c6; maximize=true)
    # "Chained data": welfare if the planner does nothing, implied terminal via pushforward
    # of γ_data_6 through Π̄_6 — should match W_data up to model slippage.
    push_from_data6 = [sum(Π̄_group[2][k, i, j] * γ_data[T-1][i, j]
                             for i in 1:r_θ, j in 1:r_I) for k in 1:r_θ]
    W_data_chained = sum(push_from_data6 .* Q)

    println("─"^86)
    @printf "[%s]   Q(θ) = %s\n" name join([@sprintf("%.4g", q) for q in Q], "  ")
    @printf "  W_data (observed p_θ_T)           = %12.5g\n" W_data
    @printf "  W_data (γ_data_6 pushforward)     = %12.5g\n" W_data_chained
    @printf "  W_static (OT only at t=6)         = %12.5g    (+%.2f%% vs data)\n" w_stat6 (100*(w_stat6-W_data)/abs(W_data))
    @printf "  W_sequential (reallocate at all t)= %12.5g    (+%.2f%% vs data,  +%.2f%% vs static)\n" W_seq (100*(W_seq-W_data)/abs(W_data)) (100*(W_seq-w_stat6)/abs(w_stat6))
end

# ─────────────────────────────────────────────────────────────────
# Paper-styled multi-panel figures. Legends live in a dedicated strip
# (nothing drawn in the data region) so no panel is ever obscured.
# ─────────────────────────────────────────────────────────────────

function E_I_given_θ(γ, p_θ)
    r_θ, r_I = size(γ); out = zeros(r_θ)
    for j in 1:r_θ
        p_θ[j] < 1e-12 && (out[j] = NaN; continue)
        s = 0.0
        for i in 1:r_I; s += i * γ[j, i]; end
        out[j] = s / p_θ[j]
    end
    out
end

_lg(s, k, d) = haskey(s, k) ? getfield(s, k) : d
function legend_strip(specs; ncol = length(specs))
    leg = plot(framestyle = :none, grid = false, ticks = nothing,
               xlims = (0, 1), ylims = (0, 1), legend = :inside,
               legend_columns = ncol, legendfontsize = FS_L)
    for s in specs
        plot!(leg, [NaN], [NaN]; color = s.color, ls = _lg(s, :ls, :solid),
              marker = _lg(s, :mk, :none), lw = _lg(s, :lw, 2.6),
              ms = _lg(s, :ms, 6), label = s.label)
    end
    leg
end

age_labels = ["0–1", "2–3", "4–5", "6–7", "8–9", "10–11"]
out_colors = OUTCOME_COLORS
EI_lay     = @layout [a b c; d e f; g{0.12h}]

# Precompute per-(t, anchor) E[I|θ]: dynamic (sequential) and myopic static-per-t
# (one-shot OT on the EMPIRICAL marginals, reward c_t = E[Q(θ_{t+1})|θ,I]).
E_dyn_tm  = [[E_I_given_θ(seq_results[m].γs[t], seq_results[m].Fs[t]) for m in eachindex(Qs)] for t in 1:T-1]
E_stat_tm = Vector{Vector{Vector{Float64}}}(undef, T-1)
for t in 1:T-1
    E_stat_tm[t] = Vector{Vector{Float64}}(undef, length(Qs))
    for m in eachindex(Qs)
        c_t = expected_Q(Qs[m], Π̄_group[group_of(t)])
        _, γ_stat = solve_ot(p_θ_emp[t], p_I_emp[t], c_t; maximize = true)
        E_stat_tm[t][m] = E_I_given_θ(γ_stat, p_θ_emp[t])
    end
end

# base panel: data (black) + comonotone benchmark (grey dashed), no legend.
function base_EI_panel(t)
    p_θ_d = p_θ_emp[t]
    E_d   = E_I_given_θ(γ_data[t], p_θ_d)
    E_cm  = E_I_given_θ(comonotone_coupling(p_θ_d, p_I_emp[t]), p_θ_d)
    panel = plot(1:r_θ, E_d; label = "", lw = 2.5, color = :black, marker = :circle, ms = 5,
                 xlabel = "θ", ylabel = (t in (1, 4) ? "E[I | θ]" : ""),
                 title = "t=$t  (ages $(age_labels[t]))", titlefontsize = FS_T,
                 xticks = 1:r_θ, yticks = 1:r_I, ylims = (0.9, r_I + 0.1), legend = false,
                 tickfontsize = FS_TK, xlabelfontsize = FS_G, ylabelfontsize = FS_G)
    plot!(panel, 1:r_θ, E_cm; label = "", lw = 2, color = PAL_GREY, ls = :dash, marker = :diamond, ms = 4)
    panel
end

anchor_legend(mk, ls) = legend_strip(vcat(
    [(; label = "data", color = :black, mk = :circle),
     (; label = "comonotone", color = PAL_GREY, ls = :dash, mk = :diamond)],
    [(; label = adult_names[m], color = out_colors[adult_names[m]], ls = ls, mk = mk) for m in eachindex(Qs)]);
    ncol = 4)

# ── Figure A: DYNAMIC (7-period sequential) E[I|θ], per anchor ──
panels_dyn = []
for t in 1:T-1
    panel = base_EI_panel(t)
    for m in eachindex(Qs)
        plot!(panel, 1:r_θ, E_dyn_tm[t][m]; label = "", lw = 1.8,
              color = out_colors[adult_names[m]], marker = :utriangle, ms = 4)
    end
    push!(panels_dyn, panel)
end
fig_dyn = plot(panels_dyn..., anchor_legend(:utriangle, :solid); layout = EI_lay, size = (1320, 820),
               plot_title = "Optimal investment E[I | θ] — 7-period dynamic (sequential) planner",
               plot_titlefontsize = FS_PT, margin = 3Plots.mm)
save_slide_fig(fig_dyn, "cf_seq_EI_given_theta"; png_dir = OUT_DIR)
println("saved dynamic E[I | θ] figure")

# ── Figure B: myopic STATIC-per-t E[I|θ], per anchor (standalone) ──
panels_stat = []
for t in 1:T-1
    panel = base_EI_panel(t)
    for m in eachindex(Qs)
        plot!(panel, 1:r_θ, E_stat_tm[t][m]; label = "", lw = 1.8, ls = :dot,
              color = out_colors[adult_names[m]], marker = :circle, ms = 4)
    end
    push!(panels_stat, panel)
end
fig_stat = plot(panels_stat..., anchor_legend(:circle, :dot); layout = EI_lay, size = (1320, 820),
                plot_title = "Optimal investment E[I | θ] — myopic static-per-t planner",
                plot_titlefontsize = FS_PT, margin = 3Plots.mm)
save_slide_fig(fig_stat, "cf_seq_EI_static"; png_dir = OUT_DIR)
println("saved static-per-t E[I | θ] figure")

# ── Figure C: static-per-t vs dynamic (anchors collapsed to one colour each) ──
println("\nStatic-per-t vs dynamic: mean over anchors of max_θ |E_stat − E_dyn| by t")
panels_cmp = []
for t in 1:T-1
    panel = base_EI_panel(t)
    gaps = Float64[]
    for m in eachindex(Qs)
        push!(gaps, maximum(abs.(filter(!isnan, E_stat_tm[t][m] .- E_dyn_tm[t][m]))))
        plot!(panel, 1:r_θ, E_stat_tm[t][m]; label = "", lw = 1.6, ls = :dot,
              color = PAL_NAVY, marker = :circle, ms = 3)
        plot!(panel, 1:r_θ, E_dyn_tm[t][m]; label = "", lw = 1.8,
              color = PAL_RUST, marker = :utriangle, ms = 4)
    end
    @printf("  t=%d (ages %-5s): %.3f\n", t, age_labels[t], mean(gaps))
    push!(panels_cmp, panel)
end
leg_cmp = legend_strip([
    (; label = "data", color = :black, mk = :circle),
    (; label = "comonotone", color = PAL_GREY, ls = :dash, mk = :diamond),
    (; label = "static-per-t", color = PAL_NAVY, ls = :dot, mk = :circle),
    (; label = "dynamic (7-period)", color = PAL_RUST, mk = :utriangle)]; ncol = 4)
fig_cmp = plot(panels_cmp..., leg_cmp; layout = EI_lay, size = (1320, 820),
               plot_title = "E[I | θ]: myopic static-per-t (navy, dotted) vs 7-period dynamic (rust, solid)",
               plot_titlefontsize = FS_PT, margin = 3Plots.mm)
save_slide_fig(fig_cmp, "cf_seq_EI_static_vs_dynamic"; png_dir = OUT_DIR)
println("saved static-vs-dynamic E[I | θ] figure")

# ── Figure D: F_θ_t trajectories (data solid vs planner dashed), per anchor ──
traj_panels = []
for m in eachindex(Qs)
    name = adult_names[m]; Fs = seq_results[m].Fs; ts = 1:T
    p = plot(xlabel = "t", ylabel = (m in (1, 4) ? "P(θ_t = k)" : ""), title = "Q = $name",
             titlefontsize = FS_T, legend = false, xticks = (ts, string.(ts)), ylims = (0, 1),
             tickfontsize = FS_TK, xlabelfontsize = FS_G, ylabelfontsize = FS_G)
    for k in 1:r_θ
        plot!(p, ts, [p_θ_emp[t][k] for t in ts]; color = SKILL_COLORS[k], lw = 1.8, ls = :solid, marker = :circle, ms = 3)
        plot!(p, ts, [Fs[t][k] for t in ts];      color = SKILL_COLORS[k], lw = 1.8, ls = :dash, marker = :utriangle, ms = 3)
    end
    push!(traj_panels, p)
end
leg_traj = legend_strip(vcat(
    [(; label = "θ=$k", color = SKILL_COLORS[k], mk = :circle) for k in 1:r_θ],
    [(; label = "data (solid)", color = PAL_GREY, ls = :solid, mk = :circle),
     (; label = "planner (dashed)", color = PAL_GREY, ls = :dash, mk = :utriangle)]); ncol = 3)
fig_Fθ = plot(traj_panels..., leg_traj; layout = (2, 3), size = (1600, 900),
              plot_title = "Skill distribution F_θ_t: data vs 7-period sequential planner",
              plot_titlefontsize = FS_PT, margin = 3Plots.mm)
save_slide_fig(fig_Fθ, "cf_seq_Ftheta_trajectory"; png_dir = OUT_DIR)
println("saved F_θ trajectories")

# ── Figure E: terminal skill distribution F_θ_T (data vs sequential) ──
term_panels = []
for m in eachindex(Qs)
    name = adult_names[m]
    F_data = p_θ_emp[T]; F_seq = seq_results[m].Fs[T]
    xs = 1:r_θ; w = 0.36
    p = bar(xs .- w/2, F_data; bar_width = w, color = PAL_TEAL, label = "",
            xlabel = "θ_T", ylabel = (m in (1, 4) ? "P(θ_T = k)" : ""),
            title = "Q = $name", titlefontsize = FS_T, ylims = (0, 0.62),
            xticks = (xs, string.(xs)), legend = false, framestyle = :box,
            tickfontsize = FS_TK, xlabelfontsize = FS_G, ylabelfontsize = FS_G)
    bar!(p, xs .+ w/2, F_seq; bar_width = w, color = PAL_RUST, label = "")
    for k in 1:r_θ
        annotate!(p, [(k - w/2, F_data[k] + 0.02, text(@sprintf("%.2f", F_data[k]), FS_L, PAL_TEAL))])
        annotate!(p, [(k + w/2, F_seq[k]  + 0.02, text(@sprintf("%.2f", F_seq[k]),  FS_L, PAL_RUST))])
    end
    push!(term_panels, p)
end
leg_term = legend_strip([
    (; label = "data", color = PAL_TEAL, mk = :rect, lw = 0, ms = 10),
    (; label = "sequential planner", color = PAL_RUST, mk = :rect, lw = 0, ms = 10)]; ncol = 2)
fig_term = plot(term_panels..., leg_term; layout = (2, 3), size = (1500, 860),
                plot_title = "Terminal skill distribution F_θ_T (ages 12–13): data vs sequential planner",
                plot_titlefontsize = FS_PT, margin = 3Plots.mm)
save_slide_fig(fig_term, "cf_seq_terminal_Ftheta"; png_dir = OUT_DIR)
println("saved terminal F_θ_T bar chart")

@info "Done."
