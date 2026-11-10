using LinearAlgebra, Statistics, Printf, JLD2, CSV, DataFrames, Plots

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3_eps.jl"))
include(joinpath(@__DIR__, "..", "src", "counterfactuals.jl"))
include(joinpath(@__DIR__, "..", "src", "slides_palette.jl"))

# ─────────────────────────────────────────────────────────────────
# 2-period E[I | θ] figure with line segments colored by local
# super/sub-modularity of the step's regime-marginalized Π̄.
#
# For each θ-edge iθ ∈ {1..r_θ-1}, score
#   c(iθ) = (#positive − #negative) / total
# across DIDs at (iθ, iI=1..r_I-1, k=1..r_θ-1):
#   DID(k,iθ,iI) = S̄(k|θ+1,I+1) + S̄(k|θ,I) − S̄(k|θ+1,I) − S̄(k|θ,I+1).
# Color blue for c=+1, red for c=−1, purple for 0.
# ─────────────────────────────────────────────────────────────────

const FIT_PATH = get(ENV, "FIT_OVERRIDE", joinpath(@__DIR__, "..", "output", "estimation_v3_eps_reg.jld2"))
const TRANSITION_GROUPS = [[1,2,3], [4,5,6]]
const OUT_DIR = joinpath(@__DIR__, "..", "output", "counterfactuals")
isdir(OUT_DIR) || mkpath(OUT_DIR)

fit       = load(FIT_PATH)
params    = fit["params_v3_eps"]
gamma_agg = fit["gamma_agg"]
dims      = fit["data_dims"]
r_θ, r_I, r_P, r_ω, r_Y = dims.r_θ, dims.r_I, dims.r_P, dims.r_ω, dims.r_Y
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
function cumulative_transition(Π̄_1::AbstractArray{<:Real,3}, n_apply::Int)
    r_θ_out, r_θ, r_I = size(Π̄_1); r_θ_out == r_θ || error("mismatch")
    Π̄_cum = zeros(r_θ, r_θ, r_I)
    for j in 1:r_I
        Π̄_cum[:, :, j] = Π̄_1[:, :, j]^n_apply
    end
    Π̄_cum
end
Π̄_step = [cumulative_transition(Π̄_1step[g], 1) for g in 1:2]   # illustrative 2-period: 1-step transitions

# Averaged empirical marginals per age group
γ_data_all = [dropdims(mean(@view(gamma_agg[t, :, :, :]), dims=3), dims=3) for t in 1:T]
avg(v) = sum(v) / length(v)
p_θ_steps = [avg([vec(sum(γ_data_all[t], dims=2)) for t in TRANSITION_GROUPS[g]]) for g in 1:2]
p_I_steps = [avg([vec(sum(γ_data_all[t], dims=1)) for t in TRANSITION_GROUPS[g]]) for g in 1:2]
γ_data_step = [avg([γ_data_all[t] for t in TRANSITION_GROUPS[g]]) for g in 1:2]

# Earnings + Q
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
    κ, r_θ, r_ω = size(A); Q = zeros(r_θ)
    for j in 1:r_θ, v in 1:r_ω, k in 1:κ
        Q[j] += π_ω_vec[v] * ys[k] * A[k, j, v]
    end
    Q
end
adult_mats  = Vector{Array{Float64,3}}(params.adult_matrices)
adult_names = ["HS", "College", "EarnQ", "AlwaysEmp", "logEarn"]
values      = [[0.0, 1.0], [0.0, 1.0], earn_means, [0.0, 1.0]]
Qs          = [Q_from_adult(adult_mats[m], values[m], π_ω) for m in 1:4]
push!(Qs, log.(Qs[3]))

# Sequential LP per outcome (2-period)
seq_gammas = Dict{String, Vector{Matrix{Float64}}}()
for m in eachindex(Qs)
    _, γs, _ = solve_sequential_ot(p_θ_steps[1], p_I_steps, Π̄_step, Qs[m])
    seq_gammas[adult_names[m]] = γs
end

# ─────────────────────────────────────────────────────────────────
# Local complementarity score per θ-edge on a given Π̄
# ─────────────────────────────────────────────────────────────────

function edge_complementarity_scores(Π̄::AbstractArray{<:Real,3}; tol::Real=1e-10)
    r_θ_out, r_θ, r_I = size(Π̄)
    r_θ_out == r_θ || error("Π̄ must be square in first two dims")
    c = zeros(r_θ - 1)
    for iθ in 1:(r_θ - 1)
        pos = 0; neg = 0; tot = 0
        for iI in 1:(r_I - 1), k in 1:(r_θ - 1)
            S = (θ_, I_) -> sum(Π̄[j, θ_, I_] for j in (k+1):r_θ_out)
            did = S(iθ+1, iI+1) + S(iθ, iI) - S(iθ+1, iI) - S(iθ, iI+1)
            tot += 1
            if     did >  tol; pos += 1
            elseif did < -tol; neg += 1
            end
        end
        c[iθ] = tot == 0 ? 0.0 : (pos - neg) / tot
    end
    c
end

function E_I_given_θ(γ, p_θ)
    out = zeros(length(p_θ))
    for j in eachindex(p_θ)
        p_θ[j] < 1e-12 && (out[j] = NaN; continue)
        s = 0.0; for i in 1:size(γ, 2); s += i * γ[j, i]; end
        out[j] = s / p_θ[j]
    end
    out
end

# ─────────────────────────────────────────────────────────────────
# Colored-segment plotting helper
# ─────────────────────────────────────────────────────────────────

const CGRAD_CS = cgrad([PAL_RUST, PAL_PLUM, PAL_TEAL])

"""
Plot `ys` against `xs` on `panel`, with each segment (i, i+1) colored by
`c_seg[i] ∈ [-1, 1]` through CGRAD_CS. Line-style and markers controllable.
Add a single legend entry via `label`.
"""
function plot_colored_line!(panel, xs::AbstractVector, ys::AbstractVector,
                             c_seg::AbstractVector;
                             lw=3.2, ls=:solid, marker=:circle, ms=6,
                             label=nothing)
    n = length(xs)
    length(c_seg) == n - 1 || error("c_seg must have length length(xs)-1")
    # Invisible placeholder for legend entry with neutral gray
    if label !== nothing
        plot!(panel, [NaN], [NaN]; color=:black, ls=ls, lw=lw,
              marker=marker, ms=ms, label=label)
    end
    for i in 1:(n-1)
        col = get(CGRAD_CS, (c_seg[i] + 1) / 2)
        plot!(panel, xs[i:i+1], ys[i:i+1]; color=col, lw=lw, ls=ls,
              marker=:none, label=false)
    end
    for i in 1:n
        col_here = i <= n-1 ? get(CGRAD_CS, (c_seg[i] + 1) / 2) :
                              get(CGRAD_CS, (c_seg[end] + 1) / 2)
        scatter!(panel, [xs[i]], [ys[i]]; color=col_here, markersize=ms,
                 markerstrokewidth=0.4, label=false, markershape=marker)
    end
end

# ─────────────────────────────────────────────────────────────────
# Build figure: 2 panels (step 1, step 2), each with
#   data, comonotone, and sequential OT per outcome,
# all segments colored by that step's local complementarity scores.
# ─────────────────────────────────────────────────────────────────

STEP_LABELS = ["step 1: ages 0–5",
               "step 2: ages 6–11"]
c_scores = [edge_complementarity_scores(Π̄_step[s]) for s in 1:2]
@info "Edge complementarity scores (step 1)" c_scores[1]
@info "Edge complementarity scores (step 2)" c_scores[2]

const COLOR_REVEAL_OUTCOMES = ["HS", "College", "EarnQ", "AlwaysEmp"]
const COLOR_PRETTY = Dict("HS" => "planner: HS completion",
                           "College" => "planner: College completion",
                           "EarnQ" => "planner: adult earnings",
                           "AlwaysEmp" => "planner: employment")

panels = []
for s in 1:2
    p_θ_d   = p_θ_steps[s]
    γ_d     = γ_data_step[s]
    γ_cm    = comonotone_coupling(p_θ_d, p_I_steps[s])
    E_data  = E_I_given_θ(γ_d,  p_θ_d)
    E_comon = E_I_given_θ(γ_cm, p_θ_d)
    panel = plot(xlabel="θ", ylabel=(s==1 ? "E[I | θ]" : ""),
                  title=STEP_LABELS[s], titlefontsize=10,
                  xticks=1:r_θ, yticks=1:r_I, ylims=(0.9, r_I+0.1),
                  legend=(s==1 ? :topleft : false), legendfontsize=8,
                  tickfontsize=9, xlabelfontsize=10, ylabelfontsize=10,
                  framestyle=:box)
    # Data and comonotone: plain reference lines (single color, no segment coloring)
    plot!(panel, 1:r_θ, E_data;
          label="data", lw=3.2, color=PAL_TEXT, marker=:circle, ms=6)
    plot!(panel, 1:r_θ, E_comon;
          label="comonotone benchmark", lw=2.8, color=PAL_GREY, ls=:dash,
          marker=:diamond, ms=5)
    # Sequential OT per outcome: segment-colored by local complementarity
    for name in COLOR_REVEAL_OUTCOMES
        E_seq = E_I_given_θ(seq_gammas[name][s],
                              s == 1 ? p_θ_d :
                                       vec(sum(seq_gammas[name][2], dims=2)))
        plot_colored_line!(panel, 1:r_θ, E_seq, c_scores[s];
                             lw=3.0, ls=:solid, marker=:utriangle, ms=5,
                             label=(s==1 ? COLOR_PRETTY[name] : nothing))
    end
    push!(panels, panel)
end
fig = plot(panels...; layout=(1, 2), size=(1400, 560),
           top_margin=2Plots.mm, bottom_margin=3Plots.mm,
           left_margin=6Plots.mm, right_margin=3Plots.mm)

# Add a standalone color-bar panel for the red↔blue scale, with its own title
colorbar_panel = heatmap([-1.0:0.01:1.0;]', color=CGRAD_CS,
                          xticks=(1:50:201, ["-1 (all sub)", "-0.5", "0 (mixed)",
                                              "0.5", "+1 (all super)"]),
                          yticks=false, clims=(-1, 1), colorbar=false,
                          title="segment colour = complementarity score  c(iθ)",
                          titlefontsize=9, size=(1400, 70),
                          framestyle=:box,
                          top_margin=1Plots.mm, bottom_margin=4Plots.mm,
                          left_margin=6Plots.mm, right_margin=3Plots.mm)
fig_combined = plot(fig, colorbar_panel; layout=grid(2, 1, heights=[0.85, 0.15]),
                     size=(1400, 640))
save_slide_fig(fig_combined, "cf_2p_EI_given_theta_colored"; png_dir=OUT_DIR)
println("saved cf_2p_EI_given_theta_colored.{png,pdf}")

@info "Done."
