using LinearAlgebra, Statistics, Printf, JLD2, CSV, DataFrames, Plots

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3_eps.jl"))
include(joinpath(@__DIR__, "..", "src", "counterfactuals.jl"))

# ─────────────────────────────────────────────────────────────────
# Heatmaps of the static-per-t OT optimum γ_t*[θ, I] across t=1..6,
# one figure per adult outcome, with γ_data on a second row for contrast.
# Illustrative — emphasises *shape* of reallocation, not welfare magnitudes.
# ─────────────────────────────────────────────────────────────────

const FIT_PATH = get(ENV, "FIT_OVERRIDE", joinpath(@__DIR__, "..", "output", "estimation_v3_eps_reg.jld2"))
const TRANSITION_GROUPS = [[1,2,3], [4,5,6]]
const OUT_DIR = joinpath(@__DIR__, "..", "output", "counterfactuals")
isdir(OUT_DIR) || mkpath(OUT_DIR)

fit = load(FIT_PATH)
params    = fit["params_v3_eps"]
gamma_agg = fit["gamma_agg"]
dims      = fit["data_dims"]
r_θ, r_I, r_P, r_ω = dims.r_θ, dims.r_I, dims.r_P, dims.r_ω

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
group_of(t) = any(t in g for g in TRANSITION_GROUPS[1:1]) ? 1 : 2

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

# Q_m(θ)
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

function period_state(t)
    γ_data = dropdims(mean(@view(gamma_agg[t, :, :, :]), dims=3), dims=3)
    p_θ, p_I = marginals(γ_data)
    γ_data, p_θ, p_I
end

# ─────────────────────────────────────────────────────────────────
# Build heatmap panels and save one figure per outcome
# Rows: γ_data (top) vs γ_OT (bottom). Cols: t=1..6.
# Shared colour scale per figure = max mass across all its panels.
# ─────────────────────────────────────────────────────────────────

function panel(γ, title_str; clim, show_ylabel=false)
    heatmap(1:r_I, 1:r_θ, γ;
            c=:viridis, clims=clim, colorbar=false,
            xticks=(1:r_I, string.(1:r_I)),
            yticks=(1:r_θ, string.(1:r_θ)),
            title=title_str, titlefontsize=8,
            xlabel="I", ylabel=(show_ylabel ? "θ" : ""),
            aspect_ratio=:equal, framestyle=:box, tickfontsize=6,
            xlabelfontsize=7, ylabelfontsize=7)
end

for m in eachindex(Qs)
    name = adult_names[m]
    γ_ots  = Vector{Matrix{Float64}}(undef, 6)
    γ_dats = Vector{Matrix{Float64}}(undef, 6)
    for t in 1:6
        γ_d, p_θ, p_I = period_state(t)
        c = expected_Q(Qs[m], Π̄_group[group_of(t)])
        _, γ_o = solve_ot(p_θ, p_I, c; maximize=true)
        γ_ots[t]  = γ_o
        γ_dats[t] = γ_d
    end
    # shared colour scale
    mx = maximum(maximum.(γ_ots) ∪ maximum.(γ_dats))
    clim = (0.0, mx)

    plots_top = [panel(γ_dats[t], "data, t=$t"; clim, show_ylabel=(t==1)) for t in 1:6]
    plots_bot = [panel(γ_ots[t],  "OT, t=$t";   clim, show_ylabel=(t==1)) for t in 1:6]
    fig = plot(plots_top..., plots_bot...; layout=(2, 6), size=(1500, 520),
               plot_title="Q = $name    γ[θ, I] joint mass    (max = $(round(mx; digits=3)))",
               plot_titlefontsize=11, margin=2Plots.mm)
    path = joinpath(OUT_DIR, "cf_adult_$(lowercase(name))_t1to6.png")
    savefig(fig, path)
    println("  saved $path")
end

@info "Figures written to $OUT_DIR"
