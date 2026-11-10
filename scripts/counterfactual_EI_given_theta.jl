using LinearAlgebra, Statistics, Printf, JLD2, CSV, DataFrames, Plots

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3_eps.jl"))
include(joinpath(@__DIR__, "..", "src", "counterfactuals.jl"))

# ─────────────────────────────────────────────────────────────────
# E[I | θ] under data, comonotone, and OT (one per adult-outcome Q).
# Positive slope in θ = comonotone; negative slope = anti-comonotone.
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
group_of(t) = t <= 3 ? 1 : 2

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

function E_I_given_θ(γ, p_θ)
    r_θ, r_I = size(γ)
    out = zeros(r_θ)
    for j in 1:r_θ
        p_θ[j] < 1e-12 && (out[j] = NaN; continue)
        s = 0.0
        for i in 1:r_I; s += i * γ[j, i]; end
        out[j] = s / p_θ[j]
    end
    out
end

function period_state(t)
    γ_data = dropdims(mean(@view(gamma_agg[t, :, :, :]), dims=3), dims=3)
    p_θ, p_I = marginals(γ_data)
    γ_data, p_θ, p_I
end

# Collect E[I|θ] across periods
E_data  = zeros(6, r_θ)
E_comon = zeros(6, r_θ)
E_OT    = [zeros(6, r_θ) for _ in eachindex(Qs)]   # one per outcome
for t in 1:6
    γ_d, p_θ, p_I = period_state(t)
    γ_c = comonotone_coupling(p_θ, p_I)
    E_data[t, :]  = E_I_given_θ(γ_d, p_θ)
    E_comon[t, :] = E_I_given_θ(γ_c, p_θ)
    for m in eachindex(Qs)
        c = expected_Q(Qs[m], Π̄_group[group_of(t)])
        _, γ_o = solve_ot(p_θ, p_I, c; maximize=true)
        E_OT[m][t, :] = E_I_given_θ(γ_o, p_θ)
    end
end

# ─────────────────────────────────────────────────────────────────
# Plot: one panel per t, lines for data / comon / 4 OT
# ─────────────────────────────────────────────────────────────────

age_labels = ["0-1", "2-3", "4-5", "6-7", "8-9", "10-11"]
out_colors = Dict("HS"=>:orange, "College"=>:red, "EarnQ"=>:purple, "AlwaysEmp"=>:brown, "logEarn"=>:teal)

panels = []
for t in 1:6
    p = plot(1:r_θ, E_data[t, :];
             label = t == 1 ? "data" : "",
             lw=2.5, color=:black, marker=:circle, ms=4,
             xlabel="θ", ylabel=(t == 1 ? "E[I | θ]" : ""),
             title="t=$t  (ages $(age_labels[t]))", titlefontsize=9,
             xticks=1:r_θ, yticks=1:r_I, ylims=(0.9, r_I+0.1),
             legendfontsize=6, legend=(t == 1 ? :topleft : false),
             tickfontsize=7, xlabelfontsize=8, ylabelfontsize=8)
    plot!(p, 1:r_θ, E_comon[t, :];
          label = t == 1 ? "comonotone" : "",
          lw=2, color=:gray, ls=:dash, marker=:diamond, ms=3.5)
    for m in eachindex(Qs)
        plot!(p, 1:r_θ, E_OT[m][t, :];
              label = t == 1 ? "OT: $(adult_names[m])" : "",
              lw=1.6, color=out_colors[adult_names[m]], marker=:utriangle, ms=3)
    end
    push!(panels, p)
end

fig = plot(panels...; layout=(2, 3), size=(1300, 720),
           plot_title="E[I | θ] under data, comonotone, and OT optima",
           plot_titlefontsize=12, margin=3Plots.mm)
path = joinpath(OUT_DIR, "cf_EI_given_theta.png")
savefig(fig, path)
println("saved $path")

# Also dump a small table
println("\nE[I | θ] summary (θ=1..$r_θ columns)")
for t in 1:6
    @printf "t=%d  data   = %s\n" t join([@sprintf("%.3f", x) for x in E_data[t, :]], "  ")
    @printf "     comon  = %s\n"  join([@sprintf("%.3f", x) for x in E_comon[t, :]], "  ")
    for m in eachindex(E_OT)
        @printf "     OT(%-9s) = %s\n" adult_names[m] join([@sprintf("%.3f", x) for x in E_OT[m][t, :]], "  ")
    end
end
