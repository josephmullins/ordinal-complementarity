using LinearAlgebra, Statistics, Printf, JLD2, CSV, DataFrames

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3_eps.jl"))
include(joinpath(@__DIR__, "..", "src", "counterfactuals.jl"))

# ─────────────────────────────────────────────────────────────────
# STATIC reallocation at every t ∈ 1..6, with the SAME Q (adult-outcome
# function of θ). Reward at period t is  c_t[θ, I] = E[Q(θ_{t+1}) | θ, I]
# using the regime-marginalized Π̄_t for the appropriate transition group.
#
# This is a myopic evaluation — it applies the adult-outcome Q to next-
# period skill regardless of t. Useful for tracking how the instantaneous
# supermodularity pattern and the data-vs-OT gap move with age.
# ─────────────────────────────────────────────────────────────────

const FIT_PATH = get(ENV, "FIT_OVERRIDE", joinpath(@__DIR__, "..", "output", "estimation_v3_eps_reg.jld2"))
const TRANSITION_GROUPS = [[1,2,3], [4,5,6]]

@info "Loading fit…"
fit = load(FIT_PATH)
params    = fit["params_v3_eps"]
gamma_agg = fit["gamma_agg"]            # (T, r_θ, r_I, N)
dims      = fit["data_dims"]
r_θ, r_I, r_P, r_ω = dims.r_θ, dims.r_I, dims.r_P, dims.r_ω
N, T = dims.N, dims.T

π_Pω = Array{Float64}(params.π_Pω)
π_ω  = vec(sum(π_Pω, dims=1))

# ─────────────────────────────────────────────────────────────────
# Π̄_g for each transition group (one per group, reused across t)
# ─────────────────────────────────────────────────────────────────

function build_Pi_bar(spec, π_Pω, r_θ, r_I, r_P, r_ω)
    v = pack_robit_nn_v3(spec); K, ν = spec.K, spec.ν
    Π̄ = zeros(r_θ, r_θ, r_I)
    for kP in 1:r_P, kω in 1:r_ω
        Π_z = _build_robit_nn_v3_pi(v, r_θ, r_I, K, ν, float(kP), float(kω))
        @. Π̄ += π_Pω[kP, kω] * Π_z
    end
    return Π̄
end
Π̄_group = [build_Pi_bar(params.trans_specs[g], π_Pω, r_θ, r_I, r_P, r_ω)
             for g in 1:length(TRANSITION_GROUPS)]

function group_of(t, groups=TRANSITION_GROUPS)
    for (g, ts) in enumerate(groups); t in ts && return g; end
    error("period $t not in any group")
end

# ─────────────────────────────────────────────────────────────────
# Earnings quintile means ($ real, base 2010)
# ─────────────────────────────────────────────────────────────────

function earnings_bin_means()
    ci  = CSV.read(joinpath(@__DIR__, "..", "NLSY", "ChildIncome",
                             "ChildIncome_Clean.csv"), DataFrame)
    cpi = CSV.read(joinpath(@__DIR__, "..", "output", "cpi_u_annual.csv"), DataFrame)
    cpi_tbl = Dict{Int,Float64}(Int(cpi[i, :year]) => Float64(cpi[i, :cpi])
                                 for i in 1:nrow(cpi))
    base = cpi_tbl[2010]
    ci.WageInc_num = [ismissing(x) ? missing : tryparse(Float64, string(x))
                       for x in ci.WageInc]
    ci.YEAR_int = [tryparse(Int, string(y)) for y in ci.YEAR]
    ci.WageInc_real = [
        (w === missing || w === nothing || yr === missing || yr === nothing) ? missing :
            w / (get(cpi_tbl, yr, base) / base)
        for (w, yr) in zip(ci.WageInc_num, ci.YEAR_int)]
    ci_adult = ci[(ci.Age .>= 25) .& (ci.Age .<= 30), :]
    earn = combine(groupby(ci_adult, :KID)) do sub
        w = filter(!isnothing, collect(skipmissing(sub.WageInc_real)))
        isempty(w) ? (; AvgEarnings=missing, N_obs=0) :
                      (; AvgEarnings=mean(w), N_obs=length(w))
    end
    earn = earn[earn.N_obs .>= 2, :]
    v = collect(skipmissing(earn.AvgEarnings))
    cuts = quantile(v, [0.2, 0.4, 0.6, 0.8])
    bin_of(x) = x <= cuts[1] ? 1 : x <= cuts[2] ? 2 :
                x <= cuts[3] ? 3 : x <= cuts[4] ? 4 : 5
    bins = bin_of.(v)
    [mean(v[bins .== q]) for q in 1:5]
end
earn_means = earnings_bin_means()

# ─────────────────────────────────────────────────────────────────
# Q_m(θ) from adult emissions, marginalizing ω at population π_ω
# ─────────────────────────────────────────────────────────────────

function Q_from_adult(A::Array{Float64,3}, ys::Vector{Float64}, π_ω::Vector{Float64})
    κ, r_θ, r_ω = size(A); length(ys) == κ || error("size mismatch")
    Q = zeros(r_θ)
    for j in 1:r_θ, v in 1:r_ω, k in 1:κ
        Q[j] += π_ω[v] * ys[k] * A[k, j, v]
    end
    return Q
end

adult_mats  = Vector{Array{Float64,3}}(params.adult_matrices)
adult_names = ["HS", "College", "EarnQ", "AlwaysEmp", "logEarn"]
values      = [[0.0, 1.0], [0.0, 1.0], earn_means, [0.0, 1.0]]
Qs          = [Q_from_adult(adult_mats[m], values[m], π_ω) for m in 1:4]
push!(Qs, log.(Qs[3]))   # Q(θ) = log(E[earnings | θ])

# ─────────────────────────────────────────────────────────────────
# Supermodularity DIDs (count negative)
# ─────────────────────────────────────────────────────────────────

function supermod_did(c::AbstractMatrix)
    r_θ, r_I = size(c); n_v = 0; n_t = 0; mn = Inf; mx = -Inf
    for i in 1:r_θ-1, j in 1:r_I-1
        d = c[i+1, j+1] + c[i, j] - c[i+1, j] - c[i, j+1]
        mn = min(mn, d); mx = max(mx, d)
        n_t += 1; d < -1e-10 && (n_v += 1)
    end
    (; min=mn, max=mx, n_viol=n_v, n_total=n_t)
end

# ─────────────────────────────────────────────────────────────────
# Marginals and data-coupling at each t
# ─────────────────────────────────────────────────────────────────

function period_state(t)
    γ_data = dropdims(mean(@view(gamma_agg[t, :, :, :]), dims=3), dims=3)
    p_θ, p_I = marginals(γ_data)
    return γ_data, p_θ, p_I
end

# ─────────────────────────────────────────────────────────────────
# Loop over outcomes × periods and print
# ─────────────────────────────────────────────────────────────────

println("\n" * "="^82)
println("Static reallocation at each t (θ_t, I_t) → θ_{t+1},  Q_m = adult-outcome(θ)")
println("N = $N.   Transition groups: $TRANSITION_GROUPS")
println("="^82)

for m in eachindex(Qs)
    name = adult_names[m]
    println("\n── Q_m = $name ──  Q(θ) = ", join([@sprintf("%.4g", q) for q in Qs[m]], ", "))
    @printf "%3s  %-17s  %-17s  %12s  %12s  %12s  %8s  %8s  %8s\n" "t" "p_θ_t" "p_I_t" "W_data" "W_OT" "W_comon" "OT-data%" "cm-data%" "DID<0"
    for t in 1:6
        γ_dt, p_θ, p_I = period_state(t)
        Π̄ = Π̄_group[group_of(t)]
        c = expected_Q(Qs[m], Π̄)
        w_dt = welfare(γ_dt, c)
        w_ot, γ_ot = solve_ot(p_θ, p_I, c; maximize=true)
        γ_cm = comonotone_coupling(p_θ, p_I)
        w_cm = welfare(γ_cm, c)
        sm = supermod_did(c)
        fmt_vec(v) = join([@sprintf("%.2f", x) for x in v], ",")
        @printf "%3d  %-17s  %-17s  %12.5g  %12.5g  %12.5g  %+7.2f%%  %+7.2f%%  %d/%d\n" t fmt_vec(p_θ) fmt_vec(p_I) w_dt w_ot w_cm (100*(w_ot-w_dt)/abs(w_dt)) (100*(w_cm-w_dt)/abs(w_dt)) sm.n_viol sm.n_total
    end
end

println("\n" * "="^82)
@info "Done."
