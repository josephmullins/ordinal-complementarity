using LinearAlgebra, Statistics, Printf, JLD2, CSV, DataFrames

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3_eps.jl"))
include(joinpath(@__DIR__, "..", "src", "counterfactuals.jl"))

# ─────────────────────────────────────────────────────────────────
# Static one-period reallocation counterfactual at t=6 → θ_7.
# Q_m(θ_7) is built from the estimated adult-outcome emissions,
# marginalized over ω using the population ω-marginal.
#
# For each adult outcome, we compare three couplings of (θ_6, I_6)
# sharing the same marginals (from smoothed posteriors at t=6):
#   γ_data   — data-implied joint (Σ_n γ_agg[6, θ, I, n] / N)
#   γ_OT     — LP optimum of E[Q_m(θ_7)] over all couplings
#   γ_comon  — Hoeffding–Fréchet upper coupling (= OT optimum iff c supermodular)
# ─────────────────────────────────────────────────────────────────

const FIT_PATH = get(ENV, "FIT_OVERRIDE", joinpath(@__DIR__, "..", "output", "estimation_v3_eps_reg.jld2"))
const T_REALLOC = 6          # last transition: (θ_6, I_6) → θ_7
const GROUP_OF_T6 = 2        # TRANSITION_GROUPS = [[1,2,3],[4,5,6]]; t=6 in group 2

# Load the fit
@info "Loading fit…"
fit = load(FIT_PATH)
params  = fit["params_v3_eps"]
w_z     = fit["w_z"]
gamma_agg = fit["gamma_agg"]           # (T, r_θ, r_I, N)
dims    = fit["data_dims"]
r_θ, r_I, r_P, r_ω, r_Y = dims.r_θ, dims.r_I, dims.r_P, dims.r_ω, dims.r_Y
N = dims.N

# ─────────────────────────────────────────────────────────────────
# 1. Regime-marginalized Π̄_t for t in group 2 (t=6)
#    Π_z has shape (r_θ', r_θ, r_I); Π̄ = Σ_{kP,kω} π_Pω(kP,kω) Π_{(kP,kω)}
# ─────────────────────────────────────────────────────────────────

trans_spec = params.trans_specs[GROUP_OF_T6]   # RobitNNTransitionV3

# Reconstruct π_Pω as an (r_P, r_ω) matrix (π_Pω stored as such)
π_Pω = Array{Float64}(params.π_Pω)

function build_Pi_bar(trans_spec, π_Pω, r_θ, r_I, r_P, r_ω)
    vec = pack_robit_nn_v3(trans_spec)
    K, ν = trans_spec.K, trans_spec.ν
    Π̄ = zeros(r_θ, r_θ, r_I)
    for kP in 1:r_P, kω in 1:r_ω
        Π_z = _build_robit_nn_v3_pi(vec, r_θ, r_I, K, ν, float(kP), float(kω))
        @. Π̄ += π_Pω[kP, kω] * Π_z
    end
    return Π̄
end
Π̄_6 = build_Pi_bar(trans_spec, π_Pω, r_θ, r_I, r_P, r_ω)

# ω-marginal for the adult-outcome averaging
π_ω = vec(sum(π_Pω, dims=1))
@info "π_ω (population)" π_ω

# ─────────────────────────────────────────────────────────────────
# 2. Per-quintile real-earnings means (for EarnQ bin values)
# ─────────────────────────────────────────────────────────────────

function earnings_bin_means()
    ci = CSV.read(joinpath(@__DIR__, "..", "NLSY", "ChildIncome",
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
        w = collect(skipmissing(sub.WageInc_real))
        w = filter(!isnothing, w)
        isempty(w) ? (; AvgEarnings=missing, N_obs=0) :
                      (; AvgEarnings=mean(w), N_obs=length(w))
    end
    earn = earn[earn.N_obs .>= 2, :]
    v = collect(skipmissing(earn.AvgEarnings))
    cuts = quantile(v, [0.2, 0.4, 0.6, 0.8])
    bin_of(x) = x <= cuts[1] ? 1 : x <= cuts[2] ? 2 :
                x <= cuts[3] ? 3 : x <= cuts[4] ? 4 : 5
    bins = bin_of.(v)
    means = [mean(v[bins .== q]) for q in 1:5]
    return means, cuts
end
earn_means, earn_cuts = earnings_bin_means()
@printf "Earnings quintile cutoffs: \$%.0f, \$%.0f, \$%.0f, \$%.0f\n" earn_cuts...
@printf "Earnings quintile means:   \$%.0f, \$%.0f, \$%.0f, \$%.0f, \$%.0f\n" earn_means...

# ─────────────────────────────────────────────────────────────────
# 3. Build Q_m(θ) = Σ_ω π_ω(ω) · Σ_k y_m(k) · A_m[k, θ, ω]
# ─────────────────────────────────────────────────────────────────

adult_mats  = Vector{Array{Float64,3}}(params.adult_matrices)  # A_m[k, θ, ω]
adult_names = ["HS", "College", "EarnQ", "AlwaysEmp"]
values = [
    [0.0, 1.0],            # HS        — P(grad)
    [0.0, 1.0],            # College   — P(grad)
    earn_means,            # EarnQ     — $/yr real
    [0.0, 1.0],            # AlwaysEmp — P(always employed)
]

function Q_from_adult(A::Array{Float64,3}, ys::Vector{Float64}, π_ω::Vector{Float64})
    κ, r_θ, r_ω = size(A)
    length(ys) == κ || throw(DimensionMismatch())
    Q = zeros(r_θ)
    for j in 1:r_θ, v in 1:r_ω, k in 1:κ
        Q[j] += π_ω[v] * ys[k] * A[k, j, v]
    end
    return Q
end
Qs = [Q_from_adult(adult_mats[m], values[m], π_ω) for m in 1:4]

# ─────────────────────────────────────────────────────────────────
# 4. Marginals and data-implied coupling at t=6
# ─────────────────────────────────────────────────────────────────

γ_data = dropdims(mean(@view(gamma_agg[T_REALLOC, :, :, :]), dims=3), dims=3)
p_θ, p_I = marginals(γ_data)
@assert isapprox(sum(p_θ), 1.0; atol=1e-8)

# ─────────────────────────────────────────────────────────────────
# 5. Supermodularity check (DID on c)
# ─────────────────────────────────────────────────────────────────

function supermod_did(c::AbstractMatrix)
    r_θ, r_I = size(c)
    mn, mx = Inf, -Inf
    n_viol = 0; n_total = 0
    for i in 1:r_θ-1, j in 1:r_I-1
        d = c[i+1, j+1] + c[i, j] - c[i+1, j] - c[i, j+1]
        mn = min(mn, d); mx = max(mx, d)
        n_total += 1
        d < -1e-10 && (n_viol += 1)
    end
    return (min=mn, max=mx, n_viol=n_viol, n_total=n_total)
end

# ─────────────────────────────────────────────────────────────────
# 6. Run OT per adult outcome and report
# ─────────────────────────────────────────────────────────────────

println("\n" * "="^78)
println("Static reallocation at t=6  (θ_6, I_6) → θ_7   N=", N)
println("="^78)
@printf "p_θ_6: %s\n" join([@sprintf("%.3f", x) for x in p_θ], " ")
@printf "p_I_6: %s\n" join([@sprintf("%.3f", x) for x in p_I], " ")

println()
for m in 1:4
    name = adult_names[m]
    Q    = Qs[m]
    c    = expected_Q(Q, Π̄_6)
    sm   = supermod_did(c)
    w_dt = welfare(γ_data, c)
    w_ot, γ_ot = solve_ot(p_θ, p_I, c; maximize=true)
    γ_cm = comonotone_coupling(p_θ, p_I)
    w_cm = welfare(γ_cm, c)
    gap_ot   = (w_ot - w_dt) / abs(w_dt)
    gap_comon = (w_cm - w_dt) / abs(w_dt)
    gap_ot_vs_cm = (w_ot - w_cm) / abs(w_cm)

    println("─"^78)
    @printf "[%s]\n" name
    @printf "  Q(θ)  = %s\n" join([@sprintf("%.3f", q) for q in Q], "  ")
    println("  c[θ, I]:")
    for i in 1:r_θ
        @printf "    θ=%d  %s\n" i join([@sprintf("%9.3f", c[i,j]) for j in 1:r_I], " ")
    end
    @printf "  2x2 DID on c:  min=%.4g  max=%.4g  violations (DID<0): %d/%d\n" sm.min sm.max sm.n_viol sm.n_total
    @printf "  W_data  = %12.5f\n" w_dt
    @printf "  W_OT    = %12.5f    gap vs data  = %+.2f%%\n" w_ot (100*gap_ot)
    @printf "  W_comon = %12.5f    gap vs data  = %+.2f%%    OT vs comon = %+.2f%%\n" w_cm (100*gap_comon) (100*gap_ot_vs_cm)
end

println("\n" * "="^78)
@info "Done."
