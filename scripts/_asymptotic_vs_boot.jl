# Asymptotic (Fisher's identity OPG, full 10-block) vs child-bootstrap SEs for the
# ordered-robit fits (2026-11-19). Compares β₅/δ, survivor functions S̄(k|θ,I) and their
# adjacent cross-differences (what the OC/OS tests use), for the grouped fit and the
# converged period-specific fit.  Data ENV as for the fits.  ENV TAG, FITS (grouped,period).
using LinearAlgebra, Random, Statistics, JLD2, Optim, Printf, ForwardDiff
const REPO = joinpath(@__DIR__, "..")
for f in ["parametric_models.jl", "parametric_models_v3.jl", "parametric_models_v3_eps.jl", "prepare_data.jl",
          "estimation.jl", "estimation_multiperiod.jl", "estimation_multiperiod_v3.jl",
          "estimation_multiperiod_v3_eps.jl", "parametric_models_v3_robit.jl", "investment_groups.jl",
          "inference.jl", "inference_common.jl", "inference_v3_eps.jl"]
    include(joinpath(REPO, "src", f))
end
include(joinpath(REPO, "scripts", "_ordrobit_data.jl"))
include(joinpath(REPO, "scripts", "_period_tech.jl"))
const TAG = get(ENV, "TAG", "_rI4_noPIAT3_grp_q5_tidy")
FITS = split(get(ENV, "FITS", "grouped,period"), ","); SEEDF = get(ENV, "SEEDF", "2")

r_θ = 4
surv(Π, k, p, q) = sum(Π[s, p, q] for s in k+1:r_θ)
# survivor free vector layout of delta_cov_survivor_bar: kI outer, kθ, k inner
function survvec(Π, r_I)
    [surv(Π, k, p, q) for k in 1:r_θ-1, p in 1:r_θ, q in 1:r_I] |> vec
end
# contrast: adjacent cross-differences, same (k, θ, I) ordering as elsewhere
function xd_matrix(r_I)
    L = LinearIndices((r_θ - 1, r_θ, r_I))
    rows = Vector{Vector{Float64}}()
    for k in 1:r_θ-1, p in 1:r_θ-1, q in 1:r_I-1
        v = zeros(length(L))
        v[L[k, p+1, q+1]] += 1; v[L[k, p, q]] += 1; v[L[k, p+1, q]] -= 1; v[L[k, p, q+1]] -= 1
        push!(rows, v)
    end
    permutedims(reduce(hcat, rows))
end

function summarize(lab, a, b)
    r = a ./ b; ok = isfinite.(r) .& (b .> 1e-6)
    any(ok) || (@printf("  %-34s no entry with bootstrap SE > 1e-6 (flat)\n", lab); return)
    @printf("  %-34s n=%3d  median asym/boot %.2f  [p10 %.2f, p90 %.2f]  corr(SE) %.2f  mean SE asym %.4f boot %.4f\n",
            lab, count(ok), median(r[ok]), quantile(r[ok], 0.1), quantile(r[ok], 0.9), cor(a[ok], b[ok]), mean(a[ok]), mean(b[ok]))
end

out = Dict{String,Any}()
for fit in FITS
    period = fit == "period"
    fitf = "estimation_v3_eps_ordrobit_$(period ? "period" : "pertheta")$(TAG)_seed$(SEEDF).jld2"
    bootf = "bootstrap_ordrobit$(TAG)$(period ? "_period" : "").jld2"
    F = load(joinpath(REPO, "output", fitf)); pv, psk = F["params_v3_eps"], F["params_skill"]
    BF = load(joinpath(REPO, "output", bootf)); ok = BF["ok"]
    tg = period ? TG_PERIOD : [[1, 2, 3], [4, 5, 6]]
    r_I = psk.r_I
    println("\n", "="^100, "\n", uppercase(fit), "   fit $(fitf)   boot $(bootf) (B ok = $(count(ok)))\n", "="^100)
    t0 = time()
    inf = compute_var_phi_v3_eps(pv, psk, data; transition_groups = tg, include_nuisance_blocks = true, verbose = true)
    @printf("  scores + OPG: %.1f min;  params %d, active %d, effective rank %d, cond %.2e, max|mean score| %.2e\n",
            (time() - t0) / 60, size(inf.S, 1), length(inf.active), inf.rank_effective, inf.cond_num, inf.score_check_max)
    inf4 = compute_var_phi_v3_eps(pv, psk, data; transition_groups = tg, include_nuisance_blocks = false)

    r10 = delta_se_β5_over_δ(inf, pv, psk); r4 = delta_se_β5_over_δ(inf4, pv, psk)
    @printf("  β₅/δ = %.3f   SE: asym 10-block %.3f | asym 4-block %.3f | bootstrap %.3f\n",
            r10.point, r10.se, r4.se, std(BF["boot_ratio"][ok]))

    S10 = delta_cov_survivor_bar(inf, pv, psk; transition_groups = tg)
    S4 = delta_cov_survivor_bar(inf4, pv, psk; transition_groups = tg)
    C = xd_matrix(r_I)
    Dkey = period ? "boot_period" : "boot_pibar"
    res = []
    for g in eachindex(tg)
        D = BF[Dkey][g][ok, :, :, :]
        Sb = reduce(hcat, [survvec(D[b, :, :, :], r_I) for b in 1:size(D, 1)])'   # B × n
        se_b = vec(std(Sb; dims = 1)); se_a = S10[g].ses; se_a4 = S4[g].ses
        xd_b = vec(std(Sb * C'; dims = 1))
        xd_a = sqrt.(max.(diag(C * S10[g].Var_S * C'), 0)); xd_a4 = sqrt.(max.(diag(C * S4[g].Var_S * C'), 0))
        # point-estimate sanity: delta S_hat vs stored Π̄
        P0 = period ? BF["period0"][g] : BF["pibar0"][g]
        gl = period ? "t=$g" : (g == 1 ? "ages 0–5" : "ages 6–13")
        @printf("  -- %s   (max |S_hat − S(Π̄ stored)| = %.1e)\n", gl, maximum(abs.(S10[g].S_hat .- survvec(P0, r_I))))
        summarize("survivor, 10-block", se_a, se_b)
        summarize("survivor, 4-block", se_a4, se_b)
        summarize("cross-diff, 10-block", xd_a, xd_b)
        summarize("cross-diff, 4-block", xd_a4, xd_b)
        # do the 90% significance calls agree?  (|xd| > 1.645 SE)
        x0 = C * survvec(P0, r_I)
        sa = abs.(x0) .> 1.645 .* xd_a; sb = abs.(x0) .> 1.645 .* xd_b
        @printf("     cross-diffs |x̂| > 1.645·SE:  asym %d, boot %d, agree %d/%d\n", count(sa), count(sb), count(sa .== sb), length(sa))
        push!(res, (; g, se_a, se_a4, se_b, xd_a, xd_a4, xd_b, x0))
    end
    out[fit] = (; ratio = r10.point, se_ratio10 = r10.se, se_ratio4 = r4.se, se_ratio_boot = std(BF["boot_ratio"][ok]), res,
                rank = inf.rank_effective, nact = length(inf.active), cond = inf.cond_num)
end
jldsave(joinpath(REPO, "output", "asymptotic_vs_boot$(TAG).jld2"); out)
println("\nSaved → output/asymptotic_vs_boot$(TAG).jld2")
