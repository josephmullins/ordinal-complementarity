# Monte Carlo: size of the one-sided NDM rationalizability tests in BOTH directions at the
# boundary t = 0 (2026-11-19, exploratory).
# DGP "modular": Π(·|θ,I) = ½ A(·|θ) + ½ C(·|I), built from the school-age grouped Π̄ of FITF
#   (A = Π̄ averaged over I, C = Π̄ averaged over θ). The survivor function is additive in θ and I, so every
#   cross-difference is 0 under every Q: t⁺ = t⁻ = 0 (boundary of every null).
# DGP "kink": the early-childhood Π̄ with every cross-difference lowered by t⁺(Π̄), so t⁺ = 0 exactly
#   while the cross-differences stay non-zero (only some bind at the optimum Q).
# Estimator: x̂ ~ N(x0, V), V = asymptotic Var of the free Π̄ entries of the matching group (from
# INF file). NDM draws: x*_b ~ N(x̂, V) (as in _age_inference_ndm.jl). Rejection frequencies of
#   orig  t⁺: √N t̂ < q05  (H0 t⁺ ≥ 0)      flip t⁺: √N t̂ > q95  (H0 t⁺ ≤ 0)
#   orig  t⁻: √N t̂ > q95  (H0 t⁻ ≤ 0)      flip t⁻: √N t̂ < q05  (H0 t⁻ ≥ 0)
# ENV FITF, INF, R (400), BB (300)
ENV["COMONO_MODE"] = "none"
include(joinpath(@__DIR__, "comonotonicity_test.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3_robit.jl"))
include(joinpath(@__DIR__, "..", "src", "numerical_delta.jl"))
using JLD2, Random, Statistics, Printf, LinearAlgebra
const OUTD = joinpath(@__DIR__, "..", "output")
FITF = get(ENV, "FITF", "estimation_v3_eps_ordrobit_pertheta_rI4_noPIAT3_grp_q5a_tidy_seed1.jld2")
INF = get(ENV, "INF", "age_inference_pertheta_rI4_noPIAT3_grp_q5a_tidy_seed1.jld2")
R = parse(Int, get(ENV, "R", "400")); BB = parse(Int, get(ENV, "BB", "300")); N = 9685
F = load(joinpath(OUTD, FITF)); pv = F["params_v3_eps"]; r_θ = 4; r_I = F["data_dims"].r_I
TG = [[1, 2, 3], [4, 5, 6]]
Pb = marginalize_Pi_v3(pv.trans_specs, TG, pv.π_Pω, r_θ, r_I, pv.r_P, pv.r_ω)
IR = load(joinpath(OUTD, INF))["R"]

function modular_of(Π)
    A = dropdims(mean(Π; dims = 3); dims = 3); C = dropdims(mean(Π; dims = 2); dims = 2)
    [0.5A[s, p] + 0.5C[s, q] for s in 1:r_θ, p in 1:r_θ, q in 1:r_I]
end
gapD(Π) = gap_matrices(Π)[2]
tplus(Π) = solve_shape_lp(gapD(Π); sense = :max, shape = :none)[1]

dgps = Dict{String,Tuple{Vector{Float64},Matrix{Float64}}}()
dgps["modular (school-age base, V of ages 6–13)"] = (Pi_to_free(modular_of(Pb[4]), r_θ, r_I), Matrix(IR[2].Var))
# kink: subtract λ·(p − 2.5)(q − 2.5) from every survivor S_k(θ_p, I_q). That lowers every adjacent
# cross-difference, at every threshold, by exactly λ, so t⁺(λ) = t⁺(Π̄) − λ; λ = t⁺(Π̄) puts the
# early-age technology on the boundary t⁺ = 0 with its cross-differences otherwise intact.
# (If some cell would go negative, first shrink Π̄ toward its modular projection by w: D → w·D.)
for w in (1.0, 0.75, 0.5, 0.25)
    M = modular_of(Pb[1]); Π1 = M .+ w .* (Pb[1] .- M); λ = tplus(Π1)
    λ > 0 || error("early Π̄ has t⁺ ≤ 0")
    for p in 1:r_θ, q in 1:r_I
        c = λ * (p - (r_θ + 1) / 2) * (q - (r_I + 1) / 2)
        Π1[1, p, q] += c; Π1[r_θ, p, q] -= c
    end
    minimum(Π1) > 1e-4 || continue
    @printf("kink DGP: shrink w = %.2f, shift λ = %.4f, min Π = %.4f\n", w, λ, minimum(Π1))
    dgps["kink (early-age Π̄, w=$(w), shifted to t⁺ = 0; V of ages 0–5)"] = (Pi_to_free(Π1, r_θ, r_I), Matrix(IR[1].Var))
    break
end
println("R = $R replications, $BB NDM draws each, N = $N, nominal 5%")
for (lab, (x0, V)) in sort(collect(dgps))
    L = cholesky(Symmetric(V + 1e-14I)).L; n = length(x0)
    D0 = gapD(reconstruct_Pi(x0, r_θ, r_I)[1])
    @printf("\n%s   true t⁺ = %+.2e, t⁻ = %+.2e\n", lab, solve_shape_lp(D0; sense = :max, shape = :none)[1], solve_shape_lp(D0; sense = :min, shape = :none)[1])
    rng = MersenneTwister(7)
    for e in (1/3, 1/4)
        rej = Dict(k => 0 for k in ("t⁺ orig", "t⁺ flip", "t⁻ orig", "t⁻ flip"))
        for r in 1:R
            x̂ = x0 .+ L * randn(rng, n); D̂ = gapD(reconstruct_Pi(x̂, r_θ, r_I)[1])
            Db = [gapD(reconstruct_Pi(x̂ .+ L * randn(rng, n), r_θ, r_I)[1]) for _ in 1:BB]
            for sense in (:max, :min)
                φ = M -> solve_shape_lp(M; sense, shape = :none)[1]; rt = sqrt(N) * φ(D̂)
                Z = ndm_draws(φ, D̂, Db, N; ε = N^(-e)); q05, q95 = quantile(Z, 0.05), quantile(Z, 0.95)
                s = sense === :max ? "t⁺" : "t⁻"
                rej["$s orig"] += sense === :max ? (rt < q05) : (rt > q95)
                rej["$s flip"] += sense === :max ? (rt > q95) : (rt < q05)
            end
        end
        @printf("  ε = N^-%s:  ", e == 1/3 ? "1/3" : "1/4")
        for k in ("t⁺ orig", "t⁺ flip", "t⁻ orig", "t⁻ flip"); @printf("%s %.3f   ", k, rej[k] / R); end
        println(); flush(stdout)
    end
end
