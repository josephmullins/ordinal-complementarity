# Both one-sided NDM (Hong–Li) tests for t⁺ and t⁻ from the same bootstrap draws (exploratory).
#   t⁺: orig H0 t⁺ ≥ 0 (some Q gives OC) rejects if √N t̂ < q05; flip H0 t⁺ ≤ 0 rejects if √N t̂ > q95 (confirms OC for some Q)
#   t⁻: orig H0 t⁻ ≤ 0 (some Q gives OS) rejects if √N t̂ > q95; flip H0 t⁻ ≥ 0 rejects if √N t̂ < q05 (confirms OS for some Q)
# ENV BOOTS = "label=file.jld2,label2=file2.jld2" (testfmt layout), NDM_N (9685)
ENV["COMONO_MODE"] = "none"
include(joinpath(@__DIR__, "comonotonicity_test.jl"))
include(joinpath(@__DIR__, "..", "src", "numerical_delta.jl"))
using Statistics, Printf, JLD2
N = parse(Int, get(ENV, "NDM_N", "9685")); α = 0.05
for spec in split(ENV["BOOTS"], ",")
    lab, f = split(spec, "=")
    bc = load(joinpath(@__DIR__, "..", "output", f)); ok = findall(bc["ok"]); r_θ = bc["r_θ"]
    println("# $lab: $f  (B ok = $(length(ok)), N = $N)")
    for (obj, bp, p0, rI) in (("tech", bc["boot_pibar"], bc["pibar0"], bc["r_I"]), ("induced", bc["boot_pibar_ind"], bc["pibar_ind0"], bc["n_y"]))
        gapD(v) = gap_matrices(reconstruct_Pi(v, r_θ, rI)[1])[2]
        for g in eachindex(p0)
            D̂ = gapD(p0[g]); Db = [gapD(bp[g][b, :]) for b in ok]
            for sense in (:max, :min), shape in (:none, :convex, :concave)
                φ = M -> solve_shape_lp(M; sense, shape)[1]; t̂ = φ(D̂); rt = sqrt(N) * t̂
                out = String[]
                for e in (1/3, 1/4)
                    Z = ndm_draws(φ, D̂, Db, N; ε = N^(-e)); lo, hi = quantile(Z, α), quantile(Z, 1 - α)
                    orig = sense === :max ? rt < lo : rt > hi
                    flip = sense === :max ? rt > hi : rt < lo
                    push!(out, @sprintf("%s orig %-5s flip %-5s [q05 %+6.2f q95 %+6.2f]", e == 1/3 ? "e13" : "e14", orig, flip, lo, hi))
                end
                @printf("%-10s %-7s g%d %-3s %-7s t̂=%+.4f √N t̂=%+7.2f | %s\n", lab, obj, g, sense === :max ? "t⁺" : "t⁻", shape, t̂, rt, join(out, " | "))
            end
        end
    end
end
