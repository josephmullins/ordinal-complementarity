# EXPLORATORY (not in paper): dual-certificate tests of ordinal
# non-complementarity / non-substitutability (no Q in a class rationalizes).
#
# By LP duality, t⁺ < 0  ⇔  ∃ cell weights p ≥ 0 (Σp = 1) with p'D q < 0 for
# every valuation q in the class.  For FIXED p, "some q works" implies
# max_e p'D q_e ≥ 0 over the extreme points q_e of the class, so we reject
#   H0: some Q gives supermodularity   if p'D q_e is significantly < 0 at EVERY e
# (intersection–union test: each one-sided test at level α, no correction).
# Symmetric for non-substitutability (significantly > 0 at every e).
# Each p'D q_e is linear in Π̄ ⇒ ordinary (basic) bootstrap is valid.
#
# Weights: equal (pre-specified), precision 1/s_c² (consistent plug-in), and
# the LP dual on the full sample (IN-SAMPLE SELECTION — reference only, not valid).
ENV["COMONO_MODE"] = "none"
include(joinpath(@__DIR__, "comonotonicity_test.jl"))

const α_LEVEL = 0.05
bc = load(get(ENV, "BOOT", joinpath(@__DIR__, "..", "output", "bootstrap_reg.jld2")))
ok = findall(bc["ok"]); r_θ = bc["r_θ"]; m = r_θ - 1
EXT = Dict(:none    => [Float64.(1:m .== j) for j in 1:m],
           :concave => [[i <= k ? 1 / k : 0.0 for i in 1:m] for k in 1:m],
           :convex  => [[i > m - k ? 1 / k : 0.0 for i in 1:m] for k in 1:m])

# dual weights on the cell constraints of the (shape-restricted) LP
function dual_weights(D, sense, shape)
    _, _, du, _ = solve_shape_lp(D; sense, shape)
    p = abs.(du); p ./ sum(p)
end

rows = ["object,group,side,shape,weights,max_or_min_stat,bound,reject,n_ext"]
for (obj, bp, p0, rI) in (("technology", bc["boot_pibar"], bc["pibar0"], bc["r_I"]),
                          ("induced",    bc["boot_pibar_ind"], bc["pibar_ind0"], bc["n_y"]))
    gapD(v) = gap_matrices(reconstruct_Pi(v, r_θ, rI)[1])[2]
    println("\n", "="^96, "\n", uppercase(obj), "   (reject = certificate significant at every extreme point, one-sided 5%)\n", "="^96)
    for g in 1:2
        D̂ = gapD(p0[g]); Db = [gapD(bp[g][b, :]) for b in ok]
        s = sqrt.(vec(mean(mean([(M .- D̂) .^ 2 for M in Db]), dims = 2))); s = max.(s, 0.1 * median(s))
        println("\nages ", g == 1 ? "0-5" : "6-13")
        for (side, sense, word) in (("non-complementary", :max, "all p'Dq<0"), ("non-substitutable", :min, "all p'Dq>0"))
            for shape in (:none, :concave, :convex)
                # the hard case for each side: concave Q for complementarity, convex Q for substitutability
                side == "non-complementary" && shape == :convex && continue
                side == "non-substitutable" && shape == :concave && continue
                for (wlab, p) in (("equal", fill(1 / size(D̂, 1), size(D̂, 1))),
                                  ("precision", (w = 1 ./ s .^ 2; w ./ sum(w))),
                                  ("dual-insample", dual_weights(D̂, sense, shape)))
                    Ŝ = [p' * D̂ * q for q in EXT[shape]]
                    Sb = [[p' * M * q for M in Db] for q in EXT[shape]]
                    if sense === :max   # need every p'Dq_e significantly < 0: upper bounds
                        bnd = [2Ŝ[e] - quantile(Sb[e], α_LEVEL) for e in eachindex(Ŝ)]
                        worst, wb = findmax(Ŝ)[1], maximum(bnd); rej = wb < 0
                    else                # need every p'Dq_e significantly > 0: lower bounds
                        bnd = [2Ŝ[e] - quantile(Sb[e], 1 - α_LEVEL) for e in eachindex(Ŝ)]
                        worst, wb = findmin(Ŝ)[1], minimum(bnd); rej = wb > 0
                    end
                    @printf("  %-18s %-8s %-14s least-favourable p'Dq = %+.4f   bound %+.4f   %s\n",
                            side, shape, wlab, worst, wb, rej ? "REJECT" : "  -")
                    push!(rows, @sprintf("%s,%d,%s,%s,%s,%.6f,%.6f,%s,%d", obj, g, side, shape, wlab, worst, wb, rej, length(Ŝ)))
                end
            end
        end
    end
end
csv = joinpath(@__DIR__, "..", get(ENV, "OUTDIR", "output/paper_tables_boot"), "certificate_test_exploratory.csv")
open(csv, "w") do io; foreach(r -> println(io, r), rows); end
println("\nSaved → ", csv)
