# EXPLORATORY (not in paper): "some Q works" tests at PRE-SPECIFIED valuations.
# For a fixed Q (increments α), the 12 cell cross-differences Dα are linear in Π̄.
#   supermodular-rationalizable via Q : every cell significantly > 0
#   submodular-rationalizable  via Q : every cell significantly < 0
# Intersection–union over cells (each one-sided at level a, no correction).
# Claiming rationalizability from ANY of K pre-specified Q's needs a = α/K
# (Bonferroni across the family).  Bounds: estimate ± z·(bootstrap s.e.).
# Q̂ from the LP on the full sample is shown as an (invalid) in-sample reference.
ENV["COMONO_MODE"] = "none"
include(joinpath(@__DIR__, "comonotonicity_test.jl"))
using Distributions

const α_LEVEL = 0.05
bc = load(get(ENV, "BOOT", joinpath(@__DIR__, "..", "output", "bootstrap_reg.jld2")))
ok = findall(bc["ok"]); r_θ = bc["r_θ"]
FAMILY = [("(0,1,1,1)",   [1.0, 0, 0]), ("(0,0,1,1)", [0, 1.0, 0]), ("(0,0,0,1)", [0, 0, 1.0]),
          ("linear",      [1, 1, 1] ./ 3), ("(0,½,1,1)", [0.5, 0.5, 0]), ("(0,0,½,1)", [0, 0.5, 0.5])]
K = length(FAMILY)
z1, zK = quantile(Normal(), 1 - α_LEVEL), quantile(Normal(), 1 - α_LEVEL / K)

rows = ["object,group,side,Q,n_right_sign_est,n_sig_unadj,n_sig_bonf,min_abs_z,pass_unadj,pass_bonf,in_sample"]
for (obj, bp, p0, rI) in (("technology", bc["boot_pibar"], bc["pibar0"], bc["r_I"]),
                          ("induced",    bc["boot_pibar_ind"], bc["pibar_ind0"], bc["n_y"]))
    gapD(v) = gap_matrices(reconstruct_Pi(v, r_θ, rI)[1])[2]
    println("\n", "="^100, "\n", uppercase(obj), "   (cells right-signed / significant unadj / significant Bonferroni-$K, of 12)\n", "="^100)
    for g in 1:2
        D̂ = gapD(p0[g]); Db = [gapD(bp[g][b, :]) for b in ok]
        println("\nages ", g == 1 ? "0-5" : "6-13")
        for (side, sgn, sense) in (("supermodular", +1, :max), ("submodular", -1, :min))
            α̂ = solve_shape_lp(D̂; sense)[2]
            for (qlab, a, insample) in vcat([(l, a, false) for (l, a) in FAMILY], [("Q̂ in-sample", α̂, true)])
                c = D̂ * a
                se = vec(std(reduce(hcat, [M * a for M in Db]), dims = 2))
                z = sgn .* c ./ se                                    # >0 = right sign
                n_est, n1, nK = count(>(0), z), count(>(z1), z), count(>(zK), z)
                pass1, passK = n1 == length(z), nK == length(z)
                @printf("  %-12s %-13s right-signed %2d  sig %2d  sig-Bonf %2d   weakest cell z = %+5.2f   %s\n",
                        side, qlab, n_est, n1, nK, minimum(z),
                        insample ? (pass1 ? "(in-sample: passes)" : "(in-sample)") :
                        passK ? "RATIONALIZABLE (Bonf)" : pass1 ? "passes unadjusted only" : "-")
                push!(rows, @sprintf("%s,%d,%s,%s,%d,%d,%d,%.3f,%s,%s,%s", obj, g, side, qlab,
                                     n_est, n1, nK, minimum(z), pass1, passK, insample))
            end
        end
    end
end
csv = joinpath(@__DIR__, "..", get(ENV, "OUTDIR", "output/paper_tables_boot"), "fixedQ_rationalize_test_exploratory.csv")
open(csv, "w") do io; foreach(r -> println(io, r), rows); end
println("\nSaved → ", csv)
