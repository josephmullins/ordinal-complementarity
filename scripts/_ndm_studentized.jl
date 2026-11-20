# EXPLORATORY (not in paper): studentized rationalizability statistic.
#   t̃ = max_α min_c (Dα)_c / s_c   (sense :max),   min_α max_c (Dα)_c / s_c  (:min)
# Row scaling by fixed s_c > 0 preserves the sign of every cell, so the null
# (∃ Q in the shape class with all cells ≥ 0, resp. ≤ 0) is unchanged; only the
# statistic's weighting of cells changes.  s_c = sqrt(mean_j Var_b(D*_{cj})) from
# the bootstrap draws, floored at 0.1·median (as in fang_seo studentize_scale).
# Numerical delta method (Hong-Li) on the same draws; compares to unstudentized.
ENV["COMONO_MODE"] = "none"
include(joinpath(@__DIR__, "comonotonicity_test.jl"))

const N = parse(Int, get(ENV, "NDM_N", "9686"))
bc = load(get(ENV, "BOOT", joinpath(@__DIR__, "..", "output", "bootstrap_reg.jld2")))
ok = findall(bc["ok"]); r_θ = bc["r_θ"]
steps = ((:e13, 1/3), (:e14, 1/4))
rows = ["object,group,direction,shape,stat,step,t_hat,rt_hat,crit,reject"]
for (obj, bp, p0, rI) in (("technology", bc["boot_pibar"], bc["pibar0"], bc["r_I"]),
                          ("induced",    bc["boot_pibar_ind"], bc["pibar_ind0"], bc["n_y"]))
    gapD(v) = gap_matrices(reconstruct_Pi(v, r_θ, rI)[1])[2]
    println("\n", "="^92, "\n", uppercase(obj), "\n", "="^92)
    for g in 1:2
        D̂ = gapD(p0[g]); Db = [gapD(bp[g][b, :]) for b in ok]
        V = mean([(M .- D̂) .^ 2 for M in Db])                      # cell × threshold variance
        s = sqrt.(vec(mean(V, dims = 2))); s = max.(s, 0.1 * median(s))
        W = Diagonal(1 ./ s)
        @printf("\nages %s   cell s.e. range [%.4f, %.4f] (ratio %.0f×)\n",
                g == 1 ? "0-5" : "6-13", minimum(s), maximum(s), maximum(s) / minimum(s))
        for sense in (:max, :min), shape in (:none, :convex, :concave)
            dir = sense === :max ? "comonotone" : "counter"
            out = String[]
            for (stat, T) in (("raw", I), ("stud", W))
                φ = M -> solve_shape_lp(T * M; sense, shape)[1]
                t̂ = φ(D̂)
                for (lab, e) in steps
                    r = ndm_test(t̂, ndm_draws(φ, D̂, Db, N; ε = N^(-e)), N; sense)
                    push!(rows, @sprintf("%s,%d,%s,%s,%s,%s,%.6f,%.4f,%.4f,%s",
                                         obj, g, dir, shape, stat, lab, t̂, r.rt, r.c, r.reject))
                    lab === :e13 && push!(out, @sprintf("%s √N t̂=%+7.2f crit=%+7.2f %s", stat, r.rt, r.c,
                                                       r.reject ? "REJECT" : "  -   "))
                end
            end
            @printf("  %-10s %-8s %s   |   %s\n", dir, shape, out...)
        end
    end
end
csv = joinpath(@__DIR__, "..", get(ENV, "OUTDIR", "output/paper_tables_boot"), "comonotonicity_test_ndm_studentized_exploratory.csv")
open(csv, "w") do io; foreach(r -> println(io, r), rows); end
println("\nSaved → ", csv)
