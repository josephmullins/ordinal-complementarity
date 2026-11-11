# Diagnostic: standard bootstrap vs numerical delta method (Hong-Li) for t^±,
# on the same B=200 child-bootstrap draws.  Shape = :none only.
#   S_b = √N (t*_b − t̂)                         (standard bootstrap root)
#   D_b = [φ(D̂ + ε√N(D*_b − D̂)) − t̂]/ε, ε=N^-1/3 (numerical delta)
# plus: # binding cells at the estimate, # within the NDM's implicit tolerance,
# and the three CIs (percentile = current paper, basic, NDM).
ENV["COMONO_MODE"] = "none"
include(joinpath(@__DIR__, "comonotonicity_test.jl"))

const N = 9686
bc = load(joinpath(@__DIR__, "..", "output", "bootstrap_reg.jld2"))
ok = findall(bc["ok"]); r_θ = bc["r_θ"]
q(x, p) = quantile(x, p)
for (obj, bp, p0, rI) in (("technology", bc["boot_pibar"], bc["pibar0"], bc["r_I"]),
                          ("induced",    bc["boot_pibar_ind"], bc["pibar_ind0"], bc["n_y"]))
    gapD(v) = gap_matrices(reconstruct_Pi(v, r_θ, rI)[1])[2]
    println("\n", "="^100, "\n", uppercase(obj), "\n", "="^100)
    for g in 1:2
        D̂ = gapD(p0[g]); Db = [gapD(bp[g][b, :]) for b in ok]
        ε = N^(-1/3)
        tol = ε * median(abs.(reduce(vcat, [vec(sqrt(N) .* (M .- D̂)) for M in Db])))
        for sense in (:max, :min)
            φ = M -> solve_shape_lp(M; sense)[1]
            t̂, α̂, _, bind = solve_shape_lp(D̂; sense)
            slack = abs.(D̂ * α̂ .- t̂)
            tb = [φ(M) for M in Db]
            S = sqrt(N) .* (tb .- t̂)
            D = ndm_draws(φ, D̂, Db, N; ε)
            lab = sense === :max ? "t+ (comono)" : "t- (counter)"
            @printf("\nages %s  %s   t̂=%+.4f  √N t̂=%+.2f   binding cells %d/%d, within NDM tol(%.4f): %d;  α̂ support %d/%d\n",
                    g == 1 ? "0-5 " : "6-13", lab, t̂, sqrt(N)*t̂, length(bind), length(slack), tol,
                    count(<(tol), slack), count(>(1e-8), α̂), length(α̂))
            @printf("   standard √N(t*−t̂):  mean %+6.2f  sd %5.2f   q05 %+6.2f  q50 %+6.2f  q95 %+6.2f\n",
                    mean(S), std(S), q(S,.05), q(S,.5), q(S,.95))
            @printf("   NDM (ε=N^-1/3)     :  mean %+6.2f  sd %5.2f   q05 %+6.2f  q50 %+6.2f  q95 %+6.2f\n",
                    mean(D), std(D), q(D,.05), q(D,.5), q(D,.95))
            @printf("   95%% CI  percentile [%+.4f,%+.4f]   basic [%+.4f,%+.4f]   NDM [%+.4f,%+.4f]\n",
                    q(tb,.025), q(tb,.975), t̂ - q(S,.975)/sqrt(N), t̂ - q(S,.025)/sqrt(N),
                    t̂ - q(D,.975)/sqrt(N), t̂ - q(D,.025)/sqrt(N))
        end
    end
end
