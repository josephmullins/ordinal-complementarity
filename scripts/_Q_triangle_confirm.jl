# EXPLORATORY (not in paper): CONFIRM super/submodularity (put it in H1) with
# intersection–union tests (IUT).
#   At a fixed Q (increments α):  H0: some cell ≤ 0  vs  H1: every cell > 0.
#   Reject (= confirm supermodular at Q) iff EVERY cell is individually
#   significant one-sided at 5%.  No multiplicity correction (IUT).
#   Over a polytope of Q's (cells linear in α): confirm for ALL Q in it iff
#   confirmed at every vertex  ⇒ IUT over vertices × cells.
# One-sided bounds: estimate ± 1.645·(bootstrap s.e.), B=200 child bootstrap.
ENV["COMONO_MODE"] = "none"
include(joinpath(@__DIR__, "comonotonicity_test.jl"))
ENV["PAPER_FIGS"] = "1"
include(joinpath(@__DIR__, "..", "src", "slides_palette.jl"))

const STEP = 20
const Z = 1.6448536269514722
bc = load(get(ENV, "BOOT", joinpath(@__DIR__, "..", "output", "bootstrap_reg.jld2")))
ok = findall(bc["ok"]); r_θ = bc["r_θ"]
grid = [[i, j, STEP - i - j] ./ STEP for i in 0:STEP for j in 0:(STEP - i)]
REGIONS = [("all Q (OC / OS)",          [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]),
           ("concave Q",                [[1.0, 0, 0], [0.5, 0.5, 0], [1/3, 1/3, 1/3]]),
           ("convex Q",                 [[0, 0, 1.0], [0, 0.5, 0.5], [1/3, 1/3, 1/3]]),
           ("bottom-weighted, α1 ≥ ½",   [[1.0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5]])]
xy(a) = (a[2] + a[3] / 2, a[3] * sqrt(3) / 2)

# weakest one-sided z across cells (sgn=+1: need all > 0; sgn=-1: need all < 0)
function weakest_z(D̂, Db, a, sgn)
    m = D̂ * a; se = vec(std(reduce(hcat, [M * a for M in Db]), dims = 2))
    minimum(sgn .* m ./ se)
end

rows = ["object,group,kind,label,side,weakest_z,confirmed"]
panels = []
for (obj, bp, p0, rI) in (("technology", bc["boot_pibar"], bc["pibar0"], bc["r_I"]),
                          ("induced",    bc["boot_pibar_ind"], bc["pibar_ind0"], bc["n_y"]))
    gapD(v) = gap_matrices(reconstruct_Pi(v, r_θ, rI)[1])[2]
    for g in 1:2
        D̂ = gapD(p0[g]); Db = [gapD(bp[g][b, :]) for b in ok]
        lab = "$obj, ages $(g == 1 ? "0–5" : "6–13")"
        println("\n", lab)
        zsup = [weakest_z(D̂, Db, a, +1) for a in grid]; zsub = [weakest_z(D̂, Db, a, -1) for a in grid]
        @printf("  pointwise: confirmed supermodular at %d of %d Q (best weakest-cell z %+.2f); submodular at %d (best %+.2f)\n",
                count(>(Z), zsup), length(grid), maximum(zsup), count(>(Z), zsub), maximum(zsub))
        for (k, a) in enumerate(grid)
            push!(rows, @sprintf("%s,%d,point,%.2f|%.2f|%.2f,super,%.3f,%s", obj, g, a..., zsup[k], zsup[k] > Z))
            push!(rows, @sprintf("%s,%d,point,%.2f|%.2f|%.2f,sub,%.3f,%s", obj, g, a..., zsub[k], zsub[k] > Z))
        end
        for (rlab, verts) in REGIONS, (side, sgn) in (("supermodular", +1), ("submodular", -1))
            w = minimum(weakest_z(D̂, Db, v, sgn) for v in verts)
            @printf("  region %-26s %-12s weakest cell z over vertices = %+6.2f   %s\n", rlab, side, w, w > Z ? "CONFIRMED" : "-")
            push!(rows, @sprintf("%s,%d,region,%s,%s,%.3f,%s", obj, g, rlab, side, w, w > Z))
        end
        # map: shade by the better of the two weakest-z's, solid if confirmed
        P = plot(; aspect_ratio = :equal, framestyle = :none, grid = false, legend = false,
                 title = lab, xlims = (-0.12, 1.12), ylims = (-0.12, 0.98))
        plot!(P, [0, 1, 0.5, 0], [0, 0, sqrt(3) / 2, 0]; color = :black, lw = 1.5)
        for (k, a) in enumerate(grid)
            x, y = xy(a)
            sup = zsup[k] >= zsub[k]; zz = max(zsup[k], zsub[k])
            base = sup ? PAL_RUST : PAL_TEAL
            col = zz > Z ? base : zz > 0 ? RGB(0.6 + 0.4 * (1 - zz / Z) * 0.6, 0.6 + 0.4 * (1 - zz / Z) * 0.6, 0.6 + 0.4 * (1 - zz / Z) * 0.6) : RGB(0.93, 0.93, 0.93)
            zz > 0 && zz <= Z && (col = RGB(0.5 * base.r + 0.5 * 0.92, 0.5 * base.g + 0.5 * 0.92, 0.5 * base.b + 0.5 * 0.92))
            scatter!(P, [x], [y]; color = col, msw = 0, ms = 230 / STEP, marker = :hexagon)
        end
        for (verts, ls) in ((REGIONS[2][2], :solid), (REGIONS[3][2], :dash))
            pts = [xy(v) for v in vcat(verts, [verts[1]])]
            plot!(P, first.(pts), last.(pts); color = :black, lw = 2, ls)
        end
        annotate!(P, [(0, -0.07, text("bottom step only", 9)), (1, -0.07, text("middle step only", 9)),
                      (0.5, sqrt(3) / 2 + 0.05, text("top step only", 9))])
        push!(panels, P)
    end
end
leg = plot(; framestyle = :none, legend = :left, legendfontsize = 11, grid = false)
for (c, l) in ((PAL_RUST, "supermodular confirmed (every cell > 0, 5%)"),
               (RGB(0.5 * PAL_RUST.r + 0.46, 0.5 * PAL_RUST.g + 0.46, 0.5 * PAL_RUST.b + 0.46), "all cells > 0 at estimate, not all significant"),
               (PAL_TEAL, "submodular confirmed (every cell < 0, 5%)"),
               (RGB(0.5 * PAL_TEAL.r + 0.46, 0.5 * PAL_TEAL.g + 0.46, 0.5 * PAL_TEAL.b + 0.46), "all cells < 0 at estimate, not all significant"),
               (RGB(0.93, 0.93, 0.93), "mixed signs at estimate"))
    scatter!(leg, [NaN], [NaN]; color = c, marker = :hexagon, ms = 10, msw = 0, label = l)
end
plot!(leg, [NaN], [NaN]; color = :black, lw = 2, label = "concave Q"); plot!(leg, [NaN], [NaN]; color = :black, lw = 2, ls = :dash, label = "convex Q")
fig = plot(panels..., leg; layout = @layout([a b; c d; e{0.14h}]), size = (1300, 1450),
           plot_title = "Can super/submodularity be CONFIRMED at Q? (IUT, pointwise)")
stem = joinpath(@__DIR__, "..", "figures", "Q_triangle_confirm" * get(ENV, "FIGTAG", ""))
savefig(fig, stem * ".pdf"); savefig(fig, stem * ".png"); println("\nSaved → ", stem, ".{pdf,png}")
csv = joinpath(@__DIR__, "..", get(ENV, "OUTDIR", "output/paper_tables_boot"), "Q_triangle_confirm_exploratory.csv")
open(csv, "w") do io; foreach(r -> println(io, r), rows); end
println("Saved → ", csv)
