# EXPLORATORY (not in paper): map the space of valuations Q.
# With r_θ = 4 and Q(1)=0, Q(4)=1, a valuation is its increments
#   α = (α1, α2, α3) ∈ simplex  (α1 = weight on escaping the bottom 1→2,
#   α2 = middle step 2→3, α3 = reaching the top 3→4).
# At each α on a grid, the 12 cell cross-differences m(α) = D α are linear in Π̄,
# so supermodularity (m ≥ 0) and submodularity (m ≤ 0) AT THAT Q are ordinary
# moment-inequality nulls.  Tested with
#   • RSW (Romano–Shaikh–Wolf, as in the paper: crit_value_rsw), and
#   • Fang–Seo projection test (fang_seo_test, Euclidean norm, paper tuning),
# using the B=200 child-bootstrap draws.  POINTWISE in Q (each point a 5% test).
# Corners α = e_j reproduce the paper's per-cutpoint tests.
#
# ENV: STEP (grid denominator, default 20), OBJECTS ("technology,induced")
ENV["COMONO_MODE"] = "none"
include(joinpath(@__DIR__, "comonotonicity_test.jl"))
include(joinpath(@__DIR__, "..", "src", "fang_seo_test.jl"))
ENV["PAPER_FIGS"] = "1"
include(joinpath(@__DIR__, "..", "src", "slides_palette.jl"))

const N = parse(Int, get(ENV, "NDM_N", "9686"))
const STEP = parse(Int, get(ENV, "STEP", "20"))
const OBJECTS = split(get(ENV, "OBJECTS", "technology,induced"), ",")
bc = load(get(ENV, "BOOT", joinpath(@__DIR__, "..", "output", "bootstrap_reg.jld2")))
ok = findall(bc["ok"]); r_θ = bc["r_θ"]
grid = [[i, j, STEP - i - j] ./ STEP for i in 0:STEP for j in 0:(STEP - i)]

function rsw_tests(m̂, Mb; α = 0.05, rng)
    V = cov(Mb; dims = 2); dV = diag(V)
    keep = findall(>(1e-10 * max(maximum(dV), 1.0)), dV)
    isempty(keep) && return (false, false)
    Dinv = Diagonal(1 ./ sqrt.(dV[keep]))
    Ω = Matrix(Symmetric(Dinv * V[keep, keep] * Dinv))
    mn = minimum(eigvals(Symmetric(Ω))); Ω += (max(0.0, 1e-8 - mn) + 1e-10) * I
    z = Dinv * m̂[keep]
    Tsup, Tsub = sum(min.(0.0, z) .^ 2), sum(max.(0.0, z) .^ 2)
    csup = crit_value_rsw(Hermitian(Ω), z, α; direction = :super, nsim = 50_000, rng).c
    csub = crit_value_rsw(Hermitian(Ω), z, α; direction = :sub,   nsim = 50_000, rng).c
    (Tsup > csup, Tsub > csub)
end

xy(a) = (a[2] + a[3] / 2, a[3] * sqrt(3) / 2)          # barycentric → plane
CAT_COL = Dict("neither" => RGB(0.85, 0.85, 0.85), "not complementary" => PAL_RUST,
               "not substitutable" => PAL_TEAL, "neither property" => PAL_PLUM)
classify(rsup, rsub) = rsup && rsub ? "neither property" : rsup ? "not complementary" :
                       rsub ? "not substitutable" : "neither"

rows = ["object,group,alpha1,alpha2,alpha3,test,reject_super,reject_sub"]
for obj in OBJECTS
    bp, p0, rI = obj == "technology" ? (bc["boot_pibar"], bc["pibar0"], bc["r_I"]) :
                                       (bc["boot_pibar_ind"], bc["pibar_ind0"], bc["n_y"])
    gapD(v) = gap_matrices(reconstruct_Pi(v, r_θ, rI)[1])[2]
    panels = []
    for g in 1:2
        D̂ = gapD(p0[g]); Db = [gapD(bp[g][b, :]) for b in ok]
        res = Dict{String,Vector{String}}("RSW" => String[], "Fang–Seo" => String[])
        rng = MersenneTwister(20261117)
        for a in grid
            m̂ = D̂ * a; Mb = reduce(hcat, [M * a for M in Db])
            r1 = rsw_tests(m̂, Mb; rng)
            fsup = fang_seo_test(m̂, Mb, N; A = Matrix(1.0I, length(m̂), length(m̂)), direction = :super).reject
            fsub = fang_seo_test(m̂, Mb, N; A = Matrix(1.0I, length(m̂), length(m̂)), direction = :sub).reject
            push!(res["RSW"], classify(r1...)); push!(res["Fang–Seo"], classify(fsup, fsub))
            push!(rows, @sprintf("%s,%d,%.3f,%.3f,%.3f,RSW,%s,%s", obj, g, a..., r1...))
            push!(rows, @sprintf("%s,%d,%.3f,%.3f,%.3f,FS,%s,%s", obj, g, a..., fsup, fsub))
        end
        corners = [findfirst(x -> x == e, grid) for e in ([1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0])]
        @printf("%s ages %s  corners (θ'≥2, θ'≥3, θ'≥4)  RSW: %s   FS: %s\n", obj, g == 1 ? "0-5" : "6-13",
                join(res["RSW"][corners], " | "), join(res["Fang–Seo"][corners], " | "))
        for test in ("RSW", "Fang–Seo")
            cats = res[test]
            for c in ("not complementary", "not substitutable", "neither property", "neither")
                @printf("   %-9s %-18s %3d of %d valuations\n", test, c, count(==(c), cats), length(grid))
            end
            P = plot(; aspect_ratio = :equal, framestyle = :none, grid = false, legend = false,
                     title = "$(g == 1 ? "ages 0–5" : "ages 6–13") — $test", xlims = (-0.12, 1.12), ylims = (-0.12, 0.98))
            plot!(P, [0, 1, 0.5, 0], [0, 0, sqrt(3) / 2, 0]; color = :black, lw = 1.5)
            for (k, a) in enumerate(grid)
                x, y = xy(a)
                scatter!(P, [x], [y]; color = CAT_COL[cats[k]], msw = 0, ms = 230 / STEP, marker = :hexagon)
            end
            # concave (α nonincreasing) and convex (α nondecreasing) sub-triangles
            for (verts, ls) in (([[1.0, 0, 0], [0.5, 0.5, 0], [1/3, 1/3, 1/3]], :solid),
                                ([[0, 0, 1.0], [0, 0.5, 0.5], [1/3, 1/3, 1/3]], :dash))
                pts = [xy(v) for v in vcat(verts, [verts[1]])]
                plot!(P, first.(pts), last.(pts); color = :black, lw = 2, ls)
            end
            annotate!(P, [(0, -0.07, text("bottom step only\n(0,1,1,1)", 9)),
                          (1, -0.07, text("middle step only\n(0,0,1,1)", 9)),
                          (0.5, sqrt(3) / 2 + 0.06, text("top step only (0,0,0,1)", 9)),
                          (0.22, 0.20, text("concave", 9, :black)), (0.5, 0.52, text("convex", 9, :black))])
            push!(panels, P)
        end
    end
    leg = plot(; framestyle = :none, legend = :left, legendfontsize = 11, grid = false)
    for c in ("not complementary", "not substitutable", "neither property", "neither")
        lbl = c == "neither" ? "no rejection" : c == "neither property" ? "both rejected" :
              c == "not complementary" ? "supermodularity rejected" : "submodularity rejected"
        scatter!(leg, [NaN], [NaN]; color = CAT_COL[c], marker = :hexagon, ms = 10, msw = 0, label = lbl)
    end
    plot!(leg, [NaN], [NaN]; color = :black, lw = 2, label = "concave valuations")
    plot!(leg, [NaN], [NaN]; color = :black, lw = 2, ls = :dash, label = "convex valuations")
    fig = plot(panels..., leg; layout = @layout([a b; c d; e{0.12h}]), size = (1300, 1350),
               plot_title = "Which valuations Q are rejected? ($obj, pointwise 5% tests)")
    stem = joinpath(@__DIR__, "..", "figures", "Q_triangle_$(obj)" * get(ENV, "FIGTAG", ""))
    savefig(fig, stem * ".pdf"); savefig(fig, stem * ".png"); println("Saved → ", stem, ".{pdf,png}\n")
end
csv = joinpath(@__DIR__, "..", get(ENV, "OUTDIR", "output/paper_tables_boot"), "Q_triangle_map_exploratory.csv")
open(csv, "w") do io; foreach(r -> println(io, r), rows); end
println("Saved → ", csv)
