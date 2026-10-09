# Summary comparison of the super/submodularity tests on the paper spec, and
# LaTeX tables for the draft:
#   1. MMM moment-inequality test, least-favorable critical value  (Table 2)
#   2. MMM with Romano–Shaikh–Wolf two-step moment selection       (Table 2)
#   3. Fang–Seo (2021) projection test, Euclidean norm
#
# MMM-LF / RSW numbers for the technology come from
# output/paper_tables_boot/supermodularity_by_cutpoint.csv ("adjacent" set, all
# cutpoints).  The induced-income CSV (modularity_induced.csv) has no RSW
# columns, so RSW critical values for the induced object are recomputed here
# with the identical recipe (crit_value_rsw, β = α/10, same rng seed and
# standardization as scripts/test_supermodularity_by_cutpoint.jl::run_set).
# FS numbers come from output/paper_tables_boot/fang_seo_projection_test.csv
# (default γ_n = 0.01/log N).
#
# All columns test the same H₀ per row: "super" = supermodularity in
# distribution (OC/complementarity), "sub" = submodularity (substitutability).
# reject=true ⇒ evidence against that shape.  T statistics are NOT comparable
# across test families (MMM: Σ of squared standardized violations; FS: √N ×
# projection distance) — compare verdicts/p-values, not magnitudes.
#
# Outputs:
#   output/paper_tables_boot/modularity_test_comparison.csv
#   output/paper_tables_boot/test_comparison_technology.tex
#   output/paper_tables_boot/test_comparison_induced.tex
#
# Run: julia scripts/compare_modularity_tests.jl
using CSV, DataFrames, Printf, JLD2, LinearAlgebra, Statistics, Random, Distributions
const REPO = joinpath(@__DIR__, "..")
const DIR  = joinpath(REPO, get(ENV, "OUTDIR", "output/paper_tables_boot"))
const BOOT = get(ENV, "BOOT", joinpath(REPO, "output", "bootstrap_reg.jld2"))
const ALPHA = parse(Float64, get(ENV, "ALPHA", "0.05"))
include(joinpath(REPO, "src", "inference.jl"))    # get_S_adjacent, crit_value_rsw

mmm  = CSV.read(joinpath(DIR, "supermodularity_by_cutpoint.csv"), DataFrame)
mind = CSV.read(joinpath(DIR, "modularity_induced.csv"), DataFrame)
fs   = CSV.read(joinpath(DIR, "fang_seo_projection_test.csv"), DataFrame)

# FS default γ_n = 0.01/log N; recover it as the closest value present in the sweep.
bd = load(BOOT)
N  = length(bd["idx_draws"][1])
γs = unique(fs.γ)
γ0 = γs[argmin(abs.(γs .- 0.01 / log(N)))]
fs = fs[fs.γ .== γ0, :]

# ── RSW critical values for the induced object (absent from its CSV) ─────────
# Same construction as test_supermodularity_by_cutpoint.jl::run_set: survivor
# DIDs m = D·Π_free with D = −get_S_adjacent, V from the bootstrap Π̄_free
# covariance, z = m/se, Ω = correlation, crit_value_rsw with β = α/10.
r_θ = bd["r_θ"]; n_y = bd["n_y"]; okb = findall(bd["ok"])
rng = MersenneTwister(20240601)
rsw_ind = Dict{Tuple{Int,String},Tuple{Float64,Bool}}()
let D = -get_S_adjacent(r_θ, n_y)
    for g in 1:2
        Π_vec = bd["pibar_ind0"][g]
        Vfree = cov(bd["boot_pibar_ind"][g][okb, :])
        m = D * Π_vec; V = D * Vfree * D'
        dV = diag(V); tol = 1e-10 * max(maximum(dV), 1.0)
        keep = findall(>(tol), dV)
        mk = m[keep]; Vk = V[keep, keep]; dk = dV[keep]
        Dinv = diagm(1.0 ./ sqrt.(dk))
        Ω = Matrix(Symmetric(Dinv * Vk * Dinv))
        mn = minimum(eigvals(Symmetric(Ω))); Ω += (max(0.0, 1e-8 - mn) + 1e-10) * I
        z = Dinv * mk
        T_super = sum(min.(0.0, z) .^ 2)
        T_sub   = sum(max.(0.0, z) .^ 2)
        c_su = crit_value_rsw(Hermitian(Ω), z, ALPHA; direction = :super, rng = rng)
        c_sb = crit_value_rsw(Hermitian(Ω), z, ALPHA; direction = :sub,   rng = rng)
        rsw_ind[(g, "super")] = (c_su, T_super > c_su)
        rsw_ind[(g, "sub")]   = (c_sb, T_sub   > c_sb)
    end
end

group_name = Dict("ages 0–5" => 1, "ages 6–13" => 2)
glabel = ["early (0–5)", "school (6–13)"]

rows = DataFrame(object=String[], group=String[], H0=String[],
                 mmm_T=Float64[], mmm_cLF=Float64[], mmm_rejLF=Bool[],
                 rsw_c=Float64[], rsw_rej=Bool[],
                 fs_T=Float64[], fs_c=Float64[], fs_p=Float64[], fs_rej=Bool[])

function fs_row(obj, g, restriction)
    r = fs[(fs.object .== obj) .& (fs.group .== g) .& (fs.restriction .== restriction), :]
    @assert nrow(r) == 1
    r[1, :]
end

for (obj, src, fsobj) in [("technology", mmm, "technology"),
                          ("induced_income", mind, "induced_income")]
    base = src[(src.cutpoint .== "all") .& (src.set .== "adjacent"), :]
    has_rsw = "c_rsw_super" in names(src)
    for r in eachrow(base)
        g = group_name[r.group]
        for dir in ("super", "sub")
            T   = r[Symbol("T_$dir")]
            rej = r[Symbol("reject_$dir")]
            rsw_c, rsw_rej = has_rsw ? (r[Symbol("c_rsw_$dir")], r[Symbol("reject_$(dir)_rsw")]) :
                                       rsw_ind[(g, dir)]
            fe = fs_row(fsobj, g, dir == "super" ? "supermodular" : "submodular")
            push!(rows, (obj, glabel[g], dir, T, r.c_alpha, rej, rsw_c, rsw_rej,
                         fe.T_n, fe.c_α, fe.p_value, fe.reject))
        end
    end
end

# ── shape-refined tests (convex-OC / concave-OS) ─────────────────────────────
# MMM-LF + RSW from shape_refinement_*.csv (scripts/test_convex_concave_shape.jl,
# cumulative suffix/prefix constraint sets), FS from the same projection CSV.
shp_tr = CSV.read(joinpath(DIR, "shape_refinement_transition.csv"), DataFrame)
shp_in = CSV.read(joinpath(DIR, "shape_refinement_induced.csv"), DataFrame)
for (obj, src, fsobj) in [("technology", shp_tr, "technology"),
                          ("induced_income", shp_in, "induced_income")]
    for r in eachrow(src)
        g = group_name[r.group]
        for (h0, Tc, cc, rc, crsw, rrsw, fsname) in [
                ("convex_OC",  r.T_convexOC,  r.c_convex,  r.reject_convexOC,
                 r.c_convex_rsw,  r.reject_convexOC_rsw,  "convex_OC"),
                ("concave_OS", r.T_concaveOS, r.c_concave, r.reject_concaveOS,
                 r.c_concave_rsw, r.reject_concaveOS_rsw, "concave_OS")]
            fe = fs_row(fsobj, g, fsname)
            push!(rows, (obj, glabel[g], h0, Tc, cc, rc, crsw, rrsw,
                         fe.T_n, fe.c_α, fe.p_value, fe.reject))
        end
    end
end

out_csv = joinpath(DIR, "modularity_test_comparison.csv")
CSV.write(out_csv, rows)

# ── console summary ──────────────────────────────────────────────────────────
mark(rej) = rej ? "REJ  " : "—    "
markp(rej, p) = rej ? @sprintf("REJ %.3f", p) : @sprintf("—   %.3f", p)
println("Verdicts: REJ = reject H₀ at α=$(ALPHA); p = FS bootstrap p-value")
println("Constraint set: adjacent survivor DIDs (both frameworks). FS γ_n = $(round(γ0, sigdigits=3)).\n")
println(rpad("object", 16), rpad("group", 15), rpad("H₀", 7),
        rpad("MMM-LF", 7), rpad("MMM-RSW", 9), "FS")
for r in eachrow(rows)
    println(rpad(r.object, 16), rpad(r.group, 15), rpad(r.H0, 7),
            rpad(mark(r.mmm_rejLF), 7), rpad(mark(r.rsw_rej), 9),
            markp(r.fs_rej, r.fs_p))
end

# ── LaTeX tables ─────────────────────────────────────────────────────────────
fmt(x; d=1) = "\$" * @sprintf("%.*f", d, x) * "\$"
rejtex(r) = r ? "yes" : "no"
h0tex = Dict("super" => "complementarity", "sub" => "substitutability")
grptex = Dict("early (0–5)" => "ages 0--5", "school (6–13)" => "ages 6--13")

caption_common = """
\$\\hat T_{\\mathrm{MMM}}\$ is the modified-method-of-moments statistic of
Section~\\ref{subsec:test-oc} on the 36 adjacent survival cross-differences
(child-bootstrap covariance, \$B=200\$), with critical values \$c^{\\mathrm{LF}}_{0.05}\$
simulated from the least-favorable null and \$c^{\\mathrm{RSW}}_{0.05}\$ from the
moment-selection refinement of \\citet{Romano_Shaikh_Wolf-2014-ECTA} with first-stage size
\$\\beta=\\alpha/10\$. \$\\hat T_{\\mathrm{FS}}\$ is the projection statistic of
\\citet{Fang_Seo-2021-ECTA} described in Section~\\ref{subsec:test-fs}, with critical value
\$\\hat c_{0.05}\$ and bootstrap \$p\$-value computed from the same \$B=200\$ child-bootstrap
draws and tuning \$\\gamma_n=0.01/\\log N\$, \$c_n=1\$ (verdicts are unchanged for
\$\\gamma_n\\in\\{1/N,\\,0.01,\\,0.1\\}\$). Rejection is evidence against the named property."""

function write_tex(path, obj, h0s, caption, label)
    sub = rows[rows.object .== obj, :]
    open(path, "w") do io
        println(io, "\\begin{table}[htbp]")
        println(io, "\\centering\\small")
        println(io, "\\begin{tabular}{@{}llcccccccc@{}}")
        println(io, "\\toprule")
        println(io, " & & \\multicolumn{3}{c}{MMM (least favorable)} & \\multicolumn{2}{c}{MMM (RSW)} & \\multicolumn{3}{c}{Fang--Seo projection} \\\\")
        println(io, "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-10}")
        println(io, "Transition group & \$H_0\$ & \$\\hat T_{\\mathrm{MMM}}\$ & \$c^{\\mathrm{LF}}_{0.05}\$ & rej. & \$c^{\\mathrm{RSW}}_{0.05}\$ & rej. & \$\\hat T_{\\mathrm{FS}}\$ & \$\\hat c_{0.05}\$ \\,(\$p\$) & rej. \\\\ \\midrule")
        for grp in ["early (0–5)", "school (6–13)"]
            first = true
            for h0 in h0s
                r = sub[(sub.group .== grp) .& (sub.H0 .== h0), :]
                @assert nrow(r) == 1; r = r[1, :]
                gcol = first ? grptex[grp] : ""
                first = false
                println(io, "$(gcol) & $(h0tex[h0]) & $(fmt(r.mmm_T)) & $(fmt(r.mmm_cLF)) & $(rejtex(r.mmm_rejLF)) & $(fmt(r.rsw_c)) & $(rejtex(r.rsw_rej)) & $(fmt(r.fs_T)) & $(fmt(r.fs_c)) ($(@sprintf("%.2f", r.fs_p))) & $(rejtex(r.fs_rej)) \\\\")
            end
            grp == "early (0–5)" && println(io, "\\midrule")
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
        println(io, "\\caption{$(caption)}")
        println(io, "\\label{$(label)}")
        println(io, "\\end{table}")
    end
    println("wrote $path")
end

merge!(h0tex, Dict("convex_OC" => "convex complementarity",
                   "concave_OS" => "concave substitutability"))

write_tex(joinpath(DIR, "test_comparison_technology.tex"), "technology",
          ["super", "sub"],
          "Ordinal complementarity (\$H_0\$: supermodularity in distribution) and substitutability (\$H_0\$: submodularity) of the technology \$\\overline\\Pi(\\theta'\\mid\\theta,I)\$, by transition group, on the pooled adjacent constraint set of Table~\\ref{tab:test}: least-favorable, moment-selection, and projection critical values. $(caption_common)",
          "tab:test_comparison")
write_tex(joinpath(DIR, "test_comparison_induced.tex"), "induced_income",
          ["super", "sub"],
          "Ordinal complementarity and substitutability of the income-induced transition \$\\Pi^{\\mathrm{ind}}(\\theta'\\mid\\theta,y)\$ in \\eqref{eq:induced}, by transition group: same statistics, constraint set, critical values, and tuning as Table~\\ref{tab:test_comparison}, with income in place of investment.",
          "tab:test_comparison_induced")

caption_shape_tech = """
Tests of convex-ordinal complementarity (Definition~\\ref{def:convex-oc}: \$\\E[Q(\\theta')]\$
supermodular for every non-decreasing convex \$Q\$; \$H_0\$: the suffix sums
\$T_j(\\theta,I)=\\sum_{k\\geq j}\\Delta^2\\overline F(k\\mid\\theta,I)\$ are all non-negative,
Proposition~\\ref{prop:convex-oc}) and concave-ordinal substitutability (\$H_0\$: the prefix
sums \$L_j(\\theta,I)=\\sum_{k\\leq j}\\Delta^2\\overline F(k\\mid\\theta,I)\$ are all
non-positive) for the technology \$\\overline\\Pi(\\theta'\\mid\\theta,I)\$, by transition
group. Same statistics, critical values, and tuning as Table~\\ref{tab:test_comparison},
applied to the cumulative (suffix- or prefix-summed) cross-differences. Neither property is
rejected by any test."""

caption_shape_ind = """
Convex-ordinal complementarity and concave-ordinal substitutability of the income-induced
transition \$\\Pi^{\\mathrm{ind}}(\\theta'\\mid\\theta,y)\$ in \\eqref{eq:induced}: same
statistics, constraint sets, critical values, and tuning as Table~\\ref{tab:shape}, with
income in place of investment. Neither property is rejected by any test."""

write_tex(joinpath(DIR, "test_comparison_shape_technology.tex"), "technology",
          ["convex_OC", "concave_OS"], caption_shape_tech, "tab:shape")
write_tex(joinpath(DIR, "test_comparison_shape_induced.tex"), "induced_income",
          ["convex_OC", "concave_OS"], caption_shape_ind, "tab:shape_induced")

println("\nSaved → $out_csv")
