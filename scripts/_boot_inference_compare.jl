# Assemble the asymptotic-vs-bootstrap inference comparison for the regularized fit.
# Reads the three supermodularity CSVs (original MLE-asymptotic, reg-asymptotic,
# reg-bootstrap) + bootstrap_reg.jld2 (β₅/δ draws), writes a markdown summary.
using CSV, DataFrames, JLD2, Statistics, Printf
const REPO = joinpath(@__DIR__, "..")
rd(p) = isfile(p) ? CSV.read(p, DataFrame) : nothing
orig = rd(joinpath(REPO,"output","paper_tables","supermodularity_by_cutpoint.csv"))
reg  = rd(joinpath(REPO,"output","paper_tables_reg","supermodularity_by_cutpoint.csv"))
boot = rd(joinpath(REPO,"output","paper_tables_boot","supermodularity_by_cutpoint.csv"))

# adjacent-set OC test row for (group, cutpoint)
function row(df, grp, cut)
    df === nothing && return nothing
    r = df[(df.group .== grp) .& (df.cutpoint .== cut) .& (df.set .== "adjacent"), :]
    nrow(r) == 0 ? nothing : r[1, :]
end
fmt(r) = r === nothing ? "—" : @sprintf("T=%.1f c=%.1f %s", r.T_super, r.c_alpha, r.reject_super ? "REJECT" : "fail")

groups = ["ages 0–5", "ages 6–13"]
cuts = ["all", "θ'≥2", "θ'≥3", "θ'≥4"]

io = IOBuffer()
println(io, "# Bootstrap vs asymptotic inference — regularized fit\n")
println(io, "OC test = supermodularity (H0); **REJECT ⇒ ordinal complementarity violated**. Adjacent (corrected) constraint set.\n")
println(io, "Columns: OLD = original fit, asymptotic | REG-asy = regularized fit, asymptotic (delta) | REG-boot = regularized fit, **bootstrap covariance**.\n")
for g in groups
    println(io, "\n## $g\n")
    println(io, "| cutpoint | OLD (asy) | REG-asy | REG-boot |")
    println(io, "|---|---|---|---|")
    for c in cuts
        println(io, "| $c | $(fmt(row(orig,g,c))) | $(fmt(row(reg,g,c))) | $(fmt(row(boot,g,c))) |")
    end
end

# β₅/δ bootstrap CI
bf = joinpath(REPO,"output","bootstrap_reg.jld2")
if isfile(bf)
    d = load(bf); ok = findall(d["ok"]); br = d["boot_ratio"][ok]
    q = quantile(br, [0.025, 0.975])
    println(io, "\n## β₅/δ (income semi-elasticity)\n")
    @printf(io, "- point = %.4f\n- bootstrap SD = %.4f  (B=%d, n_ok=%d)\n- 95%% percentile CI = [%.3f, %.3f]\n- 95%% normal CI = [%.3f, %.3f]\n",
            d["ratio0"], std(br), d["B"], length(ok), q[1], q[2],
            d["ratio0"]-1.96std(br), d["ratio0"]+1.96std(br))
    # survivor / Π̄ bootstrap SEs
    for (gi,gl) in enumerate(groups)
        sp = vec(std(d["boot_pibar"][gi][ok,:]; dims=1)); ss = vec(std(d["boot_surv"][gi][ok,:]; dims=1))
        @printf(io, "- %s: Π̄ bootstrap SE mean=%.4f max=%.4f ; survivor SE mean=%.4f max=%.4f\n",
                gl, mean(sp), maximum(sp), mean(ss), maximum(ss))
    end
end

println(io, "\n## Notes\n")
println(io, "- Headline OC rejection should be compared across the three columns; the bootstrap covariance is the trustworthy one given the flat-ridge weak identification.")
println(io, "- Watch the `θ'≥4` row at ages 0–5 (sparse, weakly-identified cell) — the verdict there is fragile across fits/methods.")

outdir = joinpath(REPO,"output","reg_comparison"); isdir(outdir)||mkpath(outdir)
out = joinpath(outdir,"BOOTSTRAP_INFERENCE.md")
write(out, String(take!(io)))
println("Wrote ", out)
print(read(out, String))
