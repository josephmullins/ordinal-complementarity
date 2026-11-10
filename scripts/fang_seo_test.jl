# Fang & Seo (2021, ECMA) projection cone tests on the paper spec, using the
# child-bootstrap draws from scripts/bootstrap_reg.jl.  Four shape restrictions,
# each a polyhedral cone in survivor space (Euclidean norm, as in the paper):
#
#   supermodular  H₀: A_adj·s ≥ 0   (ordinal complementarity)
#   submodular    H₀: A_adj·s ≤ 0   (ordinal substitutability)
#   convex_OC     H₀: A_suf·s ≥ 0   (convex ordinal complementarity: suffix sums)
#   concave_OS    H₀: A_pre·s ≤ 0   (concave ordinal substitutability: prefix sums)
#
# T_n = √N‖ŝ − Π_Λ ŝ‖; critical value per Fang–Seo §3.3 with the data-driven
# κ̂_n (γ_n = 0.01/log N default, c_n = 1) and Ĝ_b = √N(ŝ*_b − ŝ).
# Run for the technology Π̄(θ'|θ,I) and the induced income transition
# Π_ind(θ'|θ,y), both transition groups.
#
# Run: julia scripts/fang_seo_test.jl
# ENV: BOOT (default output/bootstrap_reg.jld2), ALPHA (default 0.05),
#      OUTDIR (default output/paper_tables_boot)
using LinearAlgebra, Statistics, Random, Distributions, JLD2, Printf, DataFrames, CSV
const REPO = joinpath(@__DIR__, "..")
include(joinpath(REPO, "src", "fang_seo_test.jl"))

const BOOTFILE = get(ENV, "BOOT", "output/bootstrap_reg.jld2")
const ALPHA    = parse(Float64, get(ENV, "ALPHA", "0.05"))
const OUTDIR   = joinpath(REPO, get(ENV, "OUTDIR", "output/paper_tables_boot"))
mkpath(OUTDIR)

bd = load(joinpath(REPO, BOOTFILE))
ok  = bd["ok"];  TG = bd["transition_groups"]
r_θ = bd["r_θ"]; r_I = bd["r_I"]; n_y = bd["n_y"]
N   = length(bd["idx_draws"][1])          # children per resample = sample size
G   = length(TG)
@printf("Fit: %s | B=%d ok=%d | N=%d | r_θ=%d r_I=%d n_y=%d | α=%.2f\n",
        bd["fit"], bd["B"], count(ok), N, r_θ, r_I, n_y, ALPHA)

group_label(g) = g == 1 ? "early (ages 0-5)" : "school (ages 6-13)"

rows = DataFrame(object=String[], group=Int[], restriction=String[], γ=Float64[],
                 T_n=Float64[], c_α=Float64[], p_value=Float64[], reject=Bool[],
                 κ_n=Float64[], τ=Float64[])

# restriction -> (constraint-matrix builder, direction)
restrictions(r_x) = [
    ("supermodular", survivor_did_matrix(r_θ, r_x),                            :super),
    ("submodular",   survivor_did_matrix(r_θ, r_x),                            :sub),
    ("convex_OC",    cumulative_survivor_did_matrix(r_θ, r_x; mode=:suffix),   :super),
    ("concave_OS",   cumulative_survivor_did_matrix(r_θ, r_x; mode=:prefix),   :sub),
]

function run_block!(rows, object, s0_list, boot_list, r_x; γs)
    for (lab, A, dir) in restrictions(r_x), g in 1:G
        ŝ = s0_list[g]
        sb = Matrix(boot_list[g][ok, :]')          # d × B_ok
        for γ in γs
            r = fang_seo_test(ŝ, sb, N; A, direction=dir, α=ALPHA, γ)
            push!(rows, (object, g, lab, r.γ, r.T_n, r.c_α, r.p_value, r.reject,
                         r.κ_n, r.τ))
        end
    end
end

γ_default = 0.01 / log(N)
γ_sens = [1/N, γ_default, 0.01, 0.1]

run_block!(rows, "technology", bd["surv0"], bd["boot_surv"], r_I; γs=γ_sens)
run_block!(rows, "induced_income", bd["surv_ind0"], bd["boot_surv_ind"], n_y; γs=γ_sens)

out_csv = joinpath(OUTDIR, "fang_seo_projection_test.csv")
CSV.write(out_csv, rows)

println("\n══ Fang–Seo projection tests (γ_n = 0.01/log N) ══")
println(rpad("object", 16), rpad("group", 20), rpad("H₀", 14),
        rpad("T_n", 10), rpad("c_α", 10), rpad("p", 8), "reject")
for r in eachrow(rows)
    r.γ == γ_default || continue
    @printf("%-16s%-20s%-14s%-10.4f%-10.4f%-8.3f%s\n",
            r.object, group_label(r.group), r.restriction, r.T_n, r.c_α,
            r.p_value, r.reject ? "REJECT" : "—")
end

println("\n── γ_n sensitivity ──")
for r in eachrow(rows)
    r.γ == γ_default && continue
    @printf("%-16s g=%d %-12s γ=%-9.2g T=%-9.4f c=%-9.4f p=%-7.3f %s\n",
            r.object, r.group, r.restriction, r.γ, r.T_n, r.c_α, r.p_value,
            r.reject ? "REJECT" : "—")
end
println("\nSaved → $out_csv")
