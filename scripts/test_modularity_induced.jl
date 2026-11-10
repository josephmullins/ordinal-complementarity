# ============================================================================
# Ordinal complementarity / substitutability tests on the INDUCED reduced-form
# transition Π_ind(θ' | θ, y) — skill vs EXOGENOUS income — built by composing
# the technology with the investment policy (src/induced_transition.jl).
#
# Mirrors scripts/test_supermodularity_by_cutpoint.jl but on the income lattice:
# the second conditioning axis is an income grid (n_y bins) instead of investment,
# so the SAME survivor-DID test machinery applies with n_y in place of r_I.
#
# Point estimate Π_ind and its free vector come from the bootstrap file's
# `pibar_ind0`; the covariance Var_free_ind is the child-bootstrap covariance of
# `boot_pibar_ind` (same fixed income grid `y_levels` across resamples).
#
# Output: console table + output/paper_tables_boot/modularity_induced.csv
# Run:    julia --project=. scripts/test_modularity_induced.jl
#         BOOT=output/bootstrap_reg.jld2 (default)
# ============================================================================
using JLD2, LinearAlgebra, Statistics, Printf, Random, CSV, DataFrames

const REPO = joinpath(@__DIR__, "..")
include(joinpath(REPO, "src", "inference.jl"))          # crit_value, get_S
include(joinpath(REPO, "src", "inference_common.jl"))   # adjacent_did_matrix, two_sided_modularity_test

const BOOT = get(ENV, "BOOT", joinpath(REPO, "output", "bootstrap_reg.jld2"))
const GROUP_LABELS = ["ages 0–5", "ages 6–13"]

println("Loading induced-transition bootstrap: ", BOOT)
bc = load(BOOT)
haskey(bc, "boot_pibar_ind") || error("$(BOOT) has no induced-transition fields; re-run bootstrap_reg.jl after adding induced_transition.jl.")
boot_ind = bc["boot_pibar_ind"]
pibar_ind0 = bc["pibar_ind0"]
okrows = findall(bc["ok"])
r_θ = bc["r_θ"]; n_y = bc["n_y"]; y_levels = bc["y_levels"]; TG = bc["transition_groups"]
@printf("  n_ok=%d  r_θ=%d  n_y=%d  income grid (\$)= %s\n",
        length(okrows), r_θ, n_y, join([@sprintf("%.0f", v) for v in y_levels], " "))

ncut = r_θ - 1
# Adjacent (minimal) and full pairwise survivor-DID sets over the (θ, y) lattice.
D_adj, cut_adj, cell_adj = adjacent_did_matrix(r_θ, n_y)
D_full = -get_S(r_θ, n_y)                                   # sign flip ⇒ D·Π = survivor DID m
cut_full = [((r - 1) % ncut) + 1 for r in 1:size(D_full, 1)]
rows_of(cut, j) = findall(==(j), cut)
rng = MersenneTwister(20240601)

specs = vcat([("all", 0)], [("θ'≥$(j+1)", j) for j in 1:ncut])
res = Dict{Tuple{Int,String,String},Any}()
for (g, group) in enumerate(TG)
    Vfree   = cov(boot_ind[g][okrows, :])
    Π_vec   = pibar_ind0[g]
    for (lab, j) in specs
        Df = j == 0 ? D_full : D_full[rows_of(cut_full, j), :]
        Da = j == 0 ? D_adj  : D_adj[rows_of(cut_adj,  j), :]
        res[(g, lab, "full")]     = two_sided_modularity_test(Df, Π_vec, Vfree; rng = rng)
        res[(g, lab, "adjacent")] = two_sided_modularity_test(Da, Π_vec, Vfree; rng = rng)
    end
end

function print_table(dir::Symbol)
    Tfield = dir == :super ? :T_super : :T_sub
    Rfield = dir == :super ? :reject_super : :reject_sub
    Vfield = dir == :super ? :n_sub_sig : :n_comp_sig
    vlabel = dir == :super ? "#sub*" : "#cmp*"
    head = dir == :super ? "COMPLEMENTARITY null (H0: m≥0;  reject ⇒ income SUBSTITUTES for skill)" :
                           "SUBSTITUTABILITY null (H0: m≤0;  reject ⇒ income COMPLEMENTS skill)"
    println("\n" * "="^96); println("INDUCED Π(θ'|θ,income):  ", head); println("="^96)
    for (g, _) in enumerate(TG)
        println("\n── Group $g ($(GROUP_LABELS[g])) ──")
        @printf "  %-7s | %5s %6s %9s %8s %5s | %5s %6s %9s %8s %5s\n" "Q_j" "#cond" "rej" "T" "c.05" vlabel "#cond" "rej" "T" "c.05" vlabel
        for (lab, _) in specs
            rf = res[(g, lab, "full")]; ra = res[(g, lab, "adjacent")]
            @printf "  %-7s | %5d %6s %9.2f %8.2f %5d | %5d %6s %9.2f %8.2f %5d\n" lab rf.nrow (getfield(rf,Rfield) ? "YES" : "no") getfield(rf,Tfield) rf.c getfield(rf,Vfield) ra.nrow (getfield(ra,Rfield) ? "YES" : "no") getfield(ra,Tfield) ra.c getfield(ra,Vfield)
        end
    end
end
print_table(:super)
print_table(:sub)

# CSV (same schema as supermodularity_by_cutpoint.csv)
records = DataFrame(group=String[], cutpoint=String[], set=String[], n_cond=Int[], n_used=Int[],
                    T_super=Float64[], reject_super=Bool[], T_sub=Float64[], reject_sub=Bool[],
                    c_alpha=Float64[], n_complement=Int[], n_complement_sig=Int[],
                    n_substitute=Int[], n_substitute_sig=Int[], worst_substitute=Float64[], strongest_complement=Float64[])
for (g, _) in enumerate(TG), (lab, _) in specs, set in ("full", "adjacent")
    r = res[(g, lab, set)]
    push!(records, (GROUP_LABELS[g], lab, set, r.nrow, r.n_used, r.T_super, r.reject_super,
                    r.T_sub, r.reject_sub, r.c, r.n_comp, r.n_comp_sig, r.n_sub, r.n_sub_sig,
                    r.worst_sub, r.strongest_comp))
end
outdir = joinpath(REPO, "output", "paper_tables_boot")
isdir(outdir) || mkpath(outdir)
CSV.write(joinpath(outdir, "modularity_induced.csv"), records)
println("\nsaved ", joinpath(outdir, "modularity_induced.csv"))
