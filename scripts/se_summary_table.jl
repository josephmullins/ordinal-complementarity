# Consolidated SE comparison: asymptotic 4-block vs 10-block vs bootstrap
# (across iteration budgets).  Reads only saved JLD2 (no inference re-run).
using JLD2, Statistics, Printf
const REPO = joinpath(@__DIR__, "..")
cmp  = load(joinpath(REPO, "output", "se_block_comparison.jld2"))
boot = load(joinpath(REPO, "output", "bootstrap_estimation_v3_eps.jld2"))

budgets = boot["runs_maxiter"]
groups  = boot["transition_groups"]
pt      = boot["ratio_mle"]

a4_ratio  = cmp["se_ratio_4block"]
a10_ratio = cmp["se_ratio_10block"]
boot_ratio_by_b = boot["runs_se_ratio"]                       # vector over budgets

println("="^78)
println("  SE COMPARISON — V3+ε_Y paper spec (estimation_v3_eps.jld2), B=$(boot["B"])")
println("="^78)
@printf("\nβ₅/δ  (point = %.4f)\n", pt)
@printf("  asymptotic 4-block : %.4f\n", a4_ratio)
@printf("  asymptotic 10-block: %.4f   (×%.2f vs 4-block)\n", a10_ratio, a10_ratio/a4_ratio)
println("  bootstrap SD by EM iteration budget:")
for (i, mi) in enumerate(budgets)
    @printf("    budget %-4d : %.4f   (×%.2f vs 10-block,  ×%.2f vs 4-block)\n",
            mi, boot_ratio_by_b[i], boot_ratio_by_b[i]/a10_ratio, boot_ratio_by_b[i]/a4_ratio)
end

for g in 1:length(groups)
    a4  = cmp["se_pibar_4block"][g]
    a10 = cmp["se_pibar_10block"][g]
    @printf("\nΠ̄_free  group %d (periods %s)   [mean over 60 free entries]\n", g, groups[g])
    @printf("  asymptotic 4-block : %.4f\n", mean(a4))
    @printf("  asymptotic 10-block: %.4f   (×%.2f vs 4-block)\n", mean(a10), mean(a10)/mean(a4))
    println("  bootstrap SD by budget:")
    for (i, mi) in enumerate(budgets)
        b = mean(boot["runs_se_pibar"][i][g])
        @printf("    budget %-4d : %.4f   (×%.2f vs 10-block,  ×%.2f vs 4-block)\n",
                mi, b, b/mean(a10), b/mean(a4))
    end
end
println("\n(10-block effective rank $(cmp["rank_10block"])/$(cmp["nphi_10block"]); " *
        "4-block $(cmp["rank_4block"])/$(cmp["nphi_4block"]).  " *
        "Gap beyond 10-block = flat-ridge directions zeroed by SVD truncation.)")
