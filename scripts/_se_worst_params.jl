# Which Π̄ entries are most problematic (largest bootstrap SD and largest
# boot/asymp ratio)?  Reads saved JLD2 only.  Budget 150 (settled).
using JLD2, Statistics, Printf
const REPO = joinpath(@__DIR__, "..")
boot = load(joinpath(REPO, "output", "bootstrap_estimation_v3_eps.jld2"))
cmp  = load(joinpath(REPO, "output", "se_block_comparison.jld2"))
budgets = boot["runs_maxiter"]
bi = findfirst(==(150), budgets); bi === nothing && (bi = length(budgets))
println("Budget used: $(budgets[bi]),  B=$(boot["B"])")

r_θ, r_I = 4, 5
# Π̄_free ordering: idx = (kI-1)*r_θ*(r_θ-1) + (kθ-1)*(r_θ-1) + j ,  j=1..r_θ-1
function decode(idx)
    z = idx - 1
    j  = z % (r_θ-1) + 1
    z ÷= (r_θ-1)
    kθ = z % r_θ + 1
    kI = z ÷ r_θ + 1
    (j, kθ, kI)
end

for g in 1:2
    bse  = vec(std(boot["runs_boot_pibar"][bi][g]; dims=1))   # per-entry boot SD
    a10  = cmp["se_pibar_10block"][g]                          # 10-block asymp SE
    ratio = bse ./ max.(a10, 1e-12)
    println("\n=== Group $g — top entries by bootstrap SD ===")
    for idx in sortperm(bse; rev=true)[1:5]
        (j,kθ,kI) = decode(idx)
        @printf("  P(θ'=%d | θ=%d, I=%d):  bootSD=%.3f  asym10=%.3f  ratio=%.1f×\n",
                j, kθ, kI, bse[idx], a10[idx], ratio[idx])
    end
    println("--- top by ratio (boot/asymp10), among entries with bootSD>0.01 ---")
    mask = bse .> 0.01
    order = sortperm(ratio .* mask; rev=true)[1:5]
    for idx in order
        (j,kθ,kI) = decode(idx)
        @printf("  P(θ'=%d | θ=%d, I=%d):  bootSD=%.3f  asym10=%.3f  ratio=%.1f×\n",
                j, kθ, kI, bse[idx], a10[idx], ratio[idx])
    end
end

# Robit-NN hidden-unit degeneracy: report log_w magnitude per group (the
# compensating-pairs signature) from the fit.
fit = load(joinpath(REPO, "output", "estimation_v3_eps.jld2"))
pv = fit["params_v3_eps"]
println("\n=== Robit-NN transition log_w (compensating-pairs degeneracy) ===")
for (g, spec) in enumerate(pv.trans_specs)
    @printf("  group %d: K=%d  log_w=%s  max|log_w|=%.1f\n",
            g, spec.K, string(round.(spec.log_w; digits=2)), maximum(abs.(spec.log_w)))
end
