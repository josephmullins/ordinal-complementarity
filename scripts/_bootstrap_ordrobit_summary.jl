# Summarize bootstrap_ordrobit<tag>.jld2: SEs for β₅/δ, latent investment shares, survivor
# functions, and cross-differences (grouped and period-specific), with 90% percentile
# intervals. Cross-difference intervals are descriptive only (pointwise, near-ties invalid).
using Statistics, JLD2, Printf
const REPO = joinpath(@__DIR__, "..")
BF = load(joinpath(REPO, "output", "bootstrap_ordrobit" * ENV["TAG"] * ".jld2"))
ok = BF["ok"]; r_θ, r_I = BF["r_θ"], BF["r_I"]
@printf("%d/%d resamples ok\n", count(ok), length(ok))
@printf("β₅/δ = %.3f  (boot SE %.3f, 90%% [%.2f, %.2f])\n", BF["ratio0"], std(BF["boot_ratio"][ok]),
        quantile(BF["boot_ratio"][ok], 0.05), quantile(BF["boot_ratio"][ok], 0.95))
println("latent I shares ", round.(BF["ishares0"]; digits = 3), "  SE ", round.(vec(std(BF["boot_ish"][ok, :]; dims = 1)); digits = 3))
surv(Π, k, p, q) = sum(Π[s, p, q] for s in k+1:r_θ)
xd(Π, k, p, q) = surv(Π, k, p + 1, q + 1) - surv(Π, k, p + 1, q) - surv(Π, k, p, q + 1) + surv(Π, k, p, q)
function report(lab, Π0, D)
    D = D[ok, :, :, :]; nb = size(D, 1)
    se = [std([surv(D[b, :, :, :], k, p, q) for b in 1:nb]) for k in 1:r_θ-1, p in 1:r_θ, q in 1:r_I]
    npos = nneg = 0; cells = String[]
    for k in 1:r_θ-1, p in 1:r_θ-1, q in 1:r_I-1
        d = [xd(D[b, :, :, :], k, p, q) for b in 1:nb]; lo, hi = quantile(d, 0.05), quantile(d, 0.95)
        lo > 0 && (npos += 1; push!(cells, "+(k$(k+1),θ$p,I$q)")); hi < 0 && (nneg += 1; push!(cells, "−(k$(k+1),θ$p,I$q)"))
    end
    @printf("%-14s survivor SE mean %.3f max %.3f | cross-diffs with 90%% interval > 0: %2d, < 0: %2d of %d  %s\n",
            lab, mean(se), maximum(se), npos, nneg, (r_θ - 1)^2 * (r_I - 1), join(cells, " "))
end
get(BF, "tgfit", "grouped") == "period" || for (g, lab) in enumerate(("grouped 0–5", "grouped 6–13")); report(lab, BF["pibar0"][g], BF["boot_pibar"][g]); end
for t in 1:6; report("period $t→$(t+1)", BF["period0"][t], BF["boot_period"][t]); end
