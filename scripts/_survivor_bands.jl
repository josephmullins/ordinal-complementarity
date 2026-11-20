# Survivor functions Pr(θ′ ≥ k+1 | θ, I) with pointwise 90% child-bootstrap bands (2026-11-19).
#   Figure 1: converged period-specific EM fit (6 transitions), grouped point estimate dashed.
#   Figure 2: grouped fit (ages 0–5 / 6–13).
# Reads output/bootstrap_ordrobit<TAG>.jld2 and …<TAG>_period.jld2 (no data needed).
# Bands are pointwise percentile intervals of smooth functionals of Π̄ (descriptive).
using JLD2, Statistics, Printf
const REPO = joinpath(@__DIR__, "..")
tag = get(ENV, "TAG", "_rI4_noPIAT3_grp_q5_tidy")
ENV["PAPER_FIGS"] = "1"; include(joinpath(REPO, "src", "slides_palette.jl"))
G = load(joinpath(REPO, "output", "bootstrap_ordrobit$(tag).jld2"))
PF = joinpath(REPO, "output", "bootstrap_ordrobit$(tag)_period.jld2")
P = isfile(PF) ? load(PF) : nothing
r_θ, r_I = G["r_θ"], G["r_I"]
AGES = ["0–1", "2–3", "4–5", "6–7", "8–9", "10–11", "12–13"]
surv(Π, k, p, q) = sum(Π[s, p, q] for s in k+1:r_θ)

function panel(title, Π0, D, ok; ref = nothing, xlab = "")
    pl = plot(; title, xlabel = xlab, xticks = 1:r_I, legend = false, ylims = (0, 1.02), titlefontsize = 9)
    D = D[ok, :, :, :]
    for p in 1:r_θ
        S = [surv(D[b, :, :, :], panel_k[], p, q) for b in 1:size(D, 1), q in 1:r_I]
        lo = [quantile(S[:, q], 0.05) for q in 1:r_I]; hi = [quantile(S[:, q], 0.95) for q in 1:r_I]
        plot!(pl, 1:r_I, lo; fillrange = hi, fillalpha = 0.18, lw = 0, color = SKILL_COLORS[p])
        ref === nothing || plot!(pl, 1:r_I, [surv(ref, panel_k[], p, q) for q in 1:r_I]; color = SKILL_COLORS[p], ls = :dash, lw = 1.3)
        plot!(pl, 1:r_I, [surv(Π0, panel_k[], p, q) for q in 1:r_I]; color = SKILL_COLORS[p], lw = 2.5, marker = :circle, ms = 3)
    end
    pl
end
const panel_k = Ref(1)

function legend_panel(extra; cols = 4)
    leg = plot(; framestyle = :none, legend = :left, legendfontsize = 10, grid = false, legend_columns = cols)
    for p in 1:r_θ; plot!(leg, [NaN], [NaN]; color = SKILL_COLORS[p], lw = 4, label = "current skill θ = $p"); end
    for (lab, ls, lw) in extra; plot!(leg, [NaN], [NaN]; color = :black, ls, lw, label = lab); end
    plot!(leg, [NaN], [NaN]; color = :gray, lw = 8, alpha = 0.3, label = "90% bootstrap band (B = $(count(G["ok"])))")
    leg
end

# ---- Figure 1: converged period-specific EM (only if that bootstrap exists) ----
if P !== nothing
okP = P["ok"]; panels = []
for k in 1:r_θ-1, t in 1:6
    panel_k[] = k
    push!(panels, panel("ages $(AGES[t]) → $(AGES[t+1]):  Pr(θ′ ≥ $(k+1))", P["period0"][t], P["boot_period"][t], okP;
                        ref = G["pibar0"][t <= 3 ? 1 : 2], xlab = k == r_θ - 1 ? "investment level I" : ""))
end
leg = legend_panel([("period-specific EM (converged)", :solid, 2.5), ("grouped fit", :dash, 1.3)])
fig = plot(panels..., leg; layout = @layout([grid(3, 6); a{0.1h}]), size = (2400, 1380))
stem = joinpath(REPO, "figures", "survivor_bands_period$(tag)")
savefig(fig, stem * ".png"); savefig(fig, stem * ".pdf"); println("Saved → ", stem)
end

# ---- Figure 2: grouped fit ----
okG = G["ok"]; panels = []
for k in 1:r_θ-1, g in 1:2
    panel_k[] = k
    push!(panels, panel("ages $(g == 1 ? "0–5" : "6–13"):  Pr(θ′ ≥ $(k+1))", G["pibar0"][g], G["boot_pibar"][g], okG;
                        xlab = k == r_θ - 1 ? "investment level I" : ""))
end
leg = legend_panel([("grouped fit", :solid, 2.5)]; cols = 2)
fig = plot(panels..., leg; layout = @layout([grid(3, 2); a{0.14h}]), size = (1100, 1350))
stem = joinpath(REPO, "figures", "survivor_bands_grouped$(tag)")
savefig(fig, stem * ".png"); savefig(fig, stem * ".pdf"); println("Saved → ", stem)
