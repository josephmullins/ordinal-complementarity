# Survivor figure with BOOTSTRAP 95% CI bands, for the regularized fit.
# Self-contained: reads only output/bootstrap_reg.jld2 (surv0 = point survivor
# per group; boot_surv = B×n_free draws).  Bands = point ± 1.96·bootstrap-SD.
# ENV: BOOT (default output/bootstrap_reg.jld2), OUT (default figures_reg/survivor_groups_bootstrap.pdf)
using JLD2, Plots, Statistics, Printf
const REPO = joinpath(@__DIR__, "..")
const BOOT = get(ENV,"BOOT", joinpath(REPO,"output","bootstrap_reg.jld2"))
const OUT  = get(ENV,"OUT", joinpath(REPO,"figures_boot","survivor_groups_bootstrap.pdf"))
isdir(dirname(OUT)) || mkpath(dirname(OUT))
const SKILL_COLORS = [RGB(99/255,170/255,156/255), RGB(139/255,90/255,140/255),
                      RGB(0.90,0.72,0.33), RGB(233/255,99/255,59/255)]
const GROUP_LABELS = ["step 1: ages 0–5", "step 2: ages 6–11"]

d = load(BOOT)
r_θ, r_I = d["r_θ"], d["r_I"]; n_cut = r_θ-1
groups = d["transition_groups"]; B = d["B"]; ok = findall(d["ok"])
surv0 = d["surv0"]; boot_surv = d["boot_surv"]
surv_idx(kI,kθ,k) = (kI-1)*r_θ*n_cut + (kθ-1)*n_cut + k
se_boot = [vec(std(boot_surv[g][ok,:]; dims=1)) for g in 1:length(groups)]

gr(); default(background_color=RGB(1,1,1), foreground_color=RGB(0,0,0), framestyle=:box,
    grid=false, left_margin=10Plots.mm, bottom_margin=8Plots.mm, right_margin=3Plots.mm, top_margin=3Plots.mm)
plots = Matrix{Any}(undef, length(groups), n_cut)
for gi in 1:length(groups), k in 1:n_cut
    leg = (gi==1 && k==n_cut)
    p = plot(ylims=(0,1.05), legend = leg ? :topleft : false,
        titlefontsize=13, guidefontsize=13, tickfontsize=11, legendfontsize=11)
    gi==1 && title!(p, "k = $k"); k==1 && ylabel!(p, GROUP_LABELS[gi]*"\nsurvivor F̄(k | θ, I)")
    gi==length(groups) && xlabel!(p, "investment I")
    for kθ in 1:r_θ
        idxs = [surv_idx(kI,kθ,k) for kI in 1:r_I]
        s  = [surv0[gi][i] for i in idxs]; sb = [se_boot[gi][i] for i in idxs]
        plot!(p, 1:r_I, s, ribbon=1.96 .* sb, marker=:circle, markersize=5, linewidth=2.6,
              fillalpha=0.18, color=SKILL_COLORS[kθ], label = leg ? "θ = $kθ" : "")
    end
    plots[gi,k] = p
end
# robust interior summary
interior = vcat([(surv0[g] .> 0.05) .& (surv0[g] .< 0.95) for g in 1:length(groups)]...)
sb_all = vcat(se_boot...)
fig = plot(permutedims(plots)..., layout=(length(groups),n_cut),
    size=(460*n_cut, 360*length(groups)+40), plot_titlefontsize=12, margin=3Plots.mm,
    left_margin=12Plots.mm, top_margin=7Plots.mm,
    plot_title=@sprintf("Survivor F̄(k|θ,I) — regularized fit, bootstrap 95%% CI  ·  B=%d (n_ok=%d)", B, length(ok)))
savefig(fig, OUT)
@printf("Saved → %s   (interior survivor SE: mean %.4f, max %.4f)\n", OUT,
        mean(sb_all[interior]), maximum(sb_all))
