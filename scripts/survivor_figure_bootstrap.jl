# ═══════════════════════════════════════════════════════════════
# Survivor-function figure with BOOTSTRAP confidence bands.
#
# Re-creates the regime-marginalized survivor figure
# (cf. scripts/generate_ucla_figures.jl → survivor_groups.pdf) but
# replaces the asymptotic delta-method ±1.96·SE band with a bootstrap
# band (point ± 1.96·bootstrap-SD), and overlays the old asymptotic
# band as dashed boundary lines so the widening is visible.
#
# SELF-CONTAINED: reads ONLY output/bootstrap_estimation_v3_eps.jld2
# (survivor point estimates, bootstrap draws, asymptotic SEs are all
# stored there).  No data load / no inference → safe to run while the
# bootstrap sweep is still going.
#
# Config via ENV:
#   BOOT_FILE  bootstrap jld2 (default output/bootstrap_estimation_v3_eps.jld2)
#   BUDGET     which iteration budget to use (default = largest available)
#   OUT        output pdf (default figures/survivor_groups_bootstrap.pdf)
#
# Survivor free-entry ordering (matches delta_cov_survivor_bar /
# survivor_bar_free_vec):  idx = (kI-1)*r_θ*n_cut + (kθ-1)*n_cut + k.
# ═══════════════════════════════════════════════════════════════

using JLD2, Plots, Statistics, Printf

const REPO = joinpath(@__DIR__, "..")
const BOOT_FILE = get(ENV, "BOOT_FILE",
    joinpath(REPO, "output", "bootstrap_estimation_v3_eps.jld2"))
const OUT = get(ENV, "OUT", joinpath(REPO, "figures", "survivor_groups_bootstrap.pdf"))
isdir(dirname(OUT)) || mkpath(dirname(OUT))

# Paper spec dims (estimation_v3_eps.jld2)
const r_θ, r_I = 4, 5
const n_cut = r_θ - 1
surv_idx(kI, kθ, k) = (kI - 1) * r_θ * n_cut + (kθ - 1) * n_cut + k

# DCT palette (match generate_ucla_figures.jl)
const SKILL_COLORS = [RGB(99/255,170/255,156/255),   # θ=1 teal
                      RGB(139/255,90/255,140/255),    # θ=2 plum
                      RGB(0.90,0.72,0.33),            # θ=3 gold
                      RGB(233/255,99/255,59/255)]     # θ=4 rust
const GROUP_LABELS = ["step 1: ages 0–5", "step 2: ages 6–11"]

# ── Load bootstrap results ──
println("Loading $BOOT_FILE ...")
d = load(BOOT_FILE)
budgets   = d["runs_maxiter"]
groups    = d["transition_groups"]
B         = d["B"]
surv_mle  = d["surv_free_mle"]      # [g] → length-60 vector
se_asymp  = d["se_surv_asymp"]      # [g] → length-60 vector
runs_surv = d["runs_boot_surv"]     # [budget][g] → (B × 60)
n_ok      = d["runs_n_ok"]

# Pick budget
bsel = haskey(ENV, "BUDGET") ? parse(Int, ENV["BUDGET"]) : maximum(budgets)
bi   = findfirst(==(bsel), budgets)
bi === nothing && error("budget $bsel not in saved budgets $budgets")
println("Using budget=$bsel  (B=$B, n_ok=$(n_ok[bi]))  of budgets $budgets")

# Bootstrap SD per survivor entry, per group
se_boot = [vec(std(runs_surv[bi][g]; dims=1)) for g in 1:length(groups)]

# ── Asymptotic band for the dashed reference lines ──
# The bootstrap file stores only the ORIGINAL 4-block survivor SE.  Prefer the
# CORRECTED 10-block version: derive it from the 10-block Π̄_free covariance
# V10_list in output/se_block_comparison.jld2, since the survivor is an exact
# linear map  S̄(k|θ,I) = 1 − Σ_{j≤k} Π̄(j,θ,I)  of the free Π̄ entries.
# (Π̄_free ordering: ip(kI,kθ,j) = (kI-1)*r_θ*(r_θ-1) + (kθ-1)*(r_θ-1) + j.)
asymp_label = "4-block"
const CMP_FILE = joinpath(REPO, "output", "se_block_comparison.jld2")
if isfile(CMP_FILE)
    c = load(CMP_FILE)
    if haskey(c, "V10_list")
        V10 = c["V10_list"]
        ip(kI, kθ, j) = (kI - 1) * r_θ * (r_θ - 1) + (kθ - 1) * (r_θ - 1) + j
        se_asymp = map(1:length(groups)) do g
            V = V10[g]
            se = zeros(length(surv_mle[g]))
            for kI in 1:r_I, kθ in 1:r_θ
                idxJ = [ip(kI, kθ, j) for j in 1:(r_θ - 1)]
                Σb = V[idxJ, idxJ]                      # 3×3 cov of (π₁,π₂,π₃)
                for k in 1:n_cut
                    a = [j <= k ? 1.0 : 0.0 for j in 1:(r_θ - 1)]
                    se[surv_idx(kI, kθ, k)] = sqrt(max(a' * Σb * a, 0.0))
                end
            end
            se
        end
        asymp_label = "10-block"
        println("Dashed asymptotic band: CORRECTED 10-block (from V10_list).")
    end
else
    println("Dashed asymptotic band: 4-block (se_block_comparison.jld2 not found).")
end

# Summary ratio (boot/asymp) for the title.  Restrict to INTERIOR survivor
# entries (point ∈ [0.05, 0.95]); boundary entries (≈0/≈1) have ~0 SE in both
# and make per-entry ratios explode/meaningless.  Report a width ratio
# (mean boot SE / mean asymp SE) over interior entries — robust to per-entry
# blow-ups — plus the median per-entry interior ratio.
interior = vcat([(surv_mle[g] .> 0.05) .& (surv_mle[g] .< 0.95) for g in 1:length(groups)]...)
boot_all  = vcat(se_boot...)
asymp_all = vcat(se_asymp...)
width_ratio = sum(boot_all[interior]) / sum(asymp_all[interior])
med_ratio   = median((boot_all[interior]) ./ max.(asymp_all[interior], 1e-12))
mean_ratio  = width_ratio   # used in title
@printf("Interior entries (n=%d): mean-width ratio boot/asymp = %.2f, median per-entry = %.2f\n",
        count(interior), width_ratio, med_ratio)

gr()
default(background_color = RGB(1,1,1), foreground_color = RGB(0,0,0),
        framestyle = :box, grid = false,
        left_margin = 10Plots.mm, bottom_margin = 8Plots.mm,
        right_margin = 3Plots.mm, top_margin = 3Plots.mm)

plots = Matrix{Any}(undef, length(groups), n_cut)
for gi in 1:length(groups)
    for k in 1:n_cut
        show_legend = (gi == 1 && k == n_cut)
        p = plot(ylims = (0.0, 1.05),
                 legend = show_legend ? :topleft : false,
                 titlefontsize = 13, guidefontsize = 13,
                 tickfontsize = 11, legendfontsize = 11)
        gi == 1 && title!(p, "k = $k")
        k == 1 && ylabel!(p, GROUP_LABELS[gi] * "\nsurvivor F̄(k | θ, I)")
        gi == length(groups) && xlabel!(p, "investment I")

        for kθ in 1:r_θ
            idxs = [surv_idx(kI, kθ, k) for kI in 1:r_I]
            surv = [surv_mle[gi][i]  for i in idxs]
            sb   = [se_boot[gi][i]   for i in idxs]
            sa   = [se_asymp[gi][i]  for i in idxs]
            col  = SKILL_COLORS[kθ]
            # bootstrap band (filled ribbon)
            plot!(p, 1:r_I, surv, ribbon = 1.96 .* sb,
                  marker = :circle, markersize = 5, linewidth = 2.6,
                  fillalpha = 0.16, color = col,
                  label = show_legend ? "θ = $kθ" : "")
            # asymptotic band edges (thin dashed, same color) for comparison
            plot!(p, 1:r_I, surv .+ 1.96 .* sa, linestyle = :dash,
                  linewidth = 1.0, color = col, alpha = 0.6, label = "")
            plot!(p, 1:r_I, surv .- 1.96 .* sa, linestyle = :dash,
                  linewidth = 1.0, color = col, alpha = 0.6, label = "")
        end
        plots[gi, k] = p
    end
end

fig = plot(permutedims(plots)...,
    layout = (length(groups), n_cut),
    size = (460 * n_cut, 360 * length(groups) + 40),
    plot_titlefontsize = 11, margin = 3Plots.mm, left_margin = 12Plots.mm,
    top_margin = 7Plots.mm,
    plot_title = @sprintf("Survivor F̄(k|θ,I): bootstrap (filled) vs asymptotic %s (dashed) 95%% CI  ·  B=%d, budget=%d, ≈%.1f× wider",
                          asymp_label, B, bsel, mean_ratio))
savefig(fig, OUT)
println("Saved → $OUT")
@printf("Interior survivor bands: bootstrap ≈ %.2f× the asymptotic %s band (mean width); median per-entry %.2f×.\n",
        width_ratio, asymp_label, med_ratio)
