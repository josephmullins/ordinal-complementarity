# Survivor functions with BOTH bootstrap CI bands (filled) and asymptotic
# (delta-method) CI bands (dashed), for the regularized fit — direct visual
# comparison.  Point + bootstrap from output/bootstrap_reg.jld2; asymptotic SEs
# computed here via compute_var_phi_v3_eps + delta_cov_survivor_bar (default
# delta-method path = what the paper reports).
using JLD2, Plots, Statistics, Printf, LinearAlgebra
const REPO = joinpath(@__DIR__, "..")
for f in ["parametric_models.jl","parametric_models_v3.jl","parametric_models_v3_eps.jl",
          "prepare_data.jl","estimation.jl","estimation_multiperiod.jl",
          "estimation_multiperiod_v3.jl","estimation_multiperiod_v3_eps.jl",
          "inference_common.jl","inference_v3_eps.jl"]
    include(joinpath(REPO, "src", f))
end
const TG = [[1,2,3],[4,5,6]]
const AGES = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13)]
const ITEMS = [
 [:HowOftenRead,:HowManyBooks,:HowManyToysA,:HowManyToysB,:OutOfHouse,:HowOftenTalks,:HowOftenGrocery],
 [:HowOftenRead,:HowManyBooks,:HowManyToysA,:HowManyToysB,:OutOfHouse,:HowOftenTalks,:HowOftenGrocery,:HowManyZines,:HowOftenOuting,:FoodChoice,:LearnNumbers,:LearnAlphabet,:LearnColors,:LearnShapes,:RecordTapeCD],
 [:HowOftenRead,:HowManyBooks,:HowManyZines,:HowOftenOuting,:HowOftenMuseum,:RecordTapeCD,:FoodChoice],
 [:HowOftenRead,:HowManyBooks,:HowOftenMuseum,:HowOftenTheater,:HowOftenFamily,:HowOftenOutdoors,:ChildMusic,:SpecialActivities,:Hobbies,:EnjoyRead,:DailyNewspaper,:DiscussTV,:GradesTeacher,:GradesTalk,:GradesHelp,:GradesStudy],
 [:HowOftenRead,:HowManyBooks,:HowOftenMuseum,:HowOftenTheater,:HowOftenFamily,:HowOftenOutdoors,:HowOftenFather,:ChildMusic,:SpecialActivities,:Hobbies,:EnjoyRead,:DailyNewspaper,:DiscussTV,:GradesTeacher,:GradesTalk,:GradesHelp,:GradesStudy],
 [:HowManyBooks,:HowOftenMuseum,:HowOftenTheater,:HowOftenFamily,:HowOftenOutdoors,:HowOftenFather,:ChildMusic,:SpecialActivities,:Hobbies,:EnjoyRead,:DailyNewspaper,:DiscussTV,:GradesTeacher,:GradesTalk,:GradesHelp,:GradesStudy,:ToldPositive,:HowOftenClean,:HowOftenChores,:HowOftenTime],
 [:HowManyBooks,:HowOftenMuseum,:HowOftenTheater,:HowOftenFamily,:HowOftenOutdoors,:HowOftenFather,:ChildMusic,:SpecialActivities,:Hobbies,:EnjoyRead,:DailyNewspaper,:DiscussTV,:GradesTeacher,:GradesTalk,:GradesHelp,:GradesStudy,:ToldPositive,:HowOftenClean,:HowOftenChores,:HowOftenTime],
]
const SKILL_COLORS = [RGB(99/255,170/255,156/255), RGB(139/255,90/255,140/255),
                      RGB(0.90,0.72,0.33), RGB(233/255,99/255,59/255)]
const GROUP_LABELS = ["step 1: ages 0–5", "step 2: ages 6–13"]

println("Loading fit + data + asymptotic inference ...")
fit = load(joinpath(REPO,"output","estimation_v3_eps_reg.jld2"))
pv = fit["params_v3_eps"]; psk = fit["params_skill"]; (; r_θ, r_I) = psk
datasets = load_all_datasets()
panel = prepare_panel_data(datasets; age_ranges=AGES, invest_items=ITEMS, r_θ=r_θ, r_I=r_I)
data = merge(panel, (; r_P=3, r_ω=3))
inf = compute_var_phi_v3_eps(pv, psk, data; transition_groups=TG, verbose=false)
sb  = delta_cov_survivor_bar(inf, pv, psk; transition_groups=TG)   # per-group .ses (delta-method)

bc = load(joinpath(REPO,"output","bootstrap_reg.jld2"))
surv0 = bc["surv0"]; boot_surv = bc["boot_surv"]; ok = findall(bc["ok"]); B = bc["B"]
n_cut = r_θ - 1
surv_idx(kI,kθ,k) = (kI-1)*r_θ*n_cut + (kθ-1)*n_cut + k
se_boot = [vec(std(boot_surv[g][ok,:]; dims=1)) for g in 1:length(TG)]
se_asy  = [sb[g].ses for g in 1:length(TG)]

# report the headline ratio (interior cells)
interior = vcat([(surv0[g] .> 0.05) .& (surv0[g] .< 0.95) for g in 1:length(TG)]...)
@printf("Interior survivor SE: bootstrap mean=%.4f, asymptotic mean=%.4f, ratio=%.2f×\n",
        mean(vcat(se_boot...)[interior]), mean(vcat(se_asy...)[interior]),
        mean(vcat(se_boot...)[interior]) / mean(vcat(se_asy...)[interior]))

gr(); default(background_color=RGB(1,1,1), foreground_color=RGB(0,0,0), framestyle=:box,
    grid=false, left_margin=10Plots.mm, bottom_margin=8Plots.mm, right_margin=3Plots.mm, top_margin=3Plots.mm)
plots = Matrix{Any}(undef, length(TG), n_cut)
for gi in 1:length(TG), k in 1:n_cut
    leg = (gi==1 && k==n_cut)
    p = plot(ylims=(0,1.05), legend = leg ? :topleft : false,
        titlefontsize=13, guidefontsize=13, tickfontsize=11, legendfontsize=10)
    gi==1 && title!(p, "k = $k"); k==1 && ylabel!(p, GROUP_LABELS[gi]*"\nsurvivor F̄(k | θ, I)")
    gi==length(TG) && xlabel!(p, "investment I")
    for kθ in 1:r_θ
        idxs = [surv_idx(kI,kθ,k) for kI in 1:r_I]
        s  = [surv0[gi][i]   for i in idxs]
        sb_ = [se_boot[gi][i] for i in idxs]
        sa_ = [se_asy[gi][i]  for i in idxs]
        col = SKILL_COLORS[kθ]
        # bootstrap 95% band (filled)
        plot!(p, 1:r_I, s, ribbon=1.96 .* sb_, marker=:circle, markersize=5, linewidth=2.6,
              fillalpha=0.16, color=col, label = leg ? "θ = $kθ" : "")
        # asymptotic 95% band (dashed edges, same color, thin)
        plot!(p, 1:r_I, s .+ 1.96 .* sa_, linestyle=:dash, linewidth=1.1, color=col, alpha=0.7, label="")
        plot!(p, 1:r_I, s .- 1.96 .* sa_, linestyle=:dash, linewidth=1.1, color=col, alpha=0.7, label="")
    end
    plots[gi,k] = p
end
fig = plot(permutedims(plots)..., layout=(length(TG),n_cut),
    size=(460*n_cut, 360*length(TG)+50), plot_titlefontsize=12, margin=3Plots.mm,
    left_margin=12Plots.mm, top_margin=8Plots.mm,
    plot_title=@sprintf("Survivor F̄(k|θ,I), regularized fit:  bootstrap 95%% CI (filled)  vs  asymptotic 95%% CI (dashed)   ·   B=%d", B))
OUT = joinpath(REPO,"figures_boot","survivor_boot_vs_asymptotic.pdf")
isdir(dirname(OUT)) || mkpath(dirname(OUT))
savefig(fig, OUT)
println("Saved → $OUT")
