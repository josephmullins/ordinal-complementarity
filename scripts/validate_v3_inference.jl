using JLD2, LinearAlgebra, Statistics, ForwardDiff, Random, Printf

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3_eps.jl"))
include(joinpath(@__DIR__, "..", "src", "prepare_data.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod_v3.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod_v3_eps.jl"))
include(joinpath(@__DIR__, "..", "src", "inference_common.jl"))
include(joinpath(@__DIR__, "..", "src", "inference_v3.jl"))

# ─────────────────────────────────────────────────────────────────
# Validate: compare V3 old (buggy) vs new (correct) inference
#
# 1. Load V3 adult fit
# 2. Run old compute_psi_all_v3 → compute_Var_free_v3 → SEs on Π̄_free
# 3. Run new per_child_scores_v3 → delta_cov_survivor_bar_v3 → SEs on Π̄_free
# 4. Compare
# ─────────────────────────────────────────────────────────────────

const OUTPUT_DIR = joinpath(dirname(@__DIR__), "output")
const V3_FIT = joinpath(OUTPUT_DIR, "estimation_v3_adult.jld2")
const TRANSITION_GROUPS = [[1, 2, 3], [4, 5, 6]]

const PANEL_AGE_RANGES = [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13)]
const PANEL_INVEST_ITEMS = [
    [:HowOftenRead, :HowManyBooks, :HowManyToysA, :HowManyToysB,
     :OutOfHouse, :HowOftenTalks, :HowOftenGrocery],
    [:HowOftenRead, :HowManyBooks, :HowManyToysA, :HowManyToysB,
     :OutOfHouse, :HowOftenTalks, :HowOftenGrocery,
     :HowManyZines, :HowOftenOuting, :FoodChoice,
     :LearnNumbers, :LearnAlphabet, :LearnColors, :LearnShapes, :RecordTapeCD],
    [:HowOftenRead, :HowManyBooks, :HowManyZines, :HowOftenOuting,
     :HowOftenMuseum, :RecordTapeCD, :FoodChoice],
    [:HowOftenRead, :HowManyBooks, :HowOftenMuseum, :HowOftenTheater,
     :HowOftenFamily, :HowOftenOutdoors, :ChildMusic, :SpecialActivities,
     :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
     :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy],
    [:HowOftenRead, :HowManyBooks, :HowOftenMuseum, :HowOftenTheater,
     :HowOftenFamily, :HowOftenOutdoors, :HowOftenFather, :ChildMusic,
     :SpecialActivities, :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
     :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy],
    [:HowManyBooks, :HowOftenMuseum, :HowOftenTheater, :HowOftenFamily,
     :HowOftenOutdoors, :HowOftenFather, :ChildMusic, :SpecialActivities,
     :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
     :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy,
     :ToldPositive, :HowOftenClean, :HowOftenChores, :HowOftenTime],
    [:HowManyBooks, :HowOftenMuseum, :HowOftenTheater, :HowOftenFamily,
     :HowOftenOutdoors, :HowOftenFather, :ChildMusic, :SpecialActivities,
     :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
     :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy,
     :ToldPositive, :HowOftenClean, :HowOftenChores, :HowOftenTime],
]

# ── Load data ──
println("Loading data...")
datasets = load_all_datasets()
panel = prepare_panel_data(datasets;
    age_ranges = PANEL_AGE_RANGES,
    invest_items = PANEL_INVEST_ITEMS,
    r_θ = 4, r_I = 5)
data_v3 = merge(panel, (; r_P = 3, r_ω = 3))

# ── Load V3 adult fit ──
println("Loading V3 adult fit from $V3_FIT...")
saved = load(V3_FIT)
pv3  = saved["params_v3"]
psk  = saved["params_skill"]
ll   = saved["best_ll"]
(; r_θ, r_I, T) = psk
(; r_P, r_ω) = pv3
N = data_v3.N
println("  N=$N, T=$T, r_θ=$r_θ, r_I=$r_I, r_P=$r_P, r_ω=$r_ω, ll=$ll")

# ═══════════════════════════════════════════════════════════════
# Part A: OLD inference (buggy Π̄-denominator approach)
# ═══════════════════════════════════════════════════════════════

println("\n" * "="^60)
println("  Part A: V3 OLD inference (compute_psi_all_v3 + compute_Var_free_v3)")
println("="^60)

# Regime-marginalized Π̄
Pi_bar_old = marginalize_Pi_v3(pv3.trans_specs, TRANSITION_GROUPS, pv3.π_Z,
                                r_θ, r_I, r_P, r_ω)

# Old ψ computation
println("Computing ψ_all (old method)...")
@time ψ_old = compute_psi_all_v3(pv3, psk, data_v3, TRANSITION_GROUPS)
println("  ψ_all size: $(size(ψ_old))")

# Old variance computation
println("Computing Var_free (old method)...")
@time Var_free_old, Σ_φ_old = compute_Var_free_v3(ψ_old, Pi_bar_old, pv3, data_v3, TRANSITION_GROUPS)
se_old = [sqrt.(max.(diag(V), 0.0)) for V in Var_free_old]
for (g, group) in enumerate(TRANSITION_GROUPS)
    t = group[1]
    println("  Group $g (t=$t): max SE = $(round(maximum(se_old[t]); digits=4)), " *
            "min SE = $(round(minimum(se_old[t]); digits=4))")
end

# ═══════════════════════════════════════════════════════════════
# Part B: NEW inference (corrected regime-specific Π_z score)
# ═══════════════════════════════════════════════════════════════

println("\n" * "="^60)
println("  Part B: V3 NEW inference (per_child_scores_v3)")
println("="^60)

println("Computing per-child scores (new method)...")
@time inf_new = compute_var_phi_v3(pv3, psk, data_v3;
                                     transition_groups = TRANSITION_GROUPS,
                                     verbose = true)

println("Computing survivor-bar covariance (new method)...")
@time sb_new = delta_cov_survivor_bar_v3(inf_new, pv3, psk;
                                           transition_groups = TRANSITION_GROUPS)
for res in sb_new
    println("  Group $(res.group): max SE = $(round(maximum(res.ses); digits=4)), " *
            "min SE = $(round(minimum(res.ses); digits=4))")
end

# ═══════════════════════════════════════════════════════════════
# Part C: Comparison
# ═══════════════════════════════════════════════════════════════

println("\n" * "="^60)
println("  Part C: Comparison (old vs new SEs on Π̄_free entries)")
println("="^60)

for (g, group) in enumerate(TRANSITION_GROUPS)
    t = group[1]
    se_o = se_old[t]
    se_n = sb_new[g].ses

    # Check dimensions match
    n_free = (r_θ - 1) * r_θ * r_I
    if length(se_o) != n_free || length(se_n) != n_free
        println("  Group $g: dimension mismatch! old=$(length(se_o)), new=$(length(se_n)), expected=$n_free")
        continue
    end

    # Ratio new/old
    ratios = se_n ./ max.(se_o, 1e-15)
    finite_ratios = filter(isfinite, ratios[se_o .> 1e-8])
    if !isempty(finite_ratios)
        @printf("  Group %d:  SE ratio (new/old):  mean=%.3f  median=%.3f  min=%.3f  max=%.3f\n",
                g, mean(finite_ratios), median(finite_ratios),
                minimum(finite_ratios), maximum(finite_ratios))
    else
        println("  Group $g: no finite SE ratios (check data)")
    end

    # Print a few entries side by side
    println("  Sample entries (old SE, new SE, ratio):")
    for idx in [1, n_free÷4, n_free÷2, 3*n_free÷4, n_free]
        @printf("    [%d]  %.4f  →  %.4f  (×%.2f)\n",
                idx, se_o[idx], se_n[idx], se_n[idx] / max(se_o[idx], 1e-15))
    end
end

println("\nDone.")
