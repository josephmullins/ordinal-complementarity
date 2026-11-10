# Stage 0: validate the log_w ridge penalty.  Warm-start from the current MLE,
# run the regularized V3+ε_Y EM across a grid of λ, and for each report:
#   • final LL (should stay ≈ MLE if inference-neutral)
#   • max|log_w| per group (did the e^467 blow-up collapse to O(1)?)
#   • max|ΔΠ̄| vs the unregularized MLE (the inference-neutrality gate)
#   • β₅/δ (income channel — should be ~unchanged)
using LinearAlgebra, Random, Statistics, JLD2, Optim, Printf
const REPO = joinpath(@__DIR__, "..")
for f in ["parametric_models.jl","parametric_models_v3.jl","parametric_models_v3_eps.jl",
          "prepare_data.jl","estimation.jl","estimation_multiperiod.jl",
          "estimation_multiperiod_v3.jl","estimation_multiperiod_v3_eps.jl"]
    include(joinpath(REPO, "src", f))
end
BLAS.set_num_threads(1)
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

println("Loading data + MLE fit ...")
datasets = load_all_datasets()
panel = prepare_panel_data(datasets; age_ranges=AGES, invest_items=ITEMS, r_θ=4, r_I=5)
data = merge(panel, (; r_P=3, r_ω=3))
mle = load(joinpath(REPO,"output","estimation_v3_eps.jld2"))
pv0, psk0 = mle["params_v3_eps"], mle["params_skill"]
(; r_θ, r_I) = psk0

Pi_bar0 = marginalize_Pi_v3(pv0.trans_specs, TG, pv0.π_Pω, r_θ, r_I, pv0.r_P, pv0.r_ω)
ratio0 = pv0.ccp_spec.beta5 / pv0.income_model.delta
@printf("\nMLE baseline: LL=%.1f  β₅/δ=%.4f  max|log_w| grp1=%.1f grp2=%.1f\n",
        mle["best_ll"], ratio0,
        maximum(abs.(pv0.trans_specs[1].log_w)), maximum(abs.(pv0.trans_specs[2].log_w)))

const LAMBDAS = haskey(ENV, "LAMBDAS") ?
    [parse(Float64, s) for s in split(ENV["LAMBDAS"], ',')] :
    [0.0, 0.01, 0.1, 1.0, 10.0]
println("\n" * "="^92)
@printf("  %-8s %12s %10s %10s %10s %10s %10s\n",
        "λ", "LL", "ΔLL", "max|lw|1", "max|lw|2", "maxΔΠ̄", "β₅/δ")
println("="^92)
for λ in LAMBDAS
    pv, psk, _, _, _, ll = expectation_maximization_v3_eps(
        data, pv0, psk0; transition_groups=TG, maxiter=300, tol=1e-4,
        verbose=false, λ_logw=λ)
    Pib = marginalize_Pi_v3(pv.trans_specs, TG, pv.π_Pω, r_θ, r_I, pv.r_P, pv.r_ω)
    dΠ = maximum(maximum(abs.(Pib[g[1]] .- Pi_bar0[g[1]])) for g in TG)
    @printf("  %-8.3g %12.1f %10.2f %10.1f %10.1f %10.4f %10.4f\n",
            λ, ll, ll - mle["best_ll"],
            maximum(abs.(pv.trans_specs[1].log_w)),
            maximum(abs.(pv.trans_specs[2].log_w)),
            dΠ, pv.ccp_spec.beta5 / pv.income_model.delta)
    flush(stdout)
end
println("\nInference-neutral λ = largest λ with maxΔΠ̄ ≈ 0 and ΔLL ≈ 0 but max|log_w| now O(1).")
