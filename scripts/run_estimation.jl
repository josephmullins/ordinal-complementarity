using LinearAlgebra, Random, Statistics, JLD2

include(joinpath(@__DIR__, "..", "src", "prepare_data.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation.jl"))

# ─────────────────────────────────────────────────────────────────
# Age group configurations (from estimation plan)
# ─────────────────────────────────────────────────────────────────

const AGE_GROUP_CONFIGS = [
    (age_t1=(0,1), age_t2=(2,3), invest_items=[
        :HowOftenRead, :HowManyBooks, :HowManyToysA, :HowManyToysB,
        :OutOfHouse, :HowOftenTalks, :HowOftenGrocery]),
    (age_t1=(2,3), age_t2=(4,5), invest_items=[
        :HowOftenRead, :HowManyBooks, :HowManyToysA, :HowManyToysB,
        :OutOfHouse, :HowOftenTalks, :HowOftenGrocery,
        :HowManyZines, :HowOftenOuting, :FoodChoice,
        :LearnNumbers, :LearnAlphabet, :LearnColors, :LearnShapes, :RecordTapeCD]),
    (age_t1=(4,5), age_t2=(6,7), invest_items=[
        :HowOftenRead, :HowManyBooks, :HowManyZines, :HowOftenOuting,
        :HowOftenMuseum, :LearnNumbers, :LearnAlphabet, :LearnColors,
        :LearnShapes, :RecordTapeCD, :FoodChoice]),
    (age_t1=(6,7), age_t2=(8,9), invest_items=[
        :HowOftenRead, :HowManyBooks, :HowOftenMuseum, :HowOftenTheater,
        :HowOftenFamily, :HowOftenOutdoors, :ChildMusic, :SpecialActivities,
        :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
        :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy]),
    (age_t1=(8,9), age_t2=(10,11), invest_items=[
        :HowOftenRead, :HowManyBooks, :HowOftenMuseum, :HowOftenTheater,
        :HowOftenFamily, :HowOftenOutdoors, :HowOftenFather, :ChildMusic,
        :SpecialActivities, :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
        :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy]),
    (age_t1=(10,11), age_t2=(12,13), invest_items=[
        :HowManyBooks, :HowOftenMuseum, :HowOftenTheater, :HowOftenFamily,
        :HowOftenOutdoors, :HowOftenFather, :ChildMusic, :SpecialActivities,
        :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
        :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy,
        :ToldPositive, :HowOftenClean, :HowOftenChores, :HowOftenTime]),
]

# ─────────────────────────────────────────────────────────────────
# Load all datasets once (MSD, PPVT, PIAT, DigitSpan, HOME)
# ─────────────────────────────────────────────────────────────────

datasets = load_all_datasets()

# ─────────────────────────────────────────────────────────────────
# Run estimation for each age group
# ─────────────────────────────────────────────────────────────────

outdir = joinpath(dirname(@__DIR__), "output")
mkpath(outdir)

for (g, cfg) in enumerate(AGE_GROUP_CONFIGS)
    label = "$(cfg.age_t1[1])_$(cfg.age_t1[2])__$(cfg.age_t2[1])_$(cfg.age_t2[2])"

    println("\n" * "="^60)
    println("  AGE GROUP $g: $(cfg.age_t1) → $(cfg.age_t2)")
    println("="^60)

    # ── Data preparation ──
    println("\n  Data Preparation:")
    data = prepare_age_group_data(datasets;
        age_range_t1=cfg.age_t1,
        age_range_t2=cfg.age_t2,
        r_θ=3,
        r_I=3,
        invest_items=cfg.invest_items,
        exclude_sources_t1=hasproperty(cfg, :exclude_sources_t1) ? cfg.exclude_sources_t1 : Symbol[],
        exclude_sources_t2=hasproperty(cfg, :exclude_sources_t2) ? cfg.exclude_sources_t2 : Symbol[],
    )

    if data.N < 50
        println("  WARNING: Only $(data.N) observations — skipping this age group.")
        continue
    end

    # ── EM estimation ──
    println("\n  EM Estimation:")
    best_params, best_Q, best_ll = run_em_restarts(data)
    println("\n  Best log-likelihood: $(round(best_ll; digits=2))")

    # ── Save results ──
    outpath = joinpath(outdir, "estimation_$(label).jld2")
    @save outpath best_params best_Q data best_ll
    println("  Results saved to $outpath")
end

println("\nDone. Run analyze_results.jl to inspect estimates and test complementarity.")
