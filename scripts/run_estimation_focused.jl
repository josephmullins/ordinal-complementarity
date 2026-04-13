# ─────────────────────────────────────────────────────────────────
# run_estimation_focused.jl
#
# Estimation with FOCUSED cognitive skill measures and separate,
# age-dependent emission matrices. Each period within an age group
# uses its own best cognitive measure source — they need NOT match.
#
# Source assignments (chosen by data availability & quality):
#
#   Ages (0,1)→(2,3):   t1=MSD,  t2=MSD
#     MSD is the only source at ages 0-3.
#
#   Ages (2,3)→(4,5):   t1=MSD,  t2=MSD+PPVT
#     MSD informative at 2-3 (60 items). At 4-5, MSD has 15
#     items (.2247) near ceiling; PPVT has N≈5190. Combined
#     gives better coverage than PIAT alone (only age 5, N≈2026).
#
#   Ages (4,5)→(6,7):   t1=MSD+PPVT, t2=PIAT
#     MSD .2247 still has variation at 4-5; PPVT has N≈5190.
#     Combined gives better t1 coverage. t2 PIAT N≈7258.
#
#   Ages (6,7)→(8,9):   t1=PIAT, t2=PIAT
#     Largest N (≈7258/7334); MSD has 0 informative items.
#
#   Ages (8,9)→(10,11): t1=PIAT, t2=PIAT   (N≈7334/7311)
#   Ages (10,11)→(12,13): t1=PIAT, t2=PIAT (N≈7311/6902)
#
# Alternatives: DigitSpan (ages 7+), PPVT (ages 3+) can be
# substituted by changing skill_source_t1/t2 in the config.
#
# Benefits:
#   - Each period uses the best available measure for that age
#   - No constraint that same measures appear at both periods
#   - Separate emission matrices per period (not shared)
#   - Fewer parameters per period → better identified
# ─────────────────────────────────────────────────────────────────

using LinearAlgebra, Random, Statistics, JLD2

include(joinpath(@__DIR__, "..", "src", "prepare_data.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation.jl"))

# ─────────────────────────────────────────────────────────────────
# Age group configurations with per-period skill source
#
# skill_source_t1 / skill_source_t2 can each be a Symbol or Vector{Symbol}:
#   :msd   — use only MSD items (binary, κ=2)
#   :piat  — use only PIAT measures (MATH, RECOG, COMP; κ=10)
#   :digit — use only Digit Span (DIGITF, DIGITB; κ=10)
#   :ppvt  — use only PPVT (PPVTRaw; κ=10)
#   [:msd, :ppvt] — use both MSD and PPVT (exclude PIAT and DigitSpan)
# ─────────────────────────────────────────────────────────────────

const AGE_GROUP_CONFIGS = [
    (age_t1=(0,1), age_t2=(2,3),
     skill_source_t1=:msd, skill_source_t2=:msd,
     invest_items=[
        :HowOftenRead, :HowManyBooks, :HowManyToysA, :HowManyToysB,
        :OutOfHouse, :HowOftenTalks, :HowOftenGrocery]),

    (age_t1=(2,3), age_t2=(4,5),
     skill_source_t1=:msd, skill_source_t2=[:msd, :ppvt],
     invest_items=[
        :HowOftenRead, :HowManyBooks, :HowManyToysA, :HowManyToysB,
        :OutOfHouse, :HowOftenTalks, :HowOftenGrocery,
        :HowManyZines, :HowOftenOuting, :FoodChoice,
        :LearnNumbers, :LearnAlphabet, :LearnColors, :LearnShapes, :RecordTapeCD]),

    (age_t1=(4,5), age_t2=(6,7),
     skill_source_t1=[:msd, :ppvt], skill_source_t2=:piat,
     invest_items=[
        :HowOftenRead, :HowManyBooks, :HowManyZines, :HowOftenOuting,
        :HowOftenMuseum, :LearnNumbers, :LearnAlphabet, :LearnColors,
        :LearnShapes, :RecordTapeCD, :FoodChoice]),

    (age_t1=(6,7), age_t2=(8,9),
     skill_source_t1=:piat, skill_source_t2=:piat,
     invest_items=[
        :HowOftenRead, :HowManyBooks, :HowOftenMuseum, :HowOftenTheater,
        :HowOftenFamily, :HowOftenOutdoors, :ChildMusic, :SpecialActivities,
        :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
        :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy]),

    (age_t1=(8,9), age_t2=(10,11),
     skill_source_t1=:piat, skill_source_t2=:piat,
     invest_items=[
        :HowOftenRead, :HowManyBooks, :HowOftenMuseum, :HowOftenTheater,
        :HowOftenFamily, :HowOftenOutdoors, :HowOftenFather, :ChildMusic,
        :SpecialActivities, :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
        :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy]),

    (age_t1=(10,11), age_t2=(12,13),
     skill_source_t1=:piat, skill_source_t2=:piat,
     invest_items=[
        :HowManyBooks, :HowOftenMuseum, :HowOftenTheater, :HowOftenFamily,
        :HowOftenOutdoors, :HowOftenFather, :ChildMusic, :SpecialActivities,
        :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
        :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy,
        :ToldPositive, :HowOftenClean, :HowOftenChores, :HowOftenTime]),
]

# ─────────────────────────────────────────────────────────────────
# Map skill_source to exclude lists
# ─────────────────────────────────────────────────────────────────

const ALL_CONTINUOUS_SOURCES = [:ppvt, :piat, :digit]
const ALL_SOURCES = [:msd, :ppvt, :piat, :digit]

"""
    sources_to_exclude(skill_source) -> Vector{Symbol}

Given a source specification (Symbol or Vector{Symbol}), return the list of
continuous sources to exclude. MSD exclusion is handled separately via `uses_msd()`.
"""
sources_to_exclude(s::Symbol) = sources_to_exclude([s])
function sources_to_exclude(sources::Vector{Symbol})
    for s in sources
        s in ALL_SOURCES || error("Unknown skill_source: $s")
    end
    return [s for s in ALL_CONTINUOUS_SOURCES if s ∉ sources]
end

"Check whether MSD is included in the source specification."
uses_msd(s::Symbol) = s == :msd
uses_msd(sources::Vector{Symbol}) = :msd in sources

"Format source specification as a string for display/filenames."
source_str(s::Symbol) = string(s)
source_str(sources::Vector{Symbol}) = join(sort(string.(sources)), "+")

# ─────────────────────────────────────────────────────────────────
# Load datasets
# ─────────────────────────────────────────────────────────────────

datasets = load_all_datasets()

# ─────────────────────────────────────────────────────────────────
# Run estimation for each age group
# ─────────────────────────────────────────────────────────────────

outdir = joinpath(dirname(@__DIR__), "output")
mkpath(outdir)

for (g, cfg) in enumerate(AGE_GROUP_CONFIGS)
    label = "$(cfg.age_t1[1])_$(cfg.age_t1[2])__$(cfg.age_t2[1])_$(cfg.age_t2[2])"
    src_t1 = cfg.skill_source_t1
    src_t2 = cfg.skill_source_t2

    println("\n" * "="^60)
    println("  AGE GROUP $g: $(cfg.age_t1) → $(cfg.age_t2)")
    println("    skill_source_t1=$(source_str(src_t1)), skill_source_t2=$(source_str(src_t2))")
    println("="^60)

    # Determine what to exclude at each period
    excl_t1 = sources_to_exclude(src_t1)
    excl_t2 = sources_to_exclude(src_t2)
    exclude_msd_t1 = !uses_msd(src_t1)
    exclude_msd_t2 = !uses_msd(src_t2)

    println("  Period 1: exclude continuous=$excl_t1, exclude MSD=$exclude_msd_t1")
    println("  Period 2: exclude continuous=$excl_t2, exclude MSD=$exclude_msd_t2")

    # ── Data preparation ──
    println("\n  Data Preparation:")
    data_raw = prepare_age_group_data(datasets;
        age_range_t1=cfg.age_t1,
        age_range_t2=cfg.age_t2,
        r_θ=3,
        r_I=3,
        invest_items=cfg.invest_items,
        exclude_sources_t1=excl_t1,
        exclude_sources_t2=excl_t2,
    )

    # Post-filter MSD items (κ=2) from periods that use non-MSD sources
    needs_filter = exclude_msd_t1 || exclude_msd_t2

    if needs_filter
        keep_θ = exclude_msd_t1 ?
            [i for i in eachindex(data_raw.κ_θ) if data_raw.κ_θ[i] != 2] :
            collect(eachindex(data_raw.κ_θ))
        keep_Y = exclude_msd_t2 ?
            [i for i in eachindex(data_raw.κ_Y) if data_raw.κ_Y[i] != 2] :
            collect(eachindex(data_raw.κ_Y))

        if isempty(keep_θ)
            println("  WARNING: No $src_t1 measures at period 1 — skipping.")
            continue
        end
        if isempty(keep_Y)
            println("  WARNING: No $src_t2 measures at period 2 — skipping.")
            continue
        end

        data = (;
            N = data_raw.N,
            r_θ = data_raw.r_θ,
            r_I = data_raw.r_I,
            X_θ = data_raw.X_θ[keep_θ],
            κ_θ = data_raw.κ_θ[keep_θ],
            measure_names_θ = data_raw.measure_names_θ[keep_θ],
            X_I = data_raw.X_I,
            κ_I = data_raw.κ_I,
            measure_names_I = data_raw.measure_names_I,
            Y = data_raw.Y[keep_Y],
            κ_Y = data_raw.κ_Y[keep_Y],
            measure_names_Y = data_raw.measure_names_Y[keep_Y],
            kids = data_raw.kids,
        )
    else
        data = data_raw
    end

    println("  Period 1 skill measures ($(source_str(src_t1))): $(length(data.X_θ))  $(data.measure_names_θ)")
    println("  Period 2 skill measures ($(source_str(src_t2))): $(length(data.Y))  $(data.measure_names_Y)")
    println("  Investment measures: $(length(data.X_I))")
    println("  N = $(data.N)")

    if data.N < 50
        println("  WARNING: Only $(data.N) observations — skipping this age group.")
        continue
    end

    # ── EM estimation ──
    println("\n  EM Estimation:")
    best_params, best_Q, best_ll = run_em_restarts(data)
    println("\n  Best log-likelihood: $(round(best_ll; digits=2))")

    # ── Save results ──
    str_t1 = source_str(src_t1)
    str_t2 = source_str(src_t2)
    src_label = str_t1 == str_t2 ? str_t1 : "$(str_t1)_$(str_t2)"
    outpath = joinpath(outdir, "estimation_$(label)_$(src_label).jld2")
    @save outpath best_params best_Q data best_ll
    println("  Results saved to $outpath")
end

println("\n" * "="^60)
println("  SUMMARY OF FOCUSED ESTIMATION")
println("="^60)
println()
println("  Skill sources by age group:")
for cfg in AGE_GROUP_CONFIGS
    str_t1 = source_str(cfg.skill_source_t1)
    str_t2 = source_str(cfg.skill_source_t2)
    if str_t1 == str_t2
        println("    $(cfg.age_t1) → $(cfg.age_t2):  $str_t1 (both periods)")
    else
        println("    $(cfg.age_t1) → $(cfg.age_t2):  t1=$str_t1, t2=$str_t2")
    end
end
println()
println("  Result files saved with _<source> suffix")
println("  When sources differ: _<src_t1>_<src_t2> (e.g., _msd_piat)")
println("\nDone.")
