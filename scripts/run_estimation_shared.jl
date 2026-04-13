# ─────────────────────────────────────────────────────────────────
# run_estimation_shared.jl
#
# Same as run_estimation.jl, but constrains the skill emission
# matrices to be shared across periods (P_θ = P_Y).
#
# Because period 1 and period 2 may have different cognitive
# measures, we first filter to the subset of measures that appear
# at BOTH periods (matched by name AND κ), then run EM with
# shared_skill_emission=true.
# ─────────────────────────────────────────────────────────────────

using LinearAlgebra, Random, Statistics, JLD2

include(joinpath(@__DIR__, "..", "src", "prepare_data.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation.jl"))

# ─────────────────────────────────────────────────────────────────
# Age group configurations (identical to run_estimation.jl)
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
# Filter data to shared skill measures
# ─────────────────────────────────────────────────────────────────

"""
    make_shared_data(data) -> NamedTuple

Filter X_θ and Y to only keep measures that appear at BOTH periods
(matched by name AND same κ). Sets `shared_skill_emission=true`.

Returns a new data tuple suitable for EM with shared skill emissions.
Prints detailed information about which measures are kept and dropped.
"""
function make_shared_data(data)
    (; N, r_θ, r_I,
       X_θ, κ_θ, measure_names_θ,
       X_I, κ_I, measure_names_I,
       Y, κ_Y, measure_names_Y,
       kids) = data

    # Build lookup: measure name -> (index, κ) for each period
    t1_lookup = Dict{String, Tuple{Int, Int}}()
    for (m, name) in enumerate(measure_names_θ)
        t1_lookup[name] = (m, κ_θ[m])
    end

    t2_lookup = Dict{String, Tuple{Int, Int}}()
    for (m, name) in enumerate(measure_names_Y)
        t2_lookup[name] = (m, κ_Y[m])
    end

    # Find shared measures: present in both periods with the same κ
    shared_names = String[]
    for name in measure_names_θ
        if haskey(t2_lookup, name)
            κ1 = t1_lookup[name][2]
            κ2 = t2_lookup[name][2]
            if κ1 == κ2
                push!(shared_names, name)
            end
        end
    end
    # Deduplicate while preserving order (should already be unique, but be safe)
    shared_names = unique(shared_names)

    # Report what was kept and dropped
    t1_only = setdiff(measure_names_θ, shared_names)
    t2_only = setdiff(measure_names_Y, shared_names)
    mismatched_κ = String[]
    for name in measure_names_θ
        if haskey(t2_lookup, name)
            κ1 = t1_lookup[name][2]
            κ2 = t2_lookup[name][2]
            if κ1 != κ2
                push!(mismatched_κ, "$name (κ_t1=$κ1, κ_t2=$κ2)")
            end
        end
    end

    println("\n  Shared skill emission filtering:")
    println("    Period 1 measures: $(length(measure_names_θ))")
    println("    Period 2 measures: $(length(measure_names_Y))")
    println("    Shared measures:   $(length(shared_names))")
    if !isempty(t1_only)
        println("    Dropped (t1 only): $(length(t1_only))")
        for name in t1_only
            println("      - $name")
        end
    end
    if !isempty(t2_only)
        println("    Dropped (t2 only): $(length(t2_only))")
        for name in t2_only
            println("      - $name")
        end
    end
    if !isempty(mismatched_κ)
        println("    Dropped (κ mismatch): $(length(mismatched_κ))")
        for desc in mismatched_κ
            println("      - $desc")
        end
    end
    if !isempty(shared_names)
        println("    Kept measures:")
        for name in shared_names
            κ = t1_lookup[name][2]
            println("      + $name (κ=$κ)")
        end
    end

    if isempty(shared_names)
        error("No shared skill measures between periods — cannot use shared emission constraint.")
    end

    # Build filtered arrays
    X_θ_shared = Vector{Vector{Int}}()
    Y_shared   = Vector{Vector{Int}}()
    κ_shared   = Int[]
    names_shared = String[]

    for name in shared_names
        m1 = t1_lookup[name][1]
        m2 = t2_lookup[name][1]
        push!(X_θ_shared, X_θ[m1])
        push!(Y_shared, Y[m2])
        push!(κ_shared, t1_lookup[name][2])
        push!(names_shared, name)
    end

    return (;
        N, r_θ, r_I,
        X_θ = X_θ_shared, κ_θ = κ_shared, measure_names_θ = names_shared,
        X_I, κ_I, measure_names_I,
        Y = Y_shared, κ_Y = κ_shared, measure_names_Y = names_shared,
        kids,
        shared_skill_emission = true,
    )
end

# ─────────────────────────────────────────────────────────────────
# Load all datasets once (MSD, PPVT, PIAT, DigitSpan, HOME)
# ─────────────────────────────────────────────────────────────────

datasets = load_all_datasets()

# ─────────────────────────────────────────────────────────────────
# Run estimation for each age group with shared skill emissions
# ─────────────────────────────────────────────────────────────────

outdir = joinpath(dirname(@__DIR__), "output")
mkpath(outdir)

for (g, cfg) in enumerate(AGE_GROUP_CONFIGS)
    label = "$(cfg.age_t1[1])_$(cfg.age_t1[2])__$(cfg.age_t2[1])_$(cfg.age_t2[2])"

    println("\n" * "="^60)
    println("  AGE GROUP $g: $(cfg.age_t1) → $(cfg.age_t2)  [shared skill emission]")
    println("="^60)

    # ── Data preparation ──
    println("\n  Data Preparation:")
    data_full = prepare_age_group_data(datasets;
        age_range_t1=cfg.age_t1,
        age_range_t2=cfg.age_t2,
        r_θ=3,
        r_I=3,
        invest_items=cfg.invest_items,
        exclude_sources_t1=hasproperty(cfg, :exclude_sources_t1) ? cfg.exclude_sources_t1 : Symbol[],
        exclude_sources_t2=hasproperty(cfg, :exclude_sources_t2) ? cfg.exclude_sources_t2 : Symbol[],
    )

    if data_full.N < 50
        println("  WARNING: Only $(data_full.N) observations — skipping this age group.")
        continue
    end

    # ── Filter to shared measures ──
    data = try
        make_shared_data(data_full)
    catch e
        println("  ERROR: $e")
        println("  Skipping this age group.")
        continue
    end

    # ── EM estimation with shared skill emissions ──
    println("\n  EM Estimation (shared_skill_emission=true):")
    best_params, best_Q, best_ll = run_em_restarts(data;
        shared_skill_emission=true)
    println("\n  Best log-likelihood: $(round(best_ll; digits=2))")

    # ── Save results with _shared suffix ──
    outpath = joinpath(outdir, "estimation_$(label)_shared.jld2")
    @save outpath best_params best_Q data best_ll
    println("  Results saved to $outpath")
end

println("\nDone. Run analyze_results.jl to inspect estimates and test complementarity.")
