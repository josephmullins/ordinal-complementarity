using CSV, DataFrames, Statistics

# ─────────────────────────────────────────────────────────────────
# Data paths
# ─────────────────────────────────────────────────────────────────
const NLSY = joinpath(dirname(@__DIR__), "NLSY")

const DATA_PATHS = (
    PPVT      = joinpath(NLSY, "PPVT", "PPVT_Clean.csv"),
    MSD       = joinpath(NLSY, "Motor & Social Development", "MSD_Clean.csv"),
    HomePanel = joinpath(dirname(@__DIR__), "output", "HOME_panel.csv"),
    DigitSpan = joinpath(NLSY, "DigitSpan", "DigitSpan_Clean.csv"),
    PIAT      = joinpath(NLSY, "PIAT", "PIAT_Clean.csv"),
)

# ─────────────────────────────────────────────────────────────────
# Shared ID / demographic columns
# ─────────────────────────────────────────────────────────────────
const ID_COLS = [:KID, :MID, :CHILDRACE, :CHILDSEX, :YEARBIRTH, :RVersion, :YEAR]

"""
    load_clean(path) -> DataFrame

Load a clean CSV, parse YEAR as Int, compute AGE = YEAR - YEARBIRTH.
Filters to AGE >= 0 (drops rows where YEARBIRTH > YEAR).
"""
function load_clean(path::String)
    df = CSV.read(path, DataFrame; missingstring=["NA",""])
    if hasproperty(df, :Column1)
        select!(df, Not(:Column1))
    end
    df.YEAR = [ismissing(y) ? missing : parse(Int, string(y)) for y in df.YEAR]
    dropmissing!(df, [:YEAR, :YEARBIRTH])
    df.AGE = df.YEAR .- df.YEARBIRTH
    # Filter to valid ages
    filter!(r -> r.AGE >= 0, df)
    return df
end

"""
    measure_cols(df) -> Vector{Symbol}

Return columns that are measures (not IDs or AGE).
"""
function measure_cols(df::DataFrame)
    skip = Set(vcat(ID_COLS, [:AGE]))
    return [c for c in propertynames(df) if c ∉ skip]
end

# Minimum N for reporting a measure as "available" at an age
const MIN_N = 30

# ─────────────────────────────────────────────────────────────────
# Output file
# ─────────────────────────────────────────────────────────────────
const OUTPUT_DIR = joinpath(dirname(@__DIR__), "output")
mkpath(OUTPUT_DIR)
const OUTPUT_FILE = joinpath(OUTPUT_DIR, "data_exploration.txt")

open(OUTPUT_FILE, "w") do io
redirect_stdout(io) do

println("NLSY Data Exploration Report")
println("="^80)
println("Minimum N for reporting availability: $MIN_N")
println()

# ─────────────────────────────────────────────────────────────────
# 1. PPVT (Peabody Picture Vocabulary Test)
# ─────────────────────────────────────────────────────────────────
println("="^80)
println("  1. PPVT (Peabody Picture Vocabulary Test)")
println("="^80)

df_ppvt = load_clean(DATA_PATHS.PPVT)
println("  Total valid observations: $(nrow(df_ppvt))")
println("  Unique children: $(length(unique(df_ppvt.KID)))")
println("  Age range: $(extrema(df_ppvt.AGE))")
println("  Survey years: $(sort(unique(df_ppvt.YEAR)))")

for score in [:PPVTRaw, :PPVTStandard]
    println("\n  $score scores by child age (ages with N ≥ $MIN_N):")
    println("  " * "-"^70)
    println("    $(rpad("Age", 6))$(rpad("N", 8))$(rpad("Mean", 10))$(rpad("SD", 10))$(rpad("Min", 8))$(rpad("Max", 8))")
    println("  " * "-"^70)
    for age in sort(unique(df_ppvt.AGE))
        vals = collect(skipmissing(df_ppvt[df_ppvt.AGE .== age, score]))
        if length(vals) >= MIN_N
            println("    $(rpad(age, 6))$(rpad(length(vals), 8))$(rpad(round(mean(Float64.(vals)); digits=1), 10))$(rpad(round(std(Float64.(vals)); digits=1), 10))$(rpad(round(Int, minimum(vals)), 8))$(rpad(round(Int, maximum(vals)), 8))")
        end
    end
end

println("\n  PPVT availability summary:")
println("    PPVTRaw and PPVTStandard available at ages 3-18 (primary: 3-14)")
println("    PPVT is NOT available before age 3")
println("    For estimation: use PPVTRaw (not age-normed)")

# ─────────────────────────────────────────────────────────────────
# 2. MSD (Motor & Social Development)
# ─────────────────────────────────────────────────────────────────
println("\n\n" * "="^80)
println("  2. MSD (Motor & Social Development)")
println("="^80)

df_msd = load_clean(DATA_PATHS.MSD)
println("  Total valid observations: $(nrow(df_msd))")
println("  Unique children: $(length(unique(df_msd.KID)))")
println("  Age range: $(extrema(df_msd.AGE))")

msd_cols = measure_cols(df_msd)
println("  Total MSD items: $(length(msd_cols))")

# Group items by their age-range suffix (design age in months)
age_groups = Dict{String, Vector{Symbol}}()
for c in msd_cols
    s = string(c)
    suffix = match(r"\.(\d+)$", s)
    if suffix !== nothing
        key = suffix.captures[1]
        push!(get!(age_groups, key, Symbol[]), c)
    end
end

# Design age range descriptions
design_ages = Dict(
    "03"   => "0-3 months",
    "46"   => "4-6 months",
    "79"   => "7-9 months",
    "1012" => "10-12 months",
    "1315" => "13-15 months",
    "1618" => "16-18 months",
    "1921" => "19-21 months",
    "2247" => "22-47 months",
)

println("\n  MSD item batteries (items grouped by designed age range in months):")
println("  " * "-"^70)

# For each battery, show pass rates by child age (in years)
grp_order = sort(collect(keys(age_groups)); by=x->parse(Int,x))
for grp in grp_order
    items = age_groups[grp]
    desc = get(design_ages, grp, "unknown")
    println("\n    Battery .$grp — designed for $desc ($(length(items)) items)")
    println("    Item names: $(join(string.(items), ", "))")
    println("    $(rpad("Child Age", 12))$(rpad("N children", 14))$(rpad("N item-obs", 14))$(rpad("Frac pass", 12))$(rpad("Note", 20))")
    println("    " * "-"^60)

    for age in sort(unique(df_msd.AGE))
        mask = df_msd.AGE .== age
        n_children = sum(mask)
        if n_children == 0
            continue
        end

        # Collect all item observations at this age
        vals = Float64[]
        n_nm_per_child = zeros(Int, n_children)
        idx = findall(mask)
        for (j, c) in enumerate(items)
            for (k, i) in enumerate(idx)
                v = df_msd[i, c]
                if !ismissing(v)
                    push!(vals, Float64(v))
                    n_nm_per_child[k] += 1
                end
            end
        end

        n_item_obs = length(vals)
        if n_item_obs < MIN_N
            continue
        end

        # How many children have at least one non-missing item from this battery?
        n_kids_with_data = sum(n_nm_per_child .> 0)
        frac_pass = round(mean(vals .== 1); digits=3)

        # Classify: primary vs off-target
        note = ""
        if n_kids_with_data < 50
            note = "sparse"
        elseif frac_pass > 0.95
            note = "ceiling"
        elseif frac_pass < 0.05
            note = "floor"
        end

        println("    $(rpad(age, 12))$(rpad(n_kids_with_data, 14))$(rpad(n_item_obs, 14))$(rpad(frac_pass, 12))$(note)")
    end
end

# Summary: which batteries are informative at which child ages
println("\n\n  SUMMARY: MSD battery × child age mapping (N children ≥ 100)")
println("  " * "-"^70)
println("  This table shows where each battery has substantial, informative data.")
println("  Items are only useful when they have variation (not at ceiling or floor).\n")
println("    $(rpad("Battery", 12))$(rpad("Design age", 16))$(rpad("Primary child ages (years)", 40))")
println("    " * "-"^60)

for grp in grp_order
    items = age_groups[grp]
    desc = get(design_ages, grp, "?")

    # Find ages with N_children >= 100 and 0.05 < frac_pass < 0.95
    primary_ages = Int[]
    for age in sort(unique(df_msd.AGE))
        mask = df_msd.AGE .== age
        n_children = sum(mask)
        if n_children == 0
            continue
        end

        vals = Float64[]
        n_nm_per_child = zeros(Int, n_children)
        idx = findall(mask)
        for c in items
            for (k, i) in enumerate(idx)
                v = df_msd[i, c]
                if !ismissing(v)
                    push!(vals, Float64(v))
                    n_nm_per_child[k] += 1
                end
            end
        end

        n_kids = sum(n_nm_per_child .> 0)
        if n_kids >= 100 && length(vals) >= MIN_N
            frac = mean(vals .== 1)
            if 0.05 < frac < 0.95
                push!(primary_ages, age)
            end
        end
    end

    age_str = isempty(primary_ages) ? "none" : join(primary_ages, ", ")
    println("    $(rpad(".$grp", 12))$(rpad(desc, 16))$(age_str)")
end

# ─────────────────────────────────────────────────────────────────
# 3. HOME Inventory (Unified Panel)
# ─────────────────────────────────────────────────────────────────
println("\n\n" * "="^80)
println("  3. HOME Inventory (Unified Panel)")
println("="^80)

if isfile(DATA_PATHS.HomePanel)
    df_home = CSV.read(DATA_PATHS.HomePanel, DataFrame; missingstring=["NA", ""])
    # Filter to valid ages if AGE column present
    if hasproperty(df_home, :AGE)
        filter!(r -> !ismissing(r.AGE) && r.AGE >= 0, df_home)
    end

    home_cols = measure_cols(df_home)
    # Drop unnamed C-prefixed columns (uncleaned NLSY codes)
    home_cols = [c for c in home_cols if !startswith(string(c), "C339")]

    println("  Total valid observations: $(nrow(df_home))")
    println("  Unique children: $(length(unique(df_home.KID)))")
    println("  Age range: $(extrema(skipmissing(df_home.AGE)))")
    println("  Total named measure columns: $(length(home_cols))")

    # Categorize HOME items by type
    investment_items = [
        :HowOftenRead, :HowManyBooks, :HowManyZines, :HowManyToysA, :HowManyToysB,
        :HowOftenOuting, :HowOftenMuseum, :HowOftenTheater, :HowOftenFamily,
        :HowOftenOutdoors, :HowOftenFather, :HowOftenTalks, :OutOfHouse,
        :HowOftenGrocery, :ChildMusic, :SpecialActivities, :Hobbies,
        :RecordTapePlayer, :RecordTapeCD,
        :LearnNumbers, :LearnAlphabet, :LearnColors, :LearnShapes,
        :EnjoyRead, :DailyNewspaper,
    ]
    parenting_items = [
        :HowOftenEats, :HowOftenBed, :HowOftenRoom, :HowOftenPickUp,
        :HowOftenSpank, :MotherSpank, :HowOftenAffection, :HowOftenPraised,
        :HowOftenCompliment, :ToldPositive, :HowOftenGrounded, :HowOftenPrivileges,
        :HowOftenSentRoom, :HowOftenAllowance,
        :FoodChoice, :DiscussTV, :HowOftenBathe, :HowOftenSpills,
        :HowOftenClean, :HowOftenChores, :HowOftenTime,
    ]
    discipline_items = [
        :TantrumGrounding, :TantrumSpanking, :TantrumTalk, :TantrumChores,
        :TantrumIgnore, :TantrumRoom, :TantrumAllowance, :TantrumPrivileges,
        :TantrumOther, :TantrumTimeOut,
        :HitHitBack, :HitSendToRoom, :HitSpanking, :HitTalkToChild,
        :HitIgnoreIt, :HitGiveChores, :HitOtherResponse, :HitTakeAllowance,
        :HitHoldHands, :HitTimeOut,
    ]
    grades_items = [
        :GradesTeacher, :GradesLecture, :GradesSupervise, :GradesPunish,
        :GradesTalk, :GradesImproves, :GradesStudy, :GradesHelp,
        :GradesOther, :GradesLimit,
    ]
    family_items = [
        :ChildSeeFather, :SeeFatherDaily, :BioFather, :ParentRelation,
        :CloseMother, :CloseFather, :CloseStepMom, :CloseStepDad,
        :CloseBioFather, :MotherBelief,
    ]
    tv_items = [
        :HoursTVWeekday, :HoursTVWeekend, :HoursTVOn, :HasTVHoursOn,
    ]

    categories = [
        ("Investment / Enrichment", investment_items),
        ("Parenting / Daily Routines", parenting_items),
        ("Discipline Responses", discipline_items),
        ("School / Grades", grades_items),
        ("Family Structure", family_items),
        ("TV / Screen Time", tv_items),
    ]

    for (cat_name, cat_items) in categories
        present = [c for c in cat_items if c ∈ Set(home_cols)]
        if isempty(present)
            continue
        end

        println("\n  --- $cat_name ($(length(present)) items) ---")
        println("    $(rpad("Measure", 28))$(rpad("Ages with data", 44))$(rpad("Total N", 10))")
        println("    " * "-"^80)

        for c in present
            col = df_home[!, c]
            # Find ages with substantial data
            age_ns = Tuple{Int, Int}[]
            for age in sort(unique(skipmissing(df_home.AGE)))
                mask = .!ismissing.(df_home.AGE) .& (df_home.AGE .== age)
                n_nm = sum(i -> !ismissing(col[i]), findall(mask))
                if n_nm >= MIN_N
                    push!(age_ns, (age, n_nm))
                end
            end

            if isempty(age_ns)
                continue
            end

            total_n = sum(last, age_ns)
            age_range_str = join(["$(a)($(n))" for (a, n) in age_ns], " ")

            println("    $(rpad(string(c), 28))$(rpad(age_range_str, 44))$(total_n)")
        end
    end

    # ── Key investment items: detailed statistics by age ──
    println("\n\n  DETAILED STATISTICS: Key Investment Items")
    println("  " * "-"^70)

    key_items = [:HowOftenRead, :HowManyBooks, :HowOftenMuseum, :HowOftenOuting,
                 :LearnNumbers, :LearnAlphabet, :ChildMusic, :SpecialActivities]

    for item in key_items
        item ∉ Set(home_cols) && continue
        println("\n    $item:")
        println("    $(rpad("Age", 6))$(rpad("N", 8))$(rpad("Mean", 10))$(rpad("SD", 10))$(rpad("Categories", 30))")
        println("    " * "-"^60)
        for age in sort(unique(skipmissing(df_home.AGE)))
            mask = .!ismissing.(df_home.AGE) .& (df_home.AGE .== age)
            vals = collect(skipmissing(df_home[mask, item]))
            if length(vals) >= MIN_N
                cats = sort(unique(vals))
                println("    $(rpad(age, 6))$(rpad(length(vals), 8))$(rpad(round(mean(Float64.(vals)); digits=2), 10))$(rpad(round(std(Float64.(vals)); digits=2), 10))$(cats)")
            end
        end
    end
else
    println("  WARNING: HOME_panel.csv not found. Run build_home_panel.jl first.")
end

# ─────────────────────────────────────────────────────────────────
# 4. Digit Span (Digits Forward & Backward)
# ─────────────────────────────────────────────────────────────────
println("\n\n" * "="^80)
println("  4. Digit Span (Digits Forward & Backward)")
println("="^80)

df_digit = load_clean(DATA_PATHS.DigitSpan)
println("  Total valid observations: $(nrow(df_digit))")
println("  Unique children: $(length(unique(df_digit.KID)))")
println("  Age range: $(extrema(df_digit.AGE))")
println("  Survey years: $(sort(unique(df_digit.YEAR)))")

for score in [:DIGITF, :DIGITB]
    println("\n  $score scores by child age (ages with N ≥ $MIN_N):")
    println("  " * "-"^70)
    println("    $(rpad("Age", 6))$(rpad("N", 8))$(rpad("Mean", 10))$(rpad("SD", 10))$(rpad("Min", 8))$(rpad("Max", 8))")
    println("  " * "-"^70)
    for age in sort(unique(df_digit.AGE))
        vals = collect(skipmissing(df_digit[df_digit.AGE .== age, score]))
        if length(vals) >= MIN_N
            println("    $(rpad(age, 6))$(rpad(length(vals), 8))$(rpad(round(mean(Float64.(vals)); digits=1), 10))$(rpad(round(std(Float64.(vals)); digits=1), 10))$(rpad(minimum(vals), 8))$(rpad(maximum(vals), 8))")
        end
    end
end

println("\n  Digit Span availability summary:")
println("    DIGITF and DIGITB available at ages ~5-20 (biennial 1992-2014)")

# ─────────────────────────────────────────────────────────────────
# 5. PIAT (Peabody Individual Achievement Test)
# ─────────────────────────────────────────────────────────────────
println("\n\n" * "="^80)
println("  5. PIAT (Peabody Individual Achievement Test)")
println("="^80)

df_piat = load_clean(DATA_PATHS.PIAT)
println("  Total valid observations: $(nrow(df_piat))")
println("  Unique children: $(length(unique(df_piat.KID)))")
println("  Age range: $(extrema(df_piat.AGE))")
println("  Survey years: $(sort(unique(df_piat.YEAR)))")

for score in [:MATH, :RECOG, :COMP]
    println("\n  $score scores by child age (ages with N ≥ $MIN_N):")
    println("  " * "-"^70)
    println("    $(rpad("Age", 6))$(rpad("N", 8))$(rpad("Mean", 10))$(rpad("SD", 10))$(rpad("Min", 8))$(rpad("Max", 8))")
    println("  " * "-"^70)
    for age in sort(unique(df_piat.AGE))
        vals = collect(skipmissing(df_piat[df_piat.AGE .== age, score]))
        if length(vals) >= MIN_N
            println("    $(rpad(age, 6))$(rpad(length(vals), 8))$(rpad(round(mean(Float64.(vals)); digits=1), 10))$(rpad(round(std(Float64.(vals)); digits=1), 10))$(rpad(minimum(vals), 8))$(rpad(maximum(vals), 8))")
        end
    end
end

println("\n  PIAT availability summary:")
println("    MATH available at ages ~5-18 (biennial 1986-2014)")
println("    RECOG available at ages ~5-18")
println("    COMP available at ages ~5-18")
println("    Note: PIAT is NOT available before age 5")

# ─────────────────────────────────────────────────────────────────
# 6. AGE BIN SUMMARY — Available measures for estimation
# ─────────────────────────────────────────────────────────────────
println("\n\n" * "="^80)
println("  6. AGE BIN SUMMARY — Measures Available for Estimation")
println("="^80)
println("\n  This section identifies which measures have enough data at each")
println("  age pair for two-period estimation (θ_t → θ_{t+1}).\n")

# Age bins for estimation (period 1 → period 2)
age_pairs = [
    ((0, 1), (2, 3), "Ages 0-1 → 2-3"),
    ((2, 3), (4, 5), "Ages 2-3 → 4-5"),
    ((4, 5), (6, 7), "Ages 4-5 → 6-7"),
    ((6, 7), (8, 9), "Ages 6-7 → 8-9"),
    ((8, 9), (10, 11), "Ages 8-9 → 10-11"),
    ((10, 11), (12, 13), "Ages 10-11 → 12-13"),
]

for (t1_range, t2_range, label) in age_pairs
    println("  ── $label ──")

    # MSD
    msd_t1 = filter(r -> t1_range[1] <= r.AGE <= t1_range[2], df_msd)
    msd_t2 = filter(r -> t2_range[1] <= r.AGE <= t2_range[2], df_msd)
    msd_t1_u = unique(msd_t1, :KID)
    msd_t2_u = unique(msd_t2, :KID)
    msd_both = intersect(Set(msd_t1_u.KID), Set(msd_t2_u.KID))

    println("    MSD:")
    println("      Period 1: $(length(unique(msd_t1.KID))) children")
    println("      Period 2: $(length(unique(msd_t2.KID))) children")
    println("      Both periods: $(length(msd_both)) children")

    # Count informative items (variation, not all ceiling/floor)
    if !isempty(msd_both)
        kid_set = msd_both
        msd_t1_f = filter(r -> r.KID ∈ kid_set, msd_t1_u)
        msd_t2_f = filter(r -> r.KID ∈ kid_set, msd_t2_u)

        n_informative_t1 = 0
        n_informative_t2 = 0
        for c in msd_cols
            vals_t1 = collect(skipmissing(msd_t1_f[!, c]))
            vals_t2 = collect(skipmissing(msd_t2_f[!, c]))
            if length(vals_t1) >= MIN_N && length(unique(vals_t1)) > 1
                n_informative_t1 += 1
            end
            if length(vals_t2) >= MIN_N && length(unique(vals_t2)) > 1
                n_informative_t2 += 1
            end
        end
        println("      Informative items (N≥$MIN_N, has variation): t1=$n_informative_t1, t2=$n_informative_t2")

        # Which batteries are informative?
        for grp in grp_order
            items = age_groups[grp]
            desc = get(design_ages, grp, "?")
            n_t1 = sum(length(collect(skipmissing(msd_t1_f[!, c]))) >= MIN_N && length(unique(collect(skipmissing(msd_t1_f[!, c])))) > 1 for c in items)
            n_t2 = sum(length(collect(skipmissing(msd_t2_f[!, c]))) >= MIN_N && length(unique(collect(skipmissing(msd_t2_f[!, c])))) > 1 for c in items)
            if n_t1 > 0 || n_t2 > 0
                println("        .$grp ($desc): t1=$n_t1/$(length(items)), t2=$n_t2/$(length(items))")
            end
        end
    end

    # PPVT
    ppvt_t1 = filter(r -> t1_range[1] <= r.AGE <= t1_range[2], df_ppvt)
    ppvt_t2 = filter(r -> t2_range[1] <= r.AGE <= t2_range[2], df_ppvt)
    ppvt_t1_u = unique(ppvt_t1, :KID)
    ppvt_t2_u = unique(ppvt_t2, :KID)
    ppvt_t1_nm = filter(r -> !ismissing(r.PPVTRaw), ppvt_t1_u)
    ppvt_t2_nm = filter(r -> !ismissing(r.PPVTRaw), ppvt_t2_u)

    println("    PPVT:")
    println("      Period 1: $(nrow(ppvt_t1_nm)) children with scores")
    println("      Period 2: $(nrow(ppvt_t2_nm)) children with scores")
    ppvt_both = intersect(Set(ppvt_t1_nm.KID), Set(ppvt_t2_nm.KID))
    println("      Both periods: $(length(ppvt_both)) children")

    # Digit Span
    digit_t1 = filter(r -> t1_range[1] <= r.AGE <= t1_range[2], df_digit)
    digit_t2 = filter(r -> t2_range[1] <= r.AGE <= t2_range[2], df_digit)
    digit_t1_u = unique(digit_t1, :KID)
    digit_t2_u = unique(digit_t2, :KID)
    digit_t1_nm = filter(r -> !ismissing(r.DIGITF), digit_t1_u)
    digit_t2_nm = filter(r -> !ismissing(r.DIGITF), digit_t2_u)
    digit_both = intersect(Set(digit_t1_nm.KID), Set(digit_t2_nm.KID))

    println("    Digit Span:")
    println("      Period 1: $(nrow(digit_t1_nm)) children with scores")
    println("      Period 2: $(nrow(digit_t2_nm)) children with scores")
    println("      Both periods: $(length(digit_both)) children")

    # PIAT
    piat_t1 = filter(r -> t1_range[1] <= r.AGE <= t1_range[2], df_piat)
    piat_t2 = filter(r -> t2_range[1] <= r.AGE <= t2_range[2], df_piat)
    piat_t1_u = unique(piat_t1, :KID)
    piat_t2_u = unique(piat_t2, :KID)
    piat_t1_nm = filter(r -> any(!ismissing(r[s]) for s in [:MATH, :RECOG, :COMP]), piat_t1_u)
    piat_t2_nm = filter(r -> any(!ismissing(r[s]) for s in [:MATH, :RECOG, :COMP]), piat_t2_u)
    piat_both = intersect(Set(piat_t1_nm.KID), Set(piat_t2_nm.KID))

    println("    PIAT:")
    println("      Period 1: $(nrow(piat_t1_nm)) children with scores")
    println("      Period 2: $(nrow(piat_t2_nm)) children with scores")
    println("      Both periods: $(length(piat_both)) children")

    # HOME
    if isfile(DATA_PATHS.HomePanel)
        home_t1 = filter(r -> !ismissing(r.AGE) && t1_range[1] <= r.AGE <= t1_range[2], df_home)
        home_t1_u = unique(home_t1, :KID)
        home_t1_kids = Set(home_t1_u.KID)

        # Count investment items with data
        inv_items_present = [c for c in home_cols
            if any(c in Set(cat_items) for (_, cat_items) in categories[1:1]) &&
               sum(!ismissing(v) for v in home_t1_u[!, c]) >= MIN_N]

        println("    HOME (investment):")
        println("      Period 1: $(length(home_t1_kids)) children")
        println("      Investment items with data: $(length(inv_items_present))")

        # Combined samples
        combined = intersect(msd_both, home_t1_kids)
        println("    COMBINED (MSD both + HOME t1): $(length(combined)) children")
        if !isempty(ppvt_both)
            combined_ppvt = intersect(ppvt_both, home_t1_kids)
            println("    COMBINED (PPVT both + HOME t1): $(length(combined_ppvt)) children")
        end
        if !isempty(digit_both)
            combined_digit = intersect(digit_both, home_t1_kids)
            println("    COMBINED (DigitSpan both + HOME t1): $(length(combined_digit)) children")
        end
        if !isempty(piat_both)
            combined_piat = intersect(piat_both, home_t1_kids)
            println("    COMBINED (PIAT both + HOME t1): $(length(combined_piat)) children")
        end
        # All cognitive measures combined
        all_cog_both = union(msd_both, ppvt_both, digit_both, piat_both)
        combined_all = intersect(all_cog_both, home_t1_kids)
        println("    COMBINED (any cognitive both + HOME t1): $(length(combined_all)) children")
    end

    println()
end

println("\n\nDone.")

end # redirect_stdout
end # open

println("Output written to $OUTPUT_FILE")
