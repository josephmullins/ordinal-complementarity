using CSV, DataFrames, Statistics

# ─────────────────────────────────────────────────────────────────
# Data paths
# ─────────────────────────────────────────────────────────────────
const PROJECT_ROOT = dirname(@__DIR__)
const NLSY_DIR = joinpath(PROJECT_ROOT, "NLSY")

const DATA_PATHS = (
    MSD          = joinpath(NLSY_DIR, "Motor & Social Development", "MSD_Clean.csv"),
    PPVT         = joinpath(NLSY_DIR, "PPVT", "PPVT_Clean.csv"),
    PIAT         = joinpath(NLSY_DIR, "PIAT", "PIAT_Clean.csv"),
    DigitSpan    = joinpath(NLSY_DIR, "DigitSpan", "DigitSpan_Clean.csv"),
    HomePanel    = joinpath(PROJECT_ROOT, "output", "HOME_panel.csv"),
    HomePartA    = joinpath(NLSY_DIR, "HOME (A, B, C, D)", "HomePartA", "HomePartA_Clean.csv"),
    HomePartB    = joinpath(NLSY_DIR, "HOME (A, B, C, D)", "HomePartB", "HomePartB_Clean.csv"),
    HomePartC    = joinpath(NLSY_DIR, "HOME (A, B, C, D)", "HomePartC", "HomePartC_Clean.csv"),
    HomePartD    = joinpath(NLSY_DIR, "HOME (A, B, C, D)", "HomePartD", "HomePartD_Clean.csv"),
    ParentSkills    = joinpath(NLSY_DIR, "ParentSkills", "ParentSkills_Clean.csv"),
    HouseholdIncome = joinpath(NLSY_DIR, "HouseholdIncome", "HouseholdIncome_Clean.csv"),
    AdultOutcomes   = joinpath(PROJECT_ROOT, "output", "AdultOutcomes_Clean.csv"),
)

# ASVAB sections used as parental skill measures (cognitive ability)
# Rosenberg/Rotter columns are present in the file but reserved for non-cognitive models
const ASVAB_COLS  = [:ASVAB_ARITH, :ASVAB_WORD, :ASVAB_PARA, :ASVAB_NUMER, :ASVAB_CODING, :ASVAB_MATH]
const ASVAB_NAMES = ["ASVAB_ARITH", "ASVAB_WORD", "ASVAB_PARA", "ASVAB_NUMER", "ASVAB_CODING", "ASVAB_MATH"]

const ID_COLS = [:KID, :MID, :CHILDRACE, :CHILDSEX, :YEARBIRTH, :RVersion, :YEAR]

# ─────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────

function load_clean(path::String)
    df = CSV.read(path, DataFrame; missingstring=["NA",""])
    hasproperty(df, :Column1) && select!(df, Not(:Column1))
    df.YEAR = [ismissing(y) ? missing : parse(Int, string(y)) for y in df.YEAR]
    dropmissing!(df, [:YEAR, :YEARBIRTH])
    df.AGE = df.YEAR .- df.YEARBIRTH
    return df
end

"""
    load_household_income(path) -> DataFrame

Load the cleaned HouseholdIncome CSV (one row per MID × YEAR, long format).
Columns: MID, SID, RACE, SEX, YEAR, NetIncome.
Join to children by MID + YEAR. Income is at the mother level.
"""
function load_household_income(path::String)
    df = CSV.read(path, DataFrame; missingstring=["NA", ""])
    hasproperty(df, :Column1) && select!(df, Not(:Column1))
    return df
end

"""
    load_parent_skills(path) -> DataFrame

Load the cleaned ParentSkills CSV (one row per mother, keyed by MID).
No YEAR/AGE reshaping needed — the data is cross-sectional (1979–1981 waves).
Negative values were already set to missing during cleaning.
"""
function load_parent_skills(path::String)
    df = CSV.read(path, DataFrame; missingstring=["NA", ""])
    hasproperty(df, :Column1) && select!(df, Not(:Column1))
    return df
end

# ─────────────────────────────────────────────────────────────────
# Discretize continuous scores into categories
# ─────────────────────────────────────────────────────────────────

"""
    discretize_quantiles(x, n_bins) -> (categories, cutoffs)

Discretize non-missing values of x into n_bins quantile bins.
Returns categories (0 for missing) and the cutoff values used.
"""
function discretize_quantiles(x::AbstractVector, n_bins::Int)
    non_missing = collect(skipmissing(x))
    if isempty(non_missing)
        return zeros(Int, length(x)), Float64[]
    end
    cutoffs = quantile(non_missing, range(1/n_bins, 1-1/n_bins; length=n_bins-1))
    # Make cutoffs unique
    cutoffs = unique(cutoffs)
    actual_bins = length(cutoffs) + 1

    cats = zeros(Int, length(x))
    for i in eachindex(x)
        if ismissing(x[i])
            cats[i] = 0
        else
            cats[i] = searchsortedfirst(cutoffs, x[i])
            # clamp to actual_bins
            cats[i] = min(cats[i], actual_bins)
        end
    end
    return cats, cutoffs
end

# ─────────────────────────────────────────────────────────────────
# Apply pre-computed cutoffs to a vector
# ─────────────────────────────────────────────────────────────────

"""
    apply_cutoffs(x, cutoffs) -> Vector{Int}

Bin values of `x` using pre-computed `cutoffs`.
Returns 0 for missing, 1..length(cutoffs)+1 otherwise.
"""
function apply_cutoffs(x::AbstractVector, cutoffs::Vector{Float64})
    n_bins = length(cutoffs) + 1
    cats = zeros(Int, length(x))
    for i in eachindex(x)
        if !ismissing(x[i])
            cats[i] = min(searchsortedfirst(cutoffs, x[i]), n_bins)
        end
    end
    return cats
end

"""
    compute_pooled_cutoffs(df, col; n_bins=10) -> Vector{Float64}

Compute quantile cutoffs from ALL non-missing values of `df[!, col]`.
Returns deduplicated cutoff vector.
"""
function compute_pooled_cutoffs(df::DataFrame, col::Symbol; n_bins::Int=10)
    vals = collect(skipmissing(df[!, col]))
    if isempty(vals)
        return Float64[]
    end
    cutoffs = quantile(Float64.(vals), range(1/n_bins, 1 - 1/n_bins; length=n_bins - 1))
    return unique(cutoffs)
end

# ─────────────────────────────────────────────────────────────────
# Load all datasets and compute pooled cutoffs
# ─────────────────────────────────────────────────────────────────

"""
    load_all_datasets(; n_bins_continuous=10)

Load MSD, PPVT, PIAT, DigitSpan, HOME panel, and ParentSkills DataFrames.
Compute pooled decile cutoffs for all continuous measures (child cognitive + ASVAB).
Returns a named tuple: `(; df_msd, df_ppvt, df_piat, df_digit, df_home, df_parent, cutoffs)`.
"""
function load_all_datasets(; n_bins_continuous::Int=10)
    println("Loading all datasets...")

    df_msd    = load_clean(DATA_PATHS.MSD)
    df_ppvt   = load_clean(DATA_PATHS.PPVT)
    df_piat   = load_clean(DATA_PATHS.PIAT)
    df_digit  = load_clean(DATA_PATHS.DigitSpan)
    df_home   = load_clean(DATA_PATHS.HomePanel)
    df_parent = load_parent_skills(DATA_PATHS.ParentSkills)
    df_income = load_household_income(DATA_PATHS.HouseholdIncome)
    df_adult  = CSV.read(DATA_PATHS.AdultOutcomes, DataFrame; missingstring=["NA", ""])
    hasproperty(df_adult, :Column1) && select!(df_adult, Not(:Column1))

    println("  MSD:             $(nrow(df_msd)) rows")
    println("  PPVT:            $(nrow(df_ppvt)) rows")
    println("  PIAT:            $(nrow(df_piat)) rows")
    println("  DigitSpan:       $(nrow(df_digit)) rows")
    println("  HOME:            $(nrow(df_home)) rows")
    println("  ParentSkills:    $(nrow(df_parent)) rows ($(length(unique(df_parent.MID))) mothers)")
    println("  HouseholdIncome: $(nrow(df_income)) rows ($(length(unique(df_income.MID))) mothers, $(length(unique(df_income.YEAR))) years)")
    println("  AdultOutcomes:   $(nrow(df_adult)) rows")

    # ── Pooled cutoffs for all continuous measures ──
    println("Computing pooled cutoffs ($n_bins_continuous bins)...")
    cutoffs = Dict{Symbol, Vector{Float64}}()

    # Child cognitive measures
    cutoffs[:PPVTRaw] = compute_pooled_cutoffs(df_ppvt, :PPVTRaw; n_bins=n_bins_continuous)
    cutoffs[:MATH]    = compute_pooled_cutoffs(df_piat, :MATH;    n_bins=n_bins_continuous)
    cutoffs[:RECOG]   = compute_pooled_cutoffs(df_piat, :RECOG;   n_bins=n_bins_continuous)
    cutoffs[:COMP]    = compute_pooled_cutoffs(df_piat, :COMP;    n_bins=n_bins_continuous)
    cutoffs[:DIGITF]  = compute_pooled_cutoffs(df_digit, :DIGITF; n_bins=n_bins_continuous)
    cutoffs[:DIGITB]  = compute_pooled_cutoffs(df_digit, :DIGITB; n_bins=n_bins_continuous)

    # Parental ASVAB sections
    for col in ASVAB_COLS
        cutoffs[col] = compute_pooled_cutoffs(df_parent, col; n_bins=n_bins_continuous)
    end

    for (k, v) in sort(collect(cutoffs); by=first)
        println("  $k: $(length(v)+1) bins, cutoffs = $(round.(v; digits=1))")
    end

    return (; df_msd, df_ppvt, df_piat, df_digit, df_home, df_parent, df_income, df_adult, cutoffs)
end

# ─────────────────────────────────────────────────────────────────
# Prepare estimation data for any age group
# ─────────────────────────────────────────────────────────────────

const CONTINUOUS_COG_MEASURES = [
    # (source_key, column, measure_name)
    (:ppvt,  :PPVTRaw, "PPVTRaw"),
    (:piat,  :MATH,    "MATH"),
    (:piat,  :RECOG,   "RECOG"),
    (:piat,  :COMP,    "COMP"),
    (:digit, :DIGITF,  "DIGITF"),
    (:digit, :DIGITB,  "DIGITB"),
]

"""
    prepare_age_group_data(datasets; age_range_t1, age_range_t2,
        r_θ=4, r_I=4, invest_items, n_bins_count=4, min_obs=30)

Prepare estimation data for a two-period model using all available cognitive
measures (MSD, PPVT, PIAT, Digit Span) and specified HOME investment items.

`datasets` should be the named tuple returned by `load_all_datasets()`.

Returns the data structure expected by `estimation.jl`.
"""
function prepare_age_group_data(datasets;
    age_range_t1::Tuple{Int,Int},
    age_range_t2::Tuple{Int,Int},
    r_θ::Int = 4,
    r_I::Int = 4,
    invest_items::Vector{Symbol},
    n_bins_count::Int = 4,
    min_obs::Int = 30,
    exclude_sources_t1::Vector{Symbol} = Symbol[],
    exclude_sources_t2::Vector{Symbol} = Symbol[],
)
    (; df_msd, df_ppvt, df_piat, df_digit, df_home, cutoffs) = datasets

    println("\n  Preparing data for ages $(age_range_t1) → $(age_range_t2)")

    # ── Filter each cognitive dataset by age range, deduplicate by KID ──
    _age_filter(df, lo, hi) = unique(filter(r -> lo <= r.AGE <= hi, df), :KID)

    msd_t1   = _age_filter(df_msd,   age_range_t1...)
    msd_t2   = _age_filter(df_msd,   age_range_t2...)
    ppvt_t1  = _age_filter(df_ppvt,  age_range_t1...)
    ppvt_t2  = _age_filter(df_ppvt,  age_range_t2...)
    piat_t1  = _age_filter(df_piat,  age_range_t1...)
    piat_t2  = _age_filter(df_piat,  age_range_t2...)
    digit_t1 = _age_filter(df_digit, age_range_t1...)
    digit_t2 = _age_filter(df_digit, age_range_t2...)
    home_t1  = _age_filter(df_home,  age_range_t1...)

    println("  Raw counts:")
    println("    MSD:       t1=$(nrow(msd_t1)), t2=$(nrow(msd_t2))")
    println("    PPVT:      t1=$(nrow(ppvt_t1)), t2=$(nrow(ppvt_t2))")
    println("    PIAT:      t1=$(nrow(piat_t1)), t2=$(nrow(piat_t2))")
    println("    DigitSpan: t1=$(nrow(digit_t1)), t2=$(nrow(digit_t2))")
    println("    HOME:      t1=$(nrow(home_t1))")

    # ── Determine which MSD items to include at each period ──
    msd_item_cols = [c for c in propertynames(df_msd) if c ∉ Set(vcat(ID_COLS, [:AGE]))]

    function _select_msd_items(df_period, items, threshold)
        selected = Symbol[]
        for item in items
            if nrow(df_period) > 0 && sum(!ismissing(v) for v in df_period[!, item]) >= threshold
                push!(selected, item)
            end
        end
        return selected
    end

    msd_items_t1 = _select_msd_items(msd_t1, msd_item_cols, min_obs)
    msd_items_t2 = _select_msd_items(msd_t2, msd_item_cols, min_obs)

    println("  MSD items: t1=$(length(msd_items_t1)), t2=$(length(msd_items_t2))")

    # ── Determine which continuous measures to include at each period ──
    # A continuous measure is included if its DataFrame has ≥ min_obs non-missing at that period
    function _check_continuous(df_period, col, threshold)
        nrow(df_period) > 0 && sum(!ismissing(v) for v in df_period[!, col]) >= threshold
    end

    # Build per-period DataFrames lookup for continuous measures
    cont_dfs_t1 = Dict(:ppvt => ppvt_t1, :piat => piat_t1, :digit => digit_t1)
    cont_dfs_t2 = Dict(:ppvt => ppvt_t2, :piat => piat_t2, :digit => digit_t2)

    include_t1 = Dict{Symbol, Bool}()
    include_t2 = Dict{Symbol, Bool}()
    for (src, col, name) in CONTINUOUS_COG_MEASURES
        include_t1[col] = src ∉ exclude_sources_t1 && _check_continuous(cont_dfs_t1[src], col, min_obs)
        include_t2[col] = src ∉ exclude_sources_t2 && _check_continuous(cont_dfs_t2[src], col, min_obs)
        println("  Include $name: t1=$(include_t1[col]), t2=$(include_t2[col])")
    end

    # ── Identify children with ≥1 cognitive measure at each period ──
    function _kids_with_any_cognitive(msd_df, msd_items, cont_dfs, include_flags)
        kids = Set{Int}()

        # MSD: child has ≥1 non-missing item
        if !isempty(msd_items) && nrow(msd_df) > 0
            for i in 1:nrow(msd_df)
                for item in msd_items
                    if !ismissing(msd_df[i, item])
                        push!(kids, msd_df[i, :KID])
                        break
                    end
                end
            end
        end

        # Continuous measures
        for (src, col, _) in CONTINUOUS_COG_MEASURES
            if get(include_flags, col, false) && nrow(cont_dfs[src]) > 0
                for i in 1:nrow(cont_dfs[src])
                    if !ismissing(cont_dfs[src][i, col])
                        push!(kids, cont_dfs[src][i, :KID])
                    end
                end
            end
        end

        return kids
    end

    kids_cog_t1 = _kids_with_any_cognitive(msd_t1, msd_items_t1, cont_dfs_t1, include_t1)
    kids_cog_t2 = _kids_with_any_cognitive(msd_t2, msd_items_t2, cont_dfs_t2, include_t2)
    kids_inv    = Set(home_t1.KID)

    # Final sample = intersection
    kids_final = sort(collect(intersect(kids_cog_t1, kids_cog_t2, kids_inv)))
    N = length(kids_final)

    println("  Children with ≥1 cognitive t1: $(length(kids_cog_t1))")
    println("  Children with ≥1 cognitive t2: $(length(kids_cog_t2))")
    println("  Children with investment t1:   $(length(kids_inv))")
    println("  Final sample: N = $N")

    if N == 0
        error("No children in final sample for ages $(age_range_t1) → $(age_range_t2)")
    end

    # ── Build KID → row-index lookups ──
    _kid_index(df) = Dict{Int,Int}(df[i, :KID] => i for i in 1:nrow(df))

    msd_t1_idx   = _kid_index(msd_t1)
    msd_t2_idx   = _kid_index(msd_t2)
    ppvt_t1_idx  = _kid_index(ppvt_t1)
    ppvt_t2_idx  = _kid_index(ppvt_t2)
    piat_t1_idx  = _kid_index(piat_t1)
    piat_t2_idx  = _kid_index(piat_t2)
    digit_t1_idx = _kid_index(digit_t1)
    digit_t2_idx = _kid_index(digit_t2)
    home_t1_idx  = _kid_index(home_t1)

    cont_idx_t1 = Dict(:ppvt => ppvt_t1_idx, :piat => piat_t1_idx, :digit => digit_t1_idx)
    cont_idx_t2 = Dict(:ppvt => ppvt_t2_idx, :piat => piat_t2_idx, :digit => digit_t2_idx)

    # ── Helper: encode MSD binary items for a set of children ──
    function _encode_msd(kids, items, df_period, idx)
        vecs = Vector{Vector{Int}}()
        for item in items
            vals = zeros(Int, N)
            for (n, kid) in enumerate(kids)
                row = get(idx, kid, nothing)
                if row !== nothing
                    v = df_period[row, item]
                    vals[n] = ismissing(v) ? 0 : Int(v) + 1  # missing→0, fail→1, pass→2
                end
            end
            push!(vecs, vals)
        end
        return vecs
    end

    # ── Helper: encode a continuous measure for a set of children ──
    function _encode_continuous(kids, df_period, idx, col, cutoffs_vec)
        n_bins = length(cutoffs_vec) + 1
        vals = zeros(Int, N)
        for (n, kid) in enumerate(kids)
            row = get(idx, kid, nothing)
            if row !== nothing
                v = df_period[row, col]
                if !ismissing(v)
                    vals[n] = min(searchsortedfirst(cutoffs_vec, Float64(v)), n_bins)
                end
            end
        end
        return vals, n_bins
    end

    # ── Build period 1 cognitive measures (X_θ) ──
    X_θ = Vector{Vector{Int}}()
    κ_θ = Int[]
    measure_names_θ = String[]

    # MSD items at period 1
    for (vals, item) in zip(_encode_msd(kids_final, msd_items_t1, msd_t1, msd_t1_idx),
                            msd_items_t1)
        push!(X_θ, vals)
        push!(κ_θ, 2)
        push!(measure_names_θ, string(item))
    end

    # Continuous measures at period 1
    for (src, col, name) in CONTINUOUS_COG_MEASURES
        if include_t1[col]
            vals, n_bins = _encode_continuous(kids_final, cont_dfs_t1[src],
                                              cont_idx_t1[src], col, cutoffs[col])
            push!(X_θ, vals)
            push!(κ_θ, n_bins)
            push!(measure_names_θ, name)
        end
    end

    # ── Build period 2 cognitive measures (Y) ──
    Y = Vector{Vector{Int}}()
    κ_Y = Int[]
    measure_names_Y = String[]

    # MSD items at period 2
    for (vals, item) in zip(_encode_msd(kids_final, msd_items_t2, msd_t2, msd_t2_idx),
                            msd_items_t2)
        push!(Y, vals)
        push!(κ_Y, 2)
        push!(measure_names_Y, string(item))
    end

    # Continuous measures at period 2
    for (src, col, name) in CONTINUOUS_COG_MEASURES
        if include_t2[col]
            vals, n_bins = _encode_continuous(kids_final, cont_dfs_t2[src],
                                              cont_idx_t2[src], col, cutoffs[col])
            push!(Y, vals)
            push!(κ_Y, n_bins)
            push!(measure_names_Y, name)
        end
    end

    # ── Build investment measures (X_I) ──
    X_I = Vector{Vector{Int}}()
    κ_I = Int[]
    measure_names_I = String[]

    for item in invest_items
        if !hasproperty(home_t1, item)
            println("  WARNING: HOME item $item not found in panel, skipping")
            continue
        end

        # Collect raw values for children in final sample
        raw_vals = Vector{Union{Missing, Float64}}(missing, N)
        for (n, kid) in enumerate(kids_final)
            row = get(home_t1_idx, kid, nothing)
            if row !== nothing
                raw_vals[n] = home_t1[row, item]
            end
        end

        non_missing = collect(skipmissing(raw_vals))
        if isempty(non_missing)
            println("  WARNING: HOME item $item has no non-missing values, skipping")
            continue
        end

        n_unique = length(unique(non_missing))

        if n_unique <= 10
            # Ordinal/binary: shift to 1-based encoding
            min_val = Int(minimum(non_missing))
            shift = min_val < 1 ? (1 - min_val) : 0
            vals = [ismissing(v) ? 0 : Int(v) + shift for v in raw_vals]
            n_cats = Int(maximum(non_missing)) + shift
            push!(X_I, vals)
            push!(κ_I, n_cats)
            push!(measure_names_I, string(item))
        else
            # Count-type: discretize into quantile bins
            cats, item_cutoffs = discretize_quantiles(raw_vals, n_bins_count)
            actual_bins = length(item_cutoffs) + 1
            push!(X_I, cats)
            push!(κ_I, actual_bins)
            push!(measure_names_I, string(item, "_q", actual_bins))
        end
    end

    # ── Summary ──
    nm_t1 = [sum(x .> 0) for x in X_θ]
    nm_t2 = [sum(y .> 0) for y in Y]
    println("\n  Period 1 cognitive measures: $(length(X_θ))")
    if !isempty(nm_t1)
        println("    Non-missing: min=$(minimum(nm_t1)), median=$(round(Int, median(nm_t1))), max=$(maximum(nm_t1))")
    end
    println("  Period 2 cognitive measures: $(length(Y))")
    if !isempty(nm_t2)
        println("    Non-missing: min=$(minimum(nm_t2)), median=$(round(Int, median(nm_t2))), max=$(maximum(nm_t2))")
    end
    println("  Investment measures: $(length(X_I))")
    for (m, name) in enumerate(measure_names_I)
        n_nm = sum(X_I[m] .> 0)
        println("    $name: κ=$(κ_I[m]), non-missing=$n_nm/$N")
    end

    return (;
        N, r_θ, r_I,
        X_θ, κ_θ, measure_names_θ,
        X_I, κ_I, measure_names_I,
        Y, κ_Y, measure_names_Y,
        kids = kids_final,
    )
end

# ─────────────────────────────────────────────────────────────────
# Prepare estimation data with HOME-SF binary-recoded investment
# ─────────────────────────────────────────────────────────────────

"""
    prepare_age_group_data_homesf(datasets; age_range_t1, age_range_t2,
        r_θ=4, r_I=4, home_part, scale=:C, min_obs=30, ...)

Same as `prepare_age_group_data` but uses HOME-SF binary recoding for
investment measures. Instead of raw ordinal items, investment items are
recoded to binary (0/1) following the official NLS HOME-SF coding rules
from the specified `home_part` (`:PartA`, `:PartB`, `:PartC`, `:PartD`).

All investment measures have κ=2 (binary: 1=below threshold, 2=at/above threshold).
Missing values are encoded as 0, consistent with the rest of the pipeline.
"""
function prepare_age_group_data_homesf(datasets;
    age_range_t1::Tuple{Int,Int},
    age_range_t2::Tuple{Int,Int},
    r_θ::Int = 4,
    r_I::Int = 4,
    home_part::Symbol,
    scale::Symbol = :C,
    min_obs::Int = 30,
    exclude_sources_t1::Vector{Symbol} = Symbol[],
    exclude_sources_t2::Vector{Symbol} = Symbol[],
)
    (; df_msd, df_ppvt, df_piat, df_digit, df_home, cutoffs) = datasets

    # Get recoding rules for the requested scale
    if scale == :C
        recode_items = homesf_cognitive_items(home_part)
    else
        recode_items = homesf_emotional_items(home_part)
    end

    println("\n  Preparing data for ages $(age_range_t1) → $(age_range_t2)")
    println("  HOME-SF recoding: $home_part, scale=$scale, $(length(recode_items)) items")

    # ── Filter each cognitive dataset by age range, deduplicate by KID ──
    _age_filter(df, lo, hi) = unique(filter(r -> lo <= r.AGE <= hi, df), :KID)

    msd_t1   = _age_filter(df_msd,   age_range_t1...)
    msd_t2   = _age_filter(df_msd,   age_range_t2...)
    ppvt_t1  = _age_filter(df_ppvt,  age_range_t1...)
    ppvt_t2  = _age_filter(df_ppvt,  age_range_t2...)
    piat_t1  = _age_filter(df_piat,  age_range_t1...)
    piat_t2  = _age_filter(df_piat,  age_range_t2...)
    digit_t1 = _age_filter(df_digit, age_range_t1...)
    digit_t2 = _age_filter(df_digit, age_range_t2...)
    home_t1  = _age_filter(df_home,  age_range_t1...)

    println("  Raw counts:")
    println("    MSD:       t1=$(nrow(msd_t1)), t2=$(nrow(msd_t2))")
    println("    PPVT:      t1=$(nrow(ppvt_t1)), t2=$(nrow(ppvt_t2))")
    println("    PIAT:      t1=$(nrow(piat_t1)), t2=$(nrow(piat_t2))")
    println("    DigitSpan: t1=$(nrow(digit_t1)), t2=$(nrow(digit_t2))")
    println("    HOME:      t1=$(nrow(home_t1))")

    # ── Determine which MSD items to include at each period ──
    msd_item_cols = [c for c in propertynames(df_msd) if c ∉ Set(vcat(ID_COLS, [:AGE]))]

    function _select_msd_items(df_period, items, threshold)
        selected = Symbol[]
        for item in items
            if nrow(df_period) > 0 && sum(!ismissing(v) for v in df_period[!, item]) >= threshold
                push!(selected, item)
            end
        end
        return selected
    end

    msd_items_t1 = _select_msd_items(msd_t1, msd_item_cols, min_obs)
    msd_items_t2 = _select_msd_items(msd_t2, msd_item_cols, min_obs)

    println("  MSD items: t1=$(length(msd_items_t1)), t2=$(length(msd_items_t2))")

    # ── Determine which continuous measures to include at each period ──
    function _check_continuous(df_period, col, threshold)
        nrow(df_period) > 0 && sum(!ismissing(v) for v in df_period[!, col]) >= threshold
    end

    cont_dfs_t1 = Dict(:ppvt => ppvt_t1, :piat => piat_t1, :digit => digit_t1)
    cont_dfs_t2 = Dict(:ppvt => ppvt_t2, :piat => piat_t2, :digit => digit_t2)

    include_t1 = Dict{Symbol, Bool}()
    include_t2 = Dict{Symbol, Bool}()
    for (src, col, name) in CONTINUOUS_COG_MEASURES
        include_t1[col] = src ∉ exclude_sources_t1 && _check_continuous(cont_dfs_t1[src], col, min_obs)
        include_t2[col] = src ∉ exclude_sources_t2 && _check_continuous(cont_dfs_t2[src], col, min_obs)
        println("  Include $name: t1=$(include_t1[col]), t2=$(include_t2[col])")
    end

    # ── Identify children with ≥1 cognitive measure at each period ──
    function _kids_with_any_cognitive(msd_df, msd_items, cont_dfs, include_flags)
        kids = Set{Int}()

        if !isempty(msd_items) && nrow(msd_df) > 0
            for i in 1:nrow(msd_df)
                for item in msd_items
                    if !ismissing(msd_df[i, item])
                        push!(kids, msd_df[i, :KID])
                        break
                    end
                end
            end
        end

        for (src, col, _) in CONTINUOUS_COG_MEASURES
            if get(include_flags, col, false) && nrow(cont_dfs[src]) > 0
                for i in 1:nrow(cont_dfs[src])
                    if !ismissing(cont_dfs[src][i, col])
                        push!(kids, cont_dfs[src][i, :KID])
                    end
                end
            end
        end

        return kids
    end

    kids_cog_t1 = _kids_with_any_cognitive(msd_t1, msd_items_t1, cont_dfs_t1, include_t1)
    kids_cog_t2 = _kids_with_any_cognitive(msd_t2, msd_items_t2, cont_dfs_t2, include_t2)
    kids_inv    = Set(home_t1.KID)

    kids_final = sort(collect(intersect(kids_cog_t1, kids_cog_t2, kids_inv)))
    N = length(kids_final)

    println("  Children with ≥1 cognitive t1: $(length(kids_cog_t1))")
    println("  Children with ≥1 cognitive t2: $(length(kids_cog_t2))")
    println("  Children with investment t1:   $(length(kids_inv))")
    println("  Final sample: N = $N")

    if N == 0
        error("No children in final sample for ages $(age_range_t1) → $(age_range_t2)")
    end

    # ── Build KID → row-index lookups ──
    _kid_index(df) = Dict{Int,Int}(df[i, :KID] => i for i in 1:nrow(df))

    msd_t1_idx   = _kid_index(msd_t1)
    msd_t2_idx   = _kid_index(msd_t2)
    ppvt_t1_idx  = _kid_index(ppvt_t1)
    ppvt_t2_idx  = _kid_index(ppvt_t2)
    piat_t1_idx  = _kid_index(piat_t1)
    piat_t2_idx  = _kid_index(piat_t2)
    digit_t1_idx = _kid_index(digit_t1)
    digit_t2_idx = _kid_index(digit_t2)
    home_t1_idx  = _kid_index(home_t1)

    cont_idx_t1 = Dict(:ppvt => ppvt_t1_idx, :piat => piat_t1_idx, :digit => digit_t1_idx)
    cont_idx_t2 = Dict(:ppvt => ppvt_t2_idx, :piat => piat_t2_idx, :digit => digit_t2_idx)

    # ── Helper: encode MSD binary items for a set of children ──
    function _encode_msd(kids, items, df_period, idx)
        vecs = Vector{Vector{Int}}()
        for item in items
            vals = zeros(Int, N)
            for (n, kid) in enumerate(kids)
                row = get(idx, kid, nothing)
                if row !== nothing
                    v = df_period[row, item]
                    vals[n] = ismissing(v) ? 0 : Int(v) + 1
                end
            end
            push!(vecs, vals)
        end
        return vecs
    end

    # ── Helper: encode a continuous measure for a set of children ──
    function _encode_continuous(kids, df_period, idx, col, cutoffs_vec)
        n_bins = length(cutoffs_vec) + 1
        vals = zeros(Int, N)
        for (n, kid) in enumerate(kids)
            row = get(idx, kid, nothing)
            if row !== nothing
                v = df_period[row, col]
                if !ismissing(v)
                    vals[n] = min(searchsortedfirst(cutoffs_vec, Float64(v)), n_bins)
                end
            end
        end
        return vals, n_bins
    end

    # ── Build period 1 cognitive measures (X_θ) ──
    X_θ = Vector{Vector{Int}}()
    κ_θ = Int[]
    measure_names_θ = String[]

    for (vals, item) in zip(_encode_msd(kids_final, msd_items_t1, msd_t1, msd_t1_idx),
                            msd_items_t1)
        push!(X_θ, vals)
        push!(κ_θ, 2)
        push!(measure_names_θ, string(item))
    end

    for (src, col, name) in CONTINUOUS_COG_MEASURES
        if include_t1[col]
            vals, n_bins = _encode_continuous(kids_final, cont_dfs_t1[src],
                                              cont_idx_t1[src], col, cutoffs[col])
            push!(X_θ, vals)
            push!(κ_θ, n_bins)
            push!(measure_names_θ, name)
        end
    end

    # ── Build period 2 cognitive measures (Y) ──
    Y = Vector{Vector{Int}}()
    κ_Y = Int[]
    measure_names_Y = String[]

    for (vals, item) in zip(_encode_msd(kids_final, msd_items_t2, msd_t2, msd_t2_idx),
                            msd_items_t2)
        push!(Y, vals)
        push!(κ_Y, 2)
        push!(measure_names_Y, string(item))
    end

    for (src, col, name) in CONTINUOUS_COG_MEASURES
        if include_t2[col]
            vals, n_bins = _encode_continuous(kids_final, cont_dfs_t2[src],
                                              cont_idx_t2[src], col, cutoffs[col])
            push!(Y, vals)
            push!(κ_Y, n_bins)
            push!(measure_names_Y, name)
        end
    end

    # ── Build investment measures (X_I) with HOME-SF binary recoding ──
    X_I = Vector{Vector{Int}}()
    κ_I = Int[]
    measure_names_I = String[]

    for item_rule in recode_items
        col = item_rule.column
        if !hasproperty(home_t1, col)
            println("  WARNING: HOME-SF item $col not found in panel, skipping")
            continue
        end

        vals = zeros(Int, N)
        for (n, kid) in enumerate(kids_final)
            row = get(home_t1_idx, kid, nothing)
            if row !== nothing
                v = home_t1[row, col]
                if !ismissing(v)
                    # Binary encoding: 0=missing, 1=below threshold, 2=at/above threshold
                    vals[n] = item_rule.one_if(Int(v)) ? 2 : 1
                end
            end
        end

        n_nm = sum(vals .> 0)
        if n_nm < min_obs
            println("  WARNING: HOME-SF item $col has only $n_nm non-missing, skipping")
            continue
        end

        push!(X_I, vals)
        push!(κ_I, 2)
        push!(measure_names_I, string(col))
    end

    # ── Summary ──
    nm_t1 = [sum(x .> 0) for x in X_θ]
    nm_t2 = [sum(y .> 0) for y in Y]
    println("\n  Period 1 cognitive measures: $(length(X_θ))")
    if !isempty(nm_t1)
        println("    Non-missing: min=$(minimum(nm_t1)), median=$(round(Int, median(nm_t1))), max=$(maximum(nm_t1))")
    end
    println("  Period 2 cognitive measures: $(length(Y))")
    if !isempty(nm_t2)
        println("    Non-missing: min=$(minimum(nm_t2)), median=$(round(Int, median(nm_t2))), max=$(maximum(nm_t2))")
    end
    println("  Investment measures (HOME-SF $home_part, scale=$scale): $(length(X_I))")
    for (m, name) in enumerate(measure_names_I)
        n_nm = sum(X_I[m] .> 0)
        n_one = sum(X_I[m] .== 2)
        println("    $name: κ=2, non-missing=$n_nm/$N, recode-1=$n_one/$n_nm")
    end

    return (;
        N, r_θ, r_I,
        X_θ, κ_θ, measure_names_θ,
        X_I, κ_I, measure_names_I,
        Y, κ_Y, measure_names_Y,
        kids = kids_final,
    )
end

# ─────────────────────────────────────────────────────────────────
# Multi-period panel data preparation
# ─────────────────────────────────────────────────────────────────
#
# Module-level helpers (used by prepare_panel_data)
# ─────────────────────────────────────────────────────────────────

_age_filter_panel(df, lo, hi) = unique(filter(r -> lo <= r.AGE <= hi, df), :KID)

function _select_msd_items_panel(df_period, items, threshold)
    selected = Symbol[]
    for item in items
        if nrow(df_period) > 0 && sum(!ismissing(v) for v in df_period[!, item]) >= threshold
            push!(selected, item)
        end
    end
    return selected
end

_check_continuous_panel(df_period, col, threshold) =
    nrow(df_period) > 0 && sum(!ismissing(v) for v in df_period[!, col]) >= threshold

function _kids_with_any_cognitive_panel(msd_df, msd_items, cont_dfs, include_flags)
    kids = Set{Int}()
    if !isempty(msd_items) && nrow(msd_df) > 0
        for i in 1:nrow(msd_df)
            for item in msd_items
                if !ismissing(msd_df[i, item])
                    push!(kids, msd_df[i, :KID])
                    break
                end
            end
        end
    end
    for (src, col, _) in CONTINUOUS_COG_MEASURES
        if get(include_flags, col, false) && nrow(cont_dfs[src]) > 0
            for i in 1:nrow(cont_dfs[src])
                if !ismissing(cont_dfs[src][i, col])
                    push!(kids, cont_dfs[src][i, :KID])
                end
            end
        end
    end
    return kids
end

_kid_index_panel(df) = Dict{Int,Int}(df[i, :KID] => i for i in 1:nrow(df))

function _encode_msd_panel(kids, N, items, df_period, idx)
    vecs = Vector{Vector{Int}}()
    for item in items
        vals = zeros(Int, N)
        for (n, kid) in enumerate(kids)
            row = get(idx, kid, nothing)
            if row !== nothing
                v = df_period[row, item]
                vals[n] = ismissing(v) ? 0 : Int(v) + 1  # missing→0, fail→1, pass→2
            end
        end
        push!(vecs, vals)
    end
    return vecs
end

function _encode_continuous_panel(kids, N, df_period, idx, col, cutoffs_vec)
    n_bins = length(cutoffs_vec) + 1
    vals = zeros(Int, N)
    for (n, kid) in enumerate(kids)
        row = get(idx, kid, nothing)
        if row !== nothing
            v = df_period[row, col]
            if !ismissing(v)
                vals[n] = min(searchsortedfirst(cutoffs_vec, Float64(v)), n_bins)
            end
        end
    end
    return vals, n_bins
end

function _encode_invest_items_panel(kids, N, invest_items, home_df, home_idx;
                                     n_bins_count::Int=4, min_obs::Int=30)
    X_I = Vector{Vector{Int}}()
    κ_I = Int[]
    measure_names_I = String[]

    for item in invest_items
        if !hasproperty(home_df, item)
            println("    WARNING: HOME item $item not found, skipping")
            continue
        end

        raw_vals = Vector{Union{Missing, Float64}}(missing, N)
        for (n, kid) in enumerate(kids)
            row = get(home_idx, kid, nothing)
            if row !== nothing
                raw_vals[n] = home_df[row, item]
            end
        end

        non_missing = collect(skipmissing(raw_vals))
        if isempty(non_missing)
            println("    WARNING: HOME item $item has no non-missing values, skipping")
            continue
        end

        n_unique = length(unique(non_missing))
        if n_unique <= 10
            min_val = Int(minimum(non_missing))
            shift = min_val < 1 ? (1 - min_val) : 0
            vals = [ismissing(v) ? 0 : Int(v) + shift for v in raw_vals]
            n_cats = Int(maximum(non_missing)) + shift
            push!(X_I, vals)
            push!(κ_I, n_cats)
            push!(measure_names_I, string(item))
        else
            cats, item_cutoffs = discretize_quantiles(raw_vals, n_bins_count)
            actual_bins = length(item_cutoffs) + 1
            push!(X_I, cats)
            push!(κ_I, actual_bins)
            push!(measure_names_I, string(item, "_q", actual_bins))
        end
    end

    return X_I, κ_I, measure_names_I
end

"""
    _build_kid_to_mid(dfs) -> Dict{Int,Int}

Build a KID → MID lookup by scanning all child-level DataFrames in `dfs`.
Used to join parental skill measures to the child panel.
"""
function _build_kid_to_mid(dfs)
    kid_to_mid = Dict{Int,Int}()
    for df_list in dfs
        for df in df_list
            hasproperty(df, :MID) || continue
            for i in 1:nrow(df)
                kid = df[i, :KID]
                mid = df[i, :MID]
                if !ismissing(mid)
                    kid_to_mid[kid] = Int(mid)
                end
            end
        end
    end
    return kid_to_mid
end

"""
    _encode_parent_measures(kids, N, kid_to_mid, df_parent, parent_idx, cutoffs)

For each child in `kids`, look up their mother's ASVAB scores via MID and discretize
using pre-computed pooled cutoffs. Returns (X_P, κ_P, measure_names_P).
Missing parent (no MID match or missing ASVAB value) is encoded as 0.
"""
function _encode_parent_measures(kids, N, kid_to_mid, df_parent, parent_idx, cutoffs)
    X_P    = Vector{Vector{Int}}()
    κ_P    = Int[]
    names_P = String[]

    for (col, name) in zip(ASVAB_COLS, ASVAB_NAMES)
        hasproperty(df_parent, col) || continue
        cutoffs_vec = get(cutoffs, col, Float64[])
        n_bins = length(cutoffs_vec) + 1
        vals = zeros(Int, N)
        for (n, kid) in enumerate(kids)
            mid = get(kid_to_mid, kid, nothing)
            mid === nothing && continue
            row = get(parent_idx, mid, nothing)
            row === nothing && continue
            v = df_parent[row, col]
            if !ismissing(v)
                vals[n] = min(searchsortedfirst(cutoffs_vec, Float64(v)), n_bins)
            end
        end
        push!(X_P, vals)
        push!(κ_P, n_bins)
        push!(names_P, name)
    end

    return X_P, κ_P, names_P
end

"""
    _encode_income(kids, N, T, kid_to_mid, home_filtered, home_idx,
                   msd_filtered, msd_idx, df_income) -> Matrix{Float64}

Build an N×T matrix of log household income. For each child×period, the survey
year is read from the HOME panel (falling back to MSD), then income is looked up
from df_income by (MID, YEAR). Entries are log(NetIncome) for positive income,
NaN otherwise (missing or zero income). NaN is the missing sentinel for income
throughout the estimation code.
"""
function _encode_income(kids, N, T, kid_to_mid,
                         home_filtered, home_idx,
                         msd_filtered,  msd_idx,
                         df_income)
    # Load CPI-U for deflation (base year = 2010)
    cpi_path = joinpath(PROJECT_ROOT, "output", "cpi_u_annual.csv")
    cpi_table = Dict{Int, Float64}()
    if isfile(cpi_path)
        df_cpi = CSV.read(cpi_path, DataFrame)
        for row in eachrow(df_cpi)
            cpi_table[row.year] = row.cpi
        end
    end
    cpi_base = get(cpi_table, 2010, 218.056)

    # Build (MID, YEAR) → log real income lookup
    income_lookup = Dict{Tuple{Int,Int}, Float64}()
    for i in 1:nrow(df_income)
        mid  = df_income[i, :MID]
        year = df_income[i, :YEAR]
        inc  = df_income[i, :NetIncome]
        if !ismissing(mid) && !ismissing(year) && !ismissing(inc) && inc > 0
            yr = Int(year)
            deflator = get(cpi_table, yr, cpi_base) / cpi_base
            income_lookup[(Int(mid), yr)] = log(Float64(inc) / deflator)
        end
    end

    Y_income = fill(NaN, N, T)

    for (n, kid) in enumerate(kids)
        mid = get(kid_to_mid, kid, nothing)
        mid === nothing && continue

        for t in 1:T
            # Get survey year from HOME (best coverage), fall back to MSD
            year = nothing
            row = get(home_idx[t], kid, nothing)
            if row !== nothing
                year = home_filtered[t][row, :YEAR]
            else
                row2 = get(msd_idx[t], kid, nothing)
                if row2 !== nothing
                    year = msd_filtered[t][row2, :YEAR]
                end
            end
            year === nothing && continue

            log_inc = get(income_lookup, (mid, Int(year)), nothing)
            if log_inc !== nothing
                Y_income[n, t] = log_inc
            end
        end
    end

    return Y_income
end

"""
    prepare_panel_data(datasets; age_ranges, invest_items, r_θ=4, r_I=4,
        min_obs=30, n_bins_count=4, exclude_sources=[...])

Prepare multi-period panel data for forward-backward estimation.

Arguments:
- `age_ranges`: Vector of T (lo, hi) tuples defining age bins for each period
- `invest_items`: Vector of T vectors of HOME item symbols per period
- `exclude_sources`: Vector of T vectors of source symbols to exclude per period

Returns a NamedTuple with fields:
    N, T, r_θ, r_I, X_θ, κ_θ, X_I, κ_I, measure_names_θ, measure_names_I, kids
where X_θ[t][m][n] is skill measure m at period t for child n (0=missing).
"""
function prepare_panel_data(datasets;
    age_ranges::Vector{Tuple{Int,Int}},
    invest_items::Vector{Vector{Symbol}},
    r_θ::Int = 4,
    r_I::Int = 4,
    min_obs::Int = 30,
    n_bins_count::Int = 4,
    exclude_sources::Vector{Vector{Symbol}} = [Symbol[] for _ in age_ranges],
)
    (; df_msd, df_ppvt, df_piat, df_digit, df_home, df_parent, df_income, cutoffs) = datasets
    df_adult = haskey(datasets, :df_adult) ? datasets.df_adult : nothing
    T = length(age_ranges)

    @assert length(invest_items) == T "invest_items must have $T entries (one per period)"
    @assert length(exclude_sources) == T "exclude_sources must have $T entries"

    println("\n  Preparing panel data: T=$T periods")
    for t in 1:T
        println("    Period $t: ages $(age_ranges[t])")
    end

    # ── For each period: filter datasets by age range ──
    msd_item_cols = [c for c in propertynames(df_msd) if c ∉ Set(vcat(ID_COLS, [:AGE]))]

    msd_filtered   = [_age_filter_panel(df_msd,   age_ranges[t]...) for t in 1:T]
    ppvt_filtered  = [_age_filter_panel(df_ppvt,  age_ranges[t]...) for t in 1:T]
    piat_filtered  = [_age_filter_panel(df_piat,  age_ranges[t]...) for t in 1:T]
    digit_filtered = [_age_filter_panel(df_digit, age_ranges[t]...) for t in 1:T]
    home_filtered  = [_age_filter_panel(df_home,  age_ranges[t]...) for t in 1:T]

    # ── Determine MSD items and continuous measure availability per period ──
    msd_items = [_select_msd_items_panel(msd_filtered[t], msd_item_cols, min_obs) for t in 1:T]

    cont_dfs = [Dict(:ppvt => ppvt_filtered[t], :piat => piat_filtered[t],
                      :digit => digit_filtered[t]) for t in 1:T]

    include_cont = [Dict{Symbol,Bool}() for _ in 1:T]
    for t in 1:T
        for (src, col, name) in CONTINUOUS_COG_MEASURES
            include_cont[t][col] = src ∉ exclude_sources[t] &&
                _check_continuous_panel(cont_dfs[t][src], col, min_obs)
        end
    end

    # Print availability summary
    for t in 1:T
        println("    Period $t: MSD=$(length(msd_items[t])) items, " *
                join(["$name=$(include_cont[t][col])" for (_, col, name) in CONTINUOUS_COG_MEASURES], ", "))
    end

    # ── Sample selection ──
    # Children with ≥1 cognitive measure at ANY period (union, not intersection).
    # Forward-backward handles missing periods naturally: emission likelihood = 1
    # everywhere → no information, transition just propagates through.
    # Require ≥ min_periods periods with at least one measure (cognitive or investment)
    # so the child contributes some transition information.
    kids_cog = [_kids_with_any_cognitive_panel(msd_filtered[t], msd_items[t],
                    cont_dfs[t], include_cont[t]) for t in 1:T]
    kids_inv = [Set(home_filtered[t].KID) for t in 1:T]

    # Count how many periods each child has any measure
    all_kids = union(kids_cog..., kids_inv...)
    kid_period_count = Dict{Int,Int}()
    for kid in all_kids
        n_periods = sum(1:T) do t
            kid ∈ kids_cog[t] || kid ∈ kids_inv[t]
        end
        kid_period_count[kid] = n_periods
    end

    min_periods = 2  # need at least 2 periods to inform any transition
    kids_final = sort(collect(k for (k, c) in kid_period_count if c >= min_periods))
    N = length(kids_final)

    println("  Sample:")
    for t in 1:T
        n_any = length(kids_cog[t] ∪ kids_inv[t])
        println("    Period $t: $(length(kids_cog[t])) with cognitive, $(length(kids_inv[t])) with investment ($n_any any)")
    end
    println("    Children with ≥$min_periods periods: $N")
    println("    Final panel: N = $N")

    if N == 0
        error("No children in final panel sample")
    end

    # ── Build KID → row-index lookups ──
    msd_idx   = [_kid_index_panel(msd_filtered[t]) for t in 1:T]
    cont_idx  = [Dict(:ppvt => _kid_index_panel(ppvt_filtered[t]),
                       :piat => _kid_index_panel(piat_filtered[t]),
                       :digit => _kid_index_panel(digit_filtered[t])) for t in 1:T]
    home_idx  = [_kid_index_panel(home_filtered[t]) for t in 1:T]

    # ── Build KID → MID lookup for parent skill join ──
    kid_to_mid = _build_kid_to_mid([msd_filtered, ppvt_filtered, piat_filtered, digit_filtered])
    n_matched = sum(haskey(kid_to_mid, k) for k in kids_final)
    println("    KID→MID matched: $n_matched / $N children")

    parent_idx = Dict{Int,Int}(df_parent[i, :MID] => i for i in 1:nrow(df_parent))

    # ── Build per-period measures ──
    X_θ_all = Vector{Vector{Vector{Int}}}(undef, T)
    κ_θ_all = Vector{Vector{Int}}(undef, T)
    names_θ_all = Vector{Vector{String}}(undef, T)
    X_I_all = Vector{Vector{Vector{Int}}}(undef, T)
    κ_I_all = Vector{Vector{Int}}(undef, T)
    names_I_all = Vector{Vector{String}}(undef, T)

    for t in 1:T
        # ── Cognitive measures at period t ──
        X_θ = Vector{Vector{Int}}()
        κ_θ = Int[]
        names_θ = String[]

        # MSD items
        for (vals, item) in zip(
                _encode_msd_panel(kids_final, N, msd_items[t], msd_filtered[t], msd_idx[t]),
                msd_items[t])
            push!(X_θ, vals)
            push!(κ_θ, 2)
            push!(names_θ, string(item))
        end

        # Continuous measures
        for (src, col, name) in CONTINUOUS_COG_MEASURES
            if include_cont[t][col]
                vals, n_bins = _encode_continuous_panel(kids_final, N,
                    cont_dfs[t][src], cont_idx[t][src], col, cutoffs[col])
                push!(X_θ, vals)
                push!(κ_θ, n_bins)
                push!(names_θ, name)
            end
        end

        X_θ_all[t] = X_θ
        κ_θ_all[t] = κ_θ
        names_θ_all[t] = names_θ

        # ── Investment measures at period t ──
        X_I, κ_I, names_I = _encode_invest_items_panel(
            kids_final, N, invest_items[t], home_filtered[t], home_idx[t];
            n_bins_count, min_obs)

        X_I_all[t] = X_I
        κ_I_all[t] = κ_I
        names_I_all[t] = names_I
    end

    # ── Encode parental skill measures (cross-sectional, same for all periods) ──
    X_P, κ_P, names_P = _encode_parent_measures(
        kids_final, N, kid_to_mid, df_parent, parent_idx, cutoffs)

    # ── Encode household income (continuous log income, N×T matrix) ──
    Y_income = _encode_income(kids_final, N, T, kid_to_mid,
                               home_filtered, home_idx,
                               msd_filtered, msd_idx,
                               df_income)

    # ── Summary ──
    for t in 1:T
        nm_θ = [sum(x .> 0) for x in X_θ_all[t]]
        nm_I = [sum(x .> 0) for x in X_I_all[t]]
        println("\n  Period $t:")
        println("    Cognitive measures: $(length(X_θ_all[t]))")
        if !isempty(nm_θ)
            println("      Non-missing: min=$(minimum(nm_θ)), median=$(round(Int, median(nm_θ))), max=$(maximum(nm_θ))")
        end
        println("    Investment measures: $(length(X_I_all[t]))")
        if !isempty(nm_I)
            println("      Non-missing: min=$(minimum(nm_I)), median=$(round(Int, median(nm_I))), max=$(maximum(nm_I))")
        end
    end

    println("\n  Parental skill measures (ASVAB, cross-sectional): $(length(X_P))")
    for (m, name) in enumerate(names_P)
        n_nm = sum(X_P[m] .> 0)
        println("    $name: κ=$(κ_P[m]), non-missing=$n_nm/$N")
    end

    n_inc_obs = sum(!isnan, Y_income)
    println("\n  Household income (log, N×T): $n_inc_obs / $(N*T) child-periods observed")
    for t in 1:T
        n_t = sum(!isnan, Y_income[:, t])
        obs = filter(!isnan, Y_income[:, t])
        if !isempty(obs)
            println("    Period $t: $n_t/$N obs, mean log-inc=$(round(mean(obs); digits=2))")
        else
            println("    Period $t: 0/$N obs")
        end
    end

    # ── Adult outcomes: cross-sectional measures of (θ_T, ω) ──
    # X_adult[m][n] = observed value for child n, measure m. 0 = missing.
    # Measures: 1=HS (binary 1/2), 2=College (binary 1/2),
    #           3=EarnQuintile (1-5), 4=AlwaysEmployed (binary 1/2)
    # Binary outcomes: 1=no, 2=yes (matches κ=2 emission convention where col 1 = low type)
    adult_cols   = [:HS, :College, :EarnQuintile, :AlwaysEmployed]
    adult_names  = ["HS", "College", "EarnQ", "AlwaysEmp"]
    adult_κ      = [2, 2, 5, 2]
    n_adult      = length(adult_cols)
    X_adult      = [zeros(Int, N) for _ in 1:n_adult]

    if df_adult !== nothing
        adult_kid_idx = Dict{Int,Int}(df_adult[i, :KID] => i for i in 1:nrow(df_adult))
        for (ni, kid) in enumerate(kids_final)
            !haskey(adult_kid_idx, kid) && continue
            row = adult_kid_idx[kid]
            for m in 1:n_adult
                val = df_adult[row, adult_cols[m]]
                if ismissing(val)
                    X_adult[m][ni] = 0   # missing
                elseif m == 3   # EarnQuintile: already 1-5
                    v = Int(val)
                    X_adult[m][ni] = (1 <= v <= 5) ? v : 0
                else            # binary: 0/1 → 1/2
                    X_adult[m][ni] = Int(val) + 1
                end
            end
        end

        println("\n  Adult outcomes (cross-sectional):")
        for m in 1:n_adult
            n_nm = sum(X_adult[m] .> 0)
            println("    $(adult_names[m]): κ=$(adult_κ[m]), non-missing=$n_nm/$N")
        end
    else
        println("\n  Adult outcomes: not available (df_adult is nothing)")
    end

    return (;
        N, T, r_θ, r_I,
        X_θ = X_θ_all, κ_θ = κ_θ_all, measure_names_θ = names_θ_all,
        X_I = X_I_all, κ_I = κ_I_all, measure_names_I = names_I_all,
        X_P, κ_P, measure_names_P = names_P,
        Y_income,
        X_adult, κ_adult = adult_κ, measure_names_adult = adult_names,
        kids = kids_final,
    )
end
