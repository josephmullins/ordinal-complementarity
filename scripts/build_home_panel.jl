using CSV, DataFrames, Statistics

# ─────────────────────────────────────────────────────────────────
# Build unified HOME Inventory panel from Parts A-D
# ─────────────────────────────────────────────────────────────────
#
# Step 1.1: Create a single panel dataset with one row per KID × YEAR,
# containing all HOME variables from Parts A through D.
#
# Rules:
# 1. One instance of each variable for each KID × YEAR combination
# 2. Drop observations where YEARBIRTH > YEAR (invalid/negative ages)
# 3. Drop observations where ALL measures are missing
# 4. For overlapping variables across parts, coalesce (prefer non-missing)
# 5. Document duplicate KID × YEAR combinations

const NLSY_DIR = joinpath(dirname(@__DIR__), "NLSY")
const HOME_DIR = joinpath(NLSY_DIR, "HOME (A, B, C, D)")
const OUTPUT_DIR = joinpath(dirname(@__DIR__), "output")
mkpath(OUTPUT_DIR)

const ID_COLS = [:KID, :MID, :CHILDRACE, :CHILDSEX, :YEARBIRTH, :RVersion, :YEAR]

# ─────────────────────────────────────────────────────────────────
# Load and clean one file
# ─────────────────────────────────────────────────────────────────

function load_home_part(part::String)
    path = joinpath(HOME_DIR, part, "$(part)_Clean.csv")
    df = CSV.read(path, DataFrame; missingstring=["NA", ""])
    hasproperty(df, :Column1) && select!(df, Not(:Column1))

    # Parse YEAR to Int
    df.YEAR = [ismissing(y) ? missing : parse(Int, string(y)) for y in df.YEAR]
    dropmissing!(df, [:YEAR, :YEARBIRTH])

    df.AGE = df.YEAR .- df.YEARBIRTH

    # Drop invalid ages: YEARBIRTH > YEAR
    n_before = nrow(df)
    filter!(r -> r.AGE >= 0, df)
    n_dropped_age = n_before - nrow(df)

    # Identify measure columns
    measure_cols = [c for c in propertynames(df) if c ∉ Set(vcat(ID_COLS, [:AGE]))]

    # Drop rows where ALL measures are missing
    n_before2 = nrow(df)
    mask = [any(!ismissing(df[i, c]) for c in measure_cols) for i in 1:nrow(df)]
    df = df[mask, :]
    n_dropped_missing = n_before2 - nrow(df)

    println("  $part: $(nrow(df)) valid rows (dropped $n_dropped_age negative-age, $n_dropped_missing all-missing)")
    println("    Measures: $(length(measure_cols))")
    println("    Unique KID×YEAR: $(nrow(unique(df, [:KID, :YEAR])))")
    println("    Age range: $(extrema(df.AGE))")

    return df, measure_cols
end

# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

function build_home_panel()
    println("="^80)
    println("  Building Unified HOME Inventory Panel")
    println("="^80)
    println()

    # Load all parts
    parts = ["HomePartA", "HomePartB", "HomePartC", "HomePartD"]
    dfs = Dict{String, DataFrame}()
    measure_sets = Dict{String, Vector{Symbol}}()

    for part in parts
        dfs[part], measure_sets[part] = load_home_part(part)
    end

    # ── Document overlapping columns ──
    println("\n" * "-"^80)
    println("  Variable Overlap Across Parts")
    println("-"^80)

    all_measures = Dict{Symbol, Vector{String}}()
    for (part, cols) in measure_sets
        for c in cols
            push!(get!(all_measures, c, String[]), part)
        end
    end

    shared_vars = [(var, sort(parts)) for (var, parts) in all_measures if length(parts) > 1]
    sort!(shared_vars; by=first)
    unique_vars = [(var, parts[1]) for (var, parts) in all_measures if length(parts) == 1]
    sort!(unique_vars; by=first)

    println("\n  Variables appearing in MULTIPLE parts ($(length(shared_vars))):")
    for (var, ps) in shared_vars
        println("    $(rpad(string(var), 30)) → $(join(ps, ", "))")
    end

    println("\n  Variables unique to ONE part ($(length(unique_vars))):")
    for (var, p) in unique_vars
        println("    $(rpad(string(var), 30)) → $p")
    end

    # ── Merge strategy ──
    # Start with KID × YEAR from all parts, then add measure columns
    # For overlapping columns, coalesce across parts

    println("\n" * "-"^80)
    println("  Merging...")
    println("-"^80)

    # Collect all KID × YEAR combinations across all parts
    all_kid_year = DataFrame()
    for (part, df) in dfs
        ky = unique(select(df, [:KID, :YEAR, :MID, :CHILDRACE, :CHILDSEX, :YEARBIRTH, :RVersion]))
        all_kid_year = vcat(all_kid_year, ky)
    end
    all_kid_year = unique(all_kid_year, [:KID, :YEAR])
    sort!(all_kid_year, [:KID, :YEAR])
    all_kid_year.AGE = all_kid_year.YEAR .- all_kid_year.YEARBIRTH

    println("  Total unique KID×YEAR combinations: $(nrow(all_kid_year))")

    # Check for duplicate KID × YEAR within each part
    println("\n  Duplicate KID×YEAR within each part:")
    for part in parts
        df = dfs[part]
        n_total = nrow(df)
        n_unique = nrow(unique(df, [:KID, :YEAR]))
        n_dup = n_total - n_unique
        println("    $part: $n_dup duplicates ($(n_total) rows, $(n_unique) unique KID×YEAR)")
    end

    # For each part, keep first row per KID×YEAR
    for part in parts
        dfs[part] = unique(dfs[part], [:KID, :YEAR])
    end

    # Build result: start with the ID columns
    result = copy(all_kid_year)

    # All measure columns in order
    all_measure_names = sort(collect(keys(all_measures)))

    # For each measure, coalesce across parts
    # Priority order: A, B, C, D (earlier part preferred if non-missing)
    part_priority = ["HomePartA", "HomePartB", "HomePartC", "HomePartD"]

    # Create lookup dicts for fast merging
    println("  Building lookup tables...")
    lookups = Dict{String, Dict{Tuple{Int,Int}, Int}}()
    for part in parts
        df = dfs[part]
        lk = Dict{Tuple{Int,Int}, Int}()
        for i in 1:nrow(df)
            key = (df.KID[i], df.YEAR[i])
            if !haskey(lk, key)
                lk[key] = i
            end
        end
        lookups[part] = lk
    end

    println("  Merging $(length(all_measure_names)) measure columns...")
    n_coalesced = 0
    for var in all_measure_names
        source_parts = all_measures[var]
        N = nrow(result)
        col = Vector{Union{Missing, Any}}(missing, N)

        for i in 1:N
            key = (result.KID[i], result.YEAR[i])
            for part in part_priority
                part ∉ source_parts && continue
                lk = lookups[part]
                if haskey(lk, key)
                    row = lk[key]
                    v = dfs[part][row, var]
                    if !ismissing(v)
                        col[i] = v
                        break
                    end
                end
            end
        end

        # Try to infer type
        non_missing_vals = collect(skipmissing(col))
        if !isempty(non_missing_vals)
            T = typeof(non_missing_vals[1])
            if all(v -> v isa T, non_missing_vals)
                typed_col = Vector{Union{Missing, T}}(missing, N)
                for i in 1:N
                    if !ismissing(col[i])
                        typed_col[i] = col[i]::T
                    end
                end
                result[!, var] = typed_col
            else
                result[!, var] = col
            end
        else
            result[!, var] = col
        end

        if length(source_parts) > 1
            n_coalesced += 1
        end
    end
    println("  Coalesced $n_coalesced overlapping variables")

    # ── Drop rows where ALL measures are missing ──
    n_before = nrow(result)
    mask = [any(!ismissing(result[i, c]) for c in all_measure_names) for i in 1:nrow(result)]
    result = result[mask, :]
    println("  Dropped $(n_before - nrow(result)) rows with all measures missing")

    # ── Final summary ──
    println("\n" * "="^80)
    println("  FINAL PANEL SUMMARY")
    println("="^80)
    println("  Rows: $(nrow(result))")
    println("  Unique children: $(length(unique(result.KID)))")
    println("  Unique KID×YEAR: $(nrow(unique(result, [:KID, :YEAR])))")
    println("  Age range: $(extrema(result.AGE))")
    println("  Survey years: $(sort(unique(result.YEAR)))")
    println("  Total measure columns: $(length(all_measure_names))")

    # Coverage by age bin
    age_bins = [(0,2), (3,5), (6,8), (9,11), (12,14), (15,18)]
    println("\n  Coverage by age bin:")
    for (lo, hi) in age_bins
        sub = filter(r -> lo <= r.AGE <= hi, result)
        if nrow(sub) > 0
            n_kids = length(unique(sub.KID))
            avail = sum(1 for c in all_measure_names if sum(!ismissing(v) for v in sub[!, c]) > 0)
            println("    Age $lo-$hi: $n_kids children, $avail/$(length(all_measure_names)) measures with data")
        end
    end

    # ── Document which variables come from which parts ──
    println("\n  Variable sources:")
    for var in all_measure_names
        sources = all_measures[var]
        n_nm = sum(!ismissing(v) for v in result[!, var])
        pct = round(100 * n_nm / nrow(result); digits=1)
        println("    $(rpad(string(var), 30)) $(rpad(join(sort(sources), ","), 60)) N=$n_nm ($pct%)")
    end

    # Save
    outpath = joinpath(OUTPUT_DIR, "HOME_panel.csv")
    CSV.write(outpath, result)
    println("\n  Saved to $outpath")

    return result
end

# Run
panel = build_home_panel()
