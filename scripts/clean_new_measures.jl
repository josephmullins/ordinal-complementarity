using CSV, DataFrames, Statistics

const NLSY_DIR = joinpath(dirname(@__DIR__), "NLSY")

# ─────────────────────────────────────────────────────────────────
# ID column renaming: NLSY extract names → project convention
# ─────────────────────────────────────────────────────────────────
const ID_RENAME = Dict(
    "CPUBID"      => "KID",
    "MPUBID"      => "MID",
    "CRACE"       => "CHILDRACE",
    "CSEX"        => "CHILDSEX",
    "CYRB"        => "YEARBIRTH",
    "VERSION_R29" => "RVersion",
)

"""
    clean_nlsy_extract(csv_path, output_path; id_cols, var_pattern)

Clean an NLSY extract CSV:
1. Rename ID columns to project convention
2. Set negative values to missing
3. Reshape from wide (VARYYYY) to long format with YEAR column
4. Save as _Clean.csv
"""
function clean_nlsy_extract(csv_path::String, output_path::String)
    println("Loading $csv_path ...")
    df = CSV.read(csv_path, DataFrame)
    println("  Raw: $(nrow(df)) rows × $(ncol(df)) columns")

    # Identify ID columns and measure columns
    id_col_names = [c for c in names(df) if c in keys(ID_RENAME)]
    measure_col_names = [c for c in names(df) if c ∉ keys(ID_RENAME)]
    println("  ID columns: $id_col_names")
    println("  Measure columns: $(length(measure_col_names))")

    # Rename ID columns
    for (old, new) in ID_RENAME
        if old in names(df)
            rename!(df, old => new)
        end
    end

    # Parse measure columns: extract VAR name and YEAR from "VARYYYY" pattern
    # e.g. "DIGITF1992" → ("DIGITF", 1992), "MATH1986" → ("MATH", 1986)
    var_year_map = Dict{String, Vector{Tuple{String, Int}}}()  # varname → [(colname, year)]
    for col in measure_col_names
        m = match(r"^(.+?)(\d{4})$", col)
        if m !== nothing
            varname = m.captures[1]
            year = parse(Int, m.captures[2])
            push!(get!(var_year_map, varname, Tuple{String, Int}[]), (col, year))
        else
            println("  WARNING: could not parse column '$col', skipping")
        end
    end

    var_names = sort(collect(keys(var_year_map)))
    all_years = sort(unique(y for pairs in values(var_year_map) for (_, y) in pairs))
    println("  Measures: $var_names")
    println("  Years: $all_years")

    # Build long-format DataFrame
    N = nrow(df)
    n_years = length(all_years)
    total_rows = N * n_years

    # Pre-allocate ID columns (repeated for each year)
    id_syms = [:KID, :MID, :CHILDRACE, :CHILDSEX, :YEARBIRTH, :RVersion]
    result = DataFrame()
    for sym in id_syms
        if hasproperty(df, sym)
            result[!, sym] = repeat(df[!, sym], n_years)
        end
    end
    result[!, :YEAR] = repeat(all_years, inner=N)

    # Fill in measure values
    for varname in var_names
        col_vec = Vector{Union{Missing, Int}}(missing, total_rows)
        pairs = var_year_map[varname]

        for (colname, year) in pairs
            year_idx = findfirst(==(year), all_years)
            offset = (year_idx - 1) * N
            raw_col = df[!, colname]

            for i in 1:N
                v = raw_col[i]
                # Set negative values to missing
                if !ismissing(v) && v >= 0
                    col_vec[offset + i] = Int(v)
                end
            end
        end

        result[!, Symbol(varname)] = col_vec
    end

    # Drop rows where all measures are missing
    n_before = nrow(result)
    mask = [any(!ismissing(result[i, Symbol(v)]) for v in var_names) for i in 1:nrow(result)]
    result = result[mask, :]
    println("  Long format: $(nrow(result)) rows (dropped $(n_before - nrow(result)) all-missing)")

    # Compute AGE for reporting
    result.AGE = result.YEAR .- result.YEARBIRTH

    # Drop invalid ages
    n_before2 = nrow(result)
    filter!(r -> r.AGE >= 0, result)
    println("  After dropping AGE<0: $(nrow(result)) rows (dropped $(n_before2 - nrow(result)))")

    # Remove AGE before saving (it's computed on load, not stored)
    select!(result, Not(:AGE))

    # Summary
    println("  Unique children: $(length(unique(result.KID)))")
    println("  Unique KID×YEAR: $(nrow(unique(result, [:KID, :YEAR])))")

    # Check for duplicates
    n_dup = nrow(result) - nrow(unique(result, [:KID, :YEAR]))
    if n_dup > 0
        println("  WARNING: $n_dup duplicate KID×YEAR combinations!")
    else
        println("  No duplicate KID×YEAR combinations.")
    end

    # Per-measure summary
    for varname in var_names
        sym = Symbol(varname)
        n_nm = sum(!ismissing(v) for v in result[!, sym])
        vals = collect(skipmissing(result[!, sym]))
        if !isempty(vals)
            println("  $varname: N=$n_nm, range=$(extrema(vals)), mean=$(round(mean(Float64.(vals)); digits=1))")
        else
            println("  $varname: N=0 (all missing)")
        end
    end

    # Save
    CSV.write(output_path, result)
    println("  Saved to $output_path\n")

    return result
end

# ─────────────────────────────────────────────────────────────────
# Clean Digit Span
# ─────────────────────────────────────────────────────────────────
println("="^80)
println("  Cleaning Digit Span")
println("="^80)
clean_nlsy_extract(
    joinpath(NLSY_DIR, "DigitSpan", "digit_span.csv"),
    joinpath(NLSY_DIR, "DigitSpan", "DigitSpan_Clean.csv"),
)

# ─────────────────────────────────────────────────────────────────
# Clean PIAT
# ─────────────────────────────────────────────────────────────────
println("="^80)
println("  Cleaning PIAT")
println("="^80)
clean_nlsy_extract(
    joinpath(NLSY_DIR, "PIAT", "PIAT.csv"),
    joinpath(NLSY_DIR, "PIAT", "PIAT_Clean.csv"),
)

# ─────────────────────────────────────────────────────────────────
# Clean Parent Skills (Rotter locus of control, Rosenberg self-esteem, ASVAB)
# ─────────────────────────────────────────────────────────────────
println("="^80)
println("  Cleaning Parent Skills")
println("="^80)

"""
    clean_parent_skills(csv_path, output_path)

Clean the NLSY79 parent skills extract (cross-sectional, one row per mother).
Renames R-number columns to semantic names, sets negatives to missing,
and saves as ParentSkills_Clean.csv.

Column mapping:
  R0000100 → MID           (mother ID, matches MID in children datasets)
  R0153000–R0153600 → ROTTER_1–4  (Rotter locus of control, 1979; 1=internal, 2=external)
  R0173600 → SAMPLE_ID     (cohort/sampling group)
  R0214700 → RACE          (1=Hispanic, 2=Black, 3=Non-Black Non-Hispanic)
  R0214800 → SEX           (1=Male, 2=Female)
  R0303500–R0304400 → ROSENBERG_1–10 (Rosenberg self-esteem, 1980; 1–4 Likert)
  R0615100 → ASVAB_ARITH   (Arithmetic Reasoning, 1981)
  R0615200 → ASVAB_WORD    (Word Knowledge, 1981)
  R0615300 → ASVAB_PARA    (Paragraph Comprehension, 1981)
  R0615400 → ASVAB_NUMER   (Numeric Operations, 1981)
  R0615500 → ASVAB_CODING  (Coding Speed, 1981)
  R0615700 → ASVAB_MATH    (Math Knowledge, 1981)

Rosenberg items 1,2,4,6,7 are positively worded; items 3,5,8,9,10 are negatively
worded (higher raw value = higher self-esteem on negative items). Scoring/reversing
is left to the estimation code.
"""
function clean_parent_skills(csv_path::String, output_path::String)
    println("Loading $csv_path ...")
    df = CSV.read(csv_path, DataFrame)
    println("  Raw: $(nrow(df)) rows × $(ncol(df)) columns")

    rename!(df, Dict(
        "R0000100" => "MID",
        "R0153000" => "ROTTER_1",
        "R0153200" => "ROTTER_2",
        "R0153400" => "ROTTER_3",
        "R0153600" => "ROTTER_4",
        "R0173600" => "SAMPLE_ID",
        "R0214700" => "RACE",
        "R0214800" => "SEX",
        "R0303500" => "ROSENBERG_1",
        "R0303600" => "ROSENBERG_2",
        "R0303700" => "ROSENBERG_3",
        "R0303800" => "ROSENBERG_4",
        "R0303900" => "ROSENBERG_5",
        "R0304000" => "ROSENBERG_6",
        "R0304100" => "ROSENBERG_7",
        "R0304200" => "ROSENBERG_8",
        "R0304300" => "ROSENBERG_9",
        "R0304400" => "ROSENBERG_10",
        "R0615100" => "ASVAB_ARITH",
        "R0615200" => "ASVAB_WORD",
        "R0615300" => "ASVAB_PARA",
        "R0615400" => "ASVAB_NUMER",
        "R0615500" => "ASVAB_CODING",
        "R0615700" => "ASVAB_MATH",
    ))

    # Set negative values to missing (NLSY codes: -1=refused, -2=don't know,
    # -3=invalid missing, -4=valid missing, -5=non-interview)
    for col in names(df)
        col == "MID" && continue
        df[!, col] = [(!ismissing(v) && v < 0) ? missing : v for v in df[!, col]]
    end

    println("  Unique mothers (MID): $(length(unique(df.MID)))")

    # Per-measure summary for substantive columns
    measure_cols = ["ROTTER_1","ROTTER_2","ROTTER_3","ROTTER_4",
                    "ROSENBERG_1","ROSENBERG_2","ROSENBERG_3","ROSENBERG_4",
                    "ROSENBERG_5","ROSENBERG_6","ROSENBERG_7","ROSENBERG_8",
                    "ROSENBERG_9","ROSENBERG_10",
                    "ASVAB_ARITH","ASVAB_WORD","ASVAB_PARA",
                    "ASVAB_NUMER","ASVAB_CODING","ASVAB_MATH"]
    for col in measure_cols
        vals = collect(skipmissing(df[!, col]))
        n_nm = length(vals)
        if !isempty(vals)
            println("  $col: N=$n_nm, range=$(extrema(vals)), mean=$(round(mean(Float64.(vals)); digits=1))")
        else
            println("  $col: N=0 (all missing)")
        end
    end

    CSV.write(output_path, df)
    println("  Saved to $output_path\n")

    return df
end

clean_parent_skills(
    joinpath(NLSY_DIR, "ParentSkills", "parent_skills.csv"),
    joinpath(NLSY_DIR, "ParentSkills", "ParentSkills_Clean.csv"),
)

# ─────────────────────────────────────────────────────────────────
# Clean Household Income
# ─────────────────────────────────────────────────────────────────
println("="^80)
println("  Cleaning Household Income")
println("="^80)

"""
    clean_household_income(csv_path, vnames_path, output_path)

Clean the NLSY79 household income extract (mother level, wide format).
Uses the Vnames CSV to map R-numbers to NetIncome_{year} column names,
sets negative values to missing, reshapes wide → long, and saves as
HouseholdIncome_Clean.csv.

Output columns: MID, SID, RACE, SEX, YEAR, NetIncome.
The data is at the mother (MID) level — join to children by MID + YEAR.
"""
function clean_household_income(csv_path::String, vnames_path::String, output_path::String)
    println("Loading $csv_path ...")
    df = CSV.read(csv_path, DataFrame)
    println("  Raw: $(nrow(df)) rows × $(ncol(df)) columns")

    # Load variable name mapping: Vname, Year, RNum
    vnames = CSV.read(vnames_path, DataFrame)

    # Build RNum → NetIncome_{year} rename map
    income_rename = Dict{String,String}()
    years = Int[]
    for row in eachrow(vnames)
        income_rename[string(row.RNum)] = "NetIncome_$(row.Year)"
        push!(years, row.Year)
    end
    sort!(years)

    # Rename ID columns and income columns
    full_rename = merge(Dict(
        "R0000100" => "MID",
        "R0173600" => "SID",
        "R0214700" => "RACE",
        "R0214800" => "SEX",
    ), income_rename)

    for (old, new) in full_rename
        if old in names(df)
            rename!(df, old => new)
        end
    end

    # Set negative values to missing in all income columns
    income_cols = ["NetIncome_$y" for y in years]
    for col in income_cols
        df[!, col] = [(!ismissing(v) && v < 0) ? missing : v for v in df[!, col]]
    end

    # Reshape wide → long
    id_cols = [c for c in ["MID","SID","RACE","SEX"] if c in names(df)]
    long = stack(df, income_cols, id_cols; variable_name=:YearStr, value_name=:NetIncome)
    long.YEAR = [parse(Int, split(string(y), "_")[2]) for y in long.YearStr]
    select!(long, Not(:YearStr))

    # Drop rows where NetIncome is missing
    n_before = nrow(long)
    dropmissing!(long, :NetIncome)
    println("  Long format: $(nrow(long)) rows (dropped $(n_before - nrow(long)) missing)")

    println("  Unique mothers (MID): $(length(unique(long.MID)))")
    println("  Years covered: $(minimum(long.YEAR))–$(maximum(long.YEAR))")

    vals = Float64.(long.NetIncome)
    println("  NetIncome: range=$(extrema(vals)), median=$(round(median(vals); digits=0))")
    println("  Log(income) where >0: $(sum(vals .> 0)) rows with positive income")

    CSV.write(output_path, long)
    println("  Saved to $output_path\n")

    return long
end

clean_household_income(
    joinpath(NLSY_DIR, "HouseholdIncome", "HouseholdIncome.csv"),
    joinpath(NLSY_DIR, "HouseholdIncome", "Vnames - Household Income.csv"),
    joinpath(NLSY_DIR, "HouseholdIncome", "HouseholdIncome_Clean.csv"),
)

println("Done.")
