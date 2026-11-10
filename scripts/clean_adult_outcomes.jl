using CSV, DataFrames, Statistics

# ─────────────────────────────────────────────────────────────────
# Load raw data
# ─────────────────────────────────────────────────────────────────

println("Loading adult outcome data...")

ci = CSV.read(joinpath(@__DIR__, "..", "NLSY", "ChildIncome", "ChildIncome_Clean.csv"), DataFrame)
ed = CSV.read(joinpath(@__DIR__, "..", "NLSY", "Education", "Education_Summary.csv"), DataFrame)

println("  ChildIncome: $(nrow(ci)) rows, $(length(unique(ci.KID))) KIDs")
println("  Education:   $(nrow(ed)) rows, $(length(unique(ed.KID))) KIDs")

# Parse WageInc as Float64 (comes in as string/mixed type)
ci.WageInc_num = [ismissing(x) ? missing : tryparse(Float64, string(x)) for x in ci.WageInc]

# ─────────────────────────────────────────────────────────────────
# Compute max observed age per child
# ─────────────────────────────────────────────────────────────────

max_age = combine(groupby(ci, :KID), :Age => maximum => :MaxAge)
println("  Max age range: $(minimum(max_age.MaxAge)) to $(maximum(max_age.MaxAge))")

# ─────────────────────────────────────────────────────────────────
# 1. HS graduation indicator (valid if max age ≥ 18)
# ─────────────────────────────────────────────────────────────────

println("\nBuilding HS graduation indicator...")

# HighestGrade ≥ 12 OR DegreeLastYA indicates HS/GED or higher
# DegreeLastYA codes (from NLSY): 1=GED, 2=HS diploma, 3=AA, 4=BA, 5=MA, 6=PhD/Prof
# HighestGrade = 95 is a special code — treat as missing

function safe_parse_int(x)
    ismissing(x) && return missing
    v = tryparse(Int, string(x))
    v === nothing && return missing
    v >= 90 && return missing  # special codes
    return v
end
ed.HG_num = [safe_parse_int(x) for x in ed.HighestGrade]
ed.Deg_num = [safe_parse_int(x) for x in ed.DegreeLastYA]

hs_grad = DataFrame(KID = ed.KID)
hs_grad.HS = Vector{Union{Missing, Int}}(missing, nrow(ed))

for i in 1:nrow(ed)
    hg = ed.HG_num[i]
    deg = ed.Deg_num[i]

    # Check if we can determine HS graduation
    has_hs = false
    if !ismissing(hg) && hg >= 12
        has_hs = true
    end
    if !ismissing(deg) && deg >= 1  # GED or higher
        has_hs = true
    end

    # Only valid if child old enough
    kid = ed.KID[i]
    ma_row = findfirst(max_age.KID .== kid)
    if ma_row === nothing
        continue
    end
    ma = max_age.MaxAge[ma_row]

    if ma >= 18
        if has_hs
            hs_grad.HS[i] = 1
        elseif !ismissing(hg)  # have grade info but < 12
            hs_grad.HS[i] = 0
        end
    end
end

n_valid_hs = count(!ismissing, hs_grad.HS)
n_hs = count(x -> !ismissing(x) && x == 1, hs_grad.HS)
println("  Valid: $n_valid_hs / $(nrow(ed)), HS grad rate: $(round(n_hs/n_valid_hs*100; digits=1))%")

# ─────────────────────────────────────────────────────────────────
# 2. College graduation indicator (valid if max age ≥ 24)
# ─────────────────────────────────────────────────────────────────

println("\nBuilding College graduation indicator...")

college_grad = DataFrame(KID = ed.KID)
college_grad.College = Vector{Union{Missing, Int}}(missing, nrow(ed))

for i in 1:nrow(ed)
    hg = ed.HG_num[i]
    deg = ed.Deg_num[i]

    has_college = false
    if !ismissing(hg) && hg >= 16
        has_college = true
    end
    if !ismissing(deg) && deg >= 4  # BA or higher
        has_college = true
    end

    kid = ed.KID[i]
    ma_row = findfirst(max_age.KID .== kid)
    if ma_row === nothing; continue; end
    ma = max_age.MaxAge[ma_row]

    if ma >= 24
        if has_college
            college_grad.College[i] = 1
        elseif !ismissing(hg) || !ismissing(deg)
            college_grad.College[i] = 0
        end
    end
end

n_valid_col = count(!ismissing, college_grad.College)
n_col = count(x -> !ismissing(x) && x == 1, college_grad.College)
println("  Valid: $n_valid_col / $(nrow(ed)), College grad rate: $(round(n_col/n_valid_col*100; digits=1))%")

# ─────────────────────────────────────────────────────────────────
# 3 & 4. Average earnings and employment rate (ages 25-30)
# ─────────────────────────────────────────────────────────────────

println("\nBuilding earnings and employment measures (ages 25-30)...")

# Load CPI for deflation (base year = 2010)
cpi_df = CSV.read(joinpath(@__DIR__, "..", "output", "cpi_u_annual.csv"), DataFrame)
cpi_table = Dict{Int, Float64}(row.year => row.cpi for row in eachrow(cpi_df))
cpi_base = cpi_table[2010]
println("  CPI deflation: base year 2010 (CPI=$cpi_base)")

# Deflate WageInc by survey year
ci.YEAR_int = [tryparse(Int, string(y)) for y in ci.YEAR]
ci.WageInc_real = [
    (ismissing(w) || w === nothing || ismissing(yr) || yr === nothing) ? missing :
    let deflator = get(cpi_table, yr, cpi_base) / cpi_base
        w / deflator
    end
    for (w, yr) in zip(ci.WageInc_num, ci.YEAR_int)]

ci_adult = ci[(ci.Age .>= 25) .& (ci.Age .<= 30), :]
println("  Observations ages 25-30: $(nrow(ci_adult))")

# Per-child averages (using real earnings)
earnings_df = combine(groupby(ci_adult, :KID)) do sub
    wages = collect(skipmissing(sub.WageInc_real))
    wages = filter(!isnothing, wages)
    if isempty(wages)
        return (; AvgEarnings = missing, AvgEmployment = missing, N_obs = 0)
    end
    avg_earn = mean(wages)
    avg_emp = mean(wages .> 0)
    return (; AvgEarnings = avg_earn, AvgEmployment = avg_emp, N_obs = length(wages))
end

# Only keep children with at least 2 observations in ages 25-30
earnings_valid = earnings_df[earnings_df.N_obs .>= 2, :]
println("  Children with ≥2 obs ages 25-30: $(nrow(earnings_valid))")

# Also check: which children have max age ≥ 30 but no earnings data?
kids_old_enough = max_age[max_age.MaxAge .>= 30, :KID]
println("  Children with max age ≥ 30: $(length(kids_old_enough))")

# Earnings quintiles (pooled across all valid children)
valid_earnings = collect(skipmissing(earnings_valid.AvgEarnings))
println("  Avg earnings: mean=\$$(round(mean(valid_earnings); digits=0)), " *
        "median=\$$(round(median(valid_earnings); digits=0))")

earn_cutoffs = quantile(valid_earnings, [0.2, 0.4, 0.6, 0.8])
println("  Earnings quintile cutoffs: $(round.(earn_cutoffs; digits=0))")

earnings_valid.EarnQuintile = [ismissing(x) ? missing :
    (x <= earn_cutoffs[1] ? 1 :
     x <= earn_cutoffs[2] ? 2 :
     x <= earn_cutoffs[3] ? 3 :
     x <= earn_cutoffs[4] ? 4 : 5) for x in earnings_valid.AvgEarnings]

# Employment: binary indicator (employed in ALL observed periods ages 25-30)
valid_emp = collect(skipmissing(earnings_valid.AvgEmployment))
println("  Avg employment rate: mean=$(round(mean(valid_emp); digits=3))")
println("  Always employed (rate=1.0): $(round(mean(valid_emp .== 1.0)*100; digits=1))%")

earnings_valid.AlwaysEmployed = [ismissing(x) ? missing :
    (x >= 1.0 ? 1 : 0) for x in earnings_valid.AvgEmployment]

# ─────────────────────────────────────────────────────────────────
# Merge into single output
# ─────────────────────────────────────────────────────────────────

println("\nMerging adult outcomes...")

adult = DataFrame(KID = ed.KID)
adult = leftjoin(adult, hs_grad[:, [:KID, :HS]], on = :KID)
adult = leftjoin(adult, college_grad[:, [:KID, :College]], on = :KID)
adult = leftjoin(adult, earnings_valid[:, [:KID, :EarnQuintile, :AlwaysEmployed]], on = :KID)

# Replace missing with missing (not NaN or nothing)
println("\nAdult outcomes summary:")
println("  Total KIDs: $(nrow(adult))")
println("  HS grad valid:      $(count(!ismissing, adult.HS))")
println("  College grad valid: $(count(!ismissing, adult.College))")
println("  Earnings Q valid:   $(count(!ismissing, adult.EarnQuintile))")
println("  Always employed valid: $(count(!ismissing, adult.AlwaysEmployed))")

outpath = joinpath(@__DIR__, "..", "output", "AdultOutcomes_Clean.csv")
CSV.write(outpath, adult)
println("\nSaved to $outpath")

# Also save cutoffs for reproducibility
println("\nCutoffs for reference:")
println("  Earnings quintile cutoffs: $earn_cutoffs")
