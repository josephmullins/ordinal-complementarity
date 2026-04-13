#!/usr/bin/env julia
# ─────────────────────────────────────────────────────────────────
# Diagnostic: Age-4 investment–skill survivor functions
#
# For each period, compute the average cognitive measure and average
# investment measure per child, split into quantile bins (set N_BINS),
# and plot the empirical survivor function P(θ_bin > k | θ_bin, I_bin)
# directly from the data — no latent variable model involved.
#
# This helps diagnose whether the decreasing-in-investment survivor
# pattern at age 4 is a feature of the raw data or an estimation artifact.
# ─────────────────────────────────────────────────────────────────

using Statistics, Plots, StatsPlots, LaTeXStrings

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "prepare_data.jl"))

gr(size=(900, 500))

const OUTDIR = joinpath(dirname(@__DIR__), "output", "diagnostics")
mkpath(OUTDIR)

# ─────────────────────────────────────────────────────────────────
# Configuration — match run_baseline.jl
# ─────────────────────────────────────────────────────────────────

const PANEL_AGE_RANGES = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14)]
const PERIOD_LABELS = ["$(a[1])–$(a[2])" for a in PANEL_AGE_RANGES]

const PANEL_INVEST_ITEMS = [
    # Period 1: ages 1-2 (HOME Part A)
    [:HowOftenRead, :HowManyBooks, :HowManyToysA, :HowManyToysB,
     :OutOfHouse, :HowOftenTalks, :HowOftenGrocery],
    # Period 2: ages 3-4 (HOME Part B)
    [:HowOftenRead, :HowManyBooks, :HowManyToysA, :HowManyToysB,
     :OutOfHouse, :HowOftenTalks, :HowOftenGrocery,
     :HowManyZines, :HowOftenOuting, :FoodChoice,
     :LearnNumbers, :LearnAlphabet, :LearnColors, :LearnShapes, :RecordTapeCD],
    # Period 3: ages 5-6 (HOME Part B/C)
    [:HowOftenRead, :HowManyBooks, :HowManyZines, :HowOftenOuting,
     :HowOftenMuseum, :LearnNumbers, :LearnAlphabet, :LearnColors,
     :LearnShapes, :RecordTapeCD, :FoodChoice],
    # Period 4: ages 7-8 (HOME Part C)
    [:HowOftenRead, :HowManyBooks, :HowOftenMuseum, :HowOftenTheater,
     :HowOftenFamily, :HowOftenOutdoors, :ChildMusic, :SpecialActivities,
     :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
     :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy],
    # Period 5: ages 9-10 (HOME Part C/D)
    [:HowOftenRead, :HowManyBooks, :HowOftenMuseum, :HowOftenTheater,
     :HowOftenFamily, :HowOftenOutdoors, :HowOftenFather, :ChildMusic,
     :SpecialActivities, :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
     :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy],
    # Period 6: ages 11-12 (HOME Part D)
    [:HowManyBooks, :HowOftenMuseum, :HowOftenTheater, :HowOftenFamily,
     :HowOftenOutdoors, :HowOftenFather, :ChildMusic, :SpecialActivities,
     :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
     :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy,
     :ToldPositive, :HowOftenClean, :HowOftenChores, :HowOftenTime],
    # Period 7: ages 13-14 (HOME Part D)
    [:HowManyBooks, :HowOftenMuseum, :HowOftenTheater, :HowOftenFamily,
     :HowOftenOutdoors, :HowOftenFather, :ChildMusic, :SpecialActivities,
     :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
     :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy,
     :ToldPositive, :HowOftenClean, :HowOftenChores, :HowOftenTime],
]

# ─────────────────────────────────────────────────────────────────
# Helper: compute average measure per child (ignoring missing = 0)
# ─────────────────────────────────────────────────────────────────

"""
    average_measure(X_list, κ_list)

For each child n, compute the average of non-missing measures,
normalized to [0, 1] by dividing each measure by its κ.
Returns a vector of length N (NaN if child has no non-missing measures).
"""
function average_measure(X_list, κ_list)
    N = length(X_list[1])
    M = length(X_list)
    avg = fill(NaN, N)
    for n in 1:N
        total = 0.0
        count = 0
        for m in 1:M
            x = X_list[m][n]
            if x > 0  # not missing
                total += (x - 1) / max(κ_list[m] - 1, 1)  # normalize to [0, 1]
                count += 1
            end
        end
        if count > 0
            avg[n] = total / count
        end
    end
    return avg
end

"""
    assign_quantile_bins(vals, n_bins)

Assign quantile bin labels (1, …, n_bins) based on non-NaN values.
Returns vector of same length; NaN entries get label 0.
"""
function assign_quantile_bins(vals, n_bins)
    N = length(vals)
    labels = zeros(Int, N)
    valid_idx = findall(!isnan, vals)
    if length(valid_idx) < n_bins
        return labels
    end
    valid_vals = vals[valid_idx]
    cuts = [quantile(valid_vals, k / n_bins) for k in 1:n_bins-1]
    for i in valid_idx
        labels[i] = searchsortedlast(cuts, vals[i]) + 1
        labels[i] = clamp(labels[i], 1, n_bins)
    end
    return labels
end

# ─────────────────────────────────────────────────────────────────
# Load and prepare data
# ─────────────────────────────────────────────────────────────────

println("Loading datasets...")
datasets = load_all_datasets()

println("Preparing panel data...")
data = prepare_panel_data(datasets;
    age_ranges=PANEL_AGE_RANGES,
    invest_items=PANEL_INVEST_ITEMS,
    r_θ=3, r_I=3,
)

T = data.T
N = data.N
println("N = $N children, T = $T periods")

# ─────────────────────────────────────────────────────────────────
# Compute average measures and terciles per period
# ─────────────────────────────────────────────────────────────────

avg_skill = [average_measure(data.X_θ[t], data.κ_θ[t]) for t in 1:T]
avg_invest = [average_measure(data.X_I[t], data.κ_I[t]) for t in 1:T]

const N_BINS = 5  # number of quantile bins (5 = quintiles)
const BIN_LABELS = N_BINS == 3 ? ["Low", "Mid", "High"] :
                   N_BINS == 5 ? ["Q1", "Q2", "Q3", "Q4", "Q5"] :
                   ["$k" for k in 1:N_BINS]

bin_skill = [assign_quantile_bins(avg_skill[t], N_BINS) for t in 1:T]
bin_invest = [assign_quantile_bins(avg_invest[t], N_BINS) for t in 1:T]

# Print summary statistics
println("\n" * "="^70)
println("  DIAGNOSTIC SUMMARY")
println("="^70)
for t in 1:T
    n_skill_obs = count(!isnan, avg_skill[t])
    n_invest_obs = count(!isnan, avg_invest[t])
    n_both = count(i -> bin_skill[t][i] > 0 && bin_invest[t][i] > 0, 1:N)
    println("\n  Period $t (ages $(PERIOD_LABELS[t])):")
    println("    Children with ≥1 skill measure:      $n_skill_obs")
    println("    Children with ≥1 investment measure:  $n_invest_obs")
    println("    Children with both:                   $n_both")

    if n_skill_obs > 0
        valid = filter(!isnan, avg_skill[t])
        println("    Avg skill (normalized):  mean=$(round(mean(valid); digits=3)), " *
                "sd=$(round(std(valid); digits=3)), " *
                "min=$(round(minimum(valid); digits=3)), max=$(round(maximum(valid); digits=3))")
    end
    if n_invest_obs > 0
        valid = filter(!isnan, avg_invest[t])
        println("    Avg invest (normalized): mean=$(round(mean(valid); digits=3)), " *
                "sd=$(round(std(valid); digits=3)), " *
                "min=$(round(minimum(valid); digits=3)), max=$(round(maximum(valid); digits=3))")
    end

    # Skill measure breakdown
    println("    Skill measures ($(length(data.X_θ[t]))):")
    for m in eachindex(data.X_θ[t])
        name = data.measure_names_θ[t][m]
        κ = data.κ_θ[t][m]
        n_obs = count(>(0), data.X_θ[t][m])
        println("      $name: κ=$κ, n_obs=$n_obs")
    end
    println("    Investment measures ($(length(data.X_I[t]))):")
    for m in eachindex(data.X_I[t])
        name = data.measure_names_I[t][m]
        κ = data.κ_I[t][m]
        n_obs = count(>(0), data.X_I[t][m])
        println("      $name: κ=$κ, n_obs=$n_obs")
    end
end

# ─────────────────────────────────────────────────────────────────
# Compute and plot empirical survivor functions per period
#
# For each period t, look at the NEXT period's skill bin
# conditional on current (skill_bin, invest_bin).
# This is the empirical analog of the production function.
# ─────────────────────────────────────────────────────────────────

const THETA_COLORS = [:steelblue, :darkorange, :forestgreen, :firebrick, :purple,
                      :teal, :brown, :magenta, :olive, :navy]

println("\n" * "="^70)
println("  EMPIRICAL PRODUCTION FUNCTIONS ($(N_BINS)-QUANTILE BINS)")
println("="^70)

for t in 1:T-1
    println("\n  Transition $t: ages $(PERIOD_LABELS[t]) → $(PERIOD_LABELS[t+1])")

    # Count children in each (θ_t, I_t, θ_{t+1}) cell
    counts = zeros(Int, N_BINS, N_BINS, N_BINS)  # [θ', θ, I]
    for n in 1:N
        θ = bin_skill[t][n]
        I = bin_invest[t][n]
        θ_next = bin_skill[t+1][n]
        if θ > 0 && I > 0 && θ_next > 0
            counts[θ_next, θ, I] += 1
        end
    end

    n_valid = sum(counts)
    println("    Children with valid (θ_t, I_t, θ_{t+1}): $n_valid")

    # Print counts table
    println("    Cell counts [θ', θ, I]:")
    for kI in 1:N_BINS
        println("      I=$(BIN_LABELS[kI]):")
        for kθ in 1:N_BINS
            col_total = sum(counts[:, kθ, kI])
            fracs = col_total > 0 ? round.(counts[:, kθ, kI] ./ col_total; digits=3) : zeros(N_BINS)
            println("        θ=$(BIN_LABELS[kθ]): n=$(counts[:, kθ, kI]) (total=$col_total, P=$(fracs))")
        end
    end

    # Compute empirical transition probabilities
    Π_emp = zeros(N_BINS, N_BINS, N_BINS)
    for kI in 1:N_BINS, kθ in 1:N_BINS
        total = sum(counts[:, kθ, kI])
        if total > 0
            Π_emp[:, kθ, kI] = counts[:, kθ, kI] ./ total
        else
            Π_emp[:, kθ, kI] .= 1 / N_BINS
        end
    end

    # Print survivor function values for first and last cutoffs
    println("    Survivor P(θ' > 1 | θ, I):")
    for kθ in 1:N_BINS
        vals = [round(1 - Π_emp[1, kθ, kI]; digits=3) for kI in 1:N_BINS]
        println("      θ=$(BIN_LABELS[kθ]): " * join(["I=$(BIN_LABELS[kI])→$(vals[kI])" for kI in 1:N_BINS], ", "))
    end
    println("    Survivor P(θ' > $(N_BINS-1) | θ, I):")
    for kθ in 1:N_BINS
        vals = [round(Π_emp[N_BINS, kθ, kI]; digits=3) for kI in 1:N_BINS]
        println("      θ=$(BIN_LABELS[kθ]): " * join(["I=$(BIN_LABELS[kI])→$(vals[kI])" for kI in 1:N_BINS], ", "))
    end

    # ── Plot ──
    n_cutoffs = N_BINS - 1
    plots_t = []
    for k in 1:n_cutoffs
        p = plot(
            title = "$(PERIOD_LABELS[t]) → $(PERIOD_LABELS[t+1]): " *
                    "\$\\bar{F}($k | \\theta, I)\$",
            xlabel = "Investment quintile",
            ylabel = "\$P(\\theta' > $k | \\theta, I)\$",
            ylim = (0, 1.05),
            xticks = (1:N_BINS, BIN_LABELS),
            legend = (k == n_cutoffs) ? :topright : false,
            titlefontsize = 10,
            guidefontsize = 9,
            tickfontsize = 8,
            grid = true,
            gridalpha = 0.3,
        )

        for kθ in 1:N_BINS
            surv = [1.0 - sum(Π_emp[1:k, kθ, kI]) for kI in 1:N_BINS]

            # Confidence intervals via normal approx to binomial
            n_obs = [sum(counts[:, kθ, kI]) for kI in 1:N_BINS]
            se = [n > 0 ? sqrt(s * (1 - s) / n) : 0.0
                  for (s, n) in zip(surv, n_obs)]
            ci_lo = clamp.(surv .- 1.96 .* se, 0, 1)
            ci_hi = clamp.(surv .+ 1.96 .* se, 0, 1)

            plot!(p, 1:N_BINS, surv;
                ribbon = (surv .- ci_lo, ci_hi .- surv),
                fillalpha = 0.15,
                label = "\$\\theta\$ = $(BIN_LABELS[kθ])",
                linewidth = 2.5,
                marker = :circle,
                markersize = 4,
                color = THETA_COLORS[kθ],
            )
        end

        push!(plots_t, p)
    end

    fig = plot(plots_t...,
        layout = (n_cutoffs, 1),
        size = (700, 250 * n_cutoffs),
        margin = 5Plots.mm,
        bottom_margin = 8Plots.mm,
    )
    savefig(fig, joinpath(OUTDIR, "empirical_survivor_t$(t).pdf"))
    println("    Saved: empirical_survivor_t$(t).pdf")
end

# ─────────────────────────────────────────────────────────────────
# Additional diagnostic: same-period skill–investment correlation
# ─────────────────────────────────────────────────────────────────

println("\n" * "="^70)
println("  WITHIN-PERIOD SKILL–INVESTMENT CORRELATION")
println("="^70)
for t in 1:T
    valid = [i for i in 1:N if !isnan(avg_skill[t][i]) && !isnan(avg_invest[t][i])]
    if length(valid) >= 10
        r = cor(avg_skill[t][valid], avg_invest[t][valid])
        println("  Period $t (ages $(PERIOD_LABELS[t])): r = $(round(r; digits=3)) (n=$(length(valid)))")
    else
        println("  Period $t (ages $(PERIOD_LABELS[t])): insufficient data (n=$(length(valid)))")
    end
end

# ─────────────────────────────────────────────────────────────────
# Diagnostic: cross-tab of current skill × investment × next skill
# broken out by individual age within each period
# ─────────────────────────────────────────────────────────────────

println("\n" * "="^70)
println("  AGE-SPECIFIC MEASURE AVAILABILITY")
println("="^70)

# Reload raw data to check individual ages
(; df_msd, df_ppvt, df_piat, df_digit, df_home) = datasets

for t in 1:T
    lo, hi = PANEL_AGE_RANGES[t]
    println("\n  Period $t (ages $(lo)–$(hi)):")
    for age in lo:hi
        n_msd = nrow(filter(r -> r.AGE == age, df_msd))
        n_ppvt = nrow(filter(r -> r.AGE == age, df_ppvt))
        n_piat = nrow(filter(r -> r.AGE == age, df_piat))
        n_digit = nrow(filter(r -> r.AGE == age, df_digit))
        n_home = nrow(filter(r -> r.AGE == age, df_home))
        println("    Age $age: MSD=$n_msd, PPVT=$n_ppvt, PIAT=$n_piat, DigitSpan=$n_digit, HOME=$n_home")
    end
end

# ─────────────────────────────────────────────────────────────────
# Combined plot: all transitions side by side
# ─────────────────────────────────────────────────────────────────

println("\n  Generating combined plot...")

all_plots = []
for t in 1:T-1
    # Recompute for the combined figure (just survivor k=1: P(θ'>1|θ,I))
    counts = zeros(Int, N_BINS, N_BINS, N_BINS)
    for n in 1:N
        θ = bin_skill[t][n]
        I = bin_invest[t][n]
        θ_next = bin_skill[t+1][n]
        if θ > 0 && I > 0 && θ_next > 0
            counts[θ_next, θ, I] += 1
        end
    end

    Π_emp = zeros(N_BINS, N_BINS, N_BINS)
    for kI in 1:N_BINS, kθ in 1:N_BINS
        total = sum(counts[:, kθ, kI])
        if total > 0
            Π_emp[:, kθ, kI] = counts[:, kθ, kI] ./ total
        else
            Π_emp[:, kθ, kI] .= 1 / N_BINS
        end
    end

    p = plot(
        title = "$(PERIOD_LABELS[t]) → $(PERIOD_LABELS[t+1])",
        xlabel = "Investment quintile",
        ylabel = "\$P(\\theta' > 1 | \\theta, I)\$",
        ylim = (0, 1.05),
        xticks = (1:N_BINS, BIN_LABELS),
        legend = (t == T-1) ? :topright : false,
        titlefontsize = 10,
        guidefontsize = 9,
        tickfontsize = 8,
        grid = true,
        gridalpha = 0.3,
    )

    for kθ in 1:N_BINS
        surv = [1.0 - Π_emp[1, kθ, kI] for kI in 1:N_BINS]
        n_obs = [sum(counts[:, kθ, kI]) for kI in 1:N_BINS]
        se = [n > 0 ? sqrt(s * (1 - s) / n) : 0.0 for (s, n) in zip(surv, n_obs)]
        ci_lo = clamp.(surv .- 1.96 .* se, 0, 1)
        ci_hi = clamp.(surv .+ 1.96 .* se, 0, 1)

        plot!(p, 1:N_BINS, surv;
            ribbon = (surv .- ci_lo, ci_hi .- surv),
            fillalpha = 0.15,
            label = "\$\\theta\$ = $(BIN_LABELS[kθ])",
            linewidth = 2.5,
            marker = :circle,
            markersize = 4,
            color = THETA_COLORS[kθ],
        )
    end

    push!(all_plots, p)
end

fig_combined = plot(all_plots...,
    layout = (1, T-1),
    size = (250 * (T-1), 350),
    margin = 3Plots.mm,
    bottom_margin = 8Plots.mm,
)
savefig(fig_combined, joinpath(OUTDIR, "empirical_survivor_all.pdf"))
println("  Saved: empirical_survivor_all.pdf")

println("\n  All diagnostics saved to $OUTDIR")
println("Done.")
