#!/usr/bin/env julia
# -----------------------------------------------------------------
# Generate LaTeX tables and PDF figures for the multi-period spec
# (ordered probit emissions and transitions)
#
# Reads estimation_multiperiod_parametric.jld2 from output/ and produces:
#   - Data description table (measures per period)
#   - Complementarity test results table (NP only)
#   - Transition parameter table (ordered probit β coefficients)
#   - Survivor function plots (with CIs from nonparametric score)
#   - Emission probability plots (backup slides)
#
# All output goes to slides_multiperiod/figures/
# -----------------------------------------------------------------

using LinearAlgebra, Random, Statistics, JLD2, Plots, StatsPlots, LaTeXStrings

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod.jl"))
include(joinpath(@__DIR__, "..", "src", "inference.jl"))

gr(size=(900, 500))

const OUTDIR = joinpath(dirname(@__DIR__), "slides_multiperiod", "figures")
mkpath(OUTDIR)

const AGE_RANGES = [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13)]
const PERIOD_LABELS = ["$(a[1])--$(a[2])" for a in AGE_RANGES]
const TRANSITION_LABELS = ["$(PERIOD_LABELS[t]) → $(PERIOD_LABELS[t+1])" for t in 1:6]

# -----------------------------------------------------------------
# Load results
# -----------------------------------------------------------------

function load_multiperiod_results()
    outdir = joinpath(dirname(@__DIR__), "output")
    path = joinpath(outdir, "estimation_multiperiod_parametric.jld2")
    if !isfile(path)
        error("Multi-period results not found: $path\nRun scripts/run_estimation_multiperiod.jl first.")
    end
    return load(path)
end

# -----------------------------------------------------------------
# 1. Data description table
# -----------------------------------------------------------------

function generate_data_table(data, config)
    T = data.T
    N = data.N

    lines = String[]
    push!(lines, "\\begin{table}[t]")
    push!(lines, "\\centering")
    push!(lines, "\\caption{Measures by Period}")
    push!(lines, "\\label{tab:measures}")
    push!(lines, "\\begin{tabular}{clcc}")
    push!(lines, "\\toprule")
    push!(lines, "Period & Ages & Skill Measures & Investment Measures \\\\")
    push!(lines, "\\midrule")

    for t in 1:T
        n_skill = length(data.X_θ[t])
        n_inv = length(data.X_I[t])
        push!(lines, "$t & $(PERIOD_LABELS[t]) & $n_skill & $n_inv \\\\")
    end

    push!(lines, "\\midrule")
    em_θ = config !== nothing ? string(get(config, :emission_θ, :nonparametric)) : "nonparametric"
    em_I = config !== nothing ? string(get(config, :emission_I, :nonparametric)) : "nonparametric"
    trans = config !== nothing ? string(get(config, :transition, :nonparametric)) : "nonparametric"
    push!(lines, "\\multicolumn{4}{l}{\\footnotesize \$N = $(N)\$, \$r_\\theta = $(data.r_θ)\$, \$r_I = $(data.r_I)\$, emissions: $em_θ, transition: $trans} \\\\")
    push!(lines, "\\bottomrule")
    push!(lines, "\\end{tabular}")
    push!(lines, "\\end{table}")

    return join(lines, "\n")
end

# Detailed measure table: lists measure names grouped by type and source
function generate_detailed_measure_table(data)
    T = data.T
    lines = String[]

    push!(lines, "\\begin{table}[t]")
    push!(lines, "\\centering")
    push!(lines, "\\caption{Detailed Measure Availability}")
    push!(lines, "\\label{tab:measures_detail}")
    push!(lines, "\\footnotesize")
    push!(lines, "\\begin{tabular}{cl" * repeat("c", T) * "}")
    push!(lines, "\\toprule")
    header = " & Measure"
    for t in 1:T
        header *= " & $(PERIOD_LABELS[t])"
    end
    header *= " \\\\"
    push!(lines, header)
    push!(lines, "\\midrule")

    # Skill measures: count by source type
    sources = ["MSD", "PPVTRaw", "MATH", "RECOG", "COMP", "DIGITF", "DIGITB"]
    source_labels = ["MSD (binary)", "PPVT", "PIAT Math", "PIAT Reading", "PIAT Comprehension", "Digit Span Fwd", "Digit Span Bwd"]

    push!(lines, "\\multicolumn{$(T+2)}{l}{\\textit{Cognitive Skill Measures}} \\\\")
    for (src, lbl) in zip(sources, source_labels)
        row = " & $lbl"
        for t in 1:T
            names = data.measure_names_θ[t]
            if src == "MSD"
                n = count(n -> !occursin("PPVTRaw", n) && !occursin("MATH", n) &&
                              !occursin("RECOG", n) && !occursin("COMP", n) &&
                              !occursin("DIGITF", n) && !occursin("DIGITB", n), names)
            else
                n = count(n -> n == src, names)
            end
            row *= n > 0 ? " & $n" : " & --"
        end
        row *= " \\\\"
        push!(lines, row)
    end

    push!(lines, "\\midrule")
    push!(lines, "\\multicolumn{$(T+2)}{l}{\\textit{Investment Measures (HOME)}} \\\\")

    all_inv_names = sort(unique(vcat([data.measure_names_I[t] for t in 1:T]...)))

    for inv_name in all_inv_names
        lbl = replace(inv_name, "_" => "\\_")
        row = " & \\texttt{$lbl}"
        for t in 1:T
            if inv_name ∈ data.measure_names_I[t]
                idx = findfirst(==(inv_name), data.measure_names_I[t])
                n_obs = sum(data.X_I[t][idx] .> 0)
                row *= " & $n_obs"
            else
                row *= " & --"
            end
        end
        row *= " \\\\"
        push!(lines, row)
    end

    push!(lines, "\\bottomrule")
    push!(lines, "\\end{tabular}")
    push!(lines, "\\end{table}")

    return join(lines, "\n")
end

# -----------------------------------------------------------------
# 2. Complementarity test results table (NP test only)
# -----------------------------------------------------------------

function generate_test_table(test_results, r_θ, r_I;
        caption="Complementarity Test Results (\$r_\\theta=$r_θ, r_I=$r_I\$)",
        label="tab:test_results")
    lines = String[]

    push!(lines, "\\begin{table}[t]")
    push!(lines, "\\centering")
    push!(lines, "\\caption{$caption}")
    push!(lines, "\\label{$label}")
    push!(lines, "\\begin{tabular}{lccc}")
    push!(lines, "\\toprule")
    push!(lines, "Transition & \$T_n\$ & \$c_\\alpha\$ & Reject \\\\")
    push!(lines, "\\midrule")

    n_transitions = length(TRANSITION_LABELS)
    for t in 1:n_transitions
        res = test_results[t]
        rej = res.reject ? "\\checkmark" : ""
        push!(lines, "$(TRANSITION_LABELS[t]) & $(round(res.T_n; digits=2)) & $(round(res.c_α; digits=2)) & $rej \\\\")
    end

    push!(lines, "\\bottomrule")
    push!(lines, "\\end{tabular}")
    push!(lines, "\\end{table}")

    return join(lines, "\n")
end

# -----------------------------------------------------------------
# 2b. Transition parameter table (ordered probit)
# -----------------------------------------------------------------

function generate_transition_table(specs, r_θ, r_I)
    lines = String[]

    push!(lines, "\\begin{table}[t]")
    push!(lines, "\\centering")
    push!(lines, "\\caption{Ordered Probit Transition Parameters}")
    push!(lines, "\\label{tab:transition_params}")
    push!(lines, "\\begin{tabular}{l" * repeat("c", length(TRANSITION_LABELS)) * "}")
    push!(lines, "\\toprule")

    header = "Parameter"
    for lbl in TRANSITION_LABELS
        header *= " & $lbl"
    end
    header *= " \\\\"
    push!(lines, header)
    push!(lines, "\\midrule")

    trans_specs = specs[:transition]

    # β₁ (θ)
    row = "\$\\beta_1\$ (\$\\theta\$)"
    for spec in trans_specs
        row *= " & $(round(spec.beta1; digits=3))"
    end
    row *= " \\\\"
    push!(lines, row)

    # β₂ (I)
    row = "\$\\beta_2\$ (\$I\$)"
    for spec in trans_specs
        row *= " & $(round(spec.beta2; digits=3))"
    end
    row *= " \\\\"
    push!(lines, row)

    # β₃ (θ·I)
    row = "\$\\beta_3\$ (\$\\theta \\cdot I\$)"
    for spec in trans_specs
        row *= " & $(round(spec.beta3; digits=3))"
    end
    row *= " \\\\"
    push!(lines, row)

    # Cutpoints
    push!(lines, "\\midrule")
    n_cuts = length(trans_specs[1].c)
    for k in 1:n_cuts
        row = "\$c_$(k)\$"
        for spec in trans_specs
            row *= " & $(round(spec.c[k]; digits=3))"
        end
        row *= " \\\\"
        push!(lines, row)
    end

    push!(lines, "\\midrule")
    push!(lines, "\\multicolumn{$(length(TRANSITION_LABELS)+1)}{l}{\\footnotesize \$P(\\theta' \\leq k | \\theta, I) = \\Phi(c_k - \\beta_1 \\theta - \\beta_2 I - \\beta_3 \\theta I)\$} \\\\")
    push!(lines, "\\bottomrule")
    push!(lines, "\\end{tabular}")
    push!(lines, "\\end{table}")

    return join(lines, "\n")
end

# -----------------------------------------------------------------
# 3. Survivor function plots (single Π with CIs)
# -----------------------------------------------------------------

const THETA_COLORS = [:steelblue, :darkorange, :forestgreen, :firebrick, :purple]

function compute_survivor_se_nonparametric_mp(params, test_results, N)
    n_transitions = length(params.Π)
    surv_SE = Vector{Array{Float64,3}}(undef, n_transitions)
    for t in 1:n_transitions
        Var_free = pi_free_var_nonparametric(test_results[t].Σ_score, N)
        surv_SE[t] = survivor_se_all(Var_free, params.Π[t])
    end
    return surv_SE
end

function plot_survivor(params, surv_SE, t, r_θ, r_I)
    n_cutoffs = r_θ - 1
    Π_t = params.Π[t]
    SE_t = surv_SE[t]

    plots = []
    for k in 1:n_cutoffs
        p = plot(
            title = (k == 1 ? TRANSITION_LABELS[t] * "\n" : "") *
                    "\$\\bar{F}($k | \\theta, I)\$",
            xlabel = "Investment (\$I\$)",
            ylabel = "\$\\bar{F}($k | \\theta, I)\$",
            ylim = (0, 1.05),
            xticks = 1:r_I,
            legend = (k == n_cutoffs) ? :topright : false,
            titlefontsize = 10,
            guidefontsize = 9,
            tickfontsize = 8,
            grid = true,
            gridalpha = 0.3,
        )

        for kθ in 1:r_θ
            surv = [1.0 - sum(Π_t[1:k, kθ, kI]) for kI in 1:r_I]
            se = [SE_t[k, kθ, kI] for kI in 1:r_I]
            ci_lo = clamp.(surv .- 1.96 .* se, 0, 1)
            ci_hi = clamp.(surv .+ 1.96 .* se, 0, 1)

            plot!(p, 1:r_I, surv;
                ribbon = (surv .- ci_lo, ci_hi .- surv),
                fillalpha = 0.15,
                label = "\$\\theta=$kθ\$",
                linewidth = 2.5,
                marker = :circle,
                markersize = 4,
                color = THETA_COLORS[kθ],
            )
        end

        push!(plots, p)
    end

    n_cols = min(3, n_cutoffs)
    n_rows = ceil(Int, n_cutoffs / n_cols)

    fig = plot(plots...,
        layout = (n_rows, n_cols),
        size = (350 * n_cols, 300 * n_rows),
        margin = 3Plots.mm,
    )
    return fig
end

# -----------------------------------------------------------------
# 4. Emission plots (backup slides)
# -----------------------------------------------------------------

function plot_skill_emissions(params, data, t)
    r_θ = data.r_θ
    P_list = params.P_θ[t]
    names_list = data.measure_names_θ[t]
    κ_list = data.κ_θ[t]

    # Group MSD items by suffix
    groups = Dict{String, Vector{Int}}()
    continuous_idx = Int[]
    for m in eachindex(P_list)
        if κ_list[m] == 2
            name = names_list[m]
            parts = split(name, ".")
            suffix = length(parts) >= 2 ? parts[end] : "other"
            battery = "MSD." * suffix
            push!(get!(groups, battery, Int[]), m)
        else
            push!(continuous_idx, m)
        end
    end

    summary_names = String[]
    summary_matrix = zeros(0, r_θ)

    for battery in sort(collect(keys(groups)))
        indices = groups[battery]
        avg_pass = zeros(1, r_θ)
        for m in indices, k in 1:r_θ
            avg_pass[1, k] += P_list[m][2, k]
        end
        avg_pass ./= length(indices)
        push!(summary_names, "$battery (n=$(length(indices)))")
        summary_matrix = vcat(summary_matrix, avg_pass)
    end

    for m in continuous_idx
        P = P_list[m]
        κ_m = size(P, 1)
        e_norm = zeros(1, r_θ)
        for k in 1:r_θ
            e_norm[1, k] = sum(j * P[j, k] for j in 1:κ_m) / κ_m
        end
        push!(summary_names, names_list[m])
        summary_matrix = vcat(summary_matrix, e_norm)
    end

    if isempty(summary_names)
        return plot(title="Period $t: no measures", framestyle=:none)
    end

    colors = reshape(THETA_COLORS[1:r_θ], 1, r_θ)
    fig = groupedbar(summary_names, summary_matrix;
        title = "Period $t (ages $(PERIOD_LABELS[t]))",
        ylabel = "Score",
        xrotation = 45,
        label = reshape(["\$\\theta=$k\$" for k in 1:r_θ], 1, r_θ),
        legend = :topleft,
        ylim = (0, 1.05),
        bar_width = 0.8,
        titlefontsize = 11,
        guidefontsize = 9,
        tickfontsize = 7,
        color = colors,
        size = (max(600, 50 * length(summary_names)), 400),
        margin = 5Plots.mm,
        bottom_margin = 12Plots.mm,
    )
    return fig
end

function plot_investment_emissions(params, data, t)
    r_I = data.r_I
    P_list = params.P_I[t]
    names_list = data.measure_names_I[t]
    M = length(P_list)

    if M == 0
        return plot(title="Period $t: no investment measures", framestyle=:none)
    end

    inv_colors = reshape(THETA_COLORS[1:min(r_I, length(THETA_COLORS))], 1, min(r_I, length(THETA_COLORS)))
    measure_plots = []

    for m in 1:M
        P = P_list[m]
        κ_m = size(P, 1)
        cat_labels = κ_m == 2 ? ["0", "1"] : string.(1:κ_m)

        p = groupedbar(cat_labels, P;
            title = names_list[m],
            ylabel = (mod1(m, 4) == 1) ? "\$P(\\mathrm{cat} | I)\$" : "",
            label = reshape(["\$I=$k\$" for k in 1:r_I], 1, r_I),
            legend = (m == 1) ? :topright : false,
            ylim = (0, 1.05),
            bar_width = 0.8,
            titlefontsize = 8,
            guidefontsize = 7,
            tickfontsize = 6,
            color = inv_colors,
        )
        push!(measure_plots, p)
    end

    n_cols = min(4, M)
    n_rows = ceil(Int, M / n_cols)

    fig = plot(measure_plots...,
        layout = (n_rows, n_cols),
        size = (280 * n_cols, 280 * n_rows),
        margin = 5Plots.mm,
        bottom_margin = 8Plots.mm,
    )
    return fig
end

# -----------------------------------------------------------------
# Main: generate all output
# -----------------------------------------------------------------

println("Generating multi-period output...")

res = load_multiperiod_results()
data = res["data"]
params = res["best_params"]
test_results = res["test_results"]
config = get(res, "parametric_config", nothing)
specs = get(res, "best_specs", nothing)

r_θ = data.r_θ
r_I = data.r_I
T = data.T
N = data.N
n_transitions = T - 1

println("  r_θ=$r_θ, r_I=$r_I, T=$T, N=$N")

# ── 1. Data description table ──
println("  Data tables...")
open(joinpath(OUTDIR, "table_measures.tex"), "w") do io
    write(io, generate_data_table(data, config))
end
open(joinpath(OUTDIR, "table_measures_detail.tex"), "w") do io
    write(io, generate_detailed_measure_table(data))
end

# ── 2. Test results table ──
println("  Test results table...")
open(joinpath(OUTDIR, "table_test.tex"), "w") do io
    write(io, generate_test_table(test_results, r_θ, r_I))
end

# ── 2b. Transition parameter table ──
if specs !== nothing && haskey(specs, :transition)
    println("  Transition parameter table...")
    open(joinpath(OUTDIR, "table_transition.tex"), "w") do io
        write(io, generate_transition_table(specs, r_θ, r_I))
    end
end

# ── 3. Survivor function plots ──
println("  Survivor function plots...")
Var_free_list = get(res, "Var_free", nothing)
if Var_free_list !== nothing
    println("    Using parametric Var(Π̂_free) for survivor SEs")
    surv_SE = [survivor_se_all(Var_free_list[t], params.Π[t]) for t in 1:n_transitions]
else
    surv_SE = compute_survivor_se_nonparametric_mp(params, test_results, N)
end

for t in 1:n_transitions
    fig = plot_survivor(params, surv_SE, t, r_θ, r_I)
    savefig(fig, joinpath(OUTDIR, "survivor_t$(t).pdf"))
end

# ── 4. Emission plots (backup slides) ──
println("  Emission plots...")
for t in 1:T
    fig_skill = plot_skill_emissions(params, data, t)
    savefig(fig_skill, joinpath(OUTDIR, "emissions_skill_t$(t).pdf"))

    fig_inv = plot_investment_emissions(params, data, t)
    savefig(fig_inv, joinpath(OUTDIR, "emissions_invest_t$(t).pdf"))
end

println("\n  All output saved to $OUTDIR")
println("Done.")
