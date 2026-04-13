#!/usr/bin/env julia
# -----------------------------------------------------------------
# Generate LaTeX tables and PDF figures for the grouped monotone NN spec
# (ordered probit emissions, monotone NN transitions, shared Π within groups)
#
# Reads estimation_multiperiod_nn_k4_grouped.jld2 from output/ and produces:
#   - Data description table (measures per period)
#   - Complementarity test results table (per group)
#   - Monotone NN transition parameter table (per group)
#   - Survivor function plots (one Π per group, with CIs)
#   - Transition probability bar charts (per group)
#   - Emission probability plots (backup slides)
#
# All output goes to slides_multiperiod_nn/figures/
# -----------------------------------------------------------------

using LinearAlgebra, Random, Statistics, JLD2, Plots, StatsPlots, LaTeXStrings

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod.jl"))
include(joinpath(@__DIR__, "..", "src", "inference.jl"))

gr(size=(900, 500))

const OUTDIR = joinpath(dirname(@__DIR__), "slides_multiperiod_nn", "figures")
mkpath(OUTDIR)

const AGE_RANGES = [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13)]
const PERIOD_LABELS = ["$(a[1])--$(a[2])" for a in AGE_RANGES]
const TRANSITION_LABELS = ["$(PERIOD_LABELS[t]) → $(PERIOD_LABELS[t+1])" for t in 1:6]

# -----------------------------------------------------------------
# Load results
# -----------------------------------------------------------------

function load_nn_results()
    outdir = joinpath(dirname(@__DIR__), "output")
    path = joinpath(outdir, "estimation_multiperiod_nn_k4_grouped.jld2")
    if !isfile(path)
        error("Monotone NN results not found: $path\nRun scripts/run_estimation_multiperiod_nn_k4.jl first.")
    end
    return load(path)
end

# -----------------------------------------------------------------
# 1. Data description table
# -----------------------------------------------------------------

function generate_data_table(data, config, transition_groups)
    T = data.T
    N = data.N
    K = get(config, :K, "?")

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
    group_strs = [join(["$(t)" for t in g], ",") for g in transition_groups]
    groups_desc = join(["\\{$(s)\\}" for s in group_strs], ", ")
    push!(lines, "\\multicolumn{4}{l}{\\footnotesize \$N = $(N)\$, \$r_\\theta = $(data.r_θ)\$, \$r_I = $(data.r_I)\$, monotone NN (\$K=$K\$)} \\\\")
    push!(lines, "\\multicolumn{4}{l}{\\footnotesize Transition groups: $groups_desc} \\\\")
    push!(lines, "\\bottomrule")
    push!(lines, "\\end{tabular}")
    push!(lines, "\\end{table}")

    return join(lines, "\n")
end

# Detailed measure table
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
# 2. Complementarity test results table (per group)
# -----------------------------------------------------------------

function generate_test_table(test_results, transition_groups, r_θ, r_I)
    lines = String[]

    push!(lines, "\\begin{table}[t]")
    push!(lines, "\\centering")
    push!(lines, "\\caption{Complementarity Test Results (Grouped, \$r_\\theta=$r_θ, r_I=$r_I\$)}")
    push!(lines, "\\label{tab:test_results}")
    push!(lines, "\\begin{tabular}{lccc}")
    push!(lines, "\\toprule")
    push!(lines, "Group & \$T_n\$ & \$c_\\alpha\$ & Reject \\\\")
    push!(lines, "\\midrule")

    for (g, res) in enumerate(test_results)
        group = transition_groups[g]
        label = join([TRANSITION_LABELS[t] for t in group], ", ")
        rej = res.reject ? "\\checkmark" : ""
        push!(lines, "$label & $(round(res.T_n; digits=2)) & $(round(res.c_α; digits=2)) & $rej \\\\")
    end

    push!(lines, "\\bottomrule")
    push!(lines, "\\end{tabular}")
    push!(lines, "\\end{table}")

    return join(lines, "\n")
end

# -----------------------------------------------------------------
# 2b. Monotone NN transition parameter table
# -----------------------------------------------------------------

function generate_transition_table(specs, transition_groups, r_θ, r_I)
    lines = String[]
    n_groups = length(transition_groups)
    trans_specs = specs[:transition]

    # One spec per group (representative = first transition in group)
    group_specs = [trans_specs[g[1]]::MonotoneNNTransition for g in transition_groups]
    K = group_specs[1].K

    push!(lines, "\\begin{table}[t]")
    push!(lines, "\\centering")
    push!(lines, "\\caption{Monotone NN Transition Parameters (\$K=$K\$ hidden units)}")
    push!(lines, "\\label{tab:transition_params}")
    push!(lines, "\\begin{tabular}{l" * repeat("c", n_groups) * "}")
    push!(lines, "\\toprule")

    # Header
    header = "Parameter"
    for group in transition_groups
        if length(group) == 1
            header *= " & $(TRANSITION_LABELS[group[1]])"
        else
            header *= " & $(PERIOD_LABELS[group[1]])--$(PERIOD_LABELS[group[end]+1])"
        end
    end
    header *= " \\\\"
    push!(lines, header)
    push!(lines, "\\midrule")

    # Cutpoints
    n_cuts = length(_unpack_survivor_cutoffs(group_specs[1].c_raw))
    for k in 1:n_cuts
        row = "\$c_$(k)\$"
        for spec in group_specs
            c = _unpack_survivor_cutoffs(spec.c_raw)
            row *= " & $(round(c[k]; digits=3))"
        end
        row *= " \\\\"
        push!(lines, row)
    end

    push!(lines, "\\midrule")

    # Hidden unit parameters
    for h in 1:K
        # alpha1
        row = "\$\\alpha_{1,$h}\$"
        for spec in group_specs
            row *= " & $(round(spec.alpha1[h]; digits=3))"
        end
        row *= " \\\\"
        push!(lines, row)
    end

    push!(lines, "\\midrule")

    for h in 1:K
        # alpha2 (exponentiated)
        row = "\$\\alpha_{2,$h}\$"
        for spec in group_specs
            a2 = exp(spec.alpha2_raw[h])
            row *= " & $(round(a2; digits=3))"
        end
        row *= " \\\\"
        push!(lines, row)
    end

    push!(lines, "\\midrule")

    for h in 1:K
        # bias
        row = "\$b_$h\$"
        for spec in group_specs
            row *= " & $(round(spec.b[h]; digits=3))"
        end
        row *= " \\\\"
        push!(lines, row)
    end

    push!(lines, "\\midrule")

    for h in 1:K
        # weights (exponentiated)
        row = "\$w_$h\$"
        for spec in group_specs
            w = exp(spec.log_w[h])
            row *= " & $(round(w; digits=3))"
        end
        row *= " \\\\"
        push!(lines, row)
    end

    push!(lines, "\\midrule")
    push!(lines, "\\multicolumn{$(n_groups+1)}{l}{\\footnotesize \$\\bar{F}(k|\\theta,I) = \\Phi\\bigl(c_k + \\sum_h w_h \\Phi(\\alpha_{1,h}\\theta + \\alpha_{2,h} I + b_h)\\bigr)\$} \\\\")
    push!(lines, "\\bottomrule")
    push!(lines, "\\end{tabular}")
    push!(lines, "\\end{table}")

    return join(lines, "\n")
end

# -----------------------------------------------------------------
# 3. Transition probability bar charts
# -----------------------------------------------------------------

const THETA_COLORS = [:steelblue, :darkorange, :forestgreen, :firebrick, :purple]

"""
Compute SE for each Π[j, kθ, kI] (including the last row via sum constraint).
Returns Array{Float64,3} of shape (r_θ, r_θ, r_I).
"""
function compute_transition_se(Var_free, r_θ, r_I)
    p_idx = LinearIndices((r_θ - 1, r_θ, r_I))
    SE = zeros(r_θ, r_θ, r_I)
    for kI in 1:r_I, kθ in 1:r_θ
        # Free rows 1:r_θ-1
        for j in 1:r_θ-1
            SE[j, kθ, kI] = sqrt(max(0.0, Var_free[p_idx[j, kθ, kI], p_idx[j, kθ, kI]]))
        end
        # Last row: Π[r_θ, kθ, kI] = 1 - Σ_{j<r_θ} Π[j, kθ, kI]
        # Var(last) = Σ_{j,j2} Cov(Π_j, Π_j2)
        var_last = 0.0
        for j in 1:r_θ-1, j2 in 1:r_θ-1
            var_last += Var_free[p_idx[j, kθ, kI], p_idx[j2, kθ, kI]]
        end
        SE[r_θ, kθ, kI] = sqrt(max(0.0, var_last))
    end
    return SE
end

function plot_transition_probs(params, Var_free, group, r_θ, r_I)
    t_rep = group[1]
    Π_t = params.Π[t_rep]
    SE_t = compute_transition_se(Var_free, r_θ, r_I)

    # One panel per current θ, bars grouped by θ' for each I
    subplots = []
    for kθ in 1:r_θ
        # x-axis: I values; grouped bars for each θ'
        prob_matrix = zeros(r_I, r_θ)    # (n_groups, n_bars_per_group)
        err_matrix = zeros(r_I, r_θ)
        for kI in 1:r_I, j in 1:r_θ
            prob_matrix[kI, j] = Π_t[j, kθ, kI]
            err_matrix[kI, j] = 1.96 * SE_t[j, kθ, kI]
        end

        p = groupedbar(string.(1:r_I), prob_matrix;
            title = "\$\\theta = $kθ\$",
            xlabel = (kθ > r_θ ÷ 2) ? "Investment (\$I\$)" : "",
            ylabel = (kθ == 1 || kθ == (r_θ ÷ 2 + 1)) ? "\$P(\\theta' \\mid \\theta, I)\$" : "",
            label = reshape(["\$\\theta'=$j\$" for j in 1:r_θ], 1, r_θ),
            legend = (kθ == r_θ) ? :topright : false,
            ylim = (0, 1.05),
            bar_width = 0.8,
            titlefontsize = 10,
            guidefontsize = 9,
            tickfontsize = 8,
            color = reshape(THETA_COLORS[1:r_θ], 1, r_θ),
            yerror = err_matrix,
            linecolor = :black,
        )
        push!(subplots, p)
    end

    n_cols = min(2, r_θ)
    n_rows = ceil(Int, r_θ / n_cols)

    fig = plot(subplots...,
        layout = (n_rows, n_cols),
        size = (450 * n_cols, 320 * n_rows),
        margin = 5Plots.mm,
    )
    return fig
end

# -----------------------------------------------------------------
# 4. Survivor function plots
# -----------------------------------------------------------------

function plot_survivor_grouped(params, surv_SE, group, r_θ, r_I)
    t_rep = group[1]
    n_cutoffs = r_θ - 1
    Π_t = params.Π[t_rep]
    SE_t = surv_SE[t_rep]

    if length(group) == 1
        title_prefix = TRANSITION_LABELS[group[1]]
    else
        title_prefix = join([TRANSITION_LABELS[t] for t in group], ", ")
    end

    plots = []
    for k in 1:n_cutoffs
        p = plot(
            title = (k == 1 ? title_prefix * "\n" : "") *
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
# 5. Emission plots (backup slides)
# -----------------------------------------------------------------

function plot_skill_emissions(params, data, t)
    r_θ = data.r_θ
    P_list = params.P_θ[t]
    names_list = data.measure_names_θ[t]
    κ_list = data.κ_θ[t]

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

println("Generating monotone NN multi-period output...")

res = load_nn_results()
data = res["data"]
params = res["best_params"]
test_results = res["test_results"]
config = get(res, "parametric_config", nothing)
specs = get(res, "best_specs", nothing)
transition_groups = res["transition_groups"]

r_θ = data.r_θ
r_I = data.r_I
T = data.T
N = data.N
n_transitions = T - 1
n_groups = length(transition_groups)

println("  r_θ=$r_θ, r_I=$r_I, T=$T, N=$N, K=$(config !== nothing ? get(config, :K, "?") : "?")")
for (g, group) in enumerate(transition_groups)
    labels = [TRANSITION_LABELS[t] for t in group]
    println("  Group $g: $(join(labels, ", "))")
end

# ── 1. Data description table ──
println("  Data tables...")
open(joinpath(OUTDIR, "table_measures.tex"), "w") do io
    write(io, generate_data_table(data, config, transition_groups))
end
open(joinpath(OUTDIR, "table_measures_detail.tex"), "w") do io
    write(io, generate_detailed_measure_table(data))
end

# ── 2. Test results table ──
println("  Test results table...")
open(joinpath(OUTDIR, "table_test.tex"), "w") do io
    write(io, generate_test_table(test_results, transition_groups, r_θ, r_I))
end

# ── 2b. Transition parameter table ──
if specs !== nothing && haskey(specs, :transition)
    println("  Transition parameter table...")
    open(joinpath(OUTDIR, "table_transition.tex"), "w") do io
        write(io, generate_transition_table(specs, transition_groups, r_θ, r_I))
    end
end

# ── 3. Transition probability bar charts and survivor function plots ──
println("  Transition probability and survivor function plots...")
Var_free_list = get(res, "Var_free", nothing)
if Var_free_list !== nothing
    println("    Using parametric Var(Π̂_free) for survivor SEs")
    surv_SE = [survivor_se_all(Var_free_list[t], params.Π[t]) for t in 1:n_transitions]
else
    surv_SE = Vector{Array{Float64,3}}(undef, n_transitions)
    for (g, group) in enumerate(transition_groups)
        Var_free = pi_free_var_nonparametric(test_results[g].Σ_score, N)
        se = survivor_se_all(Var_free, params.Π[group[1]])
        for t in group
            surv_SE[t] = se
        end
    end
end

for (g, group) in enumerate(transition_groups)
    fig = plot_survivor_grouped(params, surv_SE, group, r_θ, r_I)
    savefig(fig, joinpath(OUTDIR, "survivor_g$(g).pdf"))

    # Transition probability bar charts
    if Var_free_list !== nothing
        fig_trans = plot_transition_probs(params, Var_free_list[group[1]], group, r_θ, r_I)
    else
        # Zero SEs if no Var_free available
        Var_zero = zeros((r_θ-1)*r_θ*r_I, (r_θ-1)*r_θ*r_I)
        fig_trans = plot_transition_probs(params, Var_zero, group, r_θ, r_I)
    end
    savefig(fig_trans, joinpath(OUTDIR, "transition_probs_g$(g).pdf"))
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
