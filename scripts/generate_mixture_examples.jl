#!/usr/bin/env julia
# -----------------------------------------------------------------
# Generate example mixture probit technologies (r_θ=4, r_I=5, K=4)
# that exhibit ordinal complementarity, to illustrate the flexibility
# of the parametric specification.
#
# Strategy: start from candidate parameters and nudge weights via
# optimization to satisfy all supermodularity inequalities while
# preserving economically interesting patterns.
#
# Output: slides_multiperiod_mixture/figures/example_*.pdf
# -----------------------------------------------------------------

using LinearAlgebra, Plots, LaTeXStrings, Optim
using Distributions: Normal, cdf

gr(size=(900, 500))

const OUTDIR = joinpath(dirname(@__DIR__), "slides_multiperiod_mixture", "figures")
mkpath(OUTDIR)

const R_THETA = 4
const R_I = 5
const K = 4
const I_GRID = collect(1.0:R_I)
const THETA_COLORS = [:steelblue, :darkorange, :forestgreen, :firebrick]

Φ(x) = cdf(Normal(), x)

# -----------------------------------------------------------------
# Build Π from mixture probit parameters
# -----------------------------------------------------------------

function build_pi(c, a, W)
    r_θ = length(c) + 1
    r_I = length(I_GRID)
    Π = zeros(r_θ, r_θ, r_I)

    for kI in 1:r_I, kθ in 1:r_θ
        surv = zeros(r_θ - 1)
        for k in 1:r_θ-1
            for h in 1:K
                surv[k] += W[h, kθ] * (1.0 - Φ(c[k] - a[h] * I_GRID[kI]))
            end
        end
        Π[1, kθ, kI] = 1.0 - surv[1]
        for j in 2:r_θ-1
            Π[j, kθ, kI] = surv[j-1] - surv[j]
        end
        Π[r_θ, kθ, kI] = surv[r_θ-1]
    end
    return Π
end

# -----------------------------------------------------------------
# Check supermodularity and compute violations
# -----------------------------------------------------------------

function supermodularity_violations(Π)
    r_θ, _, r_I = size(Π)
    violations = Float64[]
    for k in 1:r_θ-1
        for kθ in 1:r_θ, kθ2 in kθ+1:r_θ
            for kI in 1:r_I, kI2 in kI+1:r_I
                surv(kθ_, kI_) = 1.0 - sum(Π[1:k, kθ_, kI_])
                did = surv(kθ2, kI2) + surv(kθ, kI) - surv(kθ2, kI) - surv(kθ, kI2)
                push!(violations, did)
            end
        end
    end
    return violations
end

function check_supermodularity(Π)
    viols = supermodularity_violations(Π)
    n_viol = count(v -> v < -1e-10, viols)
    max_viol = minimum(viols)
    return n_viol, max_viol
end

# -----------------------------------------------------------------
# Construct weights from logits, ensuring valid probabilities
# -----------------------------------------------------------------

function weights_from_logits(λ)
    # λ is (K-1) × r_θ; first component logit fixed at 0
    W = zeros(K, R_THETA)
    for kθ in 1:R_THETA
        logits = vcat(0.0, λ[:, kθ])
        logits .-= maximum(logits)
        exp_l = exp.(logits)
        W[:, kθ] = exp_l ./ sum(exp_l)
    end
    return W
end

# -----------------------------------------------------------------
# Find weights that satisfy supermodularity given (c, a)
#
# Minimize sum of squared violations + small penalty to keep weights
# close to a target (so the example stays economically interpretable)
# -----------------------------------------------------------------

function find_supermodular_weights(c, a, W_target; penalty=0.1)
    # Pack target into logits as starting point
    λ0 = zeros(K-1, R_THETA)
    for kθ in 1:R_THETA
        for h in 2:K
            λ0[h-1, kθ] = log(max(W_target[h, kθ], 1e-6)) - log(max(W_target[1, kθ], 1e-6))
        end
    end

    function objective(λ_vec)
        λ = reshape(λ_vec, K-1, R_THETA)
        W = weights_from_logits(λ)
        Π = build_pi(c, a, W)
        viols = supermodularity_violations(Π)

        # Penalize violations heavily
        loss = sum(min(0.0, v)^2 for v in viols) * 1000.0

        # Small penalty to stay near target
        loss += penalty * sum((W .- W_target).^2)

        return loss
    end

    result = optimize(objective, vec(λ0), NelderMead(),
                      Optim.Options(iterations=50000, g_tol=1e-12))

    λ_opt = reshape(Optim.minimizer(result), K-1, R_THETA)
    W_opt = weights_from_logits(λ_opt)
    return W_opt
end

# -----------------------------------------------------------------
# Plot survivor functions
# -----------------------------------------------------------------

function plot_example_survivor(Π, title_str)
    r_θ = size(Π, 1)
    r_I = size(Π, 3)
    n_cutoffs = r_θ - 1

    plots = []
    for k in 1:n_cutoffs
        p = plot(
            title = "\$\\bar{F}($k \\mid \\theta, I)\$",
            xlabel = "Investment (\$I\$)",
            ylabel = "\$\\bar{F}($k \\mid \\theta, I)\$",
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
            surv = [1.0 - sum(Π[1:k, kθ, kI]) for kI in 1:r_I]
            plot!(p, 1:r_I, surv;
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
        plot_title = title_str,
        plot_titlefontsize = 11,
    )
    return fig
end

# -----------------------------------------------------------------
# Examples: define (c, a, target W), then optimize W for validity
# -----------------------------------------------------------------

examples = []

# Example 1: "Self-productivity"
# High θ → weight on steep components; low θ → weight on flat
let
    c = [-0.5, 0.8, 2.0]
    a = [0.1, 0.3, 0.6, 1.0]
    W_target = [0.70 0.40 0.10 0.02;
                0.20 0.30 0.20 0.08;
                0.08 0.20 0.35 0.30;
                0.02 0.10 0.35 0.60]
    for kθ in 1:R_THETA; W_target[:, kθ] ./= sum(W_target[:, kθ]); end
    W = find_supermodular_weights(c, a, W_target)
    push!(examples, (c=c, a=a, W=W,
        name="Self-productivity",
        desc="Higher skill → stronger investment response"))
end

# Example 2: "Moderate complementarity"
# Gradual tilt toward steeper components
let
    c = [0.0, 1.2, 2.5]
    a = [0.15, 0.3, 0.5, 0.8]
    W_target = [0.40 0.30 0.20 0.10;
                0.30 0.30 0.25 0.20;
                0.20 0.25 0.30 0.30;
                0.10 0.15 0.25 0.40]
    for kθ in 1:R_THETA; W_target[:, kθ] ./= sum(W_target[:, kθ]); end
    W = find_supermodular_weights(c, a, W_target)
    push!(examples, (c=c, a=a, W=W,
        name="Moderate complementarity",
        desc="Gradual shift toward higher slopes"))
end

# Example 3: "Dynamic complementarity with persistence"
# Strong persistence (high diagonal) + investment amplified by skill
let
    c = [-1.2, 0.3, 1.5]
    a = [0.05, 0.15, 0.4, 0.9]
    W_target = [0.80 0.40 0.10 0.02;
                0.12 0.30 0.20 0.08;
                0.05 0.20 0.35 0.25;
                0.03 0.10 0.35 0.65]
    for kθ in 1:R_THETA; W_target[:, kθ] ./= sum(W_target[:, kθ]); end
    W = find_supermodular_weights(c, a, W_target)
    push!(examples, (c=c, a=a, W=W,
        name="Dynamic complementarity",
        desc="Strong persistence; investment effect amplified by skill"))
end

# -----------------------------------------------------------------
# Generate and save
# -----------------------------------------------------------------

println("Generating mixture probit examples (r_θ=$R_THETA, r_I=$R_I, K=$K)...\n")

all_valid = true
for (i, ex) in enumerate(examples)
    Π = build_pi(ex.c, ex.a, ex.W)
    n_viol, max_viol = check_supermodularity(Π)

    println("  Example $i: $(ex.name)")
    println("    c = $(round.(ex.c; digits=3))")
    println("    a = $(round.(ex.a; digits=3))")
    println("    Weights W[h, θ]:")
    for h in 1:K
        println("      h=$h: $(round.(ex.W[h, :]; digits=3))")
    end
    if n_viol == 0
        println("    ✓ Supermodularity satisfied")
    else
        println("    ✗ $n_viol violations (max = $(round(max_viol; digits=8)))")
        global all_valid = false
    end
    println()

    fig = plot_example_survivor(Π, "$(ex.name)")
    savefig(fig, joinpath(OUTDIR, "example_$(i).pdf"))
end

if !all_valid
    @warn "Some examples have supermodularity violations — check parameters"
end

# Combined figure: all examples side by side (k=2 cutoff)
println("  Combined comparison figure...")
combined_plots = []
for (i, ex) in enumerate(examples)
    Π = build_pi(ex.c, ex.a, ex.W)
    k = 2
    p = plot(
        title = "$(ex.name)\n\$\\bar{F}($k \\mid \\theta, I)\$",
        xlabel = "Investment (\$I\$)",
        ylabel = "\$\\bar{F}($k \\mid \\theta, I)\$",
        ylim = (0, 1.05),
        xticks = 1:R_I,
        legend = (i == length(examples)) ? :bottomright : false,
        titlefontsize = 9,
        guidefontsize = 8,
        tickfontsize = 7,
        grid = true,
        gridalpha = 0.3,
    )
    for kθ in 1:R_THETA
        surv = [1.0 - sum(Π[1:k, kθ, kI]) for kI in 1:R_I]
        plot!(p, 1:R_I, surv;
            label = "\$\\theta=$kθ\$",
            linewidth = 2.5,
            marker = :circle,
            markersize = 4,
            color = THETA_COLORS[kθ],
        )
    end
    push!(combined_plots, p)
end

fig_combined = plot(combined_plots...,
    layout = (1, length(examples)),
    size = (380 * length(examples), 320),
    margin = 5Plots.mm,
)
savefig(fig_combined, joinpath(OUTDIR, "examples_combined.pdf"))

println("\n  All output saved to $OUTDIR")
println("Done.")
