using LinearAlgebra, Random, Statistics, JLD2, Optim, Printf

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3_eps.jl"))
include(joinpath(@__DIR__, "..", "src", "prepare_data.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod_v3.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod_v3_eps.jl"))

# ─────────────────────────────────────────────────────────────────
# Re-fit transitions from fresh random starts, holding posteriors fixed.
#
# The V3+ε_Y fit has degenerate hidden units (log_w > 400).
# Strategy: use the saved ξ_sum_z (pairwise posterior SS), collapse
# over ε_Y to get 9-regime ξ_sum_z_trans, then re-optimise the
# transition specs from multiple random starting points.
# ─────────────────────────────────────────────────────────────────

const OUTPUT_DIR = joinpath(dirname(@__DIR__), "output")
const FIT_FILE = joinpath(OUTPUT_DIR, "estimation_v3_eps.jld2")
const TRANSITION_GROUPS = [[1, 2, 3], [4, 5, 6]]

println("Loading V3+ε_Y fit...")
saved = load(FIT_FILE)
pv3  = saved["params_v3_eps"]
psk  = saved["params_skill"]
ξ_full = saved["xi_sum_z"]      # [z_full][t] for z ∈ 1..27
dims = saved["data_dims"]
(; r_θ, r_I, r_P, r_ω, N, T) = dims
r_Y = pv3.r_Y
T_minus_1 = T - 1

println("  Current log_w values:")
for (g, spec) in enumerate(pv3.trans_specs)
    println("    Group $g: log_w = $(round.(spec.log_w; digits=1)),  exp = $(round.(exp.(spec.log_w); digits=2))")
end

# ── Collapse ξ over ε_Y ──
println("\nCollapsing ξ_sum_z over ε_Y...")
ξ_trans = collapse_xi_over_eps(ξ_full, r_P, r_ω, r_Y, T_minus_1, r_θ, r_I)
# ξ_trans[z_trans][t] :: (r_θ, r_I, r_θ, r_I), z_trans ∈ 1..9

θ_grid = collect(1.0:r_θ); I_grid = collect(1.0:r_I)
P_grid = collect(1.0:r_P); ω_grid = collect(1.0:r_ω)
opts = Optim.Options(iterations=200, g_tol=1e-7, show_trace=false)

# ── Try multiple K values and random starts ──
n_starts = 10
ν = 7.0

for K in [3, 4]
    println("\n" * "="^60)
    println("  K = $K, $n_starts random starts per group")
    println("="^60)

    for (g, group) in enumerate(TRANSITION_GROUPS)
        # Pool Q_sum across periods in group, per regime
        r_Z_trans = r_P * r_ω
        Q_sum_z = Vector{Array{Float64,3}}(undef, r_Z_trans)
        for z in 1:r_Z_trans
            Q_sum_z[z] = _qsum_from_xi(ξ_trans[z][group[1]])
            for t in group[2:end]
                Q_sum_z[z] = Q_sum_z[z] .+ _qsum_from_xi(ξ_trans[z][t])
            end
        end

        best_negll = Inf
        best_spec  = nothing

        for s in 1:n_starts
            rng = MersenneTwister(100 * g + s)
            spec0 = initialize_robit_nn_v3(r_θ, K; rng, ν)
            φ0 = pack_robit_nn_v3(spec0)

            obj = x -> transition_negll_robit_nn_v3(x, Q_sum_z, r_θ, r_I, K, ν,
                                                      r_P, r_ω;
                                                      θ_grid, I_grid, P_grid, ω_grid)
            res = try
                optimize(obj, φ0, LBFGS(), opts; autodiff=:forward)
            catch e
                nothing
            end
            if res !== nothing && Optim.converged(res)
                nl = Optim.minimum(res)
                spec_new = unpack_robit_nn_v3(Optim.minimizer(res), r_θ, K; ν)
                max_logw = maximum(abs.(spec_new.log_w))
                @printf("    Group %d start %2d: negll = %.2f, max|log_w| = %.1f %s\n",
                        g, s, nl, max_logw, max_logw > 50 ? "(degenerate)" : "")
                if nl < best_negll && max_logw < 50  # reject degenerate solutions
                    best_negll = nl
                    best_spec = spec_new
                end
            else
                @printf("    Group %d start %2d: failed\n", g, s)
            end
        end

        # Also try from the current (degenerate) spec with clamped weights
        spec_clamped = RobitNNTransitionV3(
            copy(pv3.trans_specs[g].c_raw),
            copy(pv3.trans_specs[g].alpha1),
            copy(pv3.trans_specs[g].alpha2_raw),
            copy(pv3.trans_specs[g].alpha3),
            copy(pv3.trans_specs[g].alpha4),
            copy(pv3.trans_specs[g].b),
            clamp.(pv3.trans_specs[g].log_w, -5.0, 5.0),  # clamp log_w
            K == pv3.trans_specs[g].K ? pv3.trans_specs[g].K : K,
            ν,
        )
        if K == pv3.trans_specs[g].K
            φ_cl = pack_robit_nn_v3(spec_clamped)
            obj = x -> transition_negll_robit_nn_v3(x, Q_sum_z, r_θ, r_I, K, ν,
                                                      r_P, r_ω;
                                                      θ_grid, I_grid, P_grid, ω_grid)
            res_cl = try
                optimize(obj, φ_cl, LBFGS(), opts; autodiff=:forward)
            catch; nothing; end
            if res_cl !== nothing && Optim.converged(res_cl)
                nl = Optim.minimum(res_cl)
                spec_cl = unpack_robit_nn_v3(Optim.minimizer(res_cl), r_θ, K; ν)
                max_logw = maximum(abs.(spec_cl.log_w))
                @printf("    Group %d clamped:  negll = %.2f, max|log_w| = %.1f %s\n",
                        g, nl, max_logw, max_logw > 50 ? "(degenerate)" : "")
                if nl < best_negll && max_logw < 50
                    best_negll = nl
                    best_spec = spec_cl
                end
            end
        end

        if best_spec !== nothing
            @printf("    Group %d BEST:     negll = %.2f\n", g, best_negll)
            println("      log_w = $(round.(best_spec.log_w; digits=2))")
            println("      exp(log_w) = $(round.(exp.(best_spec.log_w); digits=3))")
        else
            println("    Group $g: no non-degenerate solution found!")
        end
    end
end

# Compare with original fit's negll
println("\n" * "="^60)
println("  Original (degenerate) fit's transition negll:")
println("="^60)
r_Z_trans = r_P * r_ω
for (g, group) in enumerate(TRANSITION_GROUPS)
    Q_sum_z = Vector{Array{Float64,3}}(undef, r_Z_trans)
    for z in 1:r_Z_trans
        Q_sum_z[z] = _qsum_from_xi(ξ_trans[z][group[1]])
        for t in group[2:end]
            Q_sum_z[z] = Q_sum_z[z] .+ _qsum_from_xi(ξ_trans[z][t])
        end
    end
    spec = pv3.trans_specs[g]
    φ = pack_robit_nn_v3(spec)
    nl = transition_negll_robit_nn_v3(φ, Q_sum_z, r_θ, r_I, spec.K, ν,
                                        r_P, r_ω; θ_grid, I_grid, P_grid, ω_grid)
    @printf("  Group %d: negll = %.2f\n", g, nl)
end
