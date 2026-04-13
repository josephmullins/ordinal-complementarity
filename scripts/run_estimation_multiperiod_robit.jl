using LinearAlgebra, Random, Statistics, JLD2

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "prepare_data.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation.jl"))          # for skill_index, investment_index, get_score
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod.jl"))
include(joinpath(@__DIR__, "..", "src", "inference.jl"))

# ─────────────────────────────────────────────────────────────────
# Parametric configuration: robit NN transition
# ─────────────────────────────────────────────────────────────────

const PARAMETRIC_CONFIG = (
    emission_θ  = :ordered_probit,
    emission_I  = :ordered_probit,
    transition  = :robit_nn,
    K           = 3,
    ν           = 7.0,
    ccp = :ordered_probit_simple
)

const TRANSITION_GROUPS = [[1, 2, 3], [4, 5, 6]]

const PANEL_AGE_RANGES = [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13)]

const PANEL_INVEST_ITEMS = [
    # Period 1: ages 0-1
    [:HowOftenRead, :HowManyBooks, :HowManyToysA, :HowManyToysB,
     :OutOfHouse, :HowOftenTalks, :HowOftenGrocery],
    # Period 2: ages 2-3
    [:HowOftenRead, :HowManyBooks, :HowManyToysA, :HowManyToysB,
     :OutOfHouse, :HowOftenTalks, :HowOftenGrocery,
     :HowManyZines, :HowOftenOuting, :FoodChoice,
     :LearnNumbers, :LearnAlphabet, :LearnColors, :LearnShapes, :RecordTapeCD],
    # Period 3: ages 4-5
    [:HowOftenRead, :HowManyBooks, :HowManyZines, :HowOftenOuting,
     :HowOftenMuseum,
     :RecordTapeCD, :FoodChoice],
    # Period 4: ages 6-7
    [:HowOftenRead, :HowManyBooks, :HowOftenMuseum, :HowOftenTheater,
     :HowOftenFamily, :HowOftenOutdoors, :ChildMusic, :SpecialActivities,
     :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
     :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy],
    # Period 5: ages 8-9
    [:HowOftenRead, :HowManyBooks, :HowOftenMuseum, :HowOftenTheater,
     :HowOftenFamily, :HowOftenOutdoors, :HowOftenFather, :ChildMusic,
     :SpecialActivities, :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
     :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy],
    # Period 6: ages 10-11
    [:HowManyBooks, :HowOftenMuseum, :HowOftenTheater, :HowOftenFamily,
     :HowOftenOutdoors, :HowOftenFather, :ChildMusic, :SpecialActivities,
     :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
     :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy,
     :ToldPositive, :HowOftenClean, :HowOftenChores, :HowOftenTime],
    # Period 7: ages 12-13
    [:HowManyBooks, :HowOftenMuseum, :HowOftenTheater, :HowOftenFamily,
     :HowOftenOutdoors, :HowOftenFather, :ChildMusic, :SpecialActivities,
     :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
     :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy,
     :ToldPositive, :HowOftenClean, :HowOftenChores, :HowOftenTime],
]

# ─────────────────────────────────────────────────────────────────
# Load grouped ordered probit results as warm start
# ─────────────────────────────────────────────────────────────────

println("Loading warm-start from grouped ordered probit estimation...")
warmstart_path = joinpath(dirname(@__DIR__), "output", "estimation_multiperiod_grouped.jld2")
if !isfile(warmstart_path)
    error("Warm-start not found: $warmstart_path\nRun scripts/run_estimation_multiperiod_grouped.jl first.")
end
ws = load(warmstart_path)
ws_params = ws["best_params"]
ws_specs = ws["best_specs"]

# ─────────────────────────────────────────────────────────────────
# Load datasets and prepare panel data
# ─────────────────────────────────────────────────────────────────

println("Loading datasets...")
datasets = load_all_datasets()

println("\n" * "="^60)
println("  MULTI-PERIOD ROBIT NN ESTIMATION (WARM START)")
println("  K=$(PARAMETRIC_CONFIG.K) hidden units, ν=$(PARAMETRIC_CONFIG.ν), grouped transitions")
println("="^60)

println("\n  Transition groups:")
for (g, group) in enumerate(TRANSITION_GROUPS)
    labels = [string(PANEL_AGE_RANGES[t], "→", PANEL_AGE_RANGES[t+1]) for t in group]
    println("    Group $g: $(join(labels, ", "))")
end

println("\n  Data Preparation:")
data = prepare_panel_data(datasets;
    age_ranges=PANEL_AGE_RANGES,
    invest_items=PANEL_INVEST_ITEMS,
    r_θ=4,
    r_I=5,
)

println("  N = $(data.N) children, T = $(data.T) periods")
for t in 1:data.T
    n_skill = length(data.X_θ[t])
    n_inv = length(data.X_I[t])
    println("    Period $t (ages $(PANEL_AGE_RANGES[t])): $n_skill skill measures, $n_inv investment measures")
end

if data.N < 50
    error("Only $(data.N) observations — too few for estimation.")
end

(; r_θ, r_I, T, N) = data
K_nn = PARAMETRIC_CONFIG.K
ν_robit = PARAMETRIC_CONFIG.ν

# ─────────────────────────────────────────────────────────────────
# Build warm-start specs: keep emissions + CCP, fit robit NN
# transitions to the ordered probit posteriors
# ─────────────────────────────────────────────────────────────────

println("\n  Building warm-start parameters...")

# Keep emission and CCP specs from ordered probit fit
specs = Dict{Symbol, Any}()
specs[:probit_θ] = ws_specs[:probit_θ]
specs[:probit_I] = ws_specs[:probit_I]
if haskey(ws_specs, :ccp)
    specs[:ccp] = ws_specs[:ccp]
end

# Run E-step with ordered probit params to get fresh posteriors
γ = zeros(T, r_θ, r_I, N)
ξ_sum = [zeros(r_θ, r_I, r_θ, r_I) for _ in 1:T-1]
ll_warmstart = expectation_step_mp!(γ, ξ_sum, ws_params, data)
println("  Warm-start log-likelihood (ordered probit): $(round(ll_warmstart; digits=2))")

# Initialize robit NN specs and fit to ordered probit posteriors
println("  Fitting robit NN transitions to ordered probit posteriors...")
n_restarts = 10
best_init_specs = nothing
best_init_Π = nothing
best_init_obj = Inf

for i in 1:n_restarts
    rng = MersenneTwister(100 + i)
    init_trans = [initialize_robit_nn(r_θ, K_nn; rng, r_I, ν=ν_robit) for _ in 1:T-1]

    # Run M-step for robit NN transitions given ordered probit posteriors
    try
        new_trans, new_Π = parametric_mstep_transition_mp_grouped(
            init_trans, ξ_sum, r_θ, r_I, TRANSITION_GROUPS)

        # Evaluate: total negative log-likelihood of transitions given posteriors
        obj = 0.0
        for (g, group) in enumerate(TRANSITION_GROUPS)
            Q_sum_pooled = _qsum_from_xi(ξ_sum[group[1]])
            for t in group[2:end]
                Q_sum_pooled = Q_sum_pooled .+ _qsum_from_xi(ξ_sum[t])
            end
            φ = pack_robit_nn(new_trans[group[1]])
            obj += transition_negll_robit_nn_fast(φ, Q_sum_pooled, r_θ, r_I, K_nn, ν_robit)
        end

        println("    Init $i/$n_restarts: transition negll = $(round(obj; digits=2))")
        if obj < best_init_obj
            global best_init_obj = obj
            global best_init_specs = new_trans
            global best_init_Π = new_Π
        end
    catch e
        println("    Init $i/$n_restarts: failed ($(typeof(e)))")
    end
end

if best_init_specs === nothing
    error("All robit NN initializations failed.")
end

specs[:transition] = best_init_specs

# Assemble warm-start params
params_init = (;
    Π = best_init_Π,
    CCP = ws_params.CCP,
    π0 = ws_params.π0,
    P_θ = ws_params.P_θ,
    P_I = ws_params.P_I,
    r_θ, r_I, T
)

# Verify warm-start ll
ll_init = expectation_step_mp!(γ, ξ_sum, params_init, data)
println("\n  Warm-start log-likelihood (robit NN init): $(round(ll_init; digits=2))")

# ─────────────────────────────────────────────────────────────────
# EM estimation from warm start (no funnel, just converge)
# ─────────────────────────────────────────────────────────────────

println("\n  Parametric EM Estimation (from warm start):")
println("    emission_θ = $(PARAMETRIC_CONFIG.emission_θ)")
println("    emission_I = $(PARAMETRIC_CONFIG.emission_I)")
println("    transition = $(PARAMETRIC_CONFIG.transition), K = $K_nn, ν = $ν_robit")
println("    transition_groups = $TRANSITION_GROUPS")

best_params, best_γ, best_ξ_sum, best_specs = expectation_maximization_mp_parametric(
    data, params_init, specs, PARAMETRIC_CONFIG;
    maxiter=2000, tol=1e-5, verbose=true,
    transition_groups=TRANSITION_GROUPS)

best_ll = expectation_step_mp!(best_γ, best_ξ_sum, best_params, data)
println("\n  Final log-likelihood: $(round(best_ll; digits=2))")

# Skip normalization (parametric emissions enforce monotonicity)
println("\n  Skipping normalization (parametric emissions enforce monotonicity)")

# Display results
display_results_mp(best_params, data)

# Display robit NN parameters
if best_specs !== nothing && haskey(best_specs, :transition)
    println("\n" * "="^60)
    println("  ROBIT NN TRANSITION PARAMETERS")
    println("="^60)

    for (t, spec) in enumerate(best_specs[:transition])
        if spec isa RobitNNTransition
            age_from = PANEL_AGE_RANGES[t]
            age_to = PANEL_AGE_RANGES[t + 1]
            g = findfirst(grp -> t ∈ grp, TRANSITION_GROUPS)
            shared = length(TRANSITION_GROUPS[g]) > 1 ? " (shared with group $g)" : ""
            println("\n  Transition $t: ages $age_from → $age_to$shared")

            c = _unpack_survivor_cutoffs(spec.c_raw)
            alpha2 = exp.(spec.alpha2_raw)
            w = exp.(spec.log_w)
            println("    Cutpoints c = $(round.(c; digits=4))")
            println("    ν = $(spec.ν)")
            println("    Hidden units (K=$(spec.K)):")
            for h in 1:spec.K
                println("      h=$h: α₁=$(round(spec.alpha1[h]; digits=4)), " *
                        "α₂=$(round(alpha2[h]; digits=4)), " *
                        "b=$(round(spec.b[h]; digits=4)), " *
                        "w=$(round(w[h]; digits=4))")
            end
        end
    end

    if haskey(best_specs, :probit_θ)
        println("\n  Skill emission loadings (α) by period:")
        for t in 1:data.T
            alphas = round.(best_specs[:probit_θ][t].alpha; digits=4)
            println("    Period $t: α = $alphas")
        end
    end

    if haskey(best_specs, :probit_I)
        println("\n  Investment emission loadings (α) by period:")
        for t in 1:data.T
            alphas = round.(best_specs[:probit_I][t].alpha; digits=4)
            println("    Period $t: α = $alphas")
        end
    end
end

# ─────────────────────────────────────────────────────────────────
# Per-group complementarity tests
# ─────────────────────────────────────────────────────────────────

println("\n" * "="^60)
println("  GROUPED COMPLEMENTARITY TESTS")
println("="^60)

println("\n  Computing per-observation pairwise posteriors...")
ψ_all = compute_psi_all(best_params, data)

println("  Running grouped tests...")
test_results = get_test_stat_mp_grouped(ψ_all, best_params, data, TRANSITION_GROUPS; α=0.05)

for (g, res) in enumerate(test_results)
    labels = [string(PANEL_AGE_RANGES[t], "→", PANEL_AGE_RANGES[t+1]) for t in res.group]
    status = res.reject ? "REJECT" : "FAIL TO REJECT"
    println("\n  Group $g: $(join(labels, ", "))")
    println("    T_n = $(round(res.T_n; digits=4)),  c_α = $(round(res.c_α; digits=4))")
    println("    Result: $status ordinal complementarity (H₀)")
end

# ─────────────────────────────────────────────────────────────────
# Parametric Var(Π̂_free) per group (for survivor SEs)
# ─────────────────────────────────────────────────────────────────

println("\n  Computing parametric Var(Π̂_free) per group...")
Var_free_list = Vector{Matrix{Float64}}(undef, T - 1)
for (g, group) in enumerate(TRANSITION_GROUPS)
    t_rep = group[1]
    spec_g = best_specs[:transition][t_rep]::RobitNNTransition

    φ = pack_robit_nn(spec_g)
    K = spec_g.K
    ν = spec_g.ν
    function _pi_free_vec(φ_vec)
        Π_full = _build_robit_nn_pi(φ_vec, r_θ, r_I, K, ν)
        v = similar(φ_vec, (r_θ - 1) * r_θ * r_I)
        idx = 0
        for kI in 1:r_I, kθ in 1:r_θ, j in 1:(r_θ - 1)
            idx += 1
            v[idx] = Π_full[j, kθ, kI]
        end
        return v
    end
    J = ForwardDiff.jacobian(_pi_free_vec, φ)

    Σ_φ = J' * test_results[g].Σ_score * J
    Var_φ = pinv(Σ_φ) / N
    Var_free_g = J * Var_φ * J'

    for t in group
        Var_free_list[t] = Var_free_g
    end
end

# ─────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────

outdir = joinpath(dirname(@__DIR__), "output")
mkpath(outdir)
outpath = joinpath(outdir, "estimation_multiperiod_robit_grouped.jld2")
jldsave(outpath; best_params, best_gamma=best_γ, data, best_ll,
        test_results, best_specs, parametric_config=PARAMETRIC_CONFIG,
        transition_groups=TRANSITION_GROUPS, Var_free=Var_free_list)
println("\n  Results saved to $outpath")

println("\nDone.")
