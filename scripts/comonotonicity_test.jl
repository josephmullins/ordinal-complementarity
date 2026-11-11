# =====================================================================
# Test for existence of a nondecreasing return function Q under which
# the optimal (fixed-marginal) allocation of investment to skill is
# comonotone — and the opposite test for counter-monotone allocations.
#
# Implements comonotonicity_test_instructions.md on the estimated
# regime-marginalized technology Π̄ (V3+ε_Y, K=3, r_I=4 settled spec),
# with a parametric bootstrap from the asymptotic distribution of Π̄.
#
#   Comonotone achievable  ⟺  ∃ α ≥ 0, α≠0, with Σ_j α_j Δ_j ≥ 0 entrywise
#       → LP:  t* = max t  s.t.  Dα ≥ t·1, 1'α = 1, α ≥ 0      (t* ≥ 0 ?)
#   Counter-monotone achievable ⟺ ∃ α ≥ 0, α≠0, with Σ_j α_j Δ_j ≤ 0
#       → LP:  t* = min t  s.t.  Dα ≤ t·1, 1'α = 1, α ≥ 0      (t* ≤ 0 ?)
#
# Δ_j = supermodularity gap (in (θ,I)) of the survival function
#       V_j(θ,I) = Pr(θ' ≥ s_j | θ, I), for thresholds j = 2..K.
# =====================================================================

using JLD2, LinearAlgebra, Statistics, Random, Printf, ForwardDiff
using JuMP, HiGHS
using Plots
gr(size = (800, 500))

const SRC = joinpath(@__DIR__, "..", "src")

# model code (mirrors notebooks/analyze_results_v3_eps_k3_rI4); loaded at
# top level so methods are visible from run_real (avoids world-age errors)
for f in ("parametric_models.jl", "parametric_models_v3.jl",
          "parametric_models_v3_eps.jl", "prepare_data.jl",
          "estimation.jl", "estimation_multiperiod.jl",
          "estimation_multiperiod_v3.jl", "estimation_multiperiod_v3_eps.jl",
          "inference.jl", "inference_common.jl", "inference_v3_eps.jl")
    include(joinpath(SRC, f))
end

# ---------------------------------------------------------------------
# 1. Core building blocks (Π̄-only; self-contained)
# ---------------------------------------------------------------------

"""
    gap_matrices(Π) -> (Δ, D)

Build the threshold supermodularity-gap matrices from a transition array
`Π[θ', θ, I]` (first index = next-period skill s, so `size(Π,1)=K`).

Returns:
- `Δ` : array `(M-1, N-1, K-1)` of adjacent second cross-differences.
        `Δ[p,q,jj]` is the gap at lattice cell (θ_p..θ_{p+1}) × (I_q..I_{q+1})
        for threshold j = jj+1 (jj = 1..K-1).
- `D` : `L × (K-1)` matrix, `L = (M-1)(N-1)`, each column `d_j = vec(Δ[:,:,jj])`.
"""
function gap_matrices(Π::AbstractArray{<:Real,3})
    K, M, N = size(Π)                      # θ' count, θ count, I count
    # Survival V[θ,I,jj] = Pr(θ' ≥ jj+1 | θ,I) = Σ_{s ≥ jj+1} Π[s,θ,I],  jj=1..K-1
    V = zeros(eltype(Π), M, N, K - 1)
    for jj in 1:(K - 1), q in 1:N, p in 1:M
        V[p, q, jj] = sum(Π[s, p, q] for s in (jj + 1):K)
    end
    Δ = zeros(eltype(Π), M - 1, N - 1, K - 1)
    for jj in 1:(K - 1), q in 1:(N - 1), p in 1:(M - 1)
        Δ[p, q, jj] = V[p + 1, q + 1, jj] + V[p, q, jj] -
                      V[p + 1, q, jj] - V[p, q + 1, jj]
    end
    L = (M - 1) * (N - 1)
    D = zeros(eltype(Π), L, K - 1)
    for jj in 1:(K - 1)
        D[:, jj] = vec(Δ[:, :, jj])
    end
    return Δ, D
end

# cell label for row l of D (1-based (p,q) of the 2×2 lattice block)
cell_label(l, Mm1, Nm1) = (((l - 1) % Mm1) + 1, ((l - 1) ÷ Mm1) + 1)

"""
    solve_shape_lp(D; sense, shape) -> (t, α, duals, binding)

Solve the comonotone / counter-monotone existence LP.

- `sense = :max`  → t* = max t s.t. Dα ≥ t·1   (comonotone; t*≥0 means achievable)
- `sense = :min`  → t* = min t s.t. Dα ≤ t·1   (counter-monotone; t*≤0 means achievable)
- `shape ∈ (:none, :convex, :concave)` adds monotone increments on α
  (convex Q: α_j ≥ α_{j-1};  concave Q: α_j ≤ α_{j-1}).

Returns t*, the optimal increments α*, the constraint duals (on the
inequality rows), and the indices of the binding lattice cells.
"""
function solve_shape_lp(D::AbstractMatrix; sense::Symbol = :max, shape::Symbol = :none)
    L, m = size(D)
    model = Model(HiGHS.Optimizer); set_silent(model)
    @variable(model, α[1:m] >= 0)
    @variable(model, t)
    @constraint(model, sum(α) == 1)
    if shape == :convex
        @constraint(model, [j = 2:m], α[j] - α[j - 1] >= 0)
    elseif shape == :concave
        @constraint(model, [j = 2:m], α[j] - α[j - 1] <= 0)
    elseif shape != :none
        error("shape must be :none, :convex, or :concave")
    end
    if sense == :max
        @constraint(model, ineq[l = 1:L], sum(D[l, j] * α[j] for j in 1:m) - t >= 0)
        @objective(model, Max, t)
    elseif sense == :min
        @constraint(model, ineq[l = 1:L], sum(D[l, j] * α[j] for j in 1:m) - t <= 0)
        @objective(model, Min, t)
    else
        error("sense must be :max or :min")
    end
    optimize!(model)
    is_solved_and_feasible(model) || error("LP failed: $(termination_status(model))")
    tval = value(t); αval = value.(α)
    du = [dual(ineq[l]) for l in 1:L]
    Dα = D * αval
    binding = findall(l -> abs(Dα[l] - tval) < 1e-7, 1:L)
    return tval, αval, du, binding
end

"""
    cumulative_dominance(Δ) -> (super_convex, sub_concave, T_pos, L_neg)

Analytic sufficient conditions (§1.6), with upper/lower partial sums.
- `super_convex` : all upper partial sums T_j ≥ 0  ⇒ V_Q supermodular for *every* convex Q
- `sub_concave`  : all lower partial sums L_j ≤ 0  ⇒ V_Q submodular  for *every* concave Q
"""
function cumulative_dominance(Δ::AbstractArray{<:Real,3})
    Km1 = size(Δ, 3)
    # T_j = Σ_{k=j}^{K} Δ_k  (here columns jj index threshold; upper = suffix sums)
    T_pos = all(all(sum(Δ[:, :, jj:Km1], dims = 3) .>= -1e-12) for jj in 1:Km1)
    # L_j = Σ_{k=2}^{j} Δ_k  (prefix sums)
    L_neg = all(all(sum(Δ[:, :, 1:jj], dims = 3) .<= 1e-12) for jj in 1:Km1)
    return T_pos, L_neg
end

# Q increments α* → cumulative return function Q (anchored Q(s_1)=0)
function Q_from_alpha(α)
    Q = zeros(length(α) + 1)
    for j in 2:length(Q)
        Q[j] = Q[j - 1] + α[j - 1]
    end
    return Q
end

# ---------------------------------------------------------------------
# 2. Parametric bootstrap utilities
# ---------------------------------------------------------------------

# matrix square root via symmetric eigendecomposition (clamps tiny negatives)
function psd_sqrt(Σ)
    Σs = Symmetric((Σ + Σ') / 2)
    vals, vecs = eigen(Σs)
    vals = max.(vals, 0.0)
    return vecs * Diagonal(sqrt.(vals)) * vecs'
end

"""
    reconstruct_Pi(free_vec, r_θ, r_I) -> (Π, fixed)

Rebuild a full `(r_θ, r_θ, r_I)` transition array from the free entries
(first `r_θ-1` rows of each (θ,I) column), ordering
`for kI, for kθ, for j=1:r_θ-1`.  Last row = 1 − Σ.  Columns with any
negative mass are clamped to ≥0 and renormalized; `fixed` counts them.
"""
function reconstruct_Pi(free_vec, r_θ, r_I)
    Π = zeros(r_θ, r_θ, r_I)
    fixed = 0
    idx = 0
    for kI in 1:r_I, kθ in 1:r_θ
        s = 0.0
        for j in 1:(r_θ - 1)
            idx += 1
            Π[j, kθ, kI] = free_vec[idx]
            s += free_vec[idx]
        end
        Π[r_θ, kθ, kI] = 1.0 - s
        col = @view Π[:, kθ, kI]
        if any(col .< 0) || any(col .> 1)
            col .= max.(col, 0.0)
            col ./= sum(col)
            fixed += 1
        end
    end
    return Π, fixed
end

# free-entry vector from a full Π (matches reconstruct_Pi ordering)
function Pi_to_free(Π, r_θ, r_I)
    v = Float64[]
    for kI in 1:r_I, kθ in 1:r_θ, j in 1:(r_θ - 1)
        push!(v, Π[j, kθ, kI])
    end
    return v
end

"""
    bootstrap_tstar(Π̂, Var_free, r_θ, r_I; B, shapes, rng)

Parametric bootstrap of t* for both directions and each shape.
Draws Π̄^(b) from N(vec(Π̄_free), Var_free), reconstructs, recomputes t*.
Returns a Dict keyed by (direction, shape) → vector of B bootstrap t*'s,
plus the count of draws needing clamping.
"""
function bootstrap_tstar(Π̂, Var_free, r_θ, r_I;
                         B = 2000,
                         shapes = (:none, :convex, :concave),
                         rng = Random.default_rng())
    μ = Pi_to_free(Π̂, r_θ, r_I)
    A = psd_sqrt(Var_free)
    keys_ds = [(d, s) for d in (:max, :min) for s in shapes]
    boot = Dict(k => Float64[] for k in keys_ds)
    nfixed_draws = 0
    for _ in 1:B
        z = randn(rng, length(μ))
        draw = μ .+ A * z
        Π_b, fixed = reconstruct_Pi(draw, r_θ, r_I)
        fixed > 0 && (nfixed_draws += 1)
        _, D_b = gap_matrices(Π_b)
        for (d, s) in keys_ds
            tb, = solve_shape_lp(D_b; sense = d, shape = s)
            push!(boot[(d, s)], tb)
        end
    end
    return boot, nfixed_draws
end

# DATA bootstrap: recompute t* on each data-bootstrap Π̄ draw (free vec rows,
# same ordering as reconstruct_Pi).  Returns Dict (dir,shape) -> vector.
function databoot_tstar(boot_free_mat, r_θ, r_I; shapes = (:none, :convex, :concave))
    keys_ds = [(d, s) for d in (:max, :min) for s in shapes]
    boot = Dict(k => Float64[] for k in keys_ds)
    for b in 1:size(boot_free_mat, 1)
        Π_b, _ = reconstruct_Pi(boot_free_mat[b, :], r_θ, r_I)
        _, D_b = gap_matrices(Π_b)
        for (d, s) in keys_ds
            tb, = solve_shape_lp(D_b; sense = d, shape = s)
            push!(boot[(d, s)], tb)
        end
    end
    return boot
end

# summarize a bootstrap vector against the point estimate t̂
function summarize(t̂, tvec; sense::Symbol)
    sd = std(tvec)
    lo, hi = quantile(tvec, 0.025), quantile(tvec, 0.975)
    # achievability probability in the relevant direction
    p_achieve = sense == :max ? mean(tvec .>= 0) : mean(tvec .<= 0)
    return (; t̂, sd, ci = (lo, hi), p_achieve, boundary = t̂ / sd)
end

function report_block(title, t̂, tvec, α̂; sense, achieve_word)
    s = summarize(t̂, tvec; sense)
    @printf("  %-9s  t̂* = %+8.5f   sd = %7.5f   95%% CI [%+8.5f, %+8.5f]\n",
            title, s.t̂, s.sd, s.ci[1], s.ci[2])
    @printf("             P(%s achievable) = %.3f   boundary t̂*/sd = %+6.2f\n",
            achieve_word, s.p_achieve, s.boundary)
    if (sense == :max && t̂ >= 0) || (sense == :min && t̂ <= 0)
        Q = Q_from_alpha(α̂)
        @printf("             α* = %s\n", string(round.(α̂; digits = 4)))
        @printf("             implied Q(s) = %s\n", string(round.(Q; digits = 4)))
    end
    return s
end

# ---------------------------------------------------------------------
# 3. Synthetic validation (§3 of the instructions)
# ---------------------------------------------------------------------

"""
Build a synthetic Π with EXACT constant per-threshold gaps Δ_jj = γ[jj].
Uses survival V_j(θ,I) = base[jj] + γ[jj]·(θ-1)(I-1); the product term is
supermodular with unit second cross-difference, so Δ[p,q,jj] ≡ γ[jj].
"""
function synthetic_Pi(γ; r_θ = 4, r_I = 4, base = [0.7, 0.5, 0.3])
    @assert length(γ) == r_θ - 1 && length(base) == r_θ - 1
    V = zeros(r_θ, r_θ, r_I)             # V[jj, θ, I], jj=1..r_θ-1
    for kI in 1:r_I, kθ in 1:r_θ, jj in 1:(r_θ - 1)
        V[jj, kθ, kI] = base[jj] + γ[jj] * (kθ - 1) * (kI - 1)
    end
    Π = zeros(r_θ, r_θ, r_I)
    for kI in 1:r_I, kθ in 1:r_θ
        prev = 1.0
        for s in 1:(r_θ - 1)
            Π[s, kθ, kI] = prev - V[s, kθ, kI]
            prev = V[s, kθ, kI]
        end
        Π[r_θ, kθ, kI] = prev
        @assert all(Π[:, kθ, kI] .>= -1e-9) "invalid synthetic Π at ($kθ,$kI): $(Π[:,kθ,kI])"
    end
    return Π
end

function run_validation()
    println("="^72)
    println("SYNTHETIC VALIDATION")
    println("="^72)
    cases = [
        ("all Δ_j ≥ 0   (full complementarity)",  [0.01, 0.01, 0.01]),
        ("all Δ_j ≤ 0   (full substitutability)", [-0.01, -0.01, -0.01]),
        ("single-crossing (sub low → super high)", [-0.02, -0.01, 0.01]),
    ]
    for (name, γ) in cases
        Π = synthetic_Pi(γ)
        Δ, D = gap_matrices(Π)
        Tpos, Lneg = cumulative_dominance(Δ)
        println("\n• $name   (γ = $γ)")
        @printf("    gaps Δ_j (per threshold, constant across cells): %s\n",
                string(round.([Δ[1,1,jj] for jj in 1:size(Δ,3)]; digits = 4)))
        for s in (:none, :convex, :concave)
            tmax, = solve_shape_lp(D; sense = :max, shape = s)
            tmin, = solve_shape_lp(D; sense = :min, shape = s)
            @printf("    shape=%-8s  comonotone t*(max) = %+8.5f   counter t*(min) = %+8.5f\n",
                    s, tmax, tmin)
        end
        @printf("    cumulative-dominance flags: convex⇒super = %s   concave⇒sub = %s\n",
                Tpos, Lneg)
    end
    println("\nExpected: (1) all-positive → t*(max)>0 every shape;")
    println("          (2) all-negative → t*(max)<0 unrestricted, concave⇒sub flag = true;")
    println("          (3) single-crossing → t*(max)≥0 convex but <0 concave.")
    println()
end

# ---------------------------------------------------------------------
# 4. Real data: load fit, compute Var(Π̄), run both tests
# ---------------------------------------------------------------------

function run_real(; B = 2000, seed = 20260604,
                  fitfile = get(ENV, "COMONO_FIT", "estimation_v3_eps_reg.jld2"))
    println("="^72)
    println("ESTIMATED TECHNOLOGY — paper/slides benchmark spec ($fitfile)")
    println("="^72)

    fitpath = get(ENV, "FIT_OVERRIDE", joinpath(@__DIR__, "..", "output", fitfile))
    println("Loading fit: ", fitpath)
    fit = load(fitpath)
    pv3 = fit["params_v3_eps"]
    psk = fit["params_skill"]
    dims = fit["data_dims"]
    age_ranges = fit["age_ranges"]
    (; r_θ, r_I, r_P, r_ω, K, N, T) = dims
    println("Loaded: N=$N, T=$T, r_θ=$r_θ, r_I=$r_I, r_P=$r_P, r_ω=$r_ω, K=$K")

    TRANSITION_GROUPS = haskey(fit, "transition_groups") ? fit["transition_groups"] :
                        [[1, 2, 3], [4, 5, 6]]
    GROUP_LABELS = ["Early (ages 0-5)", "School (ages 6-13)"]

    # invest items per period (same as the analysis notebook)
    PANEL_INVEST_ITEMS = [
        [:HowOftenRead, :HowManyBooks, :HowManyToysA, :HowManyToysB,
         :OutOfHouse, :HowOftenTalks, :HowOftenGrocery],
        [:HowOftenRead, :HowManyBooks, :HowManyToysA, :HowManyToysB,
         :OutOfHouse, :HowOftenTalks, :HowOftenGrocery,
         :HowManyZines, :HowOftenOuting, :FoodChoice,
         :LearnNumbers, :LearnAlphabet, :LearnColors, :LearnShapes, :RecordTapeCD],
        [:HowOftenRead, :HowManyBooks, :HowManyZines, :HowOftenOuting,
         :HowOftenMuseum, :RecordTapeCD, :FoodChoice],
        [:HowOftenRead, :HowManyBooks, :HowOftenMuseum, :HowOftenTheater,
         :HowOftenFamily, :HowOftenOutdoors, :ChildMusic, :SpecialActivities,
         :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
         :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy],
        [:HowOftenRead, :HowManyBooks, :HowOftenMuseum, :HowOftenTheater,
         :HowOftenFamily, :HowOftenOutdoors, :HowOftenFather, :ChildMusic,
         :SpecialActivities, :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
         :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy],
        [:HowManyBooks, :HowOftenMuseum, :HowOftenTheater, :HowOftenFamily,
         :HowOftenOutdoors, :HowOftenFather, :ChildMusic, :SpecialActivities,
         :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
         :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy,
         :ToldPositive, :HowOftenClean, :HowOftenChores, :HowOftenTime],
        [:HowManyBooks, :HowOftenMuseum, :HowOftenTheater, :HowOftenFamily,
         :HowOftenOutdoors, :HowOftenFather, :ChildMusic, :SpecialActivities,
         :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
         :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy,
         :ToldPositive, :HowOftenClean, :HowOftenChores, :HowOftenTime],
    ]

    println("Loading raw data + computing Var(φ̂) via Fisher's identity...")
    datasets = load_all_datasets()
    panel = prepare_panel_data(datasets; age_ranges = age_ranges,
                               invest_items = PANEL_INVEST_ITEMS, r_θ = r_θ, r_I = r_I)
    data = merge(panel, (; r_P = r_P, r_ω = r_ω))
    inf = compute_var_phi_v3_eps(pv3, psk, data;
                                 transition_groups = TRANSITION_GROUPS, verbose = false)
    Var_free_list, Pi_bar = delta_cov_Pi_bar_free(inf, pv3, psk;
                                                  transition_groups = TRANSITION_GROUPS)

    rng = MersenneTwister(seed)
    summary_rows = String[]
    csv_rows = ["group,ages,direction,shape,t_hat,ci_lo,ci_hi,boot_sd,p_achieve,boundary,convex_super_flag,concave_sub_flag"]
    heat_plots = Any[]

    for (g, group) in enumerate(TRANSITION_GROUPS)
        t = group[1]
        Π̂ = Pi_bar[t]
        Var_free = Var_free_list[t]
        Δ, D = gap_matrices(Π̂)
        Mm1, Nm1, Km1 = size(Δ)

        println("\n", "-"^72)
        println("GROUP $g — $(GROUP_LABELS[g])  [periods $(group)]")
        println("-"^72)
        nneg = count(<(0), Δ)
        @printf("Survival-gap summary: %d / %d cells have Δ_j < 0 (substitutability); ",
                nneg, length(Δ))
        @printf("range [%+.4f, %+.4f]\n", minimum(Δ), maximum(Δ))
        Tpos, Lneg = cumulative_dominance(Δ)
        @printf("Cumulative-dominance flags: convex⇒super = %s,  concave⇒sub = %s\n", Tpos, Lneg)
        ages = "$(age_ranges[group[1]][1])-$(age_ranges[group[end]][2])"

        println("\nParametric bootstrap (B=$B draws of Π̄ from N(Π̄̂, Var_free))...")
        boot, nfixed = bootstrap_tstar(Π̂, Var_free, r_θ, r_I; B = B, rng = rng)
        nfixed > 0 && @printf("  (%d/%d draws needed clamping to the simplex)\n", nfixed, B)

        # ----- COMONOTONE test: max t -----
        println("\n[COMONOTONE]  ∃ nondecreasing Q ⇒ comonotone optimal?  (want t* ≥ 0)")
        for s in (:none, :convex, :concave)
            t̂, α̂, = solve_shape_lp(D; sense = :max, shape = s)
            sm = report_block("$s", t̂, boot[(:max, s)], α̂; sense = :max,
                              achieve_word = "comonotone")
            push!(summary_rows,
                  @sprintf("G%d %-8s comonotone  t̂*=%+.4f CI[%+.4f,%+.4f] P(≥0)=%.3f",
                           g, s, sm.t̂, sm.ci[1], sm.ci[2], sm.p_achieve))
            push!(csv_rows, @sprintf("%d,%s,comonotone,%s,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%s,%s",
                  g, ages, s, sm.t̂, sm.ci[1], sm.ci[2], sm.sd, sm.p_achieve, sm.boundary, Tpos, Lneg))
        end

        # ----- COUNTER-MONOTONE test: min t -----
        println("\n[COUNTER-MONOTONE]  ∃ nondecreasing Q ⇒ counter-monotone optimal?  (want t* ≤ 0)")
        for s in (:none, :convex, :concave)
            t̂, α̂, _, binding = solve_shape_lp(D; sense = :min, shape = s)
            sm = report_block("$s", t̂, boot[(:min, s)], α̂; sense = :min,
                              achieve_word = "counter-monotone")
            push!(summary_rows,
                  @sprintf("G%d %-8s counter     t̂*=%+.4f CI[%+.4f,%+.4f] P(≤0)=%.3f",
                           g, s, sm.t̂, sm.ci[1], sm.ci[2], sm.p_achieve))
            push!(csv_rows, @sprintf("%d,%s,counter,%s,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%s,%s",
                  g, ages, s, sm.t̂, sm.ci[1], sm.ci[2], sm.sd, sm.p_achieve, sm.boundary, Tpos, Lneg))
            if s == :none && t̂ < 0
                println("    Obstructing lattice cells (binding Dα*=t*):")
                for l in binding
                    (p, q) = cell_label(l, Mm1, Nm1)
                    @printf("      θ∈{%d,%d} × I∈{%d,%d}\n", p, p + 1, q, q + 1)
                end
            end
        end

        # heatmap of Σ α*_j Δ_j for the unrestricted comonotone solution
        _, α_un, = solve_shape_lp(D; sense = :max, shape = :none)
        ΣαΔ = sum(α_un[jj] .* Δ[:, :, jj] for jj in 1:Km1)
        hp = heatmap(1:Nm1, 1:Mm1, ΣαΔ; c = :balance, clims = (-maximum(abs.(ΣαΔ)), maximum(abs.(ΣαΔ))),
                     title = "G$g $(GROUP_LABELS[g]):  Σ αⱼ* Δⱼ",
                     xlabel = "I cell (q)", ylabel = "θ cell (p)", aspect_ratio = :equal)
        push!(heat_plots, hp)
    end

    println("\n", "="^72)
    println("SUMMARY")
    println("="^72)
    foreach(println, summary_rows)

    reg_run = get(ENV, "REG_RUN", "0") == "1"
    outdir = joinpath(@__DIR__, "..", "output", reg_run ? "counterfactuals_reg" : "counterfactuals")
    isdir(outdir) || mkpath(outdir)
    figpath = joinpath(outdir, "comonotonicity_test_heatmaps.png")
    plot(heat_plots...; layout = (1, length(heat_plots)), size = (520 * length(heat_plots), 460))
    savefig(figpath)
    println("\nHeatmap of Σ αⱼ* Δⱼ saved to: $figpath")

    tabdir = joinpath(@__DIR__, "..", "output", reg_run ? "paper_tables_reg" : "paper_tables")
    isdir(tabdir) || mkpath(tabdir)
    csvpath = joinpath(tabdir, "comonotonicity_test.csv")
    open(csvpath, "w") do io
        for r in csv_rows; println(io, r); end
    end
    println("Source-of-record CSV saved to: $csvpath")
end

# ---------------------------------------------------------------------
# DATA-bootstrap version: point t* from the fit's Π̄, CI/Pr from the data
# bootstrap draws (output/bootstrap_reg.jld2).  Writes paper_tables_boot/.
function run_databoot(; bootfile = get(ENV, "COMONO_BOOT",
                          joinpath(@__DIR__, "..", "output", "bootstrap_reg.jld2")),
                      induced::Bool = get(ENV, "INDUCED", "0") == "1")
    obj = induced ? "INDUCED Π(θ'|θ,income)" : "technology Π̄(θ'|θ,I)"
    println("="^72); println("DATA-BOOTSTRAP comonotonicity test — $obj  ($bootfile)"); println("="^72)
    bc = load(bootfile)
    okrows = findall(bc["ok"]); r_θ = bc["r_θ"]; TG = bc["transition_groups"]
    # On the induced object the second lattice axis is the income grid (n_y bins);
    # reconstruct_Pi / gap_matrices / solve_shape_lp are dimension-generic, so we
    # just feed n_y in place of r_I and read the induced free vectors.
    if induced
        haskey(bc, "boot_pibar_ind") || error("$(bootfile) has no induced fields; re-run bootstrap_reg.jl.")
        boot_pibar = bc["boot_pibar_ind"]; pibar0 = bc["pibar_ind0"]; r_I = bc["n_y"]
        @printf("  income grid (\$)= %s\n", join([@sprintf("%.0f", v) for v in bc["y_levels"]], " "))
    else
        boot_pibar = bc["boot_pibar"]; pibar0 = bc["pibar0"]; r_I = bc["r_I"]
    end
    GROUP_LABELS = ["Early (ages 0-5)", "School (ages 6-13)"]; ages_list = ["0-5", "6-13"]
    csv_rows = ["group,ages,direction,shape,t_hat,ci_lo,ci_hi,boot_sd,p_achieve,boundary,convex_super_flag,concave_sub_flag"]
    for (g, group) in enumerate(TG)
        Π̂, _ = reconstruct_Pi(pibar0[g], r_θ, r_I)
        Δ, D = gap_matrices(Π̂)
        Tpos, Lneg = cumulative_dominance(Δ)
        println("\n", "-"^72); println("GROUP $g — $(GROUP_LABELS[g])  [data bootstrap, n=$(length(okrows))]")
        nneg = count(<(0), Δ); @printf("  Δ<0 in %d/%d cells; cumdom convex⇒super=%s concave⇒sub=%s\n", nneg, length(Δ), Tpos, Lneg)
        boot = databoot_tstar(boot_pibar[g][okrows, :], r_θ, r_I)
        for (sense, word) in ((:max, "comonotone"), (:min, "counter-monotone"))
            println("[", uppercase(word), "]  (want t* ", sense == :max ? "≥ 0)" : "≤ 0)")
            for s in (:none, :convex, :concave)
                res = solve_shape_lp(D; sense = sense, shape = s)
                sm = report_block("$s", res[1], boot[(sense, s)], res[2]; sense = sense, achieve_word = word)
                dirlab = sense == :max ? "comonotone" : "counter"
                push!(csv_rows, @sprintf("%d,%s,%s,%s,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%s,%s",
                      g, ages_list[g], dirlab, s, sm.t̂, sm.ci[1], sm.ci[2], sm.sd, sm.p_achieve, sm.boundary, Tpos, Lneg))
            end
        end
    end
    tabdir = joinpath(@__DIR__, "..", "output", "paper_tables_boot"); isdir(tabdir) || mkpath(tabdir)
    csvpath = joinpath(tabdir, induced ? "comonotonicity_test_induced.csv" : "comonotonicity_test.csv")
    open(csvpath, "w") do io; for r in csv_rows; println(io, r); end; end
    println("\nSaved → ", csvpath)
end

# ---------------------------------------------------------------------
# NUMERICAL DELTA METHOD (Hong & Li 2018) on the same data-bootstrap draws.
# t^± is the value of a matrix game in the gap matrix D (linear in Π̄), so we
# perturb D̂ by ε√N (D*_b − D̂) and re-solve the LP.  Reports steps
# N^{-1/3} (baseline), N^{-1/4} (sensitivity), N^{-1/2} (= standard bootstrap).
# Writes paper_tables_boot/comonotonicity_test_ndm[_induced].csv.
include(joinpath(SRC, "numerical_delta.jl"))

function run_ndm(; bootfile = get(ENV, "COMONO_BOOT",
                     joinpath(@__DIR__, "..", "output", "bootstrap_reg.jld2")),
                 induced::Bool = get(ENV, "INDUCED", "0") == "1", α = 0.05,
                 N = parse(Int, get(ENV, "NDM_N", "9686")))
    obj = induced ? "INDUCED Π(θ'|θ,income)" : "technology Π̄(θ'|θ,I)"
    println("="^72); println("NUMERICAL-DELTA rationalizability test — $obj  (N=$N, $bootfile)"); println("="^72)
    bc = load(bootfile)
    okrows = findall(bc["ok"]); r_θ = bc["r_θ"]; TG = bc["transition_groups"]
    if induced
        boot_pibar = bc["boot_pibar_ind"]; pibar0 = bc["pibar_ind0"]; r_I = bc["n_y"]
    else
        boot_pibar = bc["boot_pibar"]; pibar0 = bc["pibar0"]; r_I = bc["r_I"]
    end
    gapD(v) = gap_matrices(reconstruct_Pi(v, r_θ, r_I)[1])[2]
    steps = ((:e13, 1/3), (:e14, 1/4), (:naive, 1/2))
    rows = ["group,ages,direction,shape,t_hat,rt_hat,step,eps,crit,reject,ci_lo,ci_hi,B"]
    ages_list = ["0-5", "6-13"]
    for (g, _) in enumerate(TG)
        D̂ = gapD(pibar0[g])
        Dboot = [gapD(boot_pibar[g][b, :]) for b in okrows]
        println("\nGROUP $g (ages $(ages_list[g]))  B=$(length(Dboot))")
        for sense in (:max, :min), shape in (:none, :convex, :concave)
            φ = M -> solve_shape_lp(M; sense, shape)[1]
            t̂ = φ(D̂)
            dirlab = sense === :max ? "comonotone" : "counter"
            for (lab, e) in steps
                ε = N^(-e)
                D = ndm_draws(φ, D̂, Dboot, N; ε)
                r = ndm_test(t̂, D, N; α, sense)
                @printf("  %-10s %-8s %-6s t̂=%+.5f √N t̂=%+7.3f crit=%+7.3f  %s  CI[%+.5f,%+.5f]\n",
                        dirlab, shape, lab, t̂, r.rt, r.c, r.reject ? "REJECT" : "  -   ", r.ci...)
                push!(rows, @sprintf("%d,%s,%s,%s,%.6f,%.4f,%s,%.5f,%.4f,%s,%.6f,%.6f,%d",
                      g, ages_list[g], dirlab, shape, t̂, r.rt, lab, ε, r.c, r.reject, r.ci..., length(D)))
            end
        end
    end
    tabdir = joinpath(@__DIR__, "..", "output", "paper_tables_boot"); isdir(tabdir) || mkpath(tabdir)
    csvpath = joinpath(tabdir, induced ? "comonotonicity_test_ndm_induced.csv" : "comonotonicity_test_ndm.csv")
    open(csvpath, "w") do io; foreach(r -> println(io, r), rows); end
    println("\nSaved → ", csvpath)
end

let
    mode = get(ENV, "COMONO_MODE", "all")
    (mode == "ndm") && run_ndm()
    (mode == "ndm_induced") && run_ndm(; induced = true)
    (mode in ("all", "val")) && run_validation()
    (mode in ("all", "real")) && run_real(; B = parse(Int, get(ENV, "COMONO_B", "2000")))
    (mode == "databoot") && run_databoot()
    (mode == "databoot_induced") && run_databoot(; induced = true)
end
