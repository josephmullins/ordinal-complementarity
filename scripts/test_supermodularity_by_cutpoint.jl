# ============================================================================
# Test supermodularity of E[Q_j | θ, I] separately for each cutpoint j, where
#   Q_j(θ') = 1{θ' ≥ j}   ⇒   E[Q_j | θ, I] = P(θ_{t+1} ≥ j | θ, I).
#
# Each Q_j isolates ONE survival cutpoint, so testing supermodularity of
# E[Q_j|·] over the (θ,I) lattice partitions the full ordinal-complementarity
# inequality set by cutpoint:  get_S() orders its rows with the cutpoint as the
# inner loop, so rows with kθ3 == (j-1) are exactly the DID conditions for Q_j.
# The conjunction over j is the full OC test (Table: local_modularity).
#
# Reuses the parametric (delta-method) covariance Var_free of Π̄ — the same
# object behind the survivor-band CIs and the per-outcome DID figure — and the
# modified-method-of-moments statistic of get_test_stat_mp_parametric.
#
# Output:  console table + output/paper_tables/supermodularity_by_cutpoint.csv
# Run:     julia --project=. scripts/test_supermodularity_by_cutpoint.jl
# ============================================================================
using JLD2, LinearAlgebra, Statistics, Printf, Random, CSV, DataFrames

const REPO = joinpath(@__DIR__, "..")
include(joinpath(REPO, "src", "parametric_models.jl"))
include(joinpath(REPO, "src", "parametric_models_v3.jl"))
include(joinpath(REPO, "src", "parametric_models_v3_eps.jl"))
include(joinpath(REPO, "src", "prepare_data.jl"))
include(joinpath(REPO, "src", "estimation.jl"))
include(joinpath(REPO, "src", "estimation_multiperiod.jl"))
include(joinpath(REPO, "src", "estimation_multiperiod_v3.jl"))
include(joinpath(REPO, "src", "estimation_multiperiod_v3_eps.jl"))
include(joinpath(REPO, "src", "inference.jl"))
include(joinpath(REPO, "src", "inference_common.jl"))
include(joinpath(REPO, "src", "inference_v3_eps.jl"))

const TRANSITION_GROUPS = [[1, 2, 3], [4, 5, 6]]
const GROUP_LABELS      = ["ages 0–5", "ages 6–13"]

# Investment items per period (matches generate_ucla_figures.jl).
const PANEL_INVEST_ITEMS_NB = [
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
     :ToldPositive, :HowOftenHomework],
    [:HowManyBooks, :HowOftenMuseum, :HowOftenTheater, :HowOftenFamily,
     :HowOftenOutdoors, :HowOftenFather, :ChildMusic, :SpecialActivities,
     :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
     :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy,
     :ToldPositive, :HowOftenHomework],
]

const REG_RUN  = get(ENV, "REG_RUN", "0") == "1"
const FIT_PATH = get(ENV, "FIT_OVERRIDE", joinpath(REPO, "output", "estimation_v3_eps_reg.jld2"))
println("Loading fit: ", FIT_PATH, " ...")
fit = load(FIT_PATH)
pv3 = fit["params_v3_eps"]; psk = fit["params_skill"]; dims = fit["data_dims"]
age_ranges = fit["age_ranges"]
(; r_θ, r_I, r_P, r_ω, N, T) = dims

println("Loading raw data ...")
datasets = load_all_datasets()
panel = prepare_panel_data(datasets; age_ranges = age_ranges,
    invest_items = PANEL_INVEST_ITEMS_NB, r_θ = r_θ, r_I = r_I)
data_nb = merge(panel, (; r_P = r_P, r_ω = r_ω))

if haskey(ENV, "BOOT_COV")
    # BOOTSTRAP covariance: skip the asymptotic Fisher path; take Π̄ from the
    # fit and Var_free from the bootstrap draws (cov of Π̄_free per group, placed
    # at each group's representative period; ordering matches delta_cov_Pi_bar_free).
    println("Using BOOTSTRAP covariance from ", ENV["BOOT_COV"], " ...")
    Pi_bar = marginalize_Pi_v3(pv3.trans_specs, TRANSITION_GROUPS, pv3.π_Pω, r_θ, r_I, r_P, r_ω)
    bc = load(ENV["BOOT_COV"]); bp = bc["boot_pibar"]; okb = findall(bc["ok"])
    nfree = (r_θ - 1) * r_θ * r_I
    Var_free_list = [zeros(nfree, nfree) for _ in 1:(T - 1)]
    for (g, group) in enumerate(TRANSITION_GROUPS)
        Var_free_list[group[1]] = cov(bp[g][okb, :])
    end
    println("  bootstrap n_ok=$(length(okb)); Var_free mean-diag/group: ",
            [round(mean(diag(Var_free_list[group[1]])); sigdigits=3) for group in TRANSITION_GROUPS])
else
    println("Running inference (Fisher's identity, ~10 sec) ...")
    inf = compute_var_phi_v3_eps(pv3, psk, data_nb; transition_groups = TRANSITION_GROUPS, verbose = false)
    Var_free_list, Pi_bar = delta_cov_Pi_bar_free(inf, pv3, psk; transition_groups = TRANSITION_GROUPS)
end

# ============================================================================
# Two constraint sets (FULL pairwise vs ADJACENT unit-cell), and BOTH one-sided
# tests, on the survivor diff-in-difference m = Δ²F̄(k|θ,I) = Δ²E[Q_{k+1}|θ,I]:
#   • SUPERMODULARITY  H0: m ≥ 0   →  T = Σ min(0, m/se)²   reject ⇒ substitutes
#   • SUBMODULARITY    H0: m ≤ 0   →  T = Σ max(0, m/se)²   reject ⇒ complements
# Each D is built so D·Π_free = m directly.  (src get_S gives −m and tests the
# submodularity side; the LF null is symmetric so one crit value serves both.)
# ============================================================================

"""
    get_S_adj_tagged(r_θ, r_I) -> (D, cut_of_row, cell_of_row)

Adjacent unit-cell survivor DIDs.  Row (cell (θ,I), cutpoint k):
    D·Π_free = F̄(k|θ+1,I+1) − F̄(k|θ+1,I) − F̄(k|θ,I+1) + F̄(k|θ,I) = Δ²E[Q_{k+1}].
"""
function get_S_adj_tagged(r_θ::Int, r_I::Int)
    p_idx = LinearIndices((r_θ - 1, r_θ, r_I))
    np    = prod(size(p_idx))
    ncut  = r_θ - 1
    nrow  = (r_θ - 1) * (r_I - 1) * ncut
    D = zeros(nrow, np)
    cut_of_row  = zeros(Int, nrow)
    cell_of_row = Vector{Tuple{Int,Int}}(undef, nrow)
    r = 1
    for kθ in 1:(r_θ - 1), kI in 1:(r_I - 1)
        for k in 1:ncut
            D[r, p_idx[1, kθ+1, kI+1]:p_idx[k, kθ+1, kI+1]] .= -1.0
            D[r, p_idx[1, kθ,   kI  ]:p_idx[k, kθ,   kI  ]] .= -1.0
            D[r, p_idx[1, kθ+1, kI  ]:p_idx[k, kθ+1, kI  ]] .=  1.0
            D[r, p_idx[1, kθ,   kI+1]:p_idx[k, kθ,   kI+1]] .=  1.0
            cut_of_row[r] = k; cell_of_row[r] = (kθ, kI); r += 1
        end
    end
    return D, cut_of_row, cell_of_row
end

ncut     = r_θ - 1
D_full   = -get_S(r_θ, r_I)                       # sign flip ⇒ D·Π = m
cut_full = [((r - 1) % ncut) + 1 for r in 1:size(D_full, 1)]
D_adj, cut_adj, cell_adj = get_S_adj_tagged(r_θ, r_I)
rng = MersenneTwister(20240601)
rows_of(cut, j) = findall(==(j), cut)

function run_set(D, Π_vec, Vfree; α = 0.05)
    m  = D * Π_vec
    V  = D * Vfree * D'
    dV = diag(V)
    n_comp = count(>(0.0), m); n_sub = count(<(0.0), m)
    worst_sub = minimum(m); strongest_comp = maximum(m)
    tol  = 1e-10 * max(maximum(dV), 1.0)
    keep = findall(>(tol), dV)
    isempty(keep) && return (; nrow=size(D,1), n_used=0, T_super=0.0, T_sub=0.0, c=0.0,
        reject_super=false, reject_sub=false, n_comp, n_sub, n_comp_sig=0, n_sub_sig=0,
        worst_sub, strongest_comp, c_rsw_super=0.0, c_rsw_sub=0.0,
        reject_super_rsw=false, reject_sub_rsw=false, nslack_super=0, nslack_sub=0)
    mk = m[keep]; Vk = V[keep, keep]; dk = dV[keep]
    Dinv = diagm(1.0 ./ sqrt.(dk))
    Ω = Matrix(Symmetric(Dinv * Vk * Dinv))
    mn = minimum(eigvals(Symmetric(Ω))); Ω += (max(0.0, 1e-8 - mn) + 1e-10) * I
    z = Dinv * mk
    T_super = sum(min.(0.0, z) .^ 2)
    T_sub   = sum(max.(0.0, z) .^ 2)
    c = crit_value(Hermitian(Ω), α; rng = rng)     # symmetric LF null ⇒ same c
    rsw_super = crit_value_rsw(Hermitian(Ω), z, α; direction = :super, rng = rng)
    rsw_sub   = crit_value_rsw(Hermitian(Ω), z, α; direction = :sub,   rng = rng)
    c_rsw_super = rsw_super.c; c_rsw_sub = rsw_sub.c
    (; nrow=size(D,1), n_used=length(keep), T_super, T_sub, c,
       reject_super = T_super > c, reject_sub = T_sub > c,
       n_comp, n_sub, n_comp_sig=count(>(1.96), z), n_sub_sig=count(<(-1.96), z),
       worst_sub, strongest_comp,
       c_rsw_super, c_rsw_sub,
       reject_super_rsw = T_super > c_rsw_super, reject_sub_rsw = T_sub > c_rsw_sub,
       nslack_super = rsw_super.n_slack, nslack_sub = rsw_sub.n_slack)
end

# Run every (set, group, cutpoint) once, cache results.
specs = vcat([("all", 0)], [("θ'≥$(j+1)", j) for j in 1:ncut])
res = Dict{Tuple{Int,String,String},Any}()
for (g, group) in enumerate(TRANSITION_GROUPS)
    Π_g = copy(Pi_bar[group[1]]); Π_g .= max.(Π_g, 1e-15)
    for kI in 1:r_I, kθ in 1:r_θ; Π_g[:, kθ, kI] ./= sum(Π_g[:, kθ, kI]); end
    Π_vec = Π_g[1:end-1, :, :][:]; Vfree = Var_free_list[group[1]]
    for (lab, j) in specs
        rf = j == 0 ? run_set(D_full, Π_vec, Vfree) : run_set(D_full[rows_of(cut_full, j), :], Π_vec, Vfree)
        ra = j == 0 ? run_set(D_adj,  Π_vec, Vfree) : run_set(D_adj[rows_of(cut_adj,  j), :], Π_vec, Vfree)
        res[(g, lab, "full")] = rf; res[(g, lab, "adjacent")] = ra
    end
end

function print_table(dir::Symbol)
    Tfield  = dir == :super ? :T_super : :T_sub
    Rfield  = dir == :super ? :reject_super : :reject_sub
    Vfield  = dir == :super ? :n_sub_sig : :n_comp_sig
    vlabel  = dir == :super ? "#sub*" : "#cmp*"
    head    = dir == :super ? "SUPERMODULARITY  (H0: m≥0;  reject ⇒ substitutability)" :
                              "SUBMODULARITY    (H0: m≤0;  reject ⇒ complementarity)"
    println("\n" * "="^104); println(head); println("="^104)
    for (g, group) in enumerate(TRANSITION_GROUPS)
        println("\n── Group $g ($(GROUP_LABELS[g])) ──")
        @printf "  %-7s | %5s %6s %9s %8s %5s | %5s %6s %9s %8s %5s\n" "Q_j" "#cond" "rej" "T" "c.05" vlabel "#cond" "rej" "T" "c.05" vlabel
        for (lab, _) in specs
            rf = res[(g, lab, "full")]; ra = res[(g, lab, "adjacent")]
            @printf "  %-7s | %5d %6s %9.2f %8.2f %5d | %5d %6s %9.2f %8.2f %5d\n" lab rf.nrow (getfield(rf,Rfield) ? "YES" : "no") getfield(rf,Tfield) rf.c getfield(rf,Vfield) ra.nrow (getfield(ra,Rfield) ? "YES" : "no") getfield(ra,Tfield) ra.c getfield(ra,Vfield)
        end
    end
end
print_table(:super)
print_table(:sub)

# ── Cell-level detail for a given (group, cutpoint k); Q_{k+1} = 1{θ'≥k+1} ─────
function print_detail(g::Int, k::Int)
    group = TRANSITION_GROUPS[g]
    println("\n" * "="^104)
    println("DETAIL — Group $g ($(GROUP_LABELS[g])), Q_$(k+1) = 1{θ'≥$(k+1)}:  m = Δ²P(θ'≥$(k+1) | θ,I) per adjacent cell")
    println("contrib_super = min(0, z)²  (substitute side);  contrib_sub = max(0, z)²  (complement side),  z = m/se")
    println("-"^104)
    Π_g = copy(Pi_bar[group[1]]); Π_g .= max.(Π_g, 1e-15)
    for kI in 1:r_I, kθ in 1:r_θ; Π_g[:, kθ, kI] ./= sum(Π_g[:, kθ, kI]); end
    Π_vec = Π_g[1:end-1, :, :][:]; Vfree = Var_free_list[group[1]]
    rows = rows_of(cut_adj, k)
    Dk = D_adj[rows, :]; cells = cell_adj[rows]
    m   = Dk * Π_vec
    dV  = diag(Dk * Vfree * Dk')
    se  = sqrt.(max.(dV, 0.0))
    tol = 1e-10 * max(maximum(dV), 1.0)            # same drop rule as the test
    keep = dV .> tol
    z   = m ./ se
    @printf "  %-12s %9s %9s %8s %20s  %-11s %9s %9s\n" "cell (θ,I)" "m" "se" "z" "95% CI" "class" "c_super" "c_sub"
    Tsu = 0.0; Tsb = 0.0
    for i in eachindex(rows)
        (kθ, kI) = cells[i]
        if !keep[i]
            @printf "  θ%d→%d,I%d→%d %9.4f %9.4f %8s   %20s  %-11s %9s %9s\n" kθ kθ+1 kI kI+1 m[i] se[i] "—" "" "degenerate" "—" "—"
            continue
        end
        lo = m[i] - 1.96se[i]; hi = m[i] + 1.96se[i]
        cls = lo > 0 ? "complement*" : hi < 0 ? "substitute*" : "ns"
        csu = min(0.0, z[i])^2; csb = max(0.0, z[i])^2; Tsu += csu; Tsb += csb
        @printf "  θ%d→%d,I%d→%d %9.4f %9.4f %8.2f   [%7.4f,%7.4f]  %-11s %9.3f %9.3f\n" kθ kθ+1 kI kI+1 m[i] se[i] z[i] lo hi cls csu csb
    end
    @printf "  %-12s %58s  T_super=%.2f   T_sub=%.2f\n" "TOTAL" "" Tsu Tsb
    println("  (adjacent Q_$(k+1) critical value c_0.05 = ",
            @sprintf("%.2f", res[(g, "θ'≥$(k+1)", "adjacent")].c), ")")
end

print_detail(1, 1)   # Q_2, early childhood
print_detail(2, 1)   # Q_2, school-age

# CSV: both directions, both sets
records = DataFrame(group=String[], cutpoint=String[], set=String[], n_cond=Int[], n_used=Int[],
                    T_super=Float64[], reject_super=Bool[], T_sub=Float64[], reject_sub=Bool[],
                    c_alpha=Float64[], c_rsw_super=Float64[], reject_super_rsw=Bool[],
                    c_rsw_sub=Float64[], reject_sub_rsw=Bool[],
                    nslack_rsw_super=Int[], nslack_rsw_sub=Int[],
                    n_complement=Int[], n_complement_sig=Int[],
                    n_substitute=Int[], n_substitute_sig=Int[], worst_substitute=Float64[], strongest_complement=Float64[])
for (g, _) in enumerate(TRANSITION_GROUPS), (lab, _) in specs, set in ("full", "adjacent")
    r = res[(g, lab, set)]
    push!(records, (GROUP_LABELS[g], lab, set, r.nrow, r.n_used, r.T_super, r.reject_super,
                    r.T_sub, r.reject_sub, r.c, r.c_rsw_super, r.reject_super_rsw,
                    r.c_rsw_sub, r.reject_sub_rsw, r.nslack_super, r.nslack_sub, r.n_comp, r.n_comp_sig, r.n_sub, r.n_sub_sig,
                    r.worst_sub, r.strongest_comp))
end
println("\n=== LF vs RSW (adjacent set): T_super, c_LF, c_RSW, reject_LF→reject_RSW ===")
for (g, _) in enumerate(TRANSITION_GROUPS), (lab, _) in specs
    r = res[(g, lab, "adjacent")]
    @printf "  %-9s %-8s | T=%7.2f  c_LF=%6.2f  c_RSW=%6.2f   %s→%s\n" GROUP_LABELS[g] lab r.T_super r.c r.c_rsw_super (r.reject_super ? "REJ" : "no") (r.reject_super_rsw ? "REJ" : "no")
end
outdir = joinpath(REPO, "output",
    haskey(ENV,"BOOT_COV") ? "paper_tables_boot" : (REG_RUN ? "paper_tables_reg" : "paper_tables"))
isdir(outdir) || mkpath(outdir)
CSV.write(joinpath(outdir, "supermodularity_by_cutpoint.csv"), records)
println("\nsaved ", joinpath(outdir, "supermodularity_by_cutpoint.csv"))

# ── Validation: corrected SOURCE get_test_stat_v3_grouped vs standalone adjacent "all" ──
println("\n" * "="^104)
println("VALIDATION — src get_test_stat_v3_grouped(corrected) should match standalone ADJACENT 'all'")
println("="^104)
src = get_test_stat_v3_grouped(Var_free_list, Pi_bar, data_nb, TRANSITION_GROUPS)
for (g, r) in enumerate(src)
    a = res[(g, "all", "adjacent")]
    @printf "  Group %d:  source T_n=%.2f c=%.2f reject=%s | standalone T_super=%.2f c=%.2f reject=%s\n" g r.T_n r.c_α r.reject a.T_super a.c a.reject_super
end
