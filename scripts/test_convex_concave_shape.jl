# ============================================================================
# SHAPE-REFINED ordinal-complementarity tests on the survivor DIDs.
#
# Background (comonotonicity_test_instructions.md §1.4–1.6): let
#   Δ²F̄(k|θ,I) = adjacent (θ,I) second cross-difference of the survival
#                function  S̄(k|θ,I)=P(θ'>k|θ,I)   at cutpoint k.
# The plain OC test (scripts/test_supermodularity_by_cutpoint.jl) tests the sign
# of each Δ²F̄(k|θ,I) directly.  Two SHAPE refinements test cumulative sums:
#
#   • CONVEX ordinal complementarity:  E[Q(θ')|θ,I] supermodular for every
#     nondecreasing CONVEX Q  ⟺  SUFFIX sums  T_j(θ,I)=Σ_{k≥j} Δ²F̄(k|θ,I) ≥ 0
#     for every cell and cutpoint j   (Abel summation, §1.6).
#     H0: T_j ≥ 0.  Reject ⇒ some suffix-sum cell is a significant NEGATIVE
#     (substitute).  → MMM "T_super" side of two_sided_modularity_test.
#
#   • CONCAVE ordinal substitutability:  E[Q(θ')|θ,I] submodular for every
#     nondecreasing CONCAVE Q  ⟺  PREFIX sums  L_j(θ,I)=Σ_{k≤j} Δ²F̄(k|θ,I) ≤ 0.
#     H0: L_j ≤ 0.  Reject ⇒ some prefix-sum cell is a significant POSITIVE
#     (complement).  → MMM "T_sub" side of two_sided_modularity_test.
#
# Cumulative summation is LINEAR in Π̄, so the suffix/prefix constraint matrices
# are row-sums of the adjacent per-cutpoint matrix `adjacent_did_matrix`, and the
# bootstrap covariance of Π̄ (boot_pibar / boot_pibar_ind in bootstrap_reg.jld2)
# propagates through unchanged — exactly mirroring
# scripts/test_supermodularity_by_cutpoint.jl and scripts/test_modularity_induced.jl.
#
# Outputs:
#   output/paper_tables_boot/shape_refinement_transition.csv   (technology Π̄)
#   output/paper_tables_boot/shape_refinement_induced.csv      (induced Π_ind(θ'|θ,y))
# Run: julia --project=. scripts/test_convex_concave_shape.jl
#      BOOT=output/bootstrap_reg.jld2  (default)
# ============================================================================
using JLD2, LinearAlgebra, Statistics, Printf, Random, CSV, DataFrames

const REPO = joinpath(@__DIR__, "..")
include(joinpath(REPO, "src", "inference.jl"))          # crit_value, get_S
include(joinpath(REPO, "src", "inference_common.jl"))   # adjacent_did_matrix, two_sided_modularity_test

const BOOT = get(ENV, "BOOT", joinpath(REPO, "output", "bootstrap_reg.jld2"))
const GROUP_LABELS = ["ages 0–5", "ages 6–13"]

# ---------------------------------------------------------------------------
# Cumulative survivor-DID constraint matrix.  Mode :suffix sums cutpoints k≥j
# (→ T_j, convex test); mode :prefix sums cutpoints k≤j (→ L_j, concave test).
# Built as row-sums of the adjacent per-cutpoint matrix so D·Π_free returns the
# cumulative gap directly and any covariance propagates linearly.
# ---------------------------------------------------------------------------
function cumulative_did_matrix(r_θ::Int, r_x::Int; mode::Symbol)
    D_adj, cut_adj, cell_adj = adjacent_did_matrix(r_θ, r_x)
    ncut  = r_θ - 1
    np    = size(D_adj, 2)
    cells = unique(cell_adj)                       # ordered (kθ,kx) lattice cells
    nrow  = length(cells) * ncut
    D = zeros(nrow, np)
    cut_of_row  = zeros(Int, nrow)
    cell_of_row = Vector{Tuple{Int,Int}}(undef, nrow)
    r = 1
    for cell in cells
        rows = findall(==(cell), cell_adj)
        rows = rows[sortperm(cut_adj[rows])]       # order by cutpoint k = 1..ncut
        for j in 1:ncut
            ks = mode === :suffix ? (j:ncut) : (1:j)
            for k in ks
                @views D[r, :] .+= D_adj[rows[k], :]
            end
            cut_of_row[r] = j; cell_of_row[r] = cell; r += 1
        end
    end
    return D, cut_of_row, cell_of_row
end

# ---------------------------------------------------------------------------
# Run both shape tests for one lattice given the free Π̄ vector and its
# bootstrap covariance.  Returns the convex-OC (suffix/T_super) and concave-OS
# (prefix/T_sub) outcomes.
# ---------------------------------------------------------------------------
function shape_tests(r_θ::Int, r_x::Int, Π_vec::AbstractVector, Vfree::AbstractMatrix;
                     rng::AbstractRNG, α::Float64 = 0.05)
    D_suf, _, _ = cumulative_did_matrix(r_θ, r_x; mode = :suffix)
    D_pre, _, _ = cumulative_did_matrix(r_θ, r_x; mode = :prefix)
    rs = two_sided_modularity_test(D_suf, Π_vec, Vfree; α = α, rng = rng)  # convex-OC
    rp = two_sided_modularity_test(D_pre, Π_vec, Vfree; α = α, rng = rng)  # concave-OS
    return (
        # convex ordinal complementarity (suffix sums T_j ≥ 0)
        n_suf          = rs.nrow,
        T_convexOC     = rs.T_super,
        c_convex       = rs.c,
        reject_convexOC= rs.reject_super,
        c_convex_rsw       = rs.c_rsw_super,
        reject_convexOC_rsw= rs.reject_super_rsw,
        nslack_convex_rsw  = rs.nslack_rsw_super,   # # moments RSW classifies as slack
        n_neg_suffix     = rs.n_sub,        # # suffix cells with negative point est
        n_neg_suffix_sig = rs.n_sub_sig,    # # significant negative (substitute) suffix cells
        worst_suffix     = rs.worst_sub,    # most negative suffix sum
        # DIRECTED (sup-type) convex-OC twin — worst single suffix violation
        T_convexOC_sup     = rs.T_super_sup,
        c_convex_sup       = rs.c_sup_super,
        reject_convexOC_sup= rs.reject_super_sup,
        # concave ordinal substitutability (prefix sums L_j ≤ 0)
        n_pre          = rp.nrow,
        T_concaveOS    = rp.T_sub,
        c_concave      = rp.c,
        reject_concaveOS = rp.reject_sub,
        c_concave_rsw       = rp.c_rsw_sub,
        reject_concaveOS_rsw= rp.reject_sub_rsw,
        nslack_concave_rsw  = rp.nslack_rsw_sub,
        n_pos_prefix     = rp.n_comp,       # # prefix cells with positive point est
        n_pos_prefix_sig = rp.n_comp_sig,   # # significant positive (complement) prefix cells
        strongest_prefix = rp.strongest_comp,  # most positive prefix sum
        # DIRECTED (sup-type) concave-OS twin — worst single prefix violation
        T_concaveOS_sup     = rp.T_sub_sup,
        c_concave_sup       = rp.c_sup_sub,
        reject_concaveOS_sup= rp.reject_sub_sup,
    )
end

# ---------------------------------------------------------------------------
# Load bootstrap output (point estimate + draws of Π̄_free, technology + induced)
# ---------------------------------------------------------------------------
println("Loading bootstrap: ", BOOT)
bc = load(BOOT)
okrows = findall(bc["ok"])
r_θ = bc["r_θ"]; r_I = bc["r_I"]; n_y = bc["n_y"]; TG = bc["transition_groups"]
y_levels = bc["y_levels"]
@printf("  n_ok=%d  r_θ=%d  r_I=%d  n_y=%d\n", length(okrows), r_θ, r_I, n_y)

rng = MersenneTwister(20240601)

function build_table(kind::Symbol)
    if kind === :transition
        pibar0 = bc["pibar0"]; boot = bc["boot_pibar"]; r_x = r_I
    else
        pibar0 = bc["pibar_ind0"]; boot = bc["boot_pibar_ind"]; r_x = n_y
    end
    df = DataFrame(group=String[], n_suffix_cond=Int[],
                   T_convexOC=Float64[], c_convex=Float64[], reject_convexOC=Bool[],
                   c_convex_rsw=Float64[], reject_convexOC_rsw=Bool[], nslack_convex_rsw=Int[],
                   T_convexOC_sup=Float64[], c_convex_sup=Float64[], reject_convexOC_sup=Bool[],
                   n_neg_suffix=Int[], n_neg_suffix_sig=Int[], worst_suffix=Float64[],
                   n_prefix_cond=Int[],
                   T_concaveOS=Float64[], c_concave=Float64[], reject_concaveOS=Bool[],
                   c_concave_rsw=Float64[], reject_concaveOS_rsw=Bool[], nslack_concave_rsw=Int[],
                   T_concaveOS_sup=Float64[], c_concave_sup=Float64[], reject_concaveOS_sup=Bool[],
                   n_pos_prefix=Int[], n_pos_prefix_sig=Int[], strongest_prefix=Float64[])
    results = []
    for (g, _) in enumerate(TG)
        Vfree = cov(boot[g][okrows, :])
        Π_vec = pibar0[g]
        r = shape_tests(r_θ, r_x, Π_vec, Vfree; rng = rng)
        push!(results, r)
        push!(df, (GROUP_LABELS[g], r.n_suf, r.T_convexOC, r.c_convex, r.reject_convexOC,
                   r.c_convex_rsw, r.reject_convexOC_rsw, r.nslack_convex_rsw,
                   r.T_convexOC_sup, r.c_convex_sup, r.reject_convexOC_sup,
                   r.n_neg_suffix, r.n_neg_suffix_sig, r.worst_suffix,
                   r.n_pre, r.T_concaveOS, r.c_concave, r.reject_concaveOS,
                   r.c_concave_rsw, r.reject_concaveOS_rsw, r.nslack_concave_rsw,
                   r.T_concaveOS_sup, r.c_concave_sup, r.reject_concaveOS_sup,
                   r.n_pos_prefix, r.n_pos_prefix_sig, r.strongest_prefix))
    end
    return df, results
end

function print_table(title, df)
    println("\n" * "="^126); println(title); println("="^126)
    @printf "  %-10s | %-52s | %-52s\n" "group" "CONVEX-OC  (suffix T_j≥0; rej⇒substitute)" "CONCAVE-OS (prefix L_j≤0; rej⇒complement)"
    @printf "  %-10s | %6s %6s %5s ‖ %6s %6s %5s | %6s %6s %5s ‖ %6s %6s %5s\n" "" "T_MMM" "c.05" "rej" "T_sup" "c.05" "rej" "T_MMM" "c.05" "rej" "T_sup" "c.05" "rej"
    for row in eachrow(df)
        @printf "  %-10s | %6.2f %6.2f %5s ‖ %6.2f %6.2f %5s | %6.2f %6.2f %5s ‖ %6.2f %6.2f %5s\n" row.group row.T_convexOC row.c_convex (row.reject_convexOC ? "YES" : "no") row.T_convexOC_sup row.c_convex_sup (row.reject_convexOC_sup ? "YES" : "no") row.T_concaveOS row.c_concave (row.reject_concaveOS ? "YES" : "no") row.T_concaveOS_sup row.c_concave_sup (row.reject_concaveOS_sup ? "YES" : "no")
    end
    println("  --- LF vs RSW moment selection (convex-OC | concave-OS): T, c_LF, c_RSW, reject_LF→reject_RSW ---")
    for row in eachrow(df)
        @printf "  %-10s | convexOC: T=%6.2f c_LF=%6.2f c_RSW=%6.2f  %s→%s | concaveOS: T=%6.2f c_LF=%6.2f c_RSW=%6.2f  %s→%s\n" row.group row.T_convexOC row.c_convex row.c_convex_rsw (row.reject_convexOC ? "REJ" : "no") (row.reject_convexOC_rsw ? "REJ" : "no") row.T_concaveOS row.c_concave row.c_concave_rsw (row.reject_concaveOS ? "REJ" : "no") (row.reject_concaveOS_rsw ? "REJ" : "no")
    end
end

outdir = joinpath(REPO, "output", "paper_tables_boot")
isdir(outdir) || mkpath(outdir)

df_tr, _  = build_table(:transition)
df_in, _  = build_table(:induced)
print_table("TECHNOLOGY  Π̄(θ'|θ,I) — shape-refined OC tests", df_tr)
print_table("INDUCED     Π_ind(θ'|θ,income) — shape-refined OC tests", df_in)

CSV.write(joinpath(outdir, "shape_refinement_transition.csv"), df_tr)
CSV.write(joinpath(outdir, "shape_refinement_induced.csv"), df_in)
println("\nsaved ", joinpath(outdir, "shape_refinement_transition.csv"))
println("saved ", joinpath(outdir, "shape_refinement_induced.csv"))

# ===========================================================================
# VALIDATION on synthetic technologies (comonotonicity_test_instructions §3).
# Build Π from a survival design  S̄(k|θ,I) = base_k + c_k·(θ-1)(I-1),  so the
# adjacent (θ,I) survivor DID at cutpoint k equals c_k for every lattice cell.
#   • c_k ≥ 0 ∀k   (all Δ_j ≥ 0)  ⇒ convex-OC holds; concave-OS may reject
#   • c_k ≤ 0 ∀k   (all Δ_j ≤ 0)  ⇒ concave-OS holds; convex-OC may reject
#   • single-crossing c=[-,0,+]    ⇒ convex-OC holds, concave-OS fails
# ===========================================================================
function synth_pi(r_θ, r_x, base, c)
    Π = zeros(r_θ, r_θ, r_x)
    for kθ in 1:r_θ, kx in 1:r_x
        Sbar = [clamp(base[k] + c[k] * (kθ - 1) * (kx - 1), 0.0, 1.0) for k in 1:(r_θ-1)]
        Π[1, kθ, kx] = 1 - Sbar[1]
        for j in 2:(r_θ-1); Π[j, kθ, kx] = Sbar[j-1] - Sbar[j]; end
        Π[r_θ, kθ, kx] = Sbar[end]
        @assert all(Π[:, kθ, kx] .>= -1e-12) "neg prob synth ($kθ,$kx): $(Π[:,kθ,kx])"
        Π[:, kθ, kx] ./= sum(Π[:, kθ, kx])
    end
    return Π
end

# Concentrated single-cell violation: I=1 column flat; a step-down shift applied to
# the I=2 column only above θ-cell `cellθ`, so exactly ONE θ-cell carries a negative
# adjacent DID and all other cells sit at the boundary (Δ=0).  The directed sup test
# should reject convex-OC on that single cell.
function synth_pi_single_cell(rθ, base; cellθ = 2, δ = 0.06)
    rx = 2
    Π = zeros(rθ, rθ, rx)
    for kθ in 1:rθ, kx in 1:rx
        shift = (kx == 2 && kθ > cellθ) ? -δ : 0.0
        Sbar = [clamp(base[k] + shift, 0.0, 1.0) for k in 1:(rθ-1)]
        Π[1, kθ, kx] = 1 - Sbar[1]
        for j in 2:(rθ-1); Π[j, kθ, kx] = Sbar[j-1] - Sbar[j]; end
        Π[rθ, kθ, kx] = Sbar[end]
        Π[:, kθ, kx] ./= sum(Π[:, kθ, kx])
    end
    return Π
end

println("\n" * "="^116)
println("VALIDATION — synthetic technologies (r_θ=4, r_x=2);  MMM sum vs DIRECTED sup on each side")
println("="^116)
rθv, rxv = 4, 2
base = [0.7, 0.5, 0.3]
nfree = (rθv - 1) * rθv * rxv
Vsyn = (0.001^2) .* Matrix(I, nfree, nfree)     # tight synthetic covariance
rngv = MersenneTwister(7)
cases = [
    ("all Δ_j ≥ 0  (c=+0.05)",        synth_pi(rθv, rxv, base, fill( 0.05, 3))),
    ("all Δ_j ≤ 0  (c=-0.05)",        synth_pi(rθv, rxv, base, fill(-0.05, 3))),
    ("single-crossing (c=[-.02,0,+.06])", synth_pi(rθv, rxv, base, [-0.02, 0.0, 0.06])),
    ("single-cell violation (θ-cell 2)",  synth_pi_single_cell(rθv, base; cellθ = 2, δ = 0.06)),
]
@printf "  %-34s | %-30s | %-30s\n" "case" "CONVEX-OC (suffix T_j≥0)" "CONCAVE-OS (prefix L_j≤0)"
@printf "  %-34s | %7s %6s ‖ %7s %6s | %7s %6s ‖ %7s %6s\n" "" "T_MMM" "rej" "T_sup" "rej" "T_MMM" "rej" "T_sup" "rej"
for (lab, Π) in cases
    Π_vec = Π[1:end-1, :, :][:]
    r = shape_tests(rθv, rxv, Π_vec, Vsyn; rng = rngv)
    @printf "  %-34s | %7.2f %6s ‖ %7.2f %6s | %7.2f %6s ‖ %7.2f %6s\n" lab r.T_convexOC (r.reject_convexOC ? "REJ" : "hold") r.T_convexOC_sup (r.reject_convexOC_sup ? "REJ" : "hold") r.T_concaveOS (r.reject_concaveOS ? "REJ" : "hold") r.T_concaveOS_sup (r.reject_concaveOS_sup ? "REJ" : "hold")
end
println("\nExpected: row1 convex-OC holds (both); row2 concave-OS holds (both);")
println("          row3 convex-OC holds & concave-OS rejects; row4 convex-OC sup rejects (single-cell).")
