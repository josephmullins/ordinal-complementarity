# ═══════════════════════════════════════════════════════════════
# Shared inference helpers for V3 and V3+ε_Y
#
# See /inference_notes.md for the design document.
#
#   • Softmax reparameterizations for probability-simplex parameters
#     (π_Pω, π_ε, π0 cells).  Reference bin = 1, α ∈ ℝ^(K-1).
#   • OPG estimator of observed-data Fisher information from per-child
#     score matrix.
#   • Delta-method utility.
# ═══════════════════════════════════════════════════════════════

using LinearAlgebra, Random

# ─────────────────────────────────────────────────────────────────
# Softmax parameterization:
#     α ∈ ℝ^(K-1)  ↔  π ∈ Δ^K   with  π_k = exp(α̃_k) / Σ_j exp(α̃_j),
#     α̃ = [0; α]   (bin 1 is the reference, log-odds zero).
# ─────────────────────────────────────────────────────────────────

"""
    softmax_to_probs(α) -> π

Map unconstrained `α ∈ ℝ^{K-1}` to a probability vector of length K,
with bin 1 as reference (α̃_1 = 0). Numerically stable via max-shift.
ForwardDiff-compatible.
"""
function softmax_to_probs(α::AbstractVector{T}) where T
    K = length(α) + 1
    z = Vector{promote_type(T, Float64)}(undef, K)
    @inbounds z[1] = zero(T)
    @inbounds for k in 2:K; z[k] = α[k-1]; end
    m = maximum(z)
    e = exp.(z .- m)
    return e ./ sum(e)
end

"""
    probs_to_softmax(π; ref=1) -> α

Inverse of `softmax_to_probs`: recover α from π (with bin `ref` as reference).
Adds a small floor to avoid log(0).
"""
function probs_to_softmax(π::AbstractVector; ref::Int=1)
    K = length(π)
    p = max.(π, 1e-300)
    return [log(p[k]) - log(p[ref]) for k in 1:K if k != ref]
end

# ─────────────────────────────────────────────────────────────────
# OPG estimator of observed-data Fisher information
# ─────────────────────────────────────────────────────────────────

"""
    opg_fisher(scores::Matrix) -> Σ_s

Outer-product-of-gradients estimator of the observed-data Fisher information
matrix, clustered at the observation (child) level.

    Σ_s = (1/N) · Σ_n s_n s_n'

`scores` is n_φ × N.  Returns n_φ × n_φ.
"""
function opg_fisher(scores::AbstractMatrix{Float64})
    n_φ, N = size(scores)
    return (scores * scores') / N
end

"""
    var_phi_hat(scores; reg=1e-10) -> Var(φ̂)

Given per-child scores, return Var(φ̂) = Σ_s⁻¹ / N.  Optional diagonal
regularization `reg` is added for numerical stability.  Falls back to
`pinv` if singular.
"""
function var_phi_hat(scores::AbstractMatrix{Float64}; reg::Float64=1e-10)
    n_φ, N = size(scores)
    Σ_s = opg_fisher(scores)
    if reg > 0
        for i in 1:n_φ; Σ_s[i,i] += reg; end
    end
    Var_φ = try
        inv(Σ_s) / N
    catch
        @warn "Σ_s near-singular; using pinv"
        pinv(Σ_s) / N
    end
    return Var_φ
end

# ─────────────────────────────────────────────────────────────────
# Delta method
# ─────────────────────────────────────────────────────────────────

"""
    delta_cov(J, Var_φ) -> Var(g(φ̂))

Delta method: Var(g) = J · Var(φ̂) · J'.
`J` is n_g × n_φ; `Var_φ` is n_φ × n_φ.
"""
function delta_cov(J::AbstractMatrix{Float64}, Var_φ::AbstractMatrix{Float64})
    return J * Var_φ * J'
end

# ─────────────────────────────────────────────────────────────────
# Simple helper to convert a full Var matrix into SEs (diagonal sqrt)
# ─────────────────────────────────────────────────────────────────

"""
    ses_from_cov(V) -> Vector

Standard errors as `sqrt(max.(diag(V), 0))`.  Negative diagonal entries
(can arise from numerical error in near-singular information matrices)
are floored at 0 before sqrt.
"""
ses_from_cov(V::AbstractMatrix) = sqrt.(max.(diag(V), 0.0))

# ═══════════════════════════════════════════════════════════════
# Two-sided modularity (ordinal complementarity / substitutability) test
#
# Survivor-DID convention (matches scripts/test_supermodularity_by_cutpoint.jl):
# the constraint matrix `D` is built so that `m = D·Π_free` is the adjacent
# survival second cross-difference  Δ²F̄(k|θ,x) = F̄(k|θ+1,x+1) − F̄(k|θ+1,x)
# − F̄(k|θ,x+1) + F̄(k|θ,x).  Ordinal COMPLEMENTARITY ⟺ m ≥ 0 everywhere;
# ordinal SUBSTITUTABILITY ⟺ m ≤ 0 everywhere.  Two one-sided MMM statistics:
#
#   • COMPLEMENTARITY null  H0: m ≥ 0  →  T_super = Σ min(0, m/se)²   reject ⇒ substitutes
#   • SUBSTITUTABILITY null H0: m ≤ 0  →  T_sub   = Σ max(0, m/se)²   reject ⇒ complements
#
# The least-favorable null is symmetric, so one critical value (from
# `crit_value`, defined in inference.jl) serves both sides.  The second
# conditioning axis `x` may be investment (r_I) OR an income grid (n_y): the
# object is structurally identical, so the SAME `D` builder and test apply with
# `r_x` in place of `r_I` — this is what the induced-transition tests use.
# ═══════════════════════════════════════════════════════════════

"""
    adjacent_did_matrix(r_θ, r_x) -> (D, cut_of_row, cell_of_row)

Minimal adjacent unit-cell survivor-DID constraint matrix over the (θ, x)
lattice (x = investment or income bin).  Row for cell (θ→θ+1, x→x+1) at
cutpoint k:  `D·Π_free = Δ²F̄(k|θ,x)`.  Free entries order
`for kx in 1:r_x, for kθ in 1:r_θ, for j in 1:r_θ-1` (matches `pibar_free`).
"""
function adjacent_did_matrix(r_θ::Int, r_x::Int)
    p_idx = LinearIndices((r_θ - 1, r_θ, r_x))
    np    = prod(size(p_idx))
    ncut  = r_θ - 1
    nrow  = (r_θ - 1) * (r_x - 1) * ncut
    D = zeros(nrow, np)
    cut_of_row  = zeros(Int, nrow)
    cell_of_row = Vector{Tuple{Int,Int}}(undef, nrow)
    r = 1
    for kθ in 1:(r_θ - 1), kx in 1:(r_x - 1)
        for k in 1:ncut
            D[r, p_idx[1, kθ+1, kx+1]:p_idx[k, kθ+1, kx+1]] .= -1.0
            D[r, p_idx[1, kθ,   kx  ]:p_idx[k, kθ,   kx  ]] .= -1.0
            D[r, p_idx[1, kθ+1, kx  ]:p_idx[k, kθ+1, kx  ]] .=  1.0
            D[r, p_idx[1, kθ,   kx+1]:p_idx[k, kθ,   kx+1]] .=  1.0
            cut_of_row[r] = k; cell_of_row[r] = (kθ, kx); r += 1
        end
    end
    return D, cut_of_row, cell_of_row
end

"""
    two_sided_modularity_test(D, Π_vec, Vfree; α=0.05, rng) -> NamedTuple

Run BOTH one-sided MMM tests on `m = D·Π_vec` with covariance `V = D·Vfree·D'`.
`Π_vec` is the free-entry vector (first r_θ-1 rows).  Returns
`(; nrow, n_used, T_super, T_sub, c, reject_super, reject_sub,
   n_comp, n_comp_sig, n_sub, n_sub_sig, worst_sub, strongest_comp)`
plus the DIRECTED (sup-type) twin computed on the SAME standardized moments z
and correlation Ω:
`(; T_super_sup, T_sub_sup, c_sup_super, c_sup_sub,
   reject_super_sup, reject_sub_sup)` where
  T_super_sup = (min(0, minimum(z)))²   (worst single suffix/complementarity gap)
  T_sub_sup   = (max(0, maximum(z)))²   (worst single substitutability gap)
and c_sup_* are the least-favourable sup critical values from `crit_value_sup`.
Requires `crit_value` and `crit_value_sup` (inference.jl) in scope at call time.
"""
function two_sided_modularity_test(D::AbstractMatrix, Π_vec::AbstractVector,
                                   Vfree::AbstractMatrix;
                                   α::Float64 = 0.05,
                                   rng::AbstractRNG = Random.default_rng())
    m  = D * Π_vec
    V  = D * Vfree * D'
    dV = diag(V)
    n_comp = count(>(0.0), m); n_sub = count(<(0.0), m)
    worst_sub = isempty(m) ? 0.0 : minimum(m)
    strongest_comp = isempty(m) ? 0.0 : maximum(m)
    tol  = 1e-10 * max(maximum(dV; init = 0.0), 1.0)
    keep = findall(>(tol), dV)
    isempty(keep) && return (; nrow = size(D, 1), n_used = 0, T_super = 0.0, T_sub = 0.0,
        c = 0.0, reject_super = false, reject_sub = false, n_comp, n_sub,
        n_comp_sig = 0, n_sub_sig = 0, worst_sub, strongest_comp,
        T_super_sup = 0.0, T_sub_sup = 0.0, c_sup_super = 0.0, c_sup_sub = 0.0,
        reject_super_sup = false, reject_sub_sup = false,
        c_rsw_super = 0.0, c_rsw_sub = 0.0,
        reject_super_rsw = false, reject_sub_rsw = false,
        nslack_rsw_super = 0, nslack_rsw_sub = 0)
    mk = m[keep]; Vk = V[keep, keep]; dk = dV[keep]
    Dinv = diagm(1.0 ./ sqrt.(dk))
    Ω = Matrix(Symmetric(Dinv * Vk * Dinv))
    mn = minimum(eigvals(Symmetric(Ω))); Ω += (max(0.0, 1e-8 - mn) + 1e-10) * I
    z = Dinv * mk
    T_super = sum(min.(0.0, z) .^ 2)     # H0: complementarity (m≥0); reject ⇒ substitutes
    T_sub   = sum(max.(0.0, z) .^ 2)     # H0: substitutability (m≤0); reject ⇒ complements
    c = crit_value(Hermitian(Ω), α; rng = rng)
    # DIRECTED (sup-type) twin on the SAME standardized moments z and Ω:
    T_super_sup = min(0.0, minimum(z))^2   # worst single complementarity-side violation
    T_sub_sup   = max(0.0, maximum(z))^2   # worst single substitutability-side violation
    csup = crit_value_sup(Hermitian(Ω), α; rng = rng)
    # ROMANO–SHAIKH–WOLF (2014) moment-selection critical values, both directions:
    rsw_super = crit_value_rsw(Hermitian(Ω), z, α; direction = :super, rng = rng)
    rsw_sub   = crit_value_rsw(Hermitian(Ω), z, α; direction = :sub,   rng = rng)
    c_rsw_super = rsw_super.c; c_rsw_sub = rsw_sub.c
    return (; nrow = size(D, 1), n_used = length(keep), T_super, T_sub, c,
            reject_super = T_super > c, reject_sub = T_sub > c,
            n_comp, n_sub, n_comp_sig = count(>(1.96), z), n_sub_sig = count(<(-1.96), z),
            worst_sub, strongest_comp,
            T_super_sup, T_sub_sup,
            c_sup_super = csup.c_super, c_sup_sub = csup.c_sub,
            reject_super_sup = T_super_sup > csup.c_super,
            reject_sub_sup   = T_sub_sup   > csup.c_sub,
            c_rsw_super, c_rsw_sub,
            reject_super_rsw = T_super > c_rsw_super,
            reject_sub_rsw   = T_sub   > c_rsw_sub,
            nslack_rsw_super = rsw_super.n_slack, nslack_rsw_sub = rsw_sub.n_slack)
end
