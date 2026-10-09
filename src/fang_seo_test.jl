# ─────────────────────────────────────────────────────────────────
# Fang & Seo (2021, ECMA; arXiv:1910.07689) projection test for shape
# restrictions that form convex cones, applied to ordinal super/submodularity
# of the technology Π(θ'|θ,I).
#
# Parameter: the survivor vector  s(k,θ,I) = F̄(k|θ,I) = P(θ' > k | θ,I),
# k = 1..r_θ-1, stacked with LinearIndices((r_θ-1, r_θ, r_I)).
# Supermodularity in distribution ⟺ A·s ≥ 0 where A is the adjacent-cell
# diff-in-difference incidence matrix (one row per (θ→θ+1, I→I+1, k)):
#   F̄(k|θ+1,I+1) + F̄(k|θ,I) − F̄(k|θ,I+1) − F̄(k|θ+1,I) ≥ 0.
# Submodularity ⟺ A·s ≤ 0.  Both {s: As ≥ 0} and {s: As ≤ 0} are polyhedral
# convex cones, so the Fang–Seo framework applies directly.
#
# Test statistic:  T_n = √N · ‖ŝ − Π_Λ(ŝ)‖₂   (projection in Euclidean norm).
# Bootstrap critical value (their §3.3, Steps 1–3), with Ĝ_b = √N(ŝ*_b − ŝ):
#   τ̂ = (1-γ_n)-quantile of ‖Ĝ_b‖;  κ̂_n = √N·c_n/τ̂  (c_n = 1 in regular,
#   √N-consistent settings);  ĉ_{1-α} = (1-α)-quantile over b of
#   ‖Ĝ_b + κ̂_n Π_Λ(ŝ) − Π_Λ(Ĝ_b + κ̂_n Π_Λ(ŝ))‖.
# Reject H₀ (the shape restriction) iff T_n > ĉ_{1-α}.
# ─────────────────────────────────────────────────────────────────

"""
    survivor_did_matrix(r_θ, r_I) -> A

Adjacent-cell diff-in-difference matrix on the survivor vector
`s = vec(F̄[k,θ,I])` (`LinearIndices((r_θ-1, r_θ, r_I))`).  Supermodularity in
distribution ⟺ `A*s ≥ 0`.  `(r_θ-1)²(r_I-1)` rows.  Same inequalities as
`get_S_adjacent` (which acts on the PMF vector; `A*s = −S*p` entrywise).
"""
function survivor_did_matrix(r_θ, r_I)
    idx  = LinearIndices((r_θ - 1, r_θ, r_I))
    nrow = (r_θ - 1) * (r_I - 1) * (r_θ - 1)
    A = zeros(nrow, length(idx))
    r = 1
    for kθ in 1:(r_θ-1), kI in 1:(r_I-1), k in 1:(r_θ-1)
        A[r, idx[k, kθ,   kI  ]] =  1.0
        A[r, idx[k, kθ+1, kI+1]] =  1.0
        A[r, idx[k, kθ,   kI+1]] = -1.0
        A[r, idx[k, kθ+1, kI  ]] = -1.0
        r += 1
    end
    return A
end

"""
    survivor_vec(Π) -> s

Survivor vector `s[k,θ,I] = P(θ' > k | θ, I)`, `k = 1..r_θ-1`, flattened in
`LinearIndices((r_θ-1, r_θ, r_I))` order, from a (column-stochastic in dim 1)
transition array `Π[θ',θ,I]`.
"""
function survivor_vec(Π::AbstractArray{<:Real,3})
    r_θ, _, r_I = size(Π)
    s = zeros(r_θ - 1, r_θ, r_I)
    for kI in 1:r_I, kθ in 1:r_θ
        tot = sum(@view Π[:, kθ, kI])
        acc = 0.0
        for k in 1:(r_θ-1)
            acc += Π[k, kθ, kI]
            s[k, kθ, kI] = 1.0 - acc / tot
        end
    end
    return s[:]
end

"""
    nnls(B, y; tol, maxiter) -> x ≥ 0 minimizing ‖B*x − y‖₂

Lawson–Hanson active-set non-negative least squares.  Exact (up to `tol`) for
the small dual problems arising in `project_cone`.
"""
function nnls(B::AbstractMatrix{Float64}, y::AbstractVector{Float64};
              tol::Float64 = 1e-12, maxiter::Int = 30 * size(B, 2))
    m, n = size(B)
    x = zeros(n)
    P = falses(n)                      # passive (unconstrained) set
    w = B' * y                         # gradient of ½‖Bx−y‖² at x=0 is −w
    resid = copy(y)
    scale = max(norm(y), 1.0)
    for _ in 1:maxiter
        w .= B' * resid
        # most-violating inactive coordinate
        j = 0; wmax = tol * scale
        for i in 1:n
            if !P[i] && w[i] > wmax
                wmax = w[i]; j = i
            end
        end
        j == 0 && break                # KKT satisfied
        P[j] = true
        # inner loop: solve LS on passive set, backtrack if any goes negative
        while true
            idx = findall(P)
            z = B[:, idx] \ y
            if all(>(tol), z)
                x .= 0.0
                x[idx] .= z
                break
            end
            # backtrack toward feasibility
            α = Inf
            for (kk, i) in enumerate(idx)
                if z[kk] ≤ tol
                    a = x[i] / (x[i] - z[kk] + eps())
                    α = min(α, a)
                end
            end
            for (kk, i) in enumerate(idx)
                x[i] += α * (z[kk] - x[i])
                if x[i] ≤ tol
                    x[i] = 0.0
                    P[i] = false
                end
            end
        end
        resid .= y .- B * x
    end
    return x
end

"""
    project_cone(v, A) -> h = argmin ‖h − v‖₂  s.t.  A*h ≥ 0

Euclidean projection onto the polyhedral cone `{h : A*h ≥ 0}`, via the dual
NNLS problem (Moreau decomposition): `h = v + A'μ*` with
`μ* = argmin_{μ≥0} ‖A'μ + v‖₂`.
"""
function project_cone(v::AbstractVector{Float64}, A::AbstractMatrix{Float64})
    μ = nnls(Matrix(A'), -v)
    return v + A' * μ
end

"""
    cumulative_survivor_did_matrix(r_θ, r_x; mode) -> A

Cumulative (over cutpoints) version of `survivor_did_matrix`, for the
shape-refined nulls (draft §convex-OC/concave-OS):

- `mode = :suffix`: row (cell, j) = Σ_{k ≥ j} Δ²F̄(k|θ,I).  Convex ordinal
  complementarity ⟺ `A*s ≥ 0` (E[Q] supermodular for every nondecreasing
  convex Q, by Abel summation).
- `mode = :prefix`: row (cell, j) = Σ_{k ≤ j} Δ²F̄(k|θ,I).  Concave ordinal
  substitutability ⟺ `A*s ≤ 0`.

Both are polyhedral cones in survivor space, so `fang_seo_test` applies with
`direction = :super` (suffix) or `:sub` (prefix).
"""
function cumulative_survivor_did_matrix(r_θ, r_x; mode::Symbol)
    A_adj = survivor_did_matrix(r_θ, r_x)
    ncut = r_θ - 1
    ncell = (r_θ - 1) * (r_x - 1)
    A = zeros(size(A_adj))
    for cell in 0:(ncell - 1)
        base = cell * ncut                 # rows base+1..base+ncut: cutpoints 1..ncut
        for j in 1:ncut
            ks = mode === :suffix ? (j:ncut) :
                 mode === :prefix ? (1:j)   :
                 throw(ArgumentError("mode must be :suffix or :prefix"))
            for k in ks
                @views A[base + j, :] .+= A_adj[base + k, :]
            end
        end
    end
    return A
end

"""
    studentize_scale(s_boot; floor_frac=0.1) -> σ

Component-wise bootstrap SDs of the survivor vector, floored at
`floor_frac · median(σ)` so near-degenerate components (survivor probabilities
pinned at 0/1 across draws, SD ~1e-5 vs median ~1e-2) do not dominate the
weighted norm.
"""
function studentize_scale(s_boot::AbstractMatrix{Float64}; floor_frac::Float64 = 0.1)
    σ = vec(std(s_boot; dims=2))
    return max.(σ, floor_frac * median(σ))
end

"""
    fang_seo_test(ŝ, s_boot, N; A, direction=:super, α=0.05, γ=:auto, c_n=1.0,
                  scale=nothing)

Fang–Seo (2021) projection cone test.

Arguments
- `ŝ`      : point estimate of the survivor vector (length d).
- `s_boot` : d × B matrix of bootstrap survivor vectors (one column per draw).
- `N`      : sample size (children), so `r_n = √N`.
- `A`      : constraint matrix; H₀ is `A*s ≥ 0` when `direction = :super`
             and `A*s ≤ 0` when `direction = :sub`.
- `γ`      : tuning parameter γ_n for the data-driven κ̂_n; `:auto` uses the
             paper's recommended `0.01/log(N)` (their simulations show results
             are insensitive to γ_n over {1/n, 0.01/log n, 0.01}).
- `c_n`    : coupling rate; 1.0 in regular √N-consistent settings (our case).
- `scale`  : optional length-d vector σ of component scales.  When given, the
             Hilbert norm is the studentized ‖x‖ = ‖x ./ σ‖₂ (equivalently the
             test runs in the transformed space s̃ = s ./ σ with constraint
             matrix A·diag(σ); the cone is unchanged, only the metric).  Use
             `studentize_scale(s_boot)` for bootstrap-SD weights.

Returns a NamedTuple with `T_n`, `c_α`, `reject`, `p_value` (bootstrap
p-value: fraction of bootstrap criteria draws ≥ T_n), `κ_n`, `τ`, `s_proj`
(in the original, unscaled space), and the per-draw bootstrap statistics
`T_boot`.
"""
function fang_seo_test(ŝ::AbstractVector{Float64}, s_boot::AbstractMatrix{Float64},
                       N::Integer; A::AbstractMatrix{Float64},
                       direction::Symbol = :super, α::Float64 = 0.05,
                       γ = :auto, c_n::Float64 = 1.0,
                       scale::Union{Nothing,AbstractVector{Float64}} = nothing)
    d, B = size(s_boot)
    @assert length(ŝ) == d "ŝ and s_boot dimension mismatch"
    Ac = direction === :super ? A :
         direction === :sub   ? -A :
         throw(ArgumentError("direction must be :super or :sub"))

    # studentized norm: run the Euclidean machinery in s̃ = s ./ σ coordinates
    σ = scale
    if σ !== nothing
        @assert length(σ) == d "scale and ŝ dimension mismatch"
        @assert all(>(0), σ) "scale must be strictly positive"
        ŝ      = ŝ ./ σ
        s_boot = s_boot ./ σ
        Ac     = Ac .* σ'          # A·diag(σ): Ã·s̃ = A·s, same cone
    end

    r_n = sqrt(N)
    γn = γ === :auto ? 0.01 / log(N) : Float64(γ)

    # Step 1: test statistic  T_n = r_n ‖ŝ − Π_Λ ŝ‖
    s_proj = project_cone(collect(ŝ), Ac)
    T_n = r_n * norm(ŝ - s_proj)

    # Step 2: bootstrap draws Ĝ_b = r_n (ŝ*_b − ŝ) and data-driven κ̂_n
    G = r_n .* (s_boot .- ŝ)
    Gnorm = [norm(@view G[:, b]) for b in 1:B]
    τ = quantile(Gnorm, 1 - γn)
    κ_n = r_n * c_n / τ

    # restricted (null-enforced) drift κ̂_n·Π_Λ(ŝ)
    drift = κ_n .* s_proj
    T_boot = zeros(B)
    for b in 1:B
        v = drift .+ @view G[:, b]
        T_boot[b] = norm(v - project_cone(v, Ac))
    end

    c_α = quantile(T_boot, 1 - α)
    p_value = mean(T_boot .>= T_n)
    s_proj_out = σ === nothing ? s_proj : σ .* s_proj
    return (; T_n, c_α, reject = T_n > c_α, p_value, κ_n, τ, γ = γn,
              s_proj = s_proj_out, T_boot, direction)
end
