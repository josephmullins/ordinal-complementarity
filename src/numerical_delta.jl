# Numerical delta method (Hong & Li 2018, JoE) for a directionally
# differentiable statistic φ of a √N-consistent estimator x̂.
#
# Given bootstrap draws x*_b with G*_b = √N (x*_b − x̂), the draws
#     D*_b = [ φ(x̂ + ε_N G*_b) − φ(x̂) ] / ε_N,   ε_N → 0, ε_N √N → ∞,
# estimate the law of φ'(x; G), the limit of √N (φ(x̂) − φ(x)), whether or not
# φ is fully differentiable at x.  ε_N = N^{-1/2} reproduces the standard
# (inconsistent at kinks) bootstrap.  See the co-authors' note
# `numerical_delta_note.tex` (RCT replication folder) for the application to
# the rationalizability statistics t^±.

"""
    ndm_draws(φ, x̂, xboot, N; ε) -> Vector

`xboot` is an iterable of bootstrap estimates (same shape as `x̂`).  Returns
D*_b for each draw.  Note x̂ + ε G*_b = x̂ + (ε√N)(x*_b − x̂).
"""
function ndm_draws(φ, x̂, xboot, N::Integer; ε::Real)
    φ̂ = φ(x̂); s = ε * sqrt(N)
    [(φ(x̂ .+ s .* (xb .- x̂)) - φ̂) / ε for xb in xboot]
end

"""
    ndm_test(t̂, D, N; α=0.05, sense) -> NamedTuple

Decision and CI from numerical-delta draws `D` for a one-sided existence null.
- `sense = :max` (t⁺, comonotone achievable):  H0: t⁺ ≥ 0, reject when √N t̂ < q_α(D).
- `sense = :min` (t⁻, counter-monotone achievable): H0: t⁻ ≤ 0, reject when √N t̂ > q_{1−α}(D).
CI: [t̂ − q_{1−α/2}/√N, t̂ − q_{α/2}/√N].
"""
function ndm_test(t̂, D, N; α = 0.05, sense::Symbol)
    rt = sqrt(N) * t̂
    if sense === :max
        c = quantile(D, α); reject = rt < c
    elseif sense === :min
        c = quantile(D, 1 - α); reject = rt > c
    else
        error("sense must be :max or :min")
    end
    ci = (t̂ - quantile(D, 1 - α / 2) / sqrt(N), t̂ - quantile(D, α / 2) / sqrt(N))
    (; rt, c, reject, ci)
end
