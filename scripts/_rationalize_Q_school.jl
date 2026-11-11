# The valuation Q that comes closest to supermodular rationalizability (t⁺ LP)
# at the estimate, and bootstrap CIs on E[Q(θ')|θ,I] and its cross-differences.
#   Q(1)=0, Q(s)=Σ_{j<s} α_j, α ∈ simplex  ⇒  E[Q(θ')|θ,I] = Σ_j α_j Pr(θ' ≥ j+1 | θ,I)
# For FIXED α̂ these are linear in Π̄, so the (basic) bootstrap is valid.
# Also re-solves α*_b on each draw to show how stable the selected Q is.
ENV["COMONO_MODE"] = "none"
include(joinpath(@__DIR__, "comonotonicity_test.jl"))

const G = parse(Int, get(ENV, "GROUP", "2"))
const SENSE = Symbol(get(ENV, "SENSE", "max"))   # :max = closest to supermodular (t⁺), :min = closest to submodular (t⁻)
const TLAB = SENSE === :max ? "t̂⁺" : "t̂⁻"
const WORD = SENSE === :max ? "complementarity" : "substitutability"
bc = load(joinpath(@__DIR__, "..", "output", "bootstrap_reg.jld2"))
ok = findall(bc["ok"]); r_θ = bc["r_θ"]; r_I = bc["r_I"]
Πof(v) = reconstruct_Pi(v, r_θ, r_I)[1]
surv(Π) = [sum(Π[s, p, q] for s in (j + 1):r_θ) for p in 1:r_θ, q in 1:r_I, j in 1:(r_θ - 1)]
EQ(Π, α) = sum(α[j] .* surv(Π)[:, :, j] for j in eachindex(α))           # r_θ × r_I
xdiff(V) = [V[p+1, q+1] - V[p+1, q] - V[p, q+1] + V[p, q] for p in 1:r_θ-1, q in 1:r_I-1]
basic(x̂, xb) = (2x̂ - quantile(xb, .975), 2x̂ - quantile(xb, .025))

Π̂ = Πof(bc["pibar0"][G]); Πb = [Πof(bc["boot_pibar"][G][b, :]) for b in ok]
_, D̂ = gap_matrices(Π̂)
t̂, α̂, _, bind = solve_shape_lp(D̂; sense = SENSE)
Q̂ = Q_from_alpha(α̂)
@printf("GROUP %d  %s  t̂ = %+.5f   α̂ = %s   Q̂(1..%d) = %s\n", G, SENSE, t̂, round.(α̂; digits = 3), r_θ, round.(Q̂; digits = 3))

# stability of the selected valuation across bootstrap draws
αb = [solve_shape_lp(gap_matrices(P)[2]; sense = SENSE)[2] for P in Πb]
A = reduce(hcat, αb)'
println("\nRe-optimized α*_b across draws (mean, 2.5%, 97.5%):")
for j in 1:size(A, 2)
    @printf("  α_%d: %.3f  [%.3f, %.3f]\n", j, mean(A[:, j]), quantile(A[:, j], .025), quantile(A[:, j], .975))
end

# E[Q̂(θ')|θ,I] with basic bootstrap CIs (Q̂ held fixed)
V̂ = EQ(Π̂, α̂); Vb = [EQ(P, α̂) for P in Πb]
println("\nE[Q̂(θ')|θ,I]   rows θ=1..$r_θ, cols I=1..$r_I   (95% basic CI)")
for p in 1:r_θ
    print("  θ=$p: ")
    for q in 1:r_I
        lo, hi = basic(V̂[p, q], [V[p, q] for V in Vb])
        @printf("%.3f[%.3f,%.3f] ", V̂[p, q], lo, hi)
    end
    println()
end

# cross-differences of E[Q̂] per lattice cell (θ→θ+1, I→I+1)
X̂ = xdiff(V̂); Xb = [xdiff(V) for V in Vb]
println("\nCross-differences Δ²E[Q̂] per cell  (* = 95% basic CI excludes 0)")
for p in 1:r_θ-1
    print("  θ $p→$(p+1): ")
    for q in 1:r_I-1
        lo, hi = basic(X̂[p, q], [X[p, q] for X in Xb])
        @printf("%+.4f[%+.4f,%+.4f]%s ", X̂[p, q], lo, hi, (lo > 0 || hi < 0) ? "*" : " ")
    end
    println()
end
@printf("\nbinding cells at the estimate: %s (linear index into the %d×%d cell grid)\n", bind, r_θ-1, r_I-1)

# ───────────────────────── figure ─────────────────────────
ENV["PAPER_FIGS"] = "1"
include(joinpath(@__DIR__, "..", "src", "slides_palette.jl"))
gl = G == 1 ? "ages 0–5" : "ages 6–13"

# (a) Q̂ with band from re-optimized Q*_b
Qb = reduce(hcat, [Q_from_alpha(a) for a in αb])'
lo_q = [quantile(Qb[:, s], .025) for s in 1:r_θ]; hi_q = [quantile(Qb[:, s], .975) for s in 1:r_θ]
pA = plot(1:r_θ, Q̂; ribbon = (Q̂ .- lo_q, hi_q .- Q̂), fillalpha = 0.2, color = PAL_RUST, lw = 3,
          marker = :circle, ms = 6, label = "Q̂ (band: re-solved on each draw)",
          xlabel = "future skill θ′", ylabel = "Q(θ′)", title = "(a) Valuation closest to $WORD",
          xticks = 1:r_θ, ylims = (-0.02, 1.05), legend = :topleft)
plot!(pA, 1:r_θ, range(0, 1; length = r_θ); ls = :dash, color = PAL_GREY, lw = 2, label = "linear Q")

# (b) returns to investment E[Q̂|θ,I] − E[Q̂|θ,1], by θ
pB = plot(xlabel = "investment I", ylabel = "E[Q̂(θ′)|θ,I] − E[Q̂(θ′)|θ,1]",
          title = "(b) Return to investment, by current skill", xticks = 1:r_I, legend = :topleft)
hline!(pB, [0]; color = :black, lw = 1, label = "")
for p in 1:r_θ
    r̂ = V̂[p, :] .- V̂[p, 1]
    rb = [V[p, :] .- V[p, 1] for V in Vb]
    ci = [basic(r̂[q], [r[q] for r in rb]) for q in 1:r_I]
    plot!(pB, 1:r_I, r̂; ribbon = (r̂ .- first.(ci), last.(ci) .- r̂), fillalpha = 0.12,
          color = SKILL_COLORS[p], lw = 3, marker = :circle, ms = 5, label = "θ = $p")
end

# (c) cross-differences with CIs, binding cells hollow-starred
pC = plot(xlabel = "investment step", ylabel = "Δ²E[Q̂]  (θ step × I step)",
          title = "(c) Cross-differences of E[Q̂]", legend = (SENSE === :max ? :topright : :bottomright),
          xticks = (1:r_I-1, ["$q→$(q+1)" for q in 1:r_I-1]))
hline!(pC, [0]; color = :black, lw = 1, label = "")
bindset = Set(CartesianIndices((r_θ-1, r_I-1))[bind]); starlab = Ref(true)
for p in 1:r_θ-1
    x = (1:r_I-1) .+ (p - 2) * 0.18
    ci = [basic(X̂[p, q], [X[p, q] for X in Xb]) for q in 1:r_I-1]
    y = X̂[p, :]
    scatter!(pC, x, y; yerror = (y .- first.(ci), last.(ci) .- y), color = SKILL_COLORS[p+1],
             ms = 6, msc = SKILL_COLORS[p+1], lw = 2, label = "θ $p→$(p+1)")
    bx = [x[q] for q in 1:r_I-1 if CartesianIndex(p, q) in bindset]
    by = [y[q] for q in 1:r_I-1 if CartesianIndex(p, q) in bindset]
    isempty(bx) || scatter!(pC, bx, by; marker = :star5, ms = 13, color = :white, msc = :black,
                            label = (starlab[] ? (starlab[] = false; "binding at $TLAB") : ""))
end
ylims!(pC, (min(minimum(X̂), 0) - 0.04, max(maximum(X̂), 0) + 0.04))
let R = [V̂[p, q] - V̂[p, 1] for p in 1:r_θ, q in 1:r_I]; ylims!(pB, (min(minimum(R), 0) - 0.03, maximum(R) + 0.03)); end

fig = plot(pA, pB, pC; layout = (1, 3), size = (1900, 600), left_margin = 16Plots.mm, right_margin = 4Plots.mm,
           bottom_margin = 14Plots.mm, top_margin = 4Plots.mm, plot_title = "Valuation closest to $WORD, $gl ($TLAB = $(round(t̂; digits=4)))")
outdir = joinpath(@__DIR__, "..", "figures"); isdir(outdir) || mkpath(outdir)
stem = joinpath(outdir, "rationalize_Q_group$(G)_$(SENSE)")
savefig(fig, stem * ".pdf"); savefig(fig, stem * ".png")
println("Saved → ", stem, ".{pdf,png}")
