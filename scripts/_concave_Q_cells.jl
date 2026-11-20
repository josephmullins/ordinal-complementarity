# Ages 6–13: cross-differences of E[Q(θ')|θ,I] under several CONCAVE Q, with
# basic bootstrap CIs — which cells stay significantly negative as Q varies?
ENV["COMONO_MODE"] = "none"
include(joinpath(@__DIR__, "comonotonicity_test.jl"))
bc = load(joinpath(@__DIR__, "..", "output", "bootstrap_reg.jld2"))
ok = findall(bc["ok"]); r_θ = bc["r_θ"]; r_I = bc["r_I"]; g = 2
Πof(v) = reconstruct_Pi(v, r_θ, r_I)[1]
Π̂ = Πof(bc["pibar0"][g]); Πb = [Πof(bc["boot_pibar"][g][b, :]) for b in ok]
cells(Π, α) = vec(gap_matrices(Π)[2] * α)                 # 12 cells, column-major (θ step fastest)
lab = ["θ$(p)→$(p+1),I$(q)→$(q+1)" for q in 1:r_I-1 for p in 1:r_θ-1]
for α in ([1/3, 1/3, 1/3], [0.5, 0.3, 0.2], [0.7, 0.2, 0.1], [1.0, 0.0, 0.0])
    ĉ = cells(Π̂, α); cb = reduce(hcat, [cells(P, α) for P in Πb])
    @printf("\nQ = %s   min cell = %+.4f\n", round.(Q_from_alpha(α); digits = 2), minimum(ĉ))
    for k in sortperm(ĉ)[1:6]
        lo = 2ĉ[k] - quantile(cb[k, :], .975); hi = 2ĉ[k] - quantile(cb[k, :], .025)
        @printf("   %-16s %+.4f  [%+.4f, %+.4f]%s\n", lab[k], ĉ[k], lo, hi, hi < 0 ? "  *" : "")
    end
end

# The concave set {α ≥ 0, Σα = 1, α nonincreasing} has extreme points
# (1,0,0), (1/2,1/2,0), (1/3,1/3,1/3); a single cell is linear in α, so its
# max over concave Q is attained at one of them.
k = findfirst(==("θ1→2,I1→2"), lab)
println("\nCell θ1→2,I1→2 at the extreme points of the concave set:")
for α in ([1.0, 0, 0], [0.5, 0.5, 0], [1/3, 1/3, 1/3])
    ĉ = cells(Π̂, α)[k]; cb = [cells(P, α)[k] for P in Πb]
    @printf("   α = %-18s %+.4f  [%+.4f, %+.4f]\n", string(round.(α; digits = 2)), ĉ, 2ĉ - quantile(cb, .975), 2ĉ - quantile(cb, .025))
end
