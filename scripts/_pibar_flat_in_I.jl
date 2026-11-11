# How much does Π̄(θ'|θ,I) change across adjacent investment levels? (per group)
ENV["COMONO_MODE"] = "none"
include(joinpath(@__DIR__, "comonotonicity_test.jl"))
bc = load(joinpath(@__DIR__, "..", "output", "bootstrap_reg.jld2")); r_θ = bc["r_θ"]; r_I = bc["r_I"]
for g in 1:2
    Π = reconstruct_Pi(bc["pibar0"][g], r_θ, r_I)[1]
    d = [maximum(abs.(Π[:, :, q+1] .- Π[:, :, q])) for q in 1:r_I-1]
    @printf("group %d  max|Π̄(·|·,I+1) − Π̄(·|·,I)|  for I = 1..%d:  %s\n", g, r_I-1, join([@sprintf("%.2e", x) for x in d], "  "))
    for p in 1:r_θ
        @printf("   θ=%d  Pr(θ'=4 | θ, I=1..5) = %s\n", p, join([@sprintf("%.4f", Π[r_θ, p, q]) for q in 1:r_I], " "))
    end
end
