# Is the early-childhood flatness of Π̄ in I (for I ≥ 3) also in the unregularized MLE?
using JLD2, Printf
for f in ["parametric_models.jl","parametric_models_v3.jl","parametric_models_v3_eps.jl","estimation_multiperiod_v3.jl"]
    include(joinpath(@__DIR__, "..", "src", f))
end
const TG = [[1,2,3],[4,5,6]]
for file in ("estimation_v3_eps.jld2", "estimation_v3_eps_reg.jld2")
    pv = load(joinpath(@__DIR__, "..", "output", file), "params_v3_eps")
    r_θ = 4; r_I = 5
    Π̄ = marginalize_Pi_v3(pv.trans_specs, TG, pv.π_Pω, r_θ, r_I, pv.r_P, pv.r_ω)
    for (g, grp) in enumerate(TG)
        Π = Π̄[grp[1]]
        d = [maximum(abs.(Π[:, :, q+1] .- Π[:, :, q])) for q in 1:r_I-1]
        @printf("%-28s group %d  max|ΔΠ̄| across I steps: %s\n", file, g, join([@sprintf("%.1e", x) for x in d], "  "))
    end
end
