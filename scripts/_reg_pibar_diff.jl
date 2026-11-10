# Standalone: compute regime-marginalized Π̄ for OLD and NEW fits and diff.
using JLD2, Printf, Statistics
const REPO = abspath(joinpath(@__DIR__, ".."))
const SRC  = joinpath(REPO, "src")
for f in ("parametric_models.jl", "parametric_models_v3.jl",
          "parametric_models_v3_eps.jl", "prepare_data.jl",
          "estimation.jl", "estimation_multiperiod.jl",
          "estimation_multiperiod_v3.jl", "estimation_multiperiod_v3_eps.jl")
    include(joinpath(SRC, f))
end

const TRANSITION_GROUPS = [[1,2,3],[4,5,6]]

function pibar_groups(fitpath)
    fit = load(fitpath)
    pv3 = fit["params_v3_eps"]
    dims = fit["data_dims"]
    (; r_θ, r_I, r_P, r_ω) = dims
    out = Vector{Array{Float64,3}}(undef, length(TRANSITION_GROUPS))
    for (g, _) in enumerate(TRANSITION_GROUPS)
        spec_g = pv3.trans_specs[g]
        φ_g = pack_robit_nn_v3(spec_g)
        Π_g = zeros(r_θ, r_θ, r_I)
        for kω in 1:r_ω, kP in 1:r_P
            w = pv3.π_Pω[kP, kω]
            Π_z = _build_robit_nn_v3_pi(φ_g, r_θ, r_I, spec_g.K, spec_g.ν,
                Float64(kP), Float64(kω); θ_grid=1.0:r_θ, I_grid=1.0:r_I)
            Π_g .+= w .* Π_z
        end
        out[g] = Π_g
    end
    return out, (r_θ, r_I)
end

old, dd = pibar_groups(joinpath(REPO, "output", "estimation_v3_eps.jld2"))
new, _  = pibar_groups(joinpath(REPO, "output", "estimation_v3_eps_reg.jld2"))
r_θ, r_I = dd

open(joinpath(REPO, "output", "reg_comparison", "pibar_diff.txt"), "w") do io
    println(io, "Regime-marginalized Π̄ : OLD (estimation_v3_eps) vs NEW (estimation_v3_eps_reg)")
    println(io, "Π̄[θ', θ, I],  size = ($r_θ,$r_θ,$r_I), per transition group")
    println(io, "="^70)
    for (g, grp) in enumerate(TRANSITION_GROUPS)
        D = new[g] .- old[g]
        amax = maximum(abs.(D)); amean = mean(abs.(D))
        @printf(io, "\nGroup %d  (periods %s):  max|ΔΠ̄| = %.4f   mean|ΔΠ̄| = %.4f\n",
                g, string(grp), amax, amean)
        # top 5 moved cells
        idxs = sortperm(vec(abs.(D)); rev=true)[1:5]
        println(io, "  top cells moved (θ', θ, I): old -> new (Δ)")
        for li in idxs
            ci = CartesianIndices(D)[li]
            tp, th, ii = ci.I
            @printf(io, "    (θ'=%d, θ=%d, I=%d): %.4f -> %.4f  (Δ=%+.4f)\n",
                    tp, th, ii, old[g][ci], new[g][ci], D[li])
        end
        # survivor F̄(k|θ,I) max change
        for k in 1:(r_θ-1)
            Sold = [sum(old[g][j,kθ,kI] for j in k+1:r_θ) for kθ in 1:r_θ, kI in 1:r_I]
            Snew = [sum(new[g][j,kθ,kI] for j in k+1:r_θ) for kθ in 1:r_θ, kI in 1:r_I]
            @printf(io, "  survivor k=%d : max|ΔS̄| = %.4f\n", k, maximum(abs.(Snew .- Sold)))
        end
    end
end
println("wrote output/reg_comparison/pibar_diff.txt")
print(read(joinpath(REPO, "output", "reg_comparison", "pibar_diff.txt"), String))
