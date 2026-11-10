# Is the degenerate robit-NN hidden unit (huge log_w) load-bearing or vestigial?
# For each group: (1) per-unit contribution wₕ·Φ(argₕ) to the inner index `mu`
# across the (θ,I) grid; (2) rebuild the regime-marginalized Π̄ with that unit
# REMOVED (log_w → −∞) and measure how much the fitted transition function moves.
using JLD2, Statistics, Printf, LinearAlgebra
const REPO = joinpath(@__DIR__, "..")
for f in ["parametric_models.jl","parametric_models_v3.jl","parametric_models_v3_eps.jl"]
    include(joinpath(REPO, "src", f))
end

fit = load(joinpath(REPO, "output", "estimation_v3_eps.jld2"))
pv  = fit["params_v3_eps"]; psk = fit["params_skill"]
r_θ, r_I = psk.r_θ, psk.r_I
r_P, r_ω = pv.r_P, pv.r_ω
π_Pω = pv.π_Pω
θg = collect(1.0:r_θ); Ig = collect(1.0:r_I); Pg = collect(1.0:r_P); ωg = collect(1.0:r_ω)

# regime-marginalized Π̄ from a spec vector
function pi_bar(φ, spec)
    Π = zeros(r_θ, r_θ, r_I)
    for kω in 1:r_ω, kP in 1:r_P
        Π .+= π_Pω[kP,kω] .* _build_robit_nn_v3_pi(φ, r_θ, r_I, spec.K, spec.ν,
                                                    Pg[kP], ωg[kω]; θ_grid=θg, I_grid=Ig)
    end
    Π
end

for (g, spec) in enumerate(pv.trans_specs)
    K = spec.K
    hd = argmax(abs.(spec.log_w))      # degenerate unit
    @printf("\n========== GROUP %d  (K=%d, degenerate unit = #%d, log_w=%.1f) ==========\n",
            g, K, hd, spec.log_w[hd])

    # ── (1) per-unit contribution wₕ·Φ(argₕ) to `mu`, at middle regime ──
    kP, kω = cld(r_P,2), cld(r_ω,2)
    a1 = spec.alpha1; a2 = exp.(spec.alpha2_raw); a3 = spec.alpha3; a4 = spec.alpha4
    b  = spec.b;      w  = exp.(spec.log_w)
    println("  Per-unit contribution wₕ·Φ(argₕ) to the index `mu` over the (θ,I) grid",
            "  [regime θᴾ=$kP, ω=$kω]:")
    for h in 1:K
        contrib = [w[h]*Φ(a1[h]*θ + a2[h]*I + a3[h]*Pg[kP] + a4[h]*ωg[kω] + b[h])
                   for θ in θg, I in Ig]
        arg = [a1[h]*θ + a2[h]*I + a3[h]*Pg[kP] + a4[h]*ωg[kω] + b[h] for θ in θg, I in Ig]
        flag = h == hd ? "  <-- degenerate" : ""
        @printf("    unit %d: contrib ∈ [%.4f, %.4f] (mean %.4f);  arg ∈ [%.1f, %.1f]%s\n",
                h, minimum(contrib), maximum(contrib), mean(contrib),
                minimum(arg), maximum(arg), flag)
    end

    # ── (2) remove the degenerate unit, compare fitted Π̄ ──
    φ_full = pack_robit_nn_v3(spec)
    spec0  = RobitNNTransitionV3(copy(spec.c_raw), copy(spec.alpha1), copy(spec.alpha2_raw),
                copy(spec.alpha3), copy(spec.alpha4), copy(spec.b),
                [i==hd ? -700.0 : spec.log_w[i] for i in 1:K], K, spec.ν)
    Πf = pi_bar(φ_full, spec)
    Π0 = pi_bar(pack_robit_nn_v3(spec0), spec0)
    dΠ = abs.(Πf .- Π0)
    @printf("  Effect of REMOVING the degenerate unit on the fitted Π̄ (transition probs):\n")
    @printf("    max|ΔΠ̄| = %.4f,  mean|ΔΠ̄| = %.4f\n", maximum(dΠ), mean(dΠ))
    # survivor change too
    Sf = [sum(Πf[j,kθ,kI] for j in k+1:r_θ) for k in 1:r_θ-1, kθ in 1:r_θ, kI in 1:r_I]
    S0 = [sum(Π0[j,kθ,kI] for j in k+1:r_θ) for k in 1:r_θ-1, kθ in 1:r_θ, kI in 1:r_I]
    @printf("    max|ΔS̄(k|θ,I)| = %.4f\n", maximum(abs.(Sf .- S0)))
end
println("\nGuide: contrib≈0 everywhere & ΔΠ̄≈0  ⇒  vestigial (K=3 fine).",
        "  contrib O(0.1+) varying & ΔΠ̄ large  ⇒  load-bearing (K=4 real).")
