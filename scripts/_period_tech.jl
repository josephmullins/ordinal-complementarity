# Period-specific technology at FIXED posteriors (2026-11-18).
# Given a converged fit (grouped technology TG) and its sufficient statistics ξ, re-fit the
# technology separately for each transition t → t+1 (same functional form, warm from the
# group's estimate) holding every posterior fixed: one transition M-step per period.
const TG_PERIOD = [[t] for t in 1:6]

function period_specific_specs(pv, ξ_full, r_θ, r_I; TG = [[1, 2, 3], [4, 5, 6]])
    ξt = collapse_xi_over_eps(ξ_full, pv.r_P, pv.r_ω, pv.r_Y, 6, r_θ, r_I)
    grp_of(t) = findfirst(g -> t in g, TG)
    [mstep_transition_v3_grouped([pv.trans_specs[grp_of(t)]], ξt, [[t]], r_θ, r_I, pv.r_P, pv.r_ω)[1] for t in 1:6]
end
period_pibar(pv, specs, r_θ, r_I) = marginalize_Pi_v3(specs, TG_PERIOD, pv.π_Pω, r_θ, r_I, pv.r_P, pv.r_ω)
