# Shared data setup for the simple ordered-robit pipeline (run_v3_eps_ordrobit.jl,
# bootstrap_ordrobit.jl, _period_specific_tech.jl). Reads ENV, defines `data`, `panel`,
# `tag`, `outp(stage)`. Requires src/ files and REPO to be loaded by the caller.
#   R_I (5), INIT (random|data), NO_PIAT3 (0/1), GROUP_INV (0/1),
#   BINS (0 = pooled deciles as in the paper; k > 0 = k quantile bins WITHIN each age period),
#   AGEBIN (0/1): with BINS > 0, form the bins within each single year of age at test
#   TIDY (0/1): drop empty categories, merge tail categories with < 1% of observations
#               into their neighbour (all measures with > 2 categories), and restart any
#               emission whose observed category has collapsed below the probability floor
include(joinpath(REPO, "scripts", "_items.jl"))
const AGES = [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13)]
const R_I       = parse(Int, get(ENV, "R_I", "5"))
const INIT      = get(ENV, "INIT", "random")
const NO_PIAT3  = get(ENV, "NO_PIAT3", "0") == "1"
const GROUP_INV = get(ENV, "GROUP_INV", "0") == "1"
const BINS      = parse(Int, get(ENV, "BINS", "0"))
const TIDY      = get(ENV, "TIDY", "0") == "1"
const AGEBIN    = get(ENV, "AGEBIN", "0") == "1"
const R_P, R_ω, R_Y, NU = 3, 3, 3, 7.0

datasets = load_all_datasets()
excl = [t == 3 && NO_PIAT3 ? [:piat] : Symbol[] for t in 1:length(AGES)]
panel = prepare_panel_data(datasets; age_ranges = AGES, invest_items = ITEMS, r_θ = 4, r_I = R_I,
                           exclude_sources = excl, cont_bins_per_period = BINS,
                           cont_bins_by_age = AGEBIN)
TIDY && (COMPOSITE_COMPRESS[] = true; EMISSION_STALL_RESTART[] = true)
if GROUP_INV
    Xg, κg, ng = group_investment_measures(panel.X_I, panel.κ_I, panel.measure_names_I)
    panel = merge(panel, (; X_I = Xg, κ_I = κg, measure_names_I = ng))
    println("Investment measures grouped: ", [length(x) for x in Xg], " per period")
end
if TIDY
    nmerge = Ref(0)
    tidy(Xs, κs) = begin
        Xo = similar(Xs); κo = similar(κs)
        for t in eachindex(Xs)
            Xo[t] = similar(Xs[t]); κo[t] = similar(κs[t])
            for m in eachindex(Xs[t])
                if κs[t][m] > 2
                    Xo[t][m], κo[t][m] = merge_sparse_tails(Xs[t][m])
                    nmerge[] += κs[t][m] - κo[t][m]
                else
                    Xo[t][m], κo[t][m] = Xs[t][m], κs[t][m]
                end
            end
        end
        Xo, κo
    end
    Xθ, κθ = tidy(panel.X_θ, panel.κ_θ); XI, κI = tidy(panel.X_I, panel.κ_I)
    panel = merge(panel, (; X_θ = Xθ, κ_θ = κθ, X_I = XI, κ_I = κI))
    println("TIDY: $(nmerge[]) sparse/empty categories merged; stalled-emission restart ON")
end
flush(stdout)
data = merge(panel, (; r_P = R_P, r_ω = R_ω))
tag = (R_I == 5 ? "" : "_rI$(R_I)") * (INIT == "data" ? "_dd" : "") * (NO_PIAT3 ? "_noPIAT3" : "") *
      (GROUP_INV ? "_grp" : "") * (BINS > 0 ? "_q$(BINS)" : "") * (AGEBIN ? "a" : "") * (TIDY ? "_tidy" : "")
outp(stage, seed) = joinpath(REPO, "output", "estimation_v3_eps_ordrobit_$(stage)$(tag)_seed$(seed).jld2")
