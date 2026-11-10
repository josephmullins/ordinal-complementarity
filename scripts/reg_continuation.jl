# Stage 1 (graduated non-convexity / continuation in λ).
#
# Tame the robit-NN compensating-pair degeneracy AND avoid local optima by
# tracking the solution down a DESCENDING λ schedule: start large (penalty
# dominates ⇒ well-conditioned, init-insensitive), warm-start each stage from
# the previous, run only a few EM iters per stage (NOT to convergence), and
# record the unpenalized log-likelihood.  Pick λ* = largest λ on the LL plateau
# (best LL, most regularization, before further λ-reduction stops mattering),
# then converge λ*.
#
# Robustness: run the continuation from the standard init (current fit, which
# the large-λ first stage de-degenerates) plus 2 random-transition starts; if
# they agree, the basin is robust.
#
# Config via ENV:
#   N_INNER   EM iters per λ stage          (default 20)
#   SCHEDULE  comma list, descending λ       (default 10,3,1,0.3,0.1,0.03,0.01,0.003,0.001)
#   PLATEAU   LL-gain tol for λ* selection   (default 1.0)
using LinearAlgebra, Random, Statistics, JLD2, Optim, Printf
const REPO = joinpath(@__DIR__, "..")
for f in ["parametric_models.jl","parametric_models_v3.jl","parametric_models_v3_eps.jl",
          "prepare_data.jl","estimation.jl","estimation_multiperiod.jl",
          "estimation_multiperiod_v3.jl","estimation_multiperiod_v3_eps.jl"]
    include(joinpath(REPO, "src", f))
end
BLAS.set_num_threads(1)
const TG = [[1,2,3],[4,5,6]]
const AGES = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13)]
const ITEMS = [
 [:HowOftenRead,:HowManyBooks,:HowManyToysA,:HowManyToysB,:OutOfHouse,:HowOftenTalks,:HowOftenGrocery],
 [:HowOftenRead,:HowManyBooks,:HowManyToysA,:HowManyToysB,:OutOfHouse,:HowOftenTalks,:HowOftenGrocery,:HowManyZines,:HowOftenOuting,:FoodChoice,:LearnNumbers,:LearnAlphabet,:LearnColors,:LearnShapes,:RecordTapeCD],
 [:HowOftenRead,:HowManyBooks,:HowManyZines,:HowOftenOuting,:HowOftenMuseum,:RecordTapeCD,:FoodChoice],
 [:HowOftenRead,:HowManyBooks,:HowOftenMuseum,:HowOftenTheater,:HowOftenFamily,:HowOftenOutdoors,:ChildMusic,:SpecialActivities,:Hobbies,:EnjoyRead,:DailyNewspaper,:DiscussTV,:GradesTeacher,:GradesTalk,:GradesHelp,:GradesStudy],
 [:HowOftenRead,:HowManyBooks,:HowOftenMuseum,:HowOftenTheater,:HowOftenFamily,:HowOftenOutdoors,:HowOftenFather,:ChildMusic,:SpecialActivities,:Hobbies,:EnjoyRead,:DailyNewspaper,:DiscussTV,:GradesTeacher,:GradesTalk,:GradesHelp,:GradesStudy],
 [:HowManyBooks,:HowOftenMuseum,:HowOftenTheater,:HowOftenFamily,:HowOftenOutdoors,:HowOftenFather,:ChildMusic,:SpecialActivities,:Hobbies,:EnjoyRead,:DailyNewspaper,:DiscussTV,:GradesTeacher,:GradesTalk,:GradesHelp,:GradesStudy,:ToldPositive,:HowOftenClean,:HowOftenChores,:HowOftenTime],
 [:HowManyBooks,:HowOftenMuseum,:HowOftenTheater,:HowOftenFamily,:HowOftenOutdoors,:HowOftenFather,:ChildMusic,:SpecialActivities,:Hobbies,:EnjoyRead,:DailyNewspaper,:DiscussTV,:GradesTeacher,:GradesTalk,:GradesHelp,:GradesStudy,:ToldPositive,:HowOftenClean,:HowOftenChores,:HowOftenTime],
]
const N_INNER  = parse(Int, get(ENV,"N_INNER","20"))
const SCHEDULE = [parse(Float64,s) for s in split(get(ENV,"SCHEDULE","10,3,1,0.3,0.1,0.03,0.01,0.003,0.001"),',')]
const PLATEAU  = parse(Float64, get(ENV,"PLATEAU","1.0"))

println("Loading data + current fit ...")
datasets = load_all_datasets()
panel = prepare_panel_data(datasets; age_ranges=AGES, invest_items=ITEMS, r_θ=4, r_I=5)
const DATA = merge(panel, (; r_P=3, r_ω=3))
mle = load(joinpath(REPO,"output","estimation_v3_eps.jld2"))
const PV0, PSK0 = mle["params_v3_eps"], mle["params_skill"]
(; r_θ, r_I) = PSK0
const PIBAR0 = marginalize_Pi_v3(PV0.trans_specs, TG, PV0.π_Pω, r_θ, r_I, PV0.r_P, PV0.r_ω)
maxlw(pv) = (maximum(abs.(pv.trans_specs[1].log_w)), maximum(abs.(pv.trans_specs[2].log_w)))
dpibar(pv) = maximum(maximum(abs.(marginalize_Pi_v3(pv.trans_specs,TG,pv.π_Pω,r_θ,r_I,pv.r_P,pv.r_ω)[g[1]] .- PIBAR0[g[1]])) for g in TG)

# Build a start with randomized transition layer (tests basin robustness)
function rand_trans_start(seed::Int)
    rng = MersenneTwister(seed)
    K = PV0.trans_specs[1].K; ν = PV0.trans_specs[1].ν
    ts = [initialize_robit_nn_v3(r_θ, K; rng, ν) for _ in TG]
    merge(PV0, (; trans_specs = ts))
end

# One continuation pass; returns (rows, per-stage params)
function continuation(pv_init, psk_init, label)
    println("\n### continuation start: $label ###")
    @printf("  %-8s %12s %10s %9s %9s %9s\n","λ","LL","ΔLL","lw1","lw2","maxΔΠ̄"); flush(stdout)
    pv, psk = pv_init, psk_init
    rows = NamedTuple[]; stage_pv = Any[]; prev_ll = NaN
    for λ in SCHEDULE
        pv, psk, _,_,_, ll = expectation_maximization_v3_eps(
            DATA, pv, psk; transition_groups=TG, maxiter=N_INNER, tol=1e-12,
            verbose=false, λ_logw=λ)
        lw1, lw2 = maxlw(pv); dΠ = dpibar(pv)
        Δ = isnan(prev_ll) ? NaN : ll - prev_ll; prev_ll = ll
        @printf("  %-8.4g %12.1f %10.2f %9.1f %9.1f %9.4f\n", λ, ll, Δ, lw1, lw2, dΠ); flush(stdout)
        push!(rows, (; λ, ll, lw1, lw2, dΠ)); push!(stage_pv, (pv, psk))
    end
    rows, stage_pv
end

# ── Run from standard init + 2 random-transition starts ──
rows_main, stages_main = continuation(PV0, PSK0, "current-fit (de-degenerated by large λ)")
rows_r1, _ = continuation(rand_trans_start(101), PSK0, "random-transition seed 101")
rows_r2, _ = continuation(rand_trans_start(202), PSK0, "random-transition seed 202")

# Basin-robustness: best LL reached by each start
bestll(rows) = maximum(r.ll for r in rows)
@printf("\nBasin check — best LL along schedule:  main=%.1f  rand1=%.1f  rand2=%.1f  (spread=%.2f)\n",
        bestll(rows_main), bestll(rows_r1), bestll(rows_r2),
        maximum([bestll(rows_main),bestll(rows_r1),bestll(rows_r2)]) -
        minimum([bestll(rows_main),bestll(rows_r1),bestll(rows_r2)]))

# ── Pick λ*: largest λ within PLATEAU of the best LL (main start) ──
llmax = bestll(rows_main)
elig  = [r for r in rows_main if r.ll >= llmax - PLATEAU]
λstar = maximum(r.λ for r in elig)            # largest λ on the plateau
istar = findfirst(r -> r.λ == λstar, rows_main)
@printf("\nλ* = %.4g  (LL=%.1f, within %.1f of best %.1f; lw=%.1f/%.1f, maxΔΠ̄=%.4f)\n",
        λstar, rows_main[istar].ll, PLATEAU, llmax,
        rows_main[istar].lw1, rows_main[istar].lw2, rows_main[istar].dΠ)

# ── Converge λ* to a final fit, warm-started from its stage params ──
println("\nConverging final fit at λ*=$λstar ...")
pv_s, psk_s = stages_main[istar]
pvf, pskf, γf, ξf, wf, llf = expectation_maximization_v3_eps(
    DATA, pv_s, psk_s; transition_groups=TG, maxiter=500, tol=1e-4, verbose=false, λ_logw=λstar)
@printf("  final: LL=%.1f (vs current MLE %.1f, Δ=%+.1f)  lw=%.1f/%.1f  maxΔΠ̄=%.4f  β₅/δ=%.4f\n",
        llf, mle["best_ll"], llf-mle["best_ll"], maxlw(pvf)..., dpibar(pvf),
        pvf.ccp_spec.beta5/pvf.income_model.delta)

const OUT = joinpath(REPO,"output","estimation_v3_eps_reg.jld2")
jldsave(OUT; params_v3_eps=pvf, params_skill=pskf, gamma_agg=γf, xi_sum_z=ξf, w_z=wf,
        best_ll=llf, lambda_logw=λstar, schedule=SCHEDULE, n_inner=N_INNER,
        continuation_main=rows_main, continuation_r1=rows_r1, continuation_r2=rows_r2,
        data_dims=(; N=DATA.N, T=PSK0.T, r_θ, r_I, r_P=3, r_ω=3, r_Y=3, K=PV0.trans_specs[1].K),
        age_ranges=AGES, transition_groups=TG)
println("\nSaved regularized fit → $OUT")
