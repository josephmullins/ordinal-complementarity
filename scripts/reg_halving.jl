# Stage 1 (replicable, efficient): successive-halving multi-start + λ-continuation.
#
# Combine graduated regularization with Hyperband-style successive halving:
# the descending-λ schedule IS the rung structure.  All N0 random starts run a
# short budget at the first (large) λ; the worst are pruned; survivors warm-start
# into the next (smaller) λ with the same short budget; prune again; ... ; the
# final survivors are run to convergence at λ_end.  Survivors thus accumulate
# more relaxation before each cut (mitigating the under-relaxation that corrupts
# a single end-of-run ranking), while pruned starts stop early (cheaper).
#
# Pruning is GENTLE early (signal is noisy under heavy reg / few iters) and
# aggressive late (signal reliable).  Seeds documented ⇒ fully replicable.
#
# Run: julia -t 8 --gcthreads=8 scripts/reg_halving.jl
# Config via ENV:
#   N0        starts at rung 1                (default 128)
#   N_INNER   EM iters per rung                (default 20)
#   SCHEDULE  descending λ (last = converge)   (default 10,3,1,0.3,0.1,0.03,0.01)
#   KEEP      survivors after each rung (len = #stages-1)  (default 96,64,32,16,8,4)
#   SEED0     base seed                        (default 2000)
using LinearAlgebra, Random, Statistics, JLD2, Optim, Printf
using Base.Threads
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
const N0       = parse(Int, get(ENV,"N0","128"))
const N_INNER  = parse(Int, get(ENV,"N_INNER","20"))
const SCHEDULE = [parse(Float64,s) for s in split(get(ENV,"SCHEDULE","10,3,1,0.3,0.1,0.03,0.01"),',')]
const KEEP     = [parse(Int,s) for s in split(get(ENV,"KEEP","96,64,32,16,8,4"),',')]
const SEED0    = parse(Int, get(ENV,"SEED0","2000"))
const λ_END    = SCHEDULE[end]
@assert length(KEEP) == length(SCHEDULE)-1 "KEEP must have one fewer entry than SCHEDULE"

println("Loading data + (benchmark) MLE ...")
datasets = load_all_datasets()
panel = prepare_panel_data(datasets; age_ranges=AGES, invest_items=ITEMS, r_θ=4, r_I=5)
const DATA = merge(panel, (; r_P=3, r_ω=3))
mle = load(joinpath(REPO,"output","estimation_v3_eps.jld2"))
const PV0, PSK0, MLE_LL = mle["params_v3_eps"], mle["params_skill"], mle["best_ll"]
(; r_θ, r_I) = PSK0
const KK = PV0.trans_specs[1].K; const NU = PV0.trans_specs[1].ν
const PIBAR0 = marginalize_Pi_v3(PV0.trans_specs, TG, PV0.π_Pω, r_θ, r_I, PV0.r_P, PV0.r_ω)
maxlw(pv) = max(maximum(abs.(pv.trans_specs[1].log_w)), maximum(abs.(pv.trans_specs[2].log_w)))
dpibar(pv) = maximum(maximum(abs.(marginalize_Pi_v3(pv.trans_specs,TG,pv.π_Pω,r_θ,r_I,pv.r_P,pv.r_ω)[g[1]] .- PIBAR0[g[1]])) for g in TG)
rand_trans_start(seed) = merge(PV0, (; trans_specs=[initialize_robit_nn_v3(r_θ, KK; rng=MersenneTwister(seed), ν=NU) for _ in TG]))

# Wrapped in a function so @threads sees a proper local scope for `active`.
function main()
    pv  = Vector{Any}(undef, N0); psk = Vector{Any}(undef, N0); ll = fill(-Inf, N0)
    extra = Vector{Any}(undef, N0)   # (γ,ξ,w) for converged finalists
    active = collect(1:N0)

    println("\nSuccessive halving: N0=$N0, schedule=$SCHEDULE, KEEP=$KEEP, N_INNER=$N_INNER, threads=$(nthreads())")
    println("MLE benchmark LL = $(round(MLE_LL; digits=1))"); flush(stdout)
    t0 = time()
    for (s, λ) in enumerate(SCHEDULE)
        islast = s == length(SCHEDULE)
        maxit  = islast ? 500 : N_INNER
        tol_s  = islast ? 1e-4 : 1e-12
        idxs = copy(active)
        @threads for idx in 1:length(idxs)
            i = idxs[idx]
            src_pv  = s == 1 ? rand_trans_start(SEED0+i) : pv[i]
            src_psk = s == 1 ? PSK0 : psk[i]
            p, k, γ, ξ, w, l = expectation_maximization_v3_eps(
                DATA, src_pv, src_psk; transition_groups=TG, maxiter=maxit, tol=tol_s,
                verbose=false, λ_logw=λ)
            pv[i]=p; psk[i]=k; ll[i]=l; if islast; extra[i]=(γ,ξ,w); end
        end
        lls = [ll[i] for i in active]
        @printf("  rung %d  λ=%-6.4g  n=%-4d  best=%.1f  median=%.1f  worst=%.1f  (%.0fs)\n",
                s, λ, length(active), maximum(lls), median(lls), minimum(lls), time()-t0); flush(stdout)
        if !islast
            keep = min(KEEP[s], length(active))
            active = active[sortperm(lls; rev=true)[1:keep]]   # survivors carry their warm state
        end
    end

    b = active[argmax([ll[i] for i in active])]
    pvf, pskf = pv[b], psk[b]; γf, ξf, wf = extra[b]
    println("\nFinalists (converged): ", [(SEED0+i, round(ll[i];digits=1)) for i in active])
    @printf("BEST (halving): seed %d  LL=%.1f  (MLE bench %.1f, Δ=%+.1f)  max|lw|=%.1f  maxΔΠ̄=%.3f  β₅/δ=%.4f  λ=%.3g\n",
            SEED0+b, ll[b], MLE_LL, ll[b]-MLE_LL, maxlw(pvf), dpibar(pvf),
            pvf.ccp_spec.beta5/pvf.income_model.delta, λ_END)

    OUT = joinpath(REPO,"output","estimation_v3_eps_reg_halving.jld2")
    jldsave(OUT; params_v3_eps=pvf, params_skill=pskf, gamma_agg=γf, xi_sum_z=ξf, w_z=wf,
            best_ll=ll[b], lambda_logw=λ_END, schedule=SCHEDULE, keep=KEEP, n_inner=N_INNER,
            n0=N0, seed0=SEED0, best_seed=SEED0+b, mle_benchmark_ll=MLE_LL,
            finalist_seeds=[SEED0+i for i in active], finalist_lls=[ll[i] for i in active],
            data_dims=(; N=DATA.N, T=PSK0.T, r_θ, r_I, r_P=3, r_ω=3, r_Y=3, K=KK),
            age_ranges=AGES, transition_groups=TG)
    println("Saved → $OUT")
end
main()
