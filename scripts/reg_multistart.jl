# Stage 1 (replicable): multi-start + graduated regularization.
#
# Fully reproducible workflow (no dependence on the existing MLE as a starting
# point — only as a benchmark).  For each of N random transition-layer starts:
#   • run a descending-λ continuation (graduated non-convexity) to a common
#     conditioned λ_end, recording the unpenalized LL there;
# then rank starts by that LL, converge the top few fully at λ_end, and keep
# the best.  Seeds are documented ⇒ the whole thing replays.
#
# Parallel across starts (independent).  Run with: julia -t 8 --gcthreads=8.
#
# Config via ENV:
#   N_STARTS  number of random starts        (default 24)
#   N_INNER   EM iters per λ stage            (default 20)
#   SCHEDULE  descending λ, ends at λ_end      (default 10,3,1,0.3,0.1,0.03,0.01)
#   N_TOP     top starts to converge fully     (default 3)
#   SEED0     base seed (starts use SEED0+i)   (default 1000)
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
const N_STARTS = parse(Int, get(ENV,"N_STARTS","24"))
const N_INNER  = parse(Int, get(ENV,"N_INNER","20"))
const SCHEDULE = [parse(Float64,s) for s in split(get(ENV,"SCHEDULE","10,3,1,0.3,0.1,0.03,0.01"),',')]
const λ_END    = SCHEDULE[end]
const N_TOP    = parse(Int, get(ENV,"N_TOP","3"))
const SEED0    = parse(Int, get(ENV,"SEED0","1000"))

println("Loading data + (benchmark) MLE ...")
datasets = load_all_datasets()
panel = prepare_panel_data(datasets; age_ranges=AGES, invest_items=ITEMS, r_θ=4, r_I=5)
const DATA = merge(panel, (; r_P=3, r_ω=3))
mle = load(joinpath(REPO,"output","estimation_v3_eps.jld2"))
const PV0, PSK0 = mle["params_v3_eps"], mle["params_skill"]
const MLE_LL = mle["best_ll"]
(; r_θ, r_I) = PSK0
const KK = PV0.trans_specs[1].K; const NU = PV0.trans_specs[1].ν
const PIBAR0 = marginalize_Pi_v3(PV0.trans_specs, TG, PV0.π_Pω, r_θ, r_I, PV0.r_P, PV0.r_ω)
maxlw(pv) = max(maximum(abs.(pv.trans_specs[1].log_w)), maximum(abs.(pv.trans_specs[2].log_w)))
dpibar(pv) = maximum(maximum(abs.(marginalize_Pi_v3(pv.trans_specs,TG,pv.π_Pω,r_θ,r_I,pv.r_P,pv.r_ω)[g[1]] .- PIBAR0[g[1]])) for g in TG)

# Random transition-layer start.  Non-transition params come from PV0 but are
# re-fit every M-step and are well-identified, so the converged fit does not
# meaningfully depend on them — only the (multimodal) transition layer is
# randomized, which is what we are exploring.
function rand_trans_start(seed::Int)
    rng = MersenneTwister(seed)
    ts = [initialize_robit_nn_v3(r_θ, KK; rng, ν=NU) for _ in TG]
    merge(PV0, (; trans_specs = ts))
end

# Continuation for one start → params + LL at λ_END (common conditioned point).
function continue_one(seed::Int)
    pv = rand_trans_start(seed); psk = PSK0; ll = NaN
    for λ in SCHEDULE
        pv, psk, _,_,_, ll = expectation_maximization_v3_eps(
            DATA, pv, psk; transition_groups=TG, maxiter=N_INNER, tol=1e-12,
            verbose=false, λ_logw=λ)
    end
    (ll, pv, psk)
end

println("\nMulti-start: $N_STARTS random starts, continuation $(SCHEDULE) (λ_end=$λ_END), " *
        "N_INNER=$N_INNER, threads=$(nthreads())")
println("MLE benchmark LL = $(round(MLE_LL; digits=1))")
start_ll  = fill(-Inf, N_STARTS)
start_pv  = Vector{Any}(undef, N_STARTS)
start_psk = Vector{Any}(undef, N_STARTS)
done = Threads.Atomic{Int}(0); t0 = time()
@threads for i in 1:N_STARTS
    seed = SEED0 + i
    ll, pv, psk = continue_one(seed)
    start_ll[i] = ll; start_pv[i] = pv; start_psk[i] = psk
    d = Threads.atomic_add!(done,1)+1
    @printf("  [%2d/%d] seed=%d  LL@λ%.3g=%.1f  max|lw|=%.1f  maxΔΠ̄=%.3f  (%.0fs)\n",
            d, N_STARTS, seed, λ_END, ll, maxlw(pv), dpibar(pv), time()-t0); flush(stdout)
end

@printf("\nStart LL@λ_end distribution: min=%.1f  median=%.1f  max=%.1f  (MLE bench %.1f)\n",
        minimum(start_ll), median(start_ll), maximum(start_ll), MLE_LL)
order = sortperm(start_ll; rev=true)
println("Top $N_TOP starts by LL@λ_end: ", [(SEED0+order[j], round(start_ll[order[j]];digits=1)) for j in 1:N_TOP])

# Converge the top N_TOP fully at λ_END, pick the global best.
println("\nConverging top $N_TOP fully at λ=$λ_END ...")
conv_ll = fill(-Inf, N_TOP); conv_pv = Vector{Any}(undef,N_TOP); conv_psk = Vector{Any}(undef,N_TOP); conv_extra = Vector{Any}(undef,N_TOP)
@threads for j in 1:N_TOP
    i = order[j]
    pv, psk, γ, ξ, w, ll = expectation_maximization_v3_eps(
        DATA, start_pv[i], start_psk[i]; transition_groups=TG, maxiter=500, tol=1e-4,
        verbose=false, λ_logw=λ_END)
    conv_ll[j]=ll; conv_pv[j]=pv; conv_psk[j]=psk; conv_extra[j]=(γ,ξ,w)
    @printf("  top %d (seed %d): converged LL=%.1f  max|lw|=%.1f  maxΔΠ̄=%.3f  β₅/δ=%.4f\n",
            j, SEED0+i, ll, maxlw(pv), dpibar(pv), pv.ccp_spec.beta5/pv.income_model.delta); flush(stdout)
end
b = argmax(conv_ll)
pvf, pskf = conv_pv[b], conv_psk[b]; γf,ξf,wf = conv_extra[b]
@printf("\nBEST replicable fit: LL=%.1f  (MLE bench %.1f, Δ=%+.1f)  max|lw|=%.1f  maxΔΠ̄=%.3f  λ=%.3g\n",
        conv_ll[b], MLE_LL, conv_ll[b]-MLE_LL, maxlw(pvf), dpibar(pvf), λ_END)

const OUT = joinpath(REPO,"output","estimation_v3_eps_reg.jld2")
jldsave(OUT; params_v3_eps=pvf, params_skill=pskf, gamma_agg=γf, xi_sum_z=ξf, w_z=wf,
        best_ll=conv_ll[b], lambda_logw=λ_END, schedule=SCHEDULE, n_inner=N_INNER,
        n_starts=N_STARTS, seed0=SEED0, best_seed=SEED0+order[b],
        start_ll=start_ll, mle_benchmark_ll=MLE_LL,
        data_dims=(; N=DATA.N, T=PSK0.T, r_θ, r_I, r_P=3, r_ω=3, r_Y=3, K=KK),
        age_ranges=AGES, transition_groups=TG)
println("Saved best replicable fit → $OUT")
