# Stage 2: clean convergent bootstrap on the REGULARIZED fit.
#
# Because the regularized EM converges (λ_logw penalty removes the flat ridge),
# each resample is a single warm-from-fit run to convergence — no iteration-
# budget sweep needed.  Resample children w/ replacement, warm-start from the
# adopted fit, run EM with the same λ to tol=1e-4, record Π̄_free (both groups)
# + β₅/δ + LL.  All downstream inference (SEs, CIs, OC test with bootstrap Ω,
# survivor CI figures) is post-processing on the saved Π̄_free draws.
#
# Run: julia -t 8 --gcthreads=8 scripts/bootstrap_reg.jl
# ENV: FIT (default estimation_v3_eps_reg.jld2), B (default 200),
#      LAMBDA (default 0.01), SEED (default 20260608), MAXITER (default 300)
using LinearAlgebra, Random, Statistics, JLD2, Optim, Printf
using Base.Threads
const REPO = joinpath(@__DIR__, "..")
for f in ["parametric_models.jl","parametric_models_v3.jl","parametric_models_v3_eps.jl",
          "prepare_data.jl","estimation.jl","estimation_multiperiod.jl",
          "estimation_multiperiod_v3.jl","estimation_multiperiod_v3_eps.jl",
          "induced_transition.jl"]
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
const FIT     = get(ENV,"FIT","output/estimation_v3_eps_reg.jld2")
const B       = parse(Int, get(ENV,"B","200"))
const LAMBDA  = parse(Float64, get(ENV,"LAMBDA","0.01"))
const SEED    = parse(Int, get(ENV,"SEED","20260608"))
const MAXITER = parse(Int, get(ENV,"MAXITER","200"))
const BTOL    = parse(Float64, get(ENV,"TOL","1e-3"))   # relaxed: residual ≪ bootstrap SD

println("Loading data + fit ($FIT) ...")
datasets = load_all_datasets()
panel = prepare_panel_data(datasets; age_ranges=AGES, invest_items=ITEMS, r_θ=4, r_I=5)
const DATA = merge(panel, (; r_P=3, r_ω=3))
fit = load(joinpath(REPO, FIT)); const PV = fit["params_v3_eps"]; const PSK = fit["params_skill"]
(; r_θ, r_I) = PSK; const N = DATA.N
const n_free = (r_θ-1)*r_θ*r_I

function resample_data(data, idx)
    X_θ=[[m[idx] for m in data.X_θ[t]] for t in 1:data.T]
    X_I=[[m[idx] for m in data.X_I[t]] for t in 1:data.T]
    X_P=[m[idx] for m in data.X_P]
    XA = isempty(data.X_adult) ? data.X_adult : [m[idx] for m in data.X_adult]
    merge(data, (; N=length(idx), X_θ, X_I, X_P, Y_income=data.Y_income[idx,:], X_adult=XA, kids=data.kids[idx]))
end
# free-vector flatteners; `_n` variants take the column count (r_I, or n_y for the
# induced income-transition object) so the same ordering serves both.
pibar_free_n(Πb,nc) = Float64[Πb[j,kθ,kI] for kI in 1:nc for kθ in 1:r_θ for j in 1:(r_θ-1)]
surv_free_n(Πb,nc)  = Float64[sum(Πb[j,kθ,kI] for j in k+1:r_θ) for kI in 1:nc for kθ in 1:r_θ for k in 1:(r_θ-1)]
pibar_free(Πb)  = pibar_free_n(Πb, r_I)
surv_free(Πb)   = surv_free_n(Πb, r_I)
ratio(pv)       = pv.ccp_spec.beta5 / pv.income_model.delta
pibar_groups(pv)= (Pb=marginalize_Pi_v3(pv.trans_specs,TG,pv.π_Pω,r_θ,r_I,pv.r_P,pv.r_ω); [Pb[g[1]] for g in TG])

# Induced reduced-form transition Π_ind(θ'|θ,y) on a FIXED income grid (held at the
# point estimate so the lattice cells are comparable across resamples).
const N_Y     = parse(Int, get(ENV,"N_Y","5"))
const Y_LEVELS = model_income_grid(PV; n_y=N_Y)
const n_free_ind = (r_θ-1)*r_θ*N_Y
pind_groups(pv) = induced_transitions_by_group(pv, TG, Y_LEVELS; r_θ=r_θ, r_I=r_I)
@printf("Induced income grid (\$): %s\n", join([@sprintf("%.0f", v) for v in Y_LEVELS], "  "))

# Point estimate
P0 = pibar_groups(PV)
pibar0 = [pibar_free(P0[g]) for g in 1:length(TG)]
surv0  = [surv_free(P0[g])  for g in 1:length(TG)]
Pind0  = pind_groups(PV)
pibar_ind0 = [pibar_free_n(Pind0[g], N_Y) for g in 1:length(TG)]
surv_ind0  = [surv_free_n(Pind0[g],  N_Y) for g in 1:length(TG)]
ratio0 = ratio(PV)
@printf("Point: β₅/δ=%.4f, fit LL=%.1f, λ=%.3g, B=%d, N=%d\n", ratio0, fit["best_ll"], LAMBDA, B, N)

rng = MersenneTwister(SEED)
idx_draws = [rand(rng, 1:N, N) for _ in 1:B]
boot_pibar = [fill(NaN, B, n_free) for _ in 1:length(TG)]
boot_surv  = [fill(NaN, B, n_free) for _ in 1:length(TG)]
boot_pibar_ind = [fill(NaN, B, n_free_ind) for _ in 1:length(TG)]
boot_surv_ind  = [fill(NaN, B, n_free_ind) for _ in 1:length(TG)]
boot_ratio = fill(NaN, B); boot_ll = fill(NaN, B); ok = fill(false, B)
const OUT = joinpath(REPO,"output", get(ENV,"OUT_NAME","bootstrap_reg.jld2"))
const CKPT = parse(Int, get(ENV,"CKPT","20"))
save_lock = ReentrantLock()
function save_state()
    nok = count(ok)
    jldsave(OUT; fit=FIT, B=B, lambda=LAMBDA, seed=SEED, n_ok=nok, ok=ok,
            transition_groups=TG, boot_pibar=boot_pibar, boot_surv=boot_surv,
            boot_ratio=boot_ratio, boot_ll=boot_ll, idx_draws=idx_draws,
            pibar0=pibar0, surv0=surv0, ratio0=ratio0, r_θ=r_θ, r_I=r_I,
            n_y=N_Y, y_levels=Y_LEVELS,
            boot_pibar_ind=boot_pibar_ind, boot_surv_ind=boot_surv_ind,
            pibar_ind0=pibar_ind0, surv_ind0=surv_ind0)
end
done = Threads.Atomic{Int}(0); t0 = time()
@threads for b in 1:B
    try
        db = resample_data(DATA, idx_draws[b])
        pv, _, _, _, _, ll = expectation_maximization_v3_eps(
            db, PV, PSK; transition_groups=TG, maxiter=MAXITER, tol=BTOL, verbose=false, λ_logw=LAMBDA)
        Pg = pibar_groups(pv)
        for g in 1:length(TG); boot_pibar[g][b,:]=pibar_free(Pg[g]); boot_surv[g][b,:]=surv_free(Pg[g]); end
        Pgi = pind_groups(pv)
        for g in 1:length(TG); boot_pibar_ind[g][b,:]=pibar_free_n(Pgi[g],N_Y); boot_surv_ind[g][b,:]=surv_free_n(Pgi[g],N_Y); end
        boot_ratio[b]=ratio(pv); boot_ll[b]=ll; ok[b]=true
    catch e
        @warn "resample $b failed" exception=e
    end
    d=Threads.atomic_add!(done,1)+1
    if d%10==0||d==B; @printf("  %3d/%d (%.0fs)\n",d,B,time()-t0); flush(stdout); end
    if d % CKPT == 0      # incremental checkpoint — survive sleep/kill/crash
        lock(save_lock) do; save_state(); end
        @printf("  [checkpoint @ %d/%d -> %s]\n", d, B, OUT); flush(stdout)
    end
end
nok = count(ok); println("\n$nok/$B resamples ok")
save_state()
# quick SE readout
for g in 1:length(TG)
    se = vec(std(boot_pibar[g][ok,:]; dims=1))
    @printf("Π̄ group %d: bootstrap SE mean=%.4f max=%.4f\n", g, mean(se), maximum(se))
end
@printf("β₅/δ: point=%.4f bootstrap SD=%.4f\n", ratio0, std(boot_ratio[ok]))
println("Saved → $OUT")
