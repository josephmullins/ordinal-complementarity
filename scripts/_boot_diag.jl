# Diagnostic: how many EM iters does a warm-from-fit bootstrap resample need,
# and how long per iter (single-threaded, clean timing)?
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
datasets = load_all_datasets()
panel = prepare_panel_data(datasets; age_ranges=AGES, invest_items=ITEMS, r_θ=4, r_I=5)
DATA = merge(panel, (; r_P=3, r_ω=3))
fit = load("output/estimation_v3_eps_reg.jld2"); PV=fit["params_v3_eps"]; PSK=fit["params_skill"]
N = DATA.N
function resample_data(data, idx)
    X_θ=[[m[idx] for m in data.X_θ[t]] for t in 1:data.T]; X_I=[[m[idx] for m in data.X_I[t]] for t in 1:data.T]
    X_P=[m[idx] for m in data.X_P]; XA=[m[idx] for m in data.X_adult]
    merge(data,(;N=length(idx),X_θ,X_I,X_P,Y_income=data.Y_income[idx,:],X_adult=XA,kids=data.kids[idx]))
end
rng=MersenneTwister(1)
for b in 1:3
    idx=rand(rng,1:N,N); db=resample_data(DATA,idx)
    println("\n### resample $b ###")
    t=@elapsed pv,_,_,_,_,ll = expectation_maximization_v3_eps(db,PV,PSK; transition_groups=TG, maxiter=400, tol=1e-4, verbose=true, λ_logw=0.01)
    @printf("resample %d: %.0fs total, ll=%.1f, β₅/δ=%.4f\n", b, t, ll, pv.ccp_spec.beta5/pv.income_model.delta)
end
