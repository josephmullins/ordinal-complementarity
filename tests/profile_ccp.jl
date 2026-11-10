using LinearAlgebra, Random, Statistics

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3.jl"))
include(joinpath(@__DIR__, "..", "src", "prepare_data.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod_v3.jl"))

# Load data
datasets = load_all_datasets()
panel_data = prepare_panel_data(datasets;
    age_ranges = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13)],
    invest_items = [
        [:HowOftenRead,:HowManyBooks,:HowManyToysA,:HowManyToysB,:OutOfHouse,:HowOftenTalks,:HowOftenGrocery],
        [:HowOftenRead,:HowManyBooks,:HowManyToysA,:HowManyToysB,:OutOfHouse,:HowOftenTalks,:HowOftenGrocery,:HowManyZines,:HowOftenOuting,:FoodChoice,:LearnNumbers,:LearnAlphabet,:LearnColors,:LearnShapes,:RecordTapeCD],
        [:HowOftenRead,:HowManyBooks,:HowManyZines,:HowOftenOuting,:HowOftenMuseum,:RecordTapeCD,:FoodChoice],
        [:HowOftenRead,:HowManyBooks,:HowOftenMuseum,:HowOftenTheater,:HowOftenFamily,:HowOftenOutdoors,:ChildMusic,:SpecialActivities,:Hobbies,:EnjoyRead,:DailyNewspaper,:DiscussTV,:GradesTeacher,:GradesTalk,:GradesHelp,:GradesStudy],
        [:HowOftenRead,:HowManyBooks,:HowOftenMuseum,:HowOftenTheater,:HowOftenFamily,:HowOftenOutdoors,:HowOftenFather,:ChildMusic,:SpecialActivities,:Hobbies,:EnjoyRead,:DailyNewspaper,:DiscussTV,:GradesTeacher,:GradesTalk,:GradesHelp,:GradesStudy],
        [:HowManyBooks,:HowOftenMuseum,:HowOftenTheater,:HowOftenFamily,:HowOftenOutdoors,:HowOftenFather,:ChildMusic,:SpecialActivities,:Hobbies,:EnjoyRead,:DailyNewspaper,:DiscussTV,:GradesTeacher,:GradesTalk,:GradesHelp,:GradesStudy,:ToldPositive,:HowOftenClean,:HowOftenChores,:HowOftenTime],
        [:HowManyBooks,:HowOftenMuseum,:HowOftenTheater,:HowOftenFamily,:HowOftenOutdoors,:HowOftenFather,:ChildMusic,:SpecialActivities,:Hobbies,:EnjoyRead,:DailyNewspaper,:DiscussTV,:GradesTeacher,:GradesTalk,:GradesHelp,:GradesStudy,:ToldPositive,:HowOftenClean,:HowOftenChores,:HowOftenTime],
    ],
    r_θ=4, r_I=5,
)

(; r_θ, r_I, T, N) = panel_data
r_P = 3; r_ω = 3; r_Z = r_P * r_ω
data_v3 = merge(panel_data, (; r_P, r_ω))

# Initialize and run one E-step to fill buffers
rng = MersenneTwister(42)
pv3, psk = initialize_params_v3(data_v3, (K=4, ν=7.0, n_groups=2); rng)

γ_agg      = zeros(T, r_θ, r_I, N)
xi_sum_z   = [[zeros(r_θ, r_I, r_θ, r_I) for _ in 1:T-1] for _ in 1:r_Z]
w_z        = zeros(r_Z, N)
gamma_z_t1 = zeros(r_θ, r_I, r_Z, N)
ccp_xi_z   = zeros(r_I, r_I, r_θ, r_Z, T-1, N)
pi0_ss     = zeros(r_θ, r_P, r_ω)

θ_grid = collect(1.0:r_θ); I_grid = collect(1.0:r_I)
P_grid = collect(1.0:r_P); ω_grid = collect(1.0:r_ω)

Pi_z_pre = precompute_Pi_z(pv3.trans_specs, [[1,2,3],[4,5,6]],
                             r_θ, r_I, r_P, r_ω; θ_grid, I_grid, P_grid, ω_grid)
expectation_step_v3!(γ_agg, xi_sum_z, w_z, gamma_z_t1, ccp_xi_z, pi0_ss,
                      Pi_z_pre, pv3, psk, data_v3; θ_grid, I_grid, P_grid, ω_grid)

# Profile one call to ccp_negll_v3
x0 = pack_ccp_v3(pv3.ccp_spec)

# Warmup
ccp_negll_v3(x0, ccp_xi_z, gamma_z_t1, w_z, data_v3.Y_income,
             r_θ, r_I, r_P, r_ω, T; θ_grid, I_grid, P_grid, ω_grid)

println("\n@time for one call to ccp_negll_v3:")
@time ccp_negll_v3(x0, ccp_xi_z, gamma_z_t1, w_z, data_v3.Y_income,
                    r_θ, r_I, r_P, r_ω, T; θ_grid, I_grid, P_grid, ω_grid)

println("\n@time again:")
@time ccp_negll_v3(x0, ccp_xi_z, gamma_z_t1, w_z, data_v3.Y_income,
                    r_θ, r_I, r_P, r_ω, T; θ_grid, I_grid, P_grid, ω_grid)
