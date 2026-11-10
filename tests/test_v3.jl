"""
V3 validation tests:
  1. Smoke test — V3 loads, initializes, and runs a few EM iterations
  2. V2 replication — with r_P=1, r_ω=1 and degenerate regime, V3 E-step
     reproduces V2 log-likelihoods
  3. Regime mechanics — posteriors sum correctly, γ_agg is proper
"""

using LinearAlgebra, Random, Statistics, Test

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "parametric_models_v3.jl"))
include(joinpath(@__DIR__, "..", "src", "prepare_data.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation_multiperiod_v3.jl"))

# ─────────────────────────────────────────────────────────────────
# Load real data (same config as V2 settled spec)
# ─────────────────────────────────────────────────────────────────

println("Loading datasets...")
datasets = load_all_datasets()

const PANEL_AGE_RANGES = [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13)]
const PANEL_INVEST_ITEMS = [
    [:HowOftenRead, :HowManyBooks, :HowManyToysA, :HowManyToysB,
     :OutOfHouse, :HowOftenTalks, :HowOftenGrocery],
    [:HowOftenRead, :HowManyBooks, :HowManyToysA, :HowManyToysB,
     :OutOfHouse, :HowOftenTalks, :HowOftenGrocery,
     :HowManyZines, :HowOftenOuting, :FoodChoice,
     :LearnNumbers, :LearnAlphabet, :LearnColors, :LearnShapes, :RecordTapeCD],
    [:HowOftenRead, :HowManyBooks, :HowManyZines, :HowOftenOuting,
     :HowOftenMuseum, :RecordTapeCD, :FoodChoice],
    [:HowOftenRead, :HowManyBooks, :HowOftenMuseum, :HowOftenTheater,
     :HowOftenFamily, :HowOftenOutdoors, :ChildMusic, :SpecialActivities,
     :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
     :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy],
    [:HowOftenRead, :HowManyBooks, :HowOftenMuseum, :HowOftenTheater,
     :HowOftenFamily, :HowOftenOutdoors, :HowOftenFather, :ChildMusic,
     :SpecialActivities, :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
     :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy],
    [:HowManyBooks, :HowOftenMuseum, :HowOftenTheater, :HowOftenFamily,
     :HowOftenOutdoors, :HowOftenFather, :ChildMusic, :SpecialActivities,
     :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
     :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy,
     :ToldPositive, :HowOftenClean, :HowOftenChores, :HowOftenTime],
    [:HowManyBooks, :HowOftenMuseum, :HowOftenTheater, :HowOftenFamily,
     :HowOftenOutdoors, :HowOftenFather, :ChildMusic, :SpecialActivities,
     :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
     :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy,
     :ToldPositive, :HowOftenClean, :HowOftenChores, :HowOftenTime],
]

panel_data = prepare_panel_data(datasets;
    age_ranges   = PANEL_AGE_RANGES,
    invest_items = PANEL_INVEST_ITEMS,
    r_θ = 4, r_I = 5,
)

(; r_θ, r_I, T, N) = panel_data
println("  N=$N, T=$T, r_θ=$r_θ, r_I=$r_I")

const TRANSITION_GROUPS = [[1,2,3],[4,5,6]]

# ═════════════════════════════════════════════════════════════════
# TEST 1: Smoke test — V3 initializes and runs
# ═════════════════════════════════════════════════════════════════

@testset "V3 Smoke Test" begin
    println("\n" * "="^60)
    println("  TEST 1: V3 smoke test (r_P=3, r_ω=3)")
    println("="^60)

    data_v3 = merge(panel_data, (; r_P=3, r_ω=3))
    config = (K=4, ν=7.0, n_groups=2)
    rng = MersenneTwister(42)

    # Initialize
    pv3, psk = initialize_params_v3(data_v3, config; rng)

    @test length(pv3.trans_specs) == 2
    @test pv3.trans_specs[1].K == 4
    @test size(pv3.π0) == (r_θ, 3, 3)
    @test length(pv3.Q_P) == length(data_v3.κ_P)
    @test psk.r_θ == r_θ
    @test psk.r_I == r_I

    # Run 3 EM iterations
    pv3_out, psk_out, γ, ξz, wz, ll = expectation_maximization_v3(
        data_v3, pv3, psk;
        transition_groups = TRANSITION_GROUPS,
        maxiter = 3, tol = 0.0, verbose = true)

    @test isfinite(ll)
    @test ll < 0  # log-likelihood should be negative
    @test size(γ) == (T, r_θ, r_I, N)
    @test size(wz) == (9, N)

    # Check regime posteriors sum to 1
    for n in 1:min(N, 100)
        @test sum(wz[:, n]) ≈ 1.0 atol=1e-10
    end

    # Check γ_agg sums to ~1 for each (t, n)
    for n in 1:min(N, 100), t in 1:T
        @test sum(γ[t, :, :, n]) ≈ 1.0 atol=1e-8
    end

    println("  PASSED: V3 smoke test")
end

# ═════════════════════════════════════════════════════════════════
# TEST 2: V2 replication
#   With r_P=1, r_ω=1, V3 should reproduce V2 log-likelihoods
#   when given identical parameters.
# ═════════════════════════════════════════════════════════════════

@testset "V2 Replication" begin
    println("\n" * "="^60)
    println("  TEST 2: V2 replication (r_P=1, r_ω=1)")
    println("="^60)

    rng = MersenneTwister(123)
    K = 4; ν = 7.0

    # ── Initialize V2 params ──
    probit_θ = [initialize_ordered_probit(panel_data.κ_θ[t], r_θ; rng) for t in 1:T]
    P_θ      = [emission_matrices_from_probit(probit_θ[t], r_θ) for t in 1:T]
    probit_I = [initialize_ordered_probit(panel_data.κ_I[t], r_I; rng) for t in 1:T]
    P_I      = [emission_matrices_from_probit(probit_I[t], r_I) for t in 1:T]

    # V2 transitions (one per transition, shared within groups)
    trans_v2 = [initialize_robit_nn(r_θ, K; rng, r_I, ν) for _ in 1:T-1]
    # Make transitions within groups identical (as V2 grouped does)
    for group in TRANSITION_GROUPS
        for t in group[2:end]
            trans_v2[t] = trans_v2[group[1]]
        end
    end

    # Build V2 Π arrays
    Π_v2 = [_build_robit_nn_pi(pack_robit_nn(trans_v2[t]), r_θ, r_I, K, ν) for t in 1:T-1]

    # V2 CCP: ordered probit simple — P(I'|θ', I_old)
    # We need to get the CCP from V2's format. Build a simple one.
    CCP_v2 = [let
        ccp = zeros(r_I, r_I, r_θ)
        for kθ in 1:r_θ, kI in 1:r_I
            pmf = ones(r_I) / r_I  # uniform for testing
            ccp[:, kI, kθ] = pmf
        end
        ccp
    end for _ in 1:T-1]

    # V2 π0: joint (r_θ, r_I)
    π0_v2 = ones(r_θ, r_I) / (r_θ * r_I)

    params_v2 = (; Π=Π_v2, CCP=CCP_v2, π0=π0_v2, P_θ, P_I, r_θ, r_I, T)

    # ── Run V2 E-step ──
    γ_v2     = zeros(T, r_θ, r_I, N)
    ξ_sum_v2 = [zeros(r_θ, r_I, r_θ, r_I) for _ in 1:T-1]
    ll_v2 = expectation_step_mp!(γ_v2, ξ_sum_v2, params_v2, panel_data)
    println("  V2 log-likelihood: $(round(ll_v2; digits=6))")

    # ── Build equivalent V3 params with r_P=1, r_ω=1 ──
    r_P = 1; r_ω = 1

    # Convert V2 transitions to V3 format (alpha3=0, alpha4=0)
    trans_v3_specs = Vector{RobitNNTransitionV3}(undef, length(TRANSITION_GROUPS))
    for (g, group) in enumerate(TRANSITION_GROUPS)
        s2 = trans_v2[group[1]]
        trans_v3_specs[g] = RobitNNTransitionV3(
            s2.c_raw, s2.alpha1, s2.alpha2_raw,
            zeros(K),   # alpha3 = 0
            zeros(K),   # alpha4 = 0
            s2.b, s2.log_w, K, ν)
    end

    # V3 CCP: ordered probit with beta3=beta4=beta5=0
    # We need to match the uniform CCP. With all betas = 0, the ordered probit
    # gives P(I'≤k) = Φ(c_k), independent of inputs.
    # For uniform: P(I'=k) = 1/r_I, so P(I'≤k) = k/r_I.
    # Set cutpoints c_k = Φ⁻¹(k/r_I).
    using Distributions
    ccp_cutpoints = [quantile(Normal(), k / r_I) for k in 1:r_I-1]
    ccp_v3 = OrderedProbitCCPV3(ccp_cutpoints, 0.0, 0.0, 0.0, 0.0, 0.0)

    # V3 π0: (r_θ, 1, 1) — uniform over θ
    π0_v3 = ones(r_θ, 1, 1) / r_θ
    π_Z_v3 = ones(1, 1)

    # Degenerate Q_P (one regime, one value per measure = certain)
    # With r_P=1, ASVAB likelihoods should all be 1 regardless of data
    Q_P_v3 = [ones(panel_data.κ_P[m], 1) for m in eachindex(panel_data.κ_P)]
    # Each column sums to κ_m, but regime_loglik does log(Q_P[m][x, kP])
    # For Q_P to not affect the likelihood, we need Q_P[x, 1] to be constant across x
    # and properly normalized. With r_P=1 the normalization in mstep doesn't matter
    # for the E-step — what matters is that all observations get the same log-likelihood
    # contribution from ASVAB. Since Q_P[x, 1] = 1.0 for all x, log(1.0) = 0. Good.
    # But wait — emission matrices should be column-normalized (sum over x = 1).
    # The regime_loglik just does log(Q_P[m][x, kP]) — not normalized.
    # So Q_P[x, 1] = 1.0 gives log(1) = 0 for all x. The ASVAB measures contribute
    # nothing to regime discrimination. That's what we want.

    # Income model: set σ very large so income contributes ~nothing
    # Actually, easier: just make beta_P = 0, beta_ω = 0, and the income
    # log-likelihood is the same for all regimes. With r_P=1, r_ω=1 there's
    # only one regime so it doesn't matter — w_z will be 1.0 regardless.
    income_v3 = IncomeModel(0.0, 0.0, 0.0, log(10.0))

    params_v3 = (;
        r_P, r_ω,
        π0 = π0_v3,
        π_Z = π_Z_v3,
        trans_specs = trans_v3_specs,
        ccp_spec = ccp_v3,
        Q_P = Q_P_v3,
        income_model = income_v3,
    )
    params_skill = (; r_θ, r_I, T, P_θ, P_I, probit_θ, probit_I)

    # Degenerate data: add r_P, r_ω to panel_data
    data_v3 = merge(panel_data, (; r_P, r_ω))

    # ── Allocate V3 E-step buffers ──
    r_Z = 1
    γ_agg      = zeros(T, r_θ, r_I, N)
    xi_sum_z   = [[zeros(r_θ, r_I, r_θ, r_I) for _ in 1:T-1] for _ in 1:r_Z]
    w_z        = zeros(r_Z, N)
    gamma_z_t1 = zeros(r_θ, r_I, r_Z, N)
    pi0_ss     = zeros(r_θ, r_P, r_ω)

    # Bin income and precompute CCPs
    bin_map, bin_values = bin_income(data_v3.Y_income; n_bins=20)
    n_total_bins = length(bin_values)
    ccp_init_ss  = zeros(r_θ, r_I, r_Z, n_total_bins)
    ccp_trans_ss = zeros(r_I, r_I, r_θ, r_Z, T-1, n_total_bins)

    # Precompute V3 transitions
    Pi_z_pre = precompute_Pi_z(trans_v3_specs, TRANSITION_GROUPS,
                                 r_θ, r_I, r_P, r_ω)

    # Verify V3 Π matches V2 Π
    for t in 1:T-1
        @test Pi_z_pre[1][t] ≈ Π_v2[t] atol=1e-10
    end
    println("  V3 Π matches V2 Π: PASSED")

    # Precompute CCP arrays
    CCP_init_pre, CCP_trans_pre = precompute_CCP_z(
        ccp_v3, r_Z, r_P, r_ω, bin_values, r_θ, r_I;
        θ_grid=1.0:r_θ, I_grid=1.0:r_I, P_grid=1.0:r_P, ω_grid=1.0:r_ω)

    # ── Run V3 E-step ──
    ll_v3 = expectation_step_v3!(
        γ_agg, xi_sum_z, w_z, gamma_z_t1, ccp_init_ss, ccp_trans_ss, pi0_ss,
        Pi_z_pre, CCP_init_pre, CCP_trans_pre, bin_map,
        params_v3, params_skill, data_v3)
    println("  V3 log-likelihood: $(round(ll_v3; digits=6))")

    # The log-likelihoods won't match exactly because:
    # - V2 uses π0 as a joint (r_θ, r_I), V3 uses π0 as (r_θ,) × CCP_init
    # - V2's CCP arrays are direct, V3's CCP is parametric (ordered probit)
    #
    # But with uniform π0 and uniform CCP in both, the initial joint should be:
    # V2: π0[kθ, kI] = 1/(r_θ*r_I)
    # V3: P(θ|z) * CCP_init(I|θ) = (1/r_θ) * (1/r_I) = 1/(r_θ*r_I)
    # So the initial distributions should match.
    #
    # The transition CCPs also need to match:
    # V2: CCP[kI', kI, kθ'] = 1/r_I (uniform)
    # V3: build_ccp_v3_trans with all betas=0, cutpoints = Φ⁻¹(k/r_I)
    #     gives P(I'=k) = 1/r_I ✓

    # V3 log-likelihood includes additive terms that V2 doesn't:
    #   - log(π_Z) for each child (= 0 when r_P=r_ω=1)
    #   - income_loglik for each (child, period) with observed income
    # The posteriors (γ, ξ) are unaffected since there's only one regime.
    # Compute the expected income offset to verify.
    income_offset = 0.0
    for n in 1:N, t in 1:T
        income_offset += income_loglik(data_v3.Y_income[n, t], 1.0, 1.0, income_v3)
    end
    ll_v3_adjusted = ll_v3 - income_offset
    println("  Income offset: $(round(income_offset; digits=2))")
    println("  Adjusted V3 ll: $(round(ll_v3_adjusted; digits=6))")
    println("  Difference (adjusted): $(abs(ll_v3_adjusted - ll_v2))")
    @test ll_v3_adjusted ≈ ll_v2 atol=1e-2

    # Check posteriors match
    max_γ_diff = maximum(abs.(γ_agg .- γ_v2))
    println("  Max |γ_v3 - γ_v2|: $max_γ_diff")
    @test max_γ_diff < 0.01

    # Check ξ_sum match
    max_ξ_diff = maximum(maximum(abs.(xi_sum_z[1][t] .- ξ_sum_v2[t])) for t in 1:T-1)
    println("  Max |ξ_v3 - ξ_v2|: $max_ξ_diff")
    @test max_ξ_diff < 0.01

    # All regime weights should be 1.0 (only one regime)
    @test all(w_z .≈ 1.0)
    println("  All w_z = 1.0: PASSED")

    println("  PASSED: V2 replication test")
end

# ═════════════════════════════════════════════════════════════════
# TEST 3: Regime mechanics
# ═════════════════════════════════════════════════════════════════

@testset "Regime Mechanics" begin
    println("\n" * "="^60)
    println("  TEST 3: Regime mechanics (r_P=2, r_ω=2)")
    println("="^60)

    r_P = 2; r_ω = 2; r_Z = r_P * r_ω
    data_v3 = merge(panel_data, (; r_P, r_ω))
    config = (K=3, ν=7.0, n_groups=2)
    rng = MersenneTwister(99)

    pv3, psk = initialize_params_v3(data_v3, config; rng)

    # Run 1 EM iteration
    pv3, psk, γ, ξz, wz, ll = expectation_maximization_v3(
        data_v3, pv3, psk;
        transition_groups = TRANSITION_GROUPS,
        maxiter = 1, tol = 0.0, verbose = true)

    @test isfinite(ll)

    # ── regime_z / regime_kPkω round-trip ──
    for kP in 1:r_P, kω in 1:r_ω
        z = regime_z(kP, kω, r_ω)
        kP2, kω2 = regime_kPkω(z, r_ω)
        @test kP == kP2
        @test kω == kω2
    end
    println("  Regime index round-trip: PASSED")

    # ── w_z sums to 1 for each child ──
    for n in 1:N
        @test sum(wz[:, n]) ≈ 1.0 atol=1e-10
    end
    println("  w_z sums to 1: PASSED")

    # ── γ_agg sums to ~1 for each (t, n) ──
    for n in 1:min(N, 200), t in 1:T
        s = sum(γ[t, :, :, n])
        @test s ≈ 1.0 atol=1e-8
    end
    println("  γ_agg sums to 1: PASSED")

    # ── ξ_sum_z are non-negative ──
    for z in 1:r_Z, t in 1:T-1
        @test all(ξz[z][t] .>= -1e-10)
    end
    println("  ξ_sum_z non-negative: PASSED")

    # ── π0 after M-step sums to 1 ──
    @test sum(pv3.π0) ≈ 1.0 atol=1e-10
    println("  π0 sums to 1: PASSED")

    # ── Q_P columns sum to 1 ──
    for m in eachindex(pv3.Q_P)
        for kP in 1:r_P
            @test sum(pv3.Q_P[m][:, kP]) ≈ 1.0 atol=1e-8
        end
    end
    println("  Q_P columns sum to 1: PASSED")

    # ── Regime-specific Π arrays are valid probability matrices ──
    Pi_z = precompute_Pi_z(pv3.trans_specs, TRANSITION_GROUPS,
                             r_θ, r_I, r_P, r_ω)
    for z in 1:r_Z, t in 1:T-1
        Π = Pi_z[z][t]
        for kI in 1:r_I, kθ in 1:r_θ
            @test all(Π[:, kθ, kI] .>= 0)
            @test sum(Π[:, kθ, kI]) ≈ 1.0 atol=1e-8
        end
    end
    println("  Π[z][t] valid probabilities: PASSED")

    # ── Different regimes produce different Π ──
    # (alpha3/alpha4 are random, so z=1 and z=r_Z should differ)
    diff = maximum(abs.(Pi_z[1][1] .- Pi_z[r_Z][1]))
    @test diff > 1e-6
    println("  Different regimes → different Π (max diff = $(round(diff; sigdigits=3))): PASSED")

    println("  PASSED: Regime mechanics")
end

# ═════════════════════════════════════════════════════════════════
# TEST 4: IncomeModel
# ═════════════════════════════════════════════════════════════════

@testset "IncomeModel" begin
    println("\n" * "="^60)
    println("  TEST 4: IncomeModel")
    println("="^60)

    # income_loglik returns 0 for NaN
    m = IncomeModel(10.0, 0.5, -0.3, log(0.5))
    @test income_loglik(NaN, 1.0, 1.0, m) == 0.0
    println("  NaN income → 0 loglik: PASSED")

    # Round-trip: initialize from data, check moments
    Y_test = panel_data.Y_income
    m_init = initialize_income_model(Y_test)
    obs = filter(!isnan, vec(Y_test))
    @test abs(m_init.beta0 - mean(obs)) < 0.01
    @test abs(exp(m_init.log_sigma) - std(obs)) < 0.1
    println("  Initialize from data moments: PASSED")

    # WLS update with uniform weights: predicted value at the single regime
    # should equal the sample mean. With r_P=1, r_ω=1, design is [1, 1, 1]
    # so β₀ + β_P·1 + β_ω·1 = mean(obs), but β₀ alone ≠ mean.
    N_test = size(Y_test, 1)
    w_uniform = ones(1, N_test)  # r_Z=1
    m_updated = update_income_model(w_uniform, Y_test, 1, 1)
    predicted = m_updated.beta0 + m_updated.betaP * 1.0 + m_updated.betaω * 1.0
    @test abs(predicted - mean(obs)) < 0.1
    println("  WLS with uniform weights ≈ OLS: PASSED")

    println("  PASSED: IncomeModel")
end

# ═════════════════════════════════════════════════════════════════
# TEST 5: Pack/unpack round-trips
# ═════════════════════════════════════════════════════════════════

@testset "Pack/Unpack Round-trips" begin
    println("\n" * "="^60)
    println("  TEST 5: Pack/unpack round-trips")
    println("="^60)

    rng = MersenneTwister(77)

    # RobitNNTransitionV3
    s = initialize_robit_nn_v3(r_θ, 4; rng, ν=7.0)
    v = pack_robit_nn_v3(s)
    s2 = unpack_robit_nn_v3(v, r_θ, 4; ν=7.0)
    @test pack_robit_nn_v3(s2) ≈ v
    println("  RobitNNTransitionV3: PASSED")

    # OrderedProbitCCPV3
    c = initialize_ccp_v3(r_I; rng)
    v = pack_ccp_v3(c)
    c2 = unpack_ccp_v3(v, r_I)
    @test pack_ccp_v3(c2) ≈ v
    println("  OrderedProbitCCPV3: PASSED")

    println("  PASSED: Pack/unpack round-trips")
end

println("\n" * "="^60)
println("  ALL TESTS COMPLETE")
println("="^60)
