using LinearAlgebra, Random, Statistics, JLD2

include(joinpath(@__DIR__, "..", "src", "parametric_models.jl"))
include(joinpath(@__DIR__, "..", "src", "home_sf_recode.jl"))
include(joinpath(@__DIR__, "..", "src", "prepare_data.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation.jl"))
include(joinpath(@__DIR__, "..", "src", "inference.jl"))

# ─────────────────────────────────────────────────────────────────
# Parametric configuration
#
# Each component can be :nonparametric or its parametric alternative:
#   emission_θ / emission_Y / emission_I: :ordered_probit
#   transition: :flexible_lpm, :parsimonious_lpm, or :ordered_probit
# ─────────────────────────────────────────────────────────────────

const PARAMETRIC_CONFIG = (
    emission_θ  = :ordered_probit,
    emission_Y  = :ordered_probit,
    emission_I  = :ordered_probit,
    transition  = :ordered_probit,   # :parsimonious_lpm, :flexible_lpm, or :ordered_probit
)

# ─────────────────────────────────────────────────────────────────
# Age group configurations (same as run_estimation_homesf.jl)
#
# Investment: HOME-SF binary-recoded Cognitive Stimulation items
# Skills: per-period best cognitive source (MSD, PPVT, PIAT)
# ─────────────────────────────────────────────────────────────────

const AGE_GROUP_CONFIGS = [
    (age_t1=(0,1),   age_t2=(2,3),   home_part=:PartA,
     skill_source_t1=:msd, skill_source_t2=:msd),
    (age_t1=(2,3),   age_t2=(4,5),   home_part=:PartB,
     skill_source_t1=:msd, skill_source_t2=[:msd, :ppvt]),
    (age_t1=(4,5),   age_t2=(6,7),   home_part=:PartB,
     skill_source_t1=[:msd, :ppvt], skill_source_t2=:piat),
    (age_t1=(6,7),   age_t2=(8,9),   home_part=:PartC,
     skill_source_t1=:piat, skill_source_t2=:piat),
    (age_t1=(8,9),   age_t2=(10,11), home_part=:PartC,
     skill_source_t1=:piat, skill_source_t2=:piat),
    (age_t1=(10,11), age_t2=(12,13), home_part=:PartD,
     skill_source_t1=:piat, skill_source_t2=:piat),
]

# ─────────────────────────────────────────────────────────────────
# Helpers for skill source selection (same as run_estimation_homesf.jl)
# ─────────────────────────────────────────────────────────────────

const ALL_CONTINUOUS_SOURCES = [:ppvt, :piat, :digit]
const ALL_SOURCES = [:msd, :ppvt, :piat, :digit]

sources_to_exclude(s::Symbol) = sources_to_exclude([s])
function sources_to_exclude(sources::Vector{Symbol})
    for s in sources
        s in ALL_SOURCES || error("Unknown skill_source: $s")
    end
    return [s for s in ALL_CONTINUOUS_SOURCES if s ∉ sources]
end

uses_msd(s::Symbol) = s == :msd
uses_msd(sources::Vector{Symbol}) = :msd in sources

source_str(s::Symbol) = string(s)
source_str(sources::Vector{Symbol}) = join(sort(string.(sources)), "+")

# ─────────────────────────────────────────────────────────────────
# Load all datasets
# ─────────────────────────────────────────────────────────────────

datasets = load_all_datasets()

# ─────────────────────────────────────────────────────────────────
# Run estimation for each age group
# ─────────────────────────────────────────────────────────────────

outdir = joinpath(dirname(@__DIR__), "output")
mkpath(outdir)

config_str = "$(PARAMETRIC_CONFIG.transition)"

for (g, cfg) in enumerate(AGE_GROUP_CONFIGS)
    label = "$(cfg.age_t1[1])_$(cfg.age_t1[2])__$(cfg.age_t2[1])_$(cfg.age_t2[2])_parametric"
    src_t1 = cfg.skill_source_t1
    src_t2 = cfg.skill_source_t2

    println("\n" * "="^60)
    println("  AGE GROUP $g: $(cfg.age_t1) → $(cfg.age_t2)  [HOME-SF $(cfg.home_part)]")
    println("    skill_source_t1=$(source_str(src_t1)), skill_source_t2=$(source_str(src_t2))")
    println("    transition=$(PARAMETRIC_CONFIG.transition), emissions=$(PARAMETRIC_CONFIG.emission_θ)")
    println("="^60)

    # Determine what to exclude at each period
    excl_t1 = sources_to_exclude(src_t1)
    excl_t2 = sources_to_exclude(src_t2)
    exclude_msd_t1 = !uses_msd(src_t1)
    exclude_msd_t2 = !uses_msd(src_t2)

    # ── Data preparation with HOME-SF recoding ──
    println("\n  Data Preparation:")
    data_raw = prepare_age_group_data_homesf(datasets;
        age_range_t1=cfg.age_t1,
        age_range_t2=cfg.age_t2,
        r_θ=3,
        r_I=5,
        home_part=cfg.home_part,
        exclude_sources_t1=excl_t1,
        exclude_sources_t2=excl_t2,
    )

    # Post-filter MSD items from periods that use non-MSD sources
    needs_filter = exclude_msd_t1 || exclude_msd_t2

    if needs_filter
        keep_θ = exclude_msd_t1 ?
            [i for i in eachindex(data_raw.κ_θ) if data_raw.κ_θ[i] != 2] :
            collect(eachindex(data_raw.κ_θ))
        keep_Y = exclude_msd_t2 ?
            [i for i in eachindex(data_raw.κ_Y) if data_raw.κ_Y[i] != 2] :
            collect(eachindex(data_raw.κ_Y))

        if isempty(keep_θ)
            println("  WARNING: No $(source_str(src_t1)) measures at period 1 — skipping.")
            continue
        end
        if isempty(keep_Y)
            println("  WARNING: No $(source_str(src_t2)) measures at period 2 — skipping.")
            continue
        end

        data = (;
            N = data_raw.N,
            r_θ = data_raw.r_θ,
            r_I = data_raw.r_I,
            X_θ = data_raw.X_θ[keep_θ],
            κ_θ = data_raw.κ_θ[keep_θ],
            measure_names_θ = data_raw.measure_names_θ[keep_θ],
            X_I = data_raw.X_I,
            κ_I = data_raw.κ_I,
            measure_names_I = data_raw.measure_names_I,
            Y = data_raw.Y[keep_Y],
            κ_Y = data_raw.κ_Y[keep_Y],
            measure_names_Y = data_raw.measure_names_Y[keep_Y],
            kids = data_raw.kids,
        )
    else
        data = data_raw
    end

    println("  Period 1 skill measures ($(source_str(src_t1))): $(length(data.X_θ))")
    println("  Period 2 skill measures ($(source_str(src_t2))): $(length(data.Y))")
    println("  Investment measures (HOME-SF): $(length(data.X_I))")
    println("  N = $(data.N)")

    if data.N < 50
        println("  WARNING: Only $(data.N) observations — skipping this age group.")
        continue
    end

    # ── Parametric EM estimation ──
    println("\n  Parametric EM Estimation ($(PARAMETRIC_CONFIG.transition)):")
    best_params, best_Q, best_ll, best_specs = run_em_restarts_parametric(
        data, PARAMETRIC_CONFIG)
    println("\n  Best log-likelihood: $(round(best_ll; digits=2))")

    # Display parametric parameter values
    if PARAMETRIC_CONFIG.transition == :parsimonious_lpm
        spec = best_specs[:transition]::ParsimoniousLPMTransition
        println("\n  Parsimonious LPM parameters:")
        println("    c = $(round.(spec.c; digits=4))")
        println("    γ₁ (I) = $(round(spec.gamma1; digits=4))")
        println("    γ₂ (θ) = $(round(spec.gamma2; digits=4))")
        println("    γ₃ (θ·I, complementarity) = $(round(spec.gamma3; digits=4))")
    elseif PARAMETRIC_CONFIG.transition == :flexible_lpm
        spec = best_specs[:transition]::FlexibleLPMTransition
        println("\n  Flexible LPM parameters:")
        println("    c = $(round.(spec.c; digits=4))")
        println("    γ₀(θ) = $(round.(spec.gamma0; digits=4))")
        println("    γ₁(θ) = $(round.(spec.gamma1; digits=4))")
    elseif PARAMETRIC_CONFIG.transition == :ordered_probit
        spec = best_specs[:transition]::OrderedProbitTransition
        println("\n  Ordered Probit Transition parameters:")
        println("    c (cutpoints) = $(round.(spec.c; digits=4))")
        println("    β₁ (θ) = $(round(spec.beta1; digits=4))")
        println("    β₂ (I) = $(round(spec.beta2; digits=4))")
        println("    β₃ (θ·I) = $(round(spec.beta3; digits=4))")
    end

    # ── Save results ──
    outpath = joinpath(outdir, "estimation_$(label).jld2")
    parametric_config = PARAMETRIC_CONFIG
    @save outpath best_params best_Q data best_ll best_specs parametric_config
    println("  Results saved to $outpath")
end

println("\n" * "="^60)
println("  SUMMARY OF PARAMETRIC ESTIMATION")
println("="^60)
println()
println("  Model: transition=$(PARAMETRIC_CONFIG.transition), emissions=$(PARAMETRIC_CONFIG.emission_θ)")
println("  Skill sources by age group:")
for cfg in AGE_GROUP_CONFIGS
    str_t1 = source_str(cfg.skill_source_t1)
    str_t2 = source_str(cfg.skill_source_t2)
    if str_t1 == str_t2
        println("    $(cfg.age_t1) → $(cfg.age_t2):  $str_t1 (both periods), HOME-SF $(cfg.home_part)")
    else
        println("    $(cfg.age_t1) → $(cfg.age_t2):  t1=$str_t1, t2=$str_t2, HOME-SF $(cfg.home_part)")
    end
end
println("\nDone. Run notebooks/analyze_results_parametric.ipynb to inspect estimates.")
