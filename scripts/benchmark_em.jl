using LinearAlgebra, Random, Statistics, Dates

include(joinpath(@__DIR__, "..", "src", "prepare_data.jl"))
include(joinpath(@__DIR__, "..", "src", "estimation.jl"))

# ─────────────────────────────────────────────────────────────────
# Benchmark E-step and M-step performance
# ─────────────────────────────────────────────────────────────────

outfile = joinpath(dirname(@__DIR__), "output", "benchmark_results.txt")
io = open(outfile, "w")

function tee(msg)
    println(msg)
    println(io, msg)
    flush(io)
end

tee("=" ^ 60)
tee("  EM Benchmark: E-step and M-step")
tee("  $(Dates.now())")
tee("=" ^ 60)

# ── Load data ──
datasets = load_all_datasets()
data = prepare_age_group_data(datasets;
    age_range_t1=(0, 1),
    age_range_t2=(2, 3),
    r_θ=4,
    r_I=4,
    invest_items=[
        :HowOftenRead, :HowManyBooks, :HowManyToysA, :HowManyToysB,
        :OutOfHouse, :HowOftenTalks, :HowOftenGrocery],
)

tee("\nData: N=$(data.N), r_θ=$(data.r_θ), r_I=$(data.r_I)")
tee("  Skill measures: $(length(data.X_θ))")
tee("  Investment measures: $(length(data.X_I))")
tee("  Period 2 measures: $(length(data.Y))")

# ── Initialize ──
rng = MersenneTwister(42)
params = initialize_params(data; rng)
Q = zeros(data.r_θ, data.r_θ, data.r_I, data.N)

shared = false
tee("  Shared skill emission: $shared")

# ── Warmup (compile) ──
tee("\nWarming up (compilation)...")
expectation_step!(Q, params, data)
maximization_step(Q, data; shared_skill_emission=shared)

# ── Benchmark E-step ──
tee("\n" * "-" ^ 60)
tee("  E-step benchmarks")
tee("-" ^ 60)

n_runs = 5
e_times = Float64[]
e_bytes = Int[]

for run in 1:n_runs
    stats = @timed expectation_step!(Q, params, data)
    push!(e_times, stats.time)
    push!(e_bytes, stats.bytes)
    tee("  Run $run: $(round(stats.time * 1000; digits=1)) ms, $(round(stats.bytes / 1e6; digits=2)) MB allocated")
end

tee("\n  E-step summary ($n_runs runs):")
tee("    Mean time:  $(round(mean(e_times) * 1000; digits=1)) ms")
tee("    Min time:   $(round(minimum(e_times) * 1000; digits=1)) ms")
tee("    Max time:   $(round(maximum(e_times) * 1000; digits=1)) ms")
tee("    Mean alloc: $(round(mean(e_bytes) / 1e6; digits=2)) MB")

# ── Benchmark per-observation E-step ──
tee("\n  Per-observation E-step (single n):")
Qn = zeros(data.r_θ, data.r_θ, data.r_I)
stats_n = @timed begin
    for _ in 1:1000
        expectation_step_n!(Qn, 1, params, data)
    end
end
tee("    1000 calls: $(round(stats_n.time * 1000; digits=1)) ms ($(round(stats_n.time; digits=4)) s)")
tee("    Per call: $(round(stats_n.time / 1000 * 1e6; digits=1)) μs")
tee("    Alloc per call: $(round(stats_n.bytes / 1000; digits=0)) bytes")

# ── Benchmark M-step ──
tee("\n" * "-" ^ 60)
tee("  M-step benchmarks")
tee("-" ^ 60)

# Run E-step once to get valid Q
expectation_step!(Q, params, data)

m_times = Float64[]
m_bytes = Int[]

for run in 1:n_runs
    stats = @timed maximization_step(Q, data; shared_skill_emission=shared)
    push!(m_times, stats.time)
    push!(m_bytes, stats.bytes)
    tee("  Run $run: $(round(stats.time * 1000; digits=1)) ms, $(round(stats.bytes / 1e6; digits=2)) MB allocated")
end

tee("\n  M-step summary ($n_runs runs):")
tee("    Mean time:  $(round(mean(m_times) * 1000; digits=1)) ms")
tee("    Min time:   $(round(minimum(m_times) * 1000; digits=1)) ms")
tee("    Max time:   $(round(maximum(m_times) * 1000; digits=1)) ms")
tee("    Mean alloc: $(round(mean(m_bytes) / 1e6; digits=2)) MB")

# ── Benchmark individual M-step components ──
tee("\n  M-step components:")

stats_pi = @timed begin
    for _ in 1:10; update_Π(Q); end
end
tee("    update_Π:              $(round(stats_pi.time / 10 * 1000; digits=1)) ms, $(round(stats_pi.bytes / 10 / 1e6; digits=2)) MB")

stats_pi0 = @timed begin
    for _ in 1:10; update_π0(Q); end
end
tee("    update_π0:             $(round(stats_pi0.time / 10 * 1000; digits=1)) ms, $(round(stats_pi0.bytes / 10 / 1e6; digits=2)) MB")

stats_pe = @timed begin
    for _ in 1:10; update_emission_I(Q, data.X_I, data.κ_I); end
end
tee("    update_emission_I:     $(round(stats_pe.time / 10 * 1000; digits=1)) ms, $(round(stats_pe.bytes / 10 / 1e6; digits=2)) MB")

if shared
    stats_ps = @timed begin
        for _ in 1:10; update_emission_skill_shared(Q, data.X_θ, data.Y, data.κ_θ); end
    end
    tee("    update_skill_shared:   $(round(stats_ps.time / 10 * 1000; digits=1)) ms, $(round(stats_ps.bytes / 10 / 1e6; digits=2)) MB")
else
    stats_pt = @timed begin
        for _ in 1:10; update_emission_θ(Q, data.X_θ, data.κ_θ); end
    end
    tee("    update_emission_θ:     $(round(stats_pt.time / 10 * 1000; digits=1)) ms, $(round(stats_pt.bytes / 10 / 1e6; digits=2)) MB")

    stats_py = @timed begin
        for _ in 1:10; update_emission_Y(Q, data.Y, data.κ_Y); end
    end
    tee("    update_emission_Y:     $(round(stats_py.time / 10 * 1000; digits=1)) ms, $(round(stats_py.bytes / 10 / 1e6; digits=2)) MB")
end

# ── Memory profile ──
tee("\n" * "-" ^ 60)
tee("  Memory profile")
tee("-" ^ 60)

q_size = sizeof(Q) / 1e6
tee("  Q array: $(data.r_θ)×$(data.r_θ)×$(data.r_I)×$(data.N) = $(prod(size(Q))) entries, $(round(q_size; digits=2)) MB")

total_emission = sum(length(P) for P in params.P_θ) + sum(length(P) for P in params.P_I) + sum(length(P) for P in params.P_Y)
tee("  Emission matrices: $(total_emission) entries total")
tee("  Π array: $(size(params.Π)) = $(length(params.Π)) entries")
tee("  π0 array: $(size(params.π0)) = $(length(params.π0)) entries")

# ── Full EM iteration ──
tee("\n" * "-" ^ 60)
tee("  Full EM iteration (E + M)")
tee("-" ^ 60)

em_times = Float64[]
for run in 1:n_runs
    stats = @timed begin
        expectation_step!(Q, params, data)
        params_new = maximization_step(Q, data; shared_skill_emission=shared)
    end
    push!(em_times, stats.time)
    tee("  Run $run: $(round(stats.time * 1000; digits=1)) ms")
end

tee("\n  Full iteration summary ($n_runs runs):")
tee("    Mean time:  $(round(mean(em_times) * 1000; digits=1)) ms")
tee("    Min time:   $(round(minimum(em_times) * 1000; digits=1)) ms")
e_frac = mean(e_times) / mean(em_times) * 100
tee("    E-step fraction: $(round(e_frac; digits=1))%")
tee("    Estimated time for 2000 iterations: $(round(mean(em_times) * 2000; digits=1)) s")

tee("\n" * "=" ^ 60)
tee("  Benchmark complete. Results saved to: $outfile")
tee("=" ^ 60)

close(io)
println("\nResults written to $outfile")
