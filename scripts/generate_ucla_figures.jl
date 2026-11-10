#####
# Generate the three main PDFs for the UCLA slide deck.
#
# Outputs (relative to repo root):
#   slides/ucla_75/figures/survivor_groups.pdf
#   slides/ucla_75/figures/local_modularity.pdf
#   slides/ucla_75/figures/survivors_by_regime.pdf
#
# Logic mirrors notebooks/analyze_results_v3_eps.ipynb cells 24, 26, 30.
# Uses the same palette as the DCT-style slide deck:
#   back=#FFF5EE, c1=#221E1D, c2=#63AA9C (teal), c3=#E9633B (rust), c5=#8B5A8C.
#####

using JLD2, Plots, LinearAlgebra, Statistics, Printf, CSV, DataFrames

const REPO = joinpath(@__DIR__, "..")
include(joinpath(REPO, "src", "parametric_models.jl"))
include(joinpath(REPO, "src", "parametric_models_v3.jl"))
include(joinpath(REPO, "src", "parametric_models_v3_eps.jl"))
include(joinpath(REPO, "src", "prepare_data.jl"))
include(joinpath(REPO, "src", "estimation.jl"))
include(joinpath(REPO, "src", "estimation_multiperiod.jl"))
include(joinpath(REPO, "src", "estimation_multiperiod_v3.jl"))
include(joinpath(REPO, "src", "estimation_multiperiod_v3_eps.jl"))
include(joinpath(REPO, "src", "inference.jl"))
include(joinpath(REPO, "src", "inference_common.jl"))
include(joinpath(REPO, "src", "inference_v3_eps.jl"))

# Paper mode (ENV PAPER_FIGS=1): white background, larger fonts, figures/ dir.
# REG_RUN=1 redirects paper-mode figures to figures_reg/ (non-destructive parallel run).
const PAPER_FIGS = get(ENV, "PAPER_FIGS", "0") == "1"
const REG_RUN    = get(ENV, "REG_RUN", "0") == "1"
const OUT_DIR  = haskey(ENV, "SLIDES_FIG_DIR") ? abspath(ENV["SLIDES_FIG_DIR"]) :
                 PAPER_FIGS ? joinpath(REPO, REG_RUN ? "figures_reg" : "figures") :
                              joinpath(REPO, "slides", "ucla_75", "figures")
isdir(OUT_DIR) || mkpath(OUT_DIR)

# Paper-aware font sizes (slides values kept as the defaults).
const TF_TITLE = PAPER_FIGS ? 16 : 11   # panel title
const TF_GUIDE = PAPER_FIGS ? 15 :  9   # axis labels
const TF_TICK  = PAPER_FIGS ? 13 :  8   # tick labels
const TF_LEG   = PAPER_FIGS ? 13 :  8   # legend
const TF_ANN   = PAPER_FIGS ? 12 :  9   # in-cell annotations
const TF_PTITLE= PAPER_FIGS ? 16 : 12   # overall plot_title

# DCT palette
const PAL_BACK   = RGB(255/255, 245/255, 238/255)   # sepia background
const PAL_TEXT   = RGB(34/255,  30/255,  29/255)    # near-black
const PAL_TEAL   = RGB(99/255,  170/255, 156/255)   # c2
const PAL_RUST   = RGB(233/255, 99/255,  59/255)    # c3
const PAL_PLUM   = RGB(139/255, 90/255,  140/255)
const PAL_GREY   = RGB(0.55, 0.55, 0.55)
const SKILL_COLORS = [PAL_TEAL, PAL_PLUM, RGB(0.90, 0.72, 0.33), PAL_RUST]  # θ = 1..4

# Use GR; paper mode overrides background to white.
gr()
# Do NOT force Palatino on GR here — its Palatino lacks Greek glyphs (θ → ?).
default(
    background_color  = PAPER_FIGS ? RGB(1,1,1) : PAL_BACK,
    foreground_color  = PAPER_FIGS ? RGB(0,0,0) : PAL_TEXT,
    framestyle        = :box,
    grid              = !PAPER_FIGS,
    gridalpha         = 0.22,
    minorgrid         = false,
    left_margin       = 8Plots.mm,
    bottom_margin     = 8Plots.mm,
    right_margin      = 3Plots.mm,
    top_margin        = 3Plots.mm,
)

# ─────────────────────────────────────────────────────────────────────────
# Load fit + run inference
# ─────────────────────────────────────────────────────────────────────────
const FIT_PATH = get(ENV, "FIT_OVERRIDE", joinpath(REPO, "output", "estimation_v3_eps_reg.jld2"))
println("Loading fit: ", FIT_PATH, " ...")
data = load(FIT_PATH)
pv3  = data["params_v3_eps"]
psk  = data["params_skill"]
dims = data["data_dims"]
age_ranges = data["age_ranges"]

(; r_θ, r_I, r_P, r_ω, K, N, T) = dims

const TRANSITION_GROUPS = [[1, 2, 3], [4, 5, 6]]
# Match the step labels used by the 2-period counterfactual figures.
const GROUP_LABELS      = ["step 1: ages 0–5", "step 2: ages 6–11"]

const PANEL_INVEST_ITEMS_NB = [
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
     :ToldPositive, :HowOftenHomework],
    [:HowManyBooks, :HowOftenMuseum, :HowOftenTheater, :HowOftenFamily,
     :HowOftenOutdoors, :HowOftenFather, :ChildMusic, :SpecialActivities,
     :Hobbies, :EnjoyRead, :DailyNewspaper, :DiscussTV,
     :GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy,
     :ToldPositive, :HowOftenHomework],
]

println("Loading raw data ...")
datasets = load_all_datasets()
panel = prepare_panel_data(datasets;
    age_ranges   = age_ranges,
    invest_items = PANEL_INVEST_ITEMS_NB,
    r_θ = r_θ, r_I = r_I,
)
data_nb = merge(panel, (; r_P = r_P, r_ω = r_ω))

println("Running inference (Fisher's identity, ~10 sec) ...")
inf = compute_var_phi_v3_eps(pv3, psk, data_nb;
    transition_groups = TRANSITION_GROUPS, verbose = false)

# ─────────────────────────────────────────────────────────────────────────
# Optional BOOTSTRAP covariance (BOOT_COV=path/to/bootstrap_reg.jld2).
# When set, the two CI-bearing figures (survivor_groups + local_modularity_by_outcome)
# use the B-sample bootstrap covariance of the free Π̄ entries instead of the
# delta-method covariance.  The free-parameter ordering matches
# delta_cov_Pi_bar_free / get_S exactly:  idx = ((kI-1)*r_θ + (kθ-1))*(r_θ-1) + j
# (j = cutpoint innermost), which is how scripts/test_supermodularity_by_cutpoint.jl
# substitutes cov(boot_pibar[g]) for Var_free_list[group[1]].
# ─────────────────────────────────────────────────────────────────────────
const USE_BOOT_COV = haskey(ENV, "BOOT_COV")
boot_Var_free = Dict{Int,Matrix{Float64}}()   # group g -> Var_free (n_free × n_free)
if USE_BOOT_COV
    println("Using BOOTSTRAP covariance from ", ENV["BOOT_COV"], " ...")
    bc  = load(ENV["BOOT_COV"]); bp = bc["boot_pibar"]; okb = findall(bc["ok"])
    for (g, _) in enumerate(TRANSITION_GROUPS)
        boot_Var_free[g] = cov(bp[g][okb, :])
    end
    println("  bootstrap n_ok=$(length(okb)); Var_free mean-diag/group: ",
            [round(mean(diag(boot_Var_free[g])); sigdigits=3) for g in eachindex(TRANSITION_GROUPS)])
end

# Linear map from free-Π̄ vector (ordering kI,kθ,j; j=1..r_θ-1) to the free
# survivor vector (ordering kI,kθ,k).  Because the top row r_θ is pinned by
# Σ_j Π̄(j)=1, we have  S(k|θ,I) = Σ_{j>k} Π̄(j) = 1 − Σ_{j=1}^{k} Π̄(j),
# so Var(S(k)) = Var(Σ_{j≤k} Π̄(j)); the constant drops out under cov().
# This matches delta_cov_survivor_bar's survivor SE ordering (kI,kθ,k).
function survivor_from_pibar_W(r_θ::Int, r_I::Int)
    n_cut  = r_θ - 1
    n_free = n_cut * r_θ * r_I
    W = zeros(n_cut * r_θ * r_I, n_free)
    pi_idx(j, kθ, kI) = ((kI - 1) * r_θ + (kθ - 1)) * (r_θ - 1) + j   # j in 1..r_θ-1
    s_row = 0
    for kI in 1:r_I, kθ in 1:r_θ, k in 1:n_cut
        s_row += 1
        for j in 1:k                       # S(k) = 1 − Σ_{j≤k} Π̄(j)
            W[s_row, pi_idx(j, kθ, kI)] -= 1.0
        end
    end
    return W
end

println("Precomputing per-regime transitions ...")
Pi_z_pre = precompute_Pi_z(pv3.trans_specs, TRANSITION_GROUPS,
    r_θ, r_I, r_P, r_ω;
    θ_grid=1.0:r_θ, I_grid=1.0:r_I,
    P_grid=1.0:r_P, ω_grid=1.0:r_ω)

# ─────────────────────────────────────────────────────────────────────────
# (1) survivor_groups.pdf — regime-marginalized survivors w/ 95% CIs
# ─────────────────────────────────────────────────────────────────────────
println("\n[1/3] Building survivor_groups.pdf ...")
sb_results = delta_cov_survivor_bar(inf, pv3, psk; transition_groups = TRANSITION_GROUPS)
n_cut = r_θ - 1

# Bootstrap survivor SEs (per group), if BOOT_COV is set: push the bootstrap
# free-Π̄ covariance through the survivor map S = W·Π̄_free.  Ordering of the
# resulting ses matches delta_cov_survivor_bar (kI,kθ,k), so indexing is identical.
boot_surv_ses = Dict{Int,Vector{Float64}}()
if USE_BOOT_COV
    Wsurv = survivor_from_pibar_W(r_θ, r_I)
    for (g, _) in enumerate(TRANSITION_GROUPS)
        Var_S = Wsurv * boot_Var_free[g] * Wsurv'
        boot_surv_ses[g] = sqrt.(max.(diag(Var_S), 0.0))
    end
end

plots_surv = Matrix{Any}(undef, length(TRANSITION_GROUPS), n_cut)
for (gi, res) in enumerate(sb_results)
    g = res.group
    ses_g = USE_BOOT_COV ? boot_surv_ses[g] : res.ses
    # Reconstruct regime-marginalised Π for this group
    Π_g = zeros(r_θ, r_θ, r_I)
    spec_g = pv3.trans_specs[g]
    φ_g = pack_robit_nn_v3(spec_g)
    for kω in 1:r_ω, kP in 1:r_P
        w = pv3.π_Pω[kP, kω]
        Π_z = _build_robit_nn_v3_pi(φ_g, r_θ, r_I, spec_g.K, spec_g.ν,
            Float64(kP), Float64(kω); θ_grid=1.0:r_θ, I_grid=1.0:r_I)
        Π_g .+= w .* Π_z
    end

    for k in 1:n_cut
        show_legend = (gi == 1 && k == n_cut)
        show_ylabel = (k == 1)
        show_xlabel = (gi == length(TRANSITION_GROUPS))
        show_title  = (gi == 1)

        p = plot(ylims=(0.0, 1.05), legend = show_legend ? :topleft : false,
            titlefontsize=TF_TITLE, guidefontsize=TF_GUIDE, tickfontsize=TF_TICK, legendfontsize=TF_LEG)
        show_title  && title!(p, "k = $k")
        show_ylabel && ylabel!(p, GROUP_LABELS[gi] * "\nsurvivor F̄(k | θ, I)")
        show_xlabel && xlabel!(p, "investment I")

        for kθ in 1:r_θ
            surv   = [sum(Π_g[j, kθ, kI] for j in k+1:r_θ) for kI in 1:r_I]
            se_val = [ses_g[(kI-1)*r_θ*n_cut + (kθ-1)*n_cut + k] for kI in 1:r_I]
            plot!(p, 1:r_I, surv, ribbon = 1.96 .* se_val,
                marker=:circle, markersize=5, linewidth=2.6, fillalpha=0.18,
                color=SKILL_COLORS[kθ],
                label = show_legend ? "θ = $(kθ)" : "")
        end
        plots_surv[gi, k] = p
    end
end
fig1 = plot(permutedims(plots_surv)...,
    layout = (length(TRANSITION_GROUPS), n_cut),
    size = ((PAPER_FIGS ? 440 : 420) * n_cut, (PAPER_FIGS ? 340 : 300) * length(TRANSITION_GROUPS)),
    plot_titlefontsize = TF_PTITLE, margin = 3Plots.mm,
    left_margin = (PAPER_FIGS ? 12 : 8)Plots.mm,
    plot_title = "Regime-marginalized survivor F̄(k | θ, I) with 95% CI")
savefig(fig1, joinpath(OUT_DIR, "survivor_groups.pdf"))
println("  saved ", joinpath(OUT_DIR, "survivor_groups.pdf"))

# ─────────────────────────────────────────────────────────────────────────
# (2) local_modularity.pdf — pointwise super/sub DID grid
# ─────────────────────────────────────────────────────────────────────────
println("\n[2/3] Building local_modularity.pdf ...")

Var_free_list, Pi_bar_test = delta_cov_Pi_bar_free(inf, pv3, psk; transition_groups = TRANSITION_GROUPS)

# When BOOT_COV is set, replace the per-group free-Π̄ covariance with the
# bootstrap covariance (Pi_bar_test point estimates are unchanged).  Only the
# CI-bearing by-outcome DID figure consumes Var_free_list below.
if USE_BOOT_COV
    for (g, group) in enumerate(TRANSITION_GROUPS)
        for t in group
            Var_free_list[t] = boot_Var_free[g]
        end
    end
end

function classify_modularity(Π̄)
    nk = r_θ - 1
    labels   = Matrix{Symbol}(undef, r_θ - 1, r_I - 1)
    did_vals = Array{Float64,3}(undef, nk, r_θ - 1, r_I - 1)
    for iθ in 1:(r_θ - 1), iI in 1:(r_I - 1)
        for k in 1:nk
            S(kθ, kI) = sum(Π̄[j, kθ, kI] for j in k+1:r_θ)
            did_vals[k, iθ, iI] = S(iθ+1, iI+1) + S(iθ, iI) -
                                    S(iθ+1, iI) - S(iθ, iI+1)
        end
        d = did_vals[:, iθ, iI]
        labels[iθ, iI] = all(d .≥ -1e-4) ? :super :
                          all(d .≤  1e-4) ? :sub   : :mixed
    end
    labels, did_vals
end

plots_mod = []
for (g, group) in enumerate(TRANSITION_GROUPS)
    t_rep = group[1]
    labels, did_vals = classify_modularity(Pi_bar_test[t_rep])
    num_grid = [labels[iθ, iI] == :super ? 1.0 :
                labels[iθ, iI] == :sub   ? -1.0 : 0.0
                for iθ in 1:(r_θ - 1), iI in 1:(r_I - 1)]
    cg = cgrad([PAL_RUST, RGB(0.93,0.93,0.93), PAL_TEAL])
    p = heatmap(1.5:(r_I - 0.5), 1.5:(r_θ - 0.5), num_grid,
        color = cg, clims = (-1.1, 1.1),
        xlabel = "investment increment I → I+1",
        ylabel = "skill increment θ → θ+1",
        title = GROUP_LABELS[g],
        xticks = (1.5:(r_I - 0.5), ["$i→$(i+1)" for i in 1:(r_I - 1)]),
        yticks = (1.5:(r_θ - 0.5), ["$i→$(i+1)" for i in 1:(r_θ - 1)]),
        titlefontsize=TF_TITLE, guidefontsize=TF_GUIDE, tickfontsize=TF_TICK,
        colorbar = false, aspect_ratio = :auto)
    for iθ in 1:(r_θ - 1), iI in 1:(r_I - 1)
        ds = did_vals[:, iθ, iI]
        lbl = labels[iθ, iI] == :super ? "super" :
              labels[iθ, iI] == :sub   ? "sub"   : "mixed"
        # Show min and max DID on a single line, two decimals.
        txt = @sprintf("%s\n[%.2f, %.2f]", lbl, minimum(ds), maximum(ds))
        tcol = labels[iθ, iI] == :mixed ? PAL_TEXT : colorant"white"
        annotate!(p, iI + 0.5, iθ + 0.5, text(txt, TF_ANN, :center, tcol))
    end
    push!(plots_mod, p)
end
fig2 = plot(plots_mod..., layout = (1, 2), size = (PAPER_FIGS ? (1180, 540) : (1000, 420)),
    plot_title = "Pointwise modularity of Π̄  (teal = super, rust = sub, grey = mixed across k)",
    plot_titlefontsize = TF_PTITLE, margin = 3Plots.mm)
savefig(fig2, joinpath(OUT_DIR, "local_modularity.pdf"))
println("  saved ", joinpath(OUT_DIR, "local_modularity.pdf"))

# ─────────────────────────────────────────────────────────────────────────
# (2b) local_modularity_by_outcome.pdf — per-anchor surplus DID with 95% CIs
# DID_Q(θ,I) = Δ²E[Q(θ')|θ,I] is linear in Π̄, so its SE follows from the
# free-Π̄ covariance Var_free via SE = sqrt(w' Var_free w).  Color flags whether
# the 95% CI excludes 0; each cell reports the estimate and ± half-width.
# ─────────────────────────────────────────────────────────────────────────
println("\n[2b] Building local_modularity_by_outcome.pdf ...")

function earnings_bin_means_nb()
    ci  = CSV.read(joinpath(REPO, "NLSY", "ChildIncome", "ChildIncome_Clean.csv"), DataFrame)
    cpi = CSV.read(joinpath(REPO, "output", "cpi_u_annual.csv"), DataFrame)
    cpi_tbl = Dict{Int,Float64}(Int(cpi[i, :year]) => Float64(cpi[i, :cpi]) for i in 1:nrow(cpi))
    base = cpi_tbl[2010]
    ci.WageInc_num  = [ismissing(x) ? missing : tryparse(Float64, string(x)) for x in ci.WageInc]
    ci.YEAR_int     = [tryparse(Int, string(y)) for y in ci.YEAR]
    ci.WageInc_real = [(w === missing || w === nothing || yr === missing || yr === nothing) ? missing :
                        w / (get(cpi_tbl, yr, base) / base) for (w, yr) in zip(ci.WageInc_num, ci.YEAR_int)]
    ci_adult = ci[(ci.Age .>= 25) .& (ci.Age .<= 30), :]
    earn = combine(groupby(ci_adult, :KID)) do sub
        w = filter(!isnothing, collect(skipmissing(sub.WageInc_real)))
        isempty(w) ? (; AvgEarnings=missing, N_obs=0) : (; AvgEarnings=mean(w), N_obs=length(w))
    end
    earn = earn[earn.N_obs .>= 2, :]
    v = collect(skipmissing(earn.AvgEarnings))
    cuts = quantile(v, [0.2, 0.4, 0.6, 0.8])
    bin_of(x) = x <= cuts[1] ? 1 : x <= cuts[2] ? 2 : x <= cuts[3] ? 3 : x <= cuts[4] ? 4 : 5
    bins = bin_of.(v)
    [mean(v[bins .== q]) for q in 1:5]
end
function Q_from_adult(A, ys, πω)
    κ, r_θ_, r_ω_ = size(A); Q = zeros(r_θ_)
    for j in 1:r_θ_, vv in 1:r_ω_, k in 1:κ
        Q[j] += πω[vv] * ys[k] * A[k, j, vv]
    end
    Q
end
π_ω_marg   = vec(sum(Array{Float64}(pv3.π_Pω), dims = 1))
adult_mats = Vector{Array{Float64,3}}(pv3.adult_matrices)
earn_means = earnings_bin_means_nb()
anchor_names = ["HS", "College", "earnings", "employment", "log earnings"]
anchor_vals  = [[0.0, 1.0], [0.0, 1.0], earn_means, [0.0, 1.0]]
Qs = [Q_from_adult(adult_mats[m], anchor_vals[m], π_ω_marg) for m in 1:4]
push!(Qs, log.(Qs[3]))

freeidx(j, θ, I) = ((I - 1) * r_θ + (θ - 1)) * (r_θ - 1) + j
function did_with_ci(Q, Π̄, Vf, iθ, iI)
    est = sum(Q[t] * (Π̄[t, iθ+1, iI+1] - Π̄[t, iθ+1, iI] - Π̄[t, iθ, iI+1] + Π̄[t, iθ, iI]) for t in 1:r_θ)
    w = zeros((r_θ - 1) * r_θ * r_I)
    for (s, θc, Ic) in ((1.0, iθ+1, iI+1), (-1.0, iθ+1, iI), (-1.0, iθ, iI+1), (1.0, iθ, iI))
        for j in 1:(r_θ - 1)
            w[freeidx(j, θc, Ic)] += s * (Q[j] - Q[r_θ])
        end
    end
    (est, 1.96 * sqrt(max(dot(w, Vf * w), 0.0)))
end

TF_AN = PAPER_FIGS ? 12 : 7
fmtval(v, s) = abs(s) ≥ 10   ? (@sprintf("%.0f", v)) :
               abs(s) ≥ 1    ? (@sprintf("%.1f", v)) :
               abs(s) ≥ 0.1  ? (@sprintf("%.2f", v)) :
               abs(s) ≥ 0.01 ? (@sprintf("%.3f", v)) : (@sprintf("%.4f", v))
stripneg(s) = (startswith(s, "-") && all(c -> c in ".-0", s)) ? s[2:end] : s

cgQ = cgrad([PAL_RUST, RGB(0.93, 0.93, 0.93), PAL_TEAL])
panelsQ = []
for m in eachindex(Qs)
    Q = Qs[m]
    data_g = []
    for (g, group) in enumerate(TRANSITION_GROUPS)
        Π̄ = Pi_bar_test[group[1]]; Vf = Var_free_list[group[1]]
        E = Matrix{Float64}(undef, r_θ - 1, r_I - 1); H = similar(E)
        for iθ in 1:(r_θ - 1), iI in 1:(r_I - 1)
            E[iθ, iI], H[iθ, iI] = did_with_ci(Q, Π̄, Vf, iθ, iI)
        end
        push!(data_g, (E, H))
    end
    mx = maximum(abs, vcat(vec(data_g[1][1]), vec(data_g[2][1]))) + eps()
    for (g, group) in enumerate(TRANSITION_GROUPS)
        E, H = data_g[g]
        sig = [E[iθ, iI] - H[iθ, iI] > 0 ? 1.0 : E[iθ, iI] + H[iθ, iI] < 0 ? -1.0 : 0.0
               for iθ in 1:(r_θ - 1), iI in 1:(r_I - 1)]
        p = heatmap(1.5:(r_I - 0.5), 1.5:(r_θ - 0.5), sig; color = cgQ, clims = (-1, 1),
                    colorbar = false, aspect_ratio = :auto,
                    xlabel = (m == length(Qs) ? "I → I+1" : ""),
                    ylabel = (g == 1 ? "$(anchor_names[m])\nθ → θ+1" : ""),
                    title  = (m == 1 ? GROUP_LABELS[g] : ""),
                    xticks = (m == length(Qs) ?
                              (collect(1.5:(r_I - 0.5)), ["$i→$(i+1)" for i in 1:r_I-1]) :
                              (Float64[], String[])),
                    yticks = (g == 1 ?
                              (collect(1.5:(r_θ - 0.5)), ["$i→$(i+1)" for i in 1:r_θ-1]) :
                              (Float64[], String[])),
                    titlefontsize = TF_TITLE, guidefontsize = TF_GUIDE, tickfontsize = TF_TICK)
        for iθ in 1:(r_θ - 1), iI in 1:(r_I - 1)
            lbl = stripneg(fmtval(E[iθ, iI], mx)) * "\n±" * fmtval(H[iθ, iI], mx)
            annotate!(p, iI + 0.5, iθ + 0.5,
                      text(lbl, TF_AN, sig[iθ, iI] == 0 ? :black : colorant"white", :center))
        end
        push!(panelsQ, p)
    end
end
figQ = plot(panelsQ...; layout = (length(Qs), 2),
            size = (PAPER_FIGS ? (1040, 1520) : (900, 1300)),
            left_margin = (PAPER_FIGS ? 12 : 6)Plots.mm, margin = 2Plots.mm,
            plot_title = "Local complementarity of E[Q | θ, I] with 95% CIs",
            plot_titlefontsize = PAPER_FIGS ? 17 : 12)
savefig(figQ, joinpath(OUT_DIR, "local_modularity_by_outcome.pdf"))
println("  saved ", joinpath(OUT_DIR, "local_modularity_by_outcome.pdf"))

# ─────────────────────────────────────────────────────────────────────────
# (3) survivors_by_regime.pdf
# ─────────────────────────────────────────────────────────────────────────
println("\n[3/3] Building survivors_by_regime.pdf ...")

function surv_panel(Π, k; title="", xlabel="", ylabel="", legend=false)
    p = plot(ylims=(0.0, 1.05), legend = legend ? :topleft : false,
        titlefontsize=9, guidefontsize=8, tickfontsize=7, legendfontsize=7)
    isempty(title)  || title!(p, title)
    isempty(xlabel) || xlabel!(p, xlabel)
    isempty(ylabel) || ylabel!(p, ylabel)
    for kθ in 1:r_θ
        surv = [sum(Π[j, kθ, kI] for j in k+1:r_θ) for kI in 1:r_I]
        plot!(p, 1:r_I, surv, marker=:circle, markersize=3, linewidth=1.8,
            color=SKILL_COLORS[kθ], label = legend ? "\$\\theta=$(kθ)\$" : "")
    end
    p
end

# Left block: varying θᴾ (ω = 2).  Right block: varying ω (θᴾ = 2).
# Compact layout: one group (early) on top row, one (school) on bottom.
kω_fix = 2
kP_fix = 2
rows = length(TRANSITION_GROUPS)
cols = n_cut

plots_left  = Matrix{Any}(undef, rows * r_P, cols)  # rows × (cols)
plots_right = Matrix{Any}(undef, rows * r_ω, cols)

function build_grid(fix_dim::Symbol)
    mat = Matrix{Any}(undef, rows, 1)  # we'll fill later
    grids = []
    for (g, group) in enumerate(TRANSITION_GROUPS)
        inner = Matrix{Any}(undef,
            fix_dim == :theta_p ? r_P : r_ω, cols)
        for k in 1:cols
            for i in 1:(fix_dim == :theta_p ? r_P : r_ω)
                z = fix_dim == :theta_p ? (i - 1) * r_ω + kω_fix :
                                          (kP_fix - 1) * r_ω + i
                Π = Pi_z_pre[z][group[1]]
                show_leg = (g == 1 && i == 1 && k == cols)
                title_  = (g == 1 && i == 1) ? "k = $k" : ""
                lbl     = fix_dim == :theta_p ? "θᴾ = $i" : "ω = $i"
                ylab    = (k == 1) ? "$(GROUP_LABELS[g])\n$lbl" : ""
                xlab    = (i == (fix_dim == :theta_p ? r_P : r_ω) &&
                           g == rows) ? "Investment I" : ""
                inner[i, k] = surv_panel(Π, k;
                    title=title_, xlabel=xlab, ylabel=ylab, legend=show_leg)
            end
        end
        fig = plot(permutedims(inner)..., layout=(size(inner, 1), cols),
            size=(cols * 300, size(inner, 1) * 220),
            plot_titlefontsize = 10, margin = 2Plots.mm,
            plot_title = GROUP_LABELS[g] *
                (fix_dim == :theta_p ? "  —  by \$\\theta^P\$ (\$\\omega=$(kω_fix)\$)"
                                     : "  —  by \$\\omega\$ (\$\\theta^P=$(kP_fix)\$)"))
        push!(grids, fig)
    end
    grids
end

grids_p = build_grid(:theta_p)
grids_w = build_grid(:omega)

fig3 = plot(grids_p..., grids_w...,
    layout = (length(grids_p) + length(grids_w), 1),
    size = (cols * 300, (r_P + r_ω) * 220 * rows),
    margin = 2Plots.mm)
savefig(fig3, joinpath(OUT_DIR, "survivors_by_regime.pdf"))
println("  saved ", joinpath(OUT_DIR, "survivors_by_regime.pdf"))

println("\nDone.")
