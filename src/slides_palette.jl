#####
# Shared plot palette and defaults for the UCLA slide deck.
#
# Include this near the top of any counterfactual / analysis script whose
# figures are meant for `slides/ucla_75/`.  Sets Plots.default(...) to match
# the DCT-style beamer theme (sepia background, teal/rust accents, Palatino
# font, generous margins so axis labels don't get clipped) and exports:
#
#   SLIDES_FIG_DIR     — path to slides/ucla_75/figures/ (created if absent)
#   OUTCOME_COLORS     — Dict{String,Color} keyed by adult-outcome name
#   PAL_TEAL / PAL_RUST / PAL_PLUM / PAL_GOLD / PAL_GREY   — primary swatches
#   save_slide_fig(fig, stem)  — saves both PNG (original output dir) and
#                                 PDF (slides dir)
#####

using Plots
# Plots.jl re-exports Colors; RGB is available without a separate using Colors.

# ── Palette matching the beamer preamble (FFF5EE / 63AA9C / E9633B etc.) ──
const PAL_BACK = RGB(255/255, 245/255, 238/255)
const PAL_TEXT = RGB(34/255,  30/255,  29/255)
const PAL_TEAL = RGB(99/255,  170/255, 156/255)
const PAL_RUST = RGB(233/255, 99/255,  59/255)
const PAL_PLUM = RGB(139/255, 90/255,  140/255)
const PAL_GOLD = RGB(230/255, 184/255, 85/255)
const PAL_GREY = RGB(0.55, 0.55, 0.55)
const PAL_NAVY = RGB(55/255,  90/255,  135/255)

# Color per adult-outcome (consistent across figures)
const OUTCOME_COLORS = Dict{String,RGB{Float64}}(
    "HS"        => PAL_TEAL,
    "College"   => PAL_RUST,
    "EarnQ"     => PAL_PLUM,
    "AlwaysEmp" => PAL_GOLD,
    "logEarn"   => PAL_NAVY,
)

# Ordinal skill colors (θ = 1..4) — used where we colour lines by θ
const SKILL_COLORS = [PAL_TEAL, PAL_PLUM, PAL_GOLD, PAL_RUST]

# ── Plot defaults (Plots.jl GR backend) ──
gr()

# ── Paper-vs-slides styling toggle ──────────────────────────────────────────
# Set ENV["PAPER_FIGS"]="1" (e.g. via scripts/generate_paper_figures.sh) to
# emit publication-styled figures — white background, larger fonts, thicker
# lines, no grid — into figures/ at the repo root instead of the slide deck.
# Default (unset) reproduces the original sepia slide styling unchanged.
const PAPER_FIGS = get(ENV, "PAPER_FIGS", "0") == "1"

# NOTE on font: we deliberately DO NOT set fontfamily to Palatino here, because
# the GR backend's Palatino ships without Greek glyphs — θ/ω/α would render as
# question marks in the PDF. The default GR font handles Unicode Greek.
if PAPER_FIGS
    default(
        background_color  = :white,
        foreground_color  = :black,
        framestyle        = :box,
        grid              = false,
        size              = (760, 480),   # larger canvas ⇒ relatively larger text
        linewidth         = 2,
        markersize        = 5,
        left_margin       = 7Plots.mm,
        bottom_margin     = 7Plots.mm,
        right_margin      = 4Plots.mm,
        top_margin        = 4Plots.mm,
        titlefontsize     = 15,
        guidefontsize     = 14,
        tickfontsize      = 12,
        legendfontsize    = 12,
    )
else
    default(
        background_color  = PAL_BACK,
        foreground_color  = PAL_TEXT,
        framestyle        = :box,
        grid              = true,
        gridalpha         = 0.22,
        minorgrid         = false,
        # generous margins so axis labels are not clipped when saved as PNG/PDF
        left_margin       = 8Plots.mm,
        bottom_margin     = 8Plots.mm,
        right_margin      = 3Plots.mm,
        top_margin        = 3Plots.mm,
        titlefontsize     = 11,
        guidefontsize     = 10,
        tickfontsize      = 9,
        legendfontsize    = 9,
    )
end

# ── Output helpers ──
# In paper mode figures go to figures/ at the repo root; otherwise to the deck.
# REG_RUN=1 redirects the paper-mode figures dir to figures_reg/ so a parallel
# (regularized-fit) run does not clobber the originals. Default behavior is
# byte-identical when REG_RUN is unset.
const _REG_RUN = get(ENV, "REG_RUN", "0") == "1"
# SLIDES_FIG_DIR env var overrides the destination (e.g. a different deck's figures/).
const SLIDES_FIG_DIR = haskey(ENV, "SLIDES_FIG_DIR") ? abspath(ENV["SLIDES_FIG_DIR"]) : (PAPER_FIGS ?
    abspath(joinpath(@__DIR__, "..", _REG_RUN ? "figures_reg" : "figures")) :
    abspath(joinpath(@__DIR__, "..", "slides", "ucla_75", "figures")))
isdir(SLIDES_FIG_DIR) || mkpath(SLIDES_FIG_DIR)

"""
    save_slide_fig(fig, stem; png_dir=nothing)

Save `fig` as a PDF into the active figures directory (the slide deck, or
`figures/` when `PAPER_FIGS` is set) and, if `png_dir` is given, also as
`<stem>.png` in `png_dir`. Returns the PDF path.
"""
function save_slide_fig(fig, stem::AbstractString; png_dir=nothing)
    pdf_path = joinpath(SLIDES_FIG_DIR, "$(stem).pdf")
    savefig(fig, pdf_path)
    if png_dir !== nothing
        png_path = joinpath(png_dir, "$(stem).png")
        savefig(fig, png_path)
    end
    pdf_path
end

nothing
