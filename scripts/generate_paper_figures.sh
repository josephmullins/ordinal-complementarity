#!/usr/bin/env bash
# ============================================================================
# Regenerate paper-styled figures and persist every paper number to disk.
#
# Paper mode (PAPER_FIGS=1) makes the shared plotting layer (src/slides_palette.jl
# and scripts/generate_ucla_figures.jl) emit publication styling — white
# background, larger fonts, thicker lines, no grid — into figures/ at the repo
# root, leaving the slide-deck PDFs in slides/ucla_75/figures/ untouched.
#
# It also writes the numeric tables consumed by first_draft.tex into
# output/paper_tables/ so the paper's numbers are reproducible from disk:
#   welfare_investment.csv          (Table: investment reallocation)
#   welfare_income.csv              (Table: income reallocation)
#   ccp_income_semielasticity.csv   (β5/δ point estimate)
#
# Usage:  bash scripts/generate_paper_figures.sh
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

export PAPER_FIGS=1
mkdir -p figures output/paper_tables

echo "▶  survivors + local modularity (also exercises the inference path)"
julia --project=. scripts/generate_ucla_figures.jl

echo "▶  investment reallocation (figures + welfare_investment.csv)"
julia --project=. scripts/counterfactual_2period.jl

echo "▶  income reallocation (figures + welfare_income.csv)"
julia --project=. scripts/counterfactual_income_transfers.jl

echo "▶  β5/δ point estimate + bootstrap SE (ccp_income_semielasticity.csv)"
julia --project=. scripts/ccp_income_semielasticity.jl

echo "▶  ordinal-complementarity tests by cutpoint (Table 2: supermodularity_by_cutpoint.csv)"
julia --project=. scripts/test_supermodularity_by_cutpoint.jl

echo
echo "All paper figures written to figures/ ; tables to output/paper_tables/."
echo "Recompile first_draft.tex (its graphicspath already prefers figures/)."
