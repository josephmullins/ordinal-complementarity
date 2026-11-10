#!/usr/bin/env bash
# Regenerate every PDF in slides/ucla_75/figures/ by running each Julia
# script in its own process.  The cheap scripts run in parallel, then the
# (piecewise income-transfer scripts dropped 2026-11).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "▶  4 cheap scripts in parallel"
for s in generate_ucla_figures.jl \
         counterfactual_2period.jl \
         counterfactual_2period_colored_EI.jl \
         counterfactual_income_transfers.jl
do
  ( julia --project=. scripts/"$s" > "/tmp/regen_${s}.log" 2>&1 \
        && echo "  ✓  $s" \
        || { echo "  ✗  $s — see /tmp/regen_${s}.log"; exit 1; } ) &
done
wait

echo "All figure scripts completed."
