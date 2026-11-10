#!/usr/bin/env bash
# Post-bootstrap inference for the regularized fit (run after bootstrap_reg.jl).
# Produces: survivor CI figure (bootstrap bands), OC test with bootstrap cov,
# regenerated OC test with reg-asymptotic cov, and the comparison markdown.
set -uo pipefail
cd "$(dirname "$0")/.."
JL=$(ls -d /Users/mullinsj/.julia/juliaup/julia-*/bin/julia | head -1)
REG=output/estimation_v3_eps_reg.jld2
BOOTJLD=output/bootstrap_reg.jld2

echo "[1/4] survivor CI figure (bootstrap bands) -> figures_boot/"
"$JL" -t 2 scripts/survivor_figure_reg.jl

echo "[2/4] OC test with BOOTSTRAP covariance -> output/paper_tables_boot/"
FIT_OVERRIDE=$REG BOOT_COV=$BOOTJLD "$JL" -t 2 scripts/test_supermodularity_by_cutpoint.jl \
  > output/_oc_boot.log 2>&1; tail -3 output/_oc_boot.log

echo "[3/4] OC test with REG-ASYMPTOTIC covariance (regenerate paper_tables_reg/)"
FIT_OVERRIDE=$REG REG_RUN=1 "$JL" -t 2 scripts/test_supermodularity_by_cutpoint.jl \
  > output/_oc_regasy.log 2>&1; tail -3 output/_oc_regasy.log

echo "[4/4] assemble comparison -> output/reg_comparison/BOOTSTRAP_INFERENCE.md"
"$JL" -t 1 scripts/_boot_inference_compare.jl

echo "DONE."
