#!/bin/bash
# After bootstrap_ordrobit.jl (grouped) finishes: summary, survivor bands, asymptotic-vs-bootstrap SEs,
# the full test battery, and both-direction NDM. Usage: scripts/_postboot_grouped.sh TAG SHORT SEED
# e.g. scripts/_postboot_grouped.sh _rI4_noPIAT3_grp_q5a_tidy q5a 1   (data ENV must be exported)
set -u
cd "$(dirname "$0")/.."
TAG="$1"; SH="$2"; SEEDF="$3"
echo ">>> summary";  TAG="$TAG" julia --project=. scripts/_bootstrap_ordrobit_summary.jl > "output/bootstrap_summary_${SH}.txt" 2>&1
echo ">>> bands";    TAG="$TAG" julia --project=. scripts/_survivor_bands.jl > "output/survivor_bands_${SH}.log" 2>&1
echo ">>> asym";     TAG="$TAG" FITS=grouped SEEDF="$SEEDF" julia --project=. scripts/_asymptotic_vs_boot.jl > "output/asymptotic_vs_boot_${SH}.txt" 2>&1
echo ">>> battery";  scripts/run_test_battery.sh "output/bootstrap_ordrobit${TAG}_testfmt.jld2" "output/tests_ordrobit_${SH}" "_${SH}" 9685
echo ">>> ndm flip"; BOOTS="grouped=bootstrap_ordrobit${TAG}_testfmt.jld2" julia --project=. scripts/_ndm_flip.jl > "output/ndm_flip_${SH}.txt" 2>&1
echo "postboot done"
