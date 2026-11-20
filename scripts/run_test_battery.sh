#!/bin/bash
# Run every ordinal-complementarity test on one bootstrap draw file in bootstrap_reg.jld2
# layout (2 groups). Writes CSVs to OUTDIR (relative to repo) and logs alongside.
# Usage: scripts/run_test_battery.sh BOOTFILE OUTDIR FIGTAG [NDM_N]
set -u
cd "$(dirname "$0")/.."
BOOTF="$1"; OUT="$2"; FT="$3"; NN="${4:-9685}"
mkdir -p "$OUT"
run() { local name="$1"; shift; echo ">>> $name"; env "$@" julia --project=. "scripts/$name.jl" > "$OUT/log_${name}${LOGSFX:-}.txt" 2>&1 || echo "!!! $name FAILED (see $OUT/log_${name}${LOGSFX:-}.txt)"; }
run test_supermodularity_by_cutpoint BOOT_ONLY=1 BOOT_COV="$BOOTF" OUTDIR="$OUT"
run test_modularity_induced BOOT="$BOOTF" OUTDIR="$OUT"
run fang_seo_test BOOT="$BOOTF" OUTDIR="$OUT"
run test_convex_concave_shape BOOT="$BOOTF" OUTDIR="$OUT"
run compare_modularity_tests BOOT="$BOOTF" OUTDIR="$OUT"
LOGSFX=_databoot run comonotonicity_test COMONO_MODE=databoot COMONO_BOOT="$BOOTF" OUTDIR="$OUT"
LOGSFX=_databoot_induced run comonotonicity_test COMONO_MODE=databoot_induced COMONO_BOOT="$BOOTF" OUTDIR="$OUT"
LOGSFX=_ndm run comonotonicity_test COMONO_MODE=ndm COMONO_BOOT="$BOOTF" NDM_N="$NN" OUTDIR="$OUT"
LOGSFX=_ndm_induced run comonotonicity_test COMONO_MODE=ndm_induced COMONO_BOOT="$BOOTF" NDM_N="$NN" OUTDIR="$OUT"
run _ndm_studentized BOOT="$BOOTF" NDM_N="$NN" OUTDIR="$OUT"
run _certificate_test BOOT="$BOOTF" OUTDIR="$OUT"
run _fixedQ_rationalize_test BOOT="$BOOTF" OUTDIR="$OUT"
run _Q_triangle_map BOOT="$BOOTF" NDM_N="$NN" OUTDIR="$OUT" FIGTAG="$FT"
run _Q_triangle_confirm BOOT="$BOOTF" OUTDIR="$OUT" FIGTAG="$FT"
echo "battery done: $OUT"
