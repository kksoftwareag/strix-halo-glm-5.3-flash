#!/usr/bin/env bash
# Terminal-Bench-Mini-20 nacheinander für mehrere Quants/Engines laufen lassen.
# Ein Stream (np 1), MTP an, ein Versuch je Aufgabe (pass@1).
#
#   bench/quality/run-quants.sh                          # UD-IQ2_XXS (Standard), weitere nach Download
#   bench/quality/run-quants.sh UD-IQ2_XXS UD-Q2_K_XL    # mehrere Quants
#   TB_PRESET=unsloth-agent bench/quality/run-quants.sh  # andere Engine (A/B)
#   TB_EFFORT=max TB_AGENT_TIMEOUT=5400 bench/quality/run-quants.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
QUANTS=("$@")
[ ${#QUANTS[@]} -eq 0 ] && QUANTS=(UD-IQ2_XXS)
PRESET="${TB_PRESET:-unsloth-agent}"
TIMEOUT="${TB_AGENT_TIMEOUT:-3600}"
EFFORT="${TB_EFFORT:-high}"             # low | high | max
CTX="${TB_CTX:-131072}"
FREI_GIB="${TB_FREE_GIB:-100}"
SUFFIX="-$PRESET-$EFFORT"
mkdir -p state/quality
warte_auf_speicher() {
  for _ in $(seq 1 120); do
    frei=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
    [ "$frei" -ge "$FREI_GIB" ] && return 0
    sleep 5
  done
  echo "WARNUNG: nur ${frei} GiB frei (erwartet >= ${FREI_GIB})" >&2
  return 1
}
for q in "${QUANTS[@]}"; do
  log="state/quality/tbmini-${q}${SUFFIX}.log"
  echo "=== $q  Preset $PRESET  Start $(date '+%F %T')  Zeitlimit ${TIMEOUT}s/Aufgabe  Denkstufe $EFFORT"
  warte_auf_speicher
  python3 bench/quality/tbench.py \
    --tier full --attempts 1 --agent-timeout "$TIMEOUT" \
    --apt-mirror "${TB_APT_MIRROR:-ftp.fau.de}" \
    --preset "$PRESET" --quant "$q" --ctx "$CTX" --reasoning "$EFFORT" \
    --job-name "tbmini-${q}${SUFFIX}" > "$log" 2>&1
  rc=$?
  echo "=== $q  Ende  $(date '+%F %T')  exit $rc  Log: $log"
  grep -E "^(Results|Aggregate|Passed):" "$log" | tail -3
done
echo "=== alle Quants fertig $(date '+%F %T')"
