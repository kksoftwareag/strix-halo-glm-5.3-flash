#!/usr/bin/env bash
# Diagnose für GGML_ASSERT(width == mtp_dsa_sel_width) auf #27917: Prompt kleiner als ein Ubatch (600 Token) vs. über
# die Ubatch-Grenze (1500 Token bei -ub 1024). Wartet, bis die MTP-Serie fertig ist.
set -uo pipefail
cd "$(dirname "$0")/.."
for i in $(seq 1 400); do grep -q "##### .* fertig" state/logs/streams-mtp4.log 2>/dev/null && ! pgrep -x llama-server >/dev/null && break; sleep 10; done
sleep 15
run() { echo "##### $(date '+%F %T') $1"; shift; python3 bench/multiuser.py --levels 1 --ctx-per-slot 16384 --max-tokens 256 --min-avail-gib 3.5 --budget-gib 105 "$@"; sleep 15; }
run "tk-mtp, MTP, UD-IQ1_M, Prompt 600 Token (ein Ubatch)"       --engine tk-mtp --mtp --quant UD-IQ1_M --ctx-tokens 600
run "tk-mtp, MTP, UD-IQ1_M, Prompt 1500 Token (zwei Ubatches)"   --engine tk-mtp --mtp --quant UD-IQ1_M --ctx-tokens 1500
echo "##### $(date '+%F %T') fertig"
