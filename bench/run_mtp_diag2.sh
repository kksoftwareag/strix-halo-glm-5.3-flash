#!/usr/bin/env bash
# Diagnose 2: liegt die Grenze bei -b 4096 (ein Decode-Aufruf)? 3000 und 5000 Token Prompt, dann 8000 Token mit -b 16384.
set -uo pipefail
cd "$(dirname "$0")/.."
for i in $(seq 1 400); do grep -q "##### .* fertig" state/logs/streams-mtp6.log 2>/dev/null && ! pgrep -x llama-server >/dev/null && break; sleep 10; done
sleep 15
run() { echo "##### $(date '+%F %T') $1"; shift; python3 bench/multiuser.py --levels 1 --ctx-per-slot 16384 --max-tokens 256 --min-avail-gib 3.5 --budget-gib 105 --engine tk-mtp --mtp --quant UD-IQ1_M "$@"; sleep 15; }
run "tk-mtp, MTP, IQ1_M, Prompt 3000 Token (-b 4096)"           --ctx-tokens 3000
run "tk-mtp, MTP, IQ1_M, Prompt 5000 Token (-b 4096, zwei Decode-Aufrufe)" --ctx-tokens 5000
run "tk-mtp, MTP, IQ1_M, Prompt 8000 Token, -b 16384"           --ctx-tokens 8000 --batch 16384
echo "##### $(date '+%F %T') fertig"
