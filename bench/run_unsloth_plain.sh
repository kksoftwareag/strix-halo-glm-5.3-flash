#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
for i in $(seq 1 400); do grep -q "##### .* fertig" state/logs/streams-mtp7.log 2>/dev/null && ! pgrep -x llama-server >/dev/null && break; sleep 10; done
sleep 15
echo "##### $(date '+%F %T') unsloth, ohne MTP, UD-IQ2_XXS, 1 Slot"
python3 bench/multiuser.py --engine unsloth --quant UD-IQ2_XXS --levels 1 --ctx-per-slot 16384 --ctx-tokens 8000 --max-tokens 512 --min-avail-gib 3.5 --budget-gib 105
echo "##### $(date '+%F %T') fertig"
