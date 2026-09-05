#!/usr/bin/env bash
# Gegenstück: Engine tk ohne MTP mit einem Slot (kein kv-unified), 8k Prompt – fairer Engine-Vergleich.
set -uo pipefail
cd "$(dirname "$0")/.."
for i in $(seq 1 400); do grep -q "##### .* fertig" state/logs/streams-mtp5.log 2>/dev/null && ! pgrep -x llama-server >/dev/null && break; sleep 10; done
sleep 15
echo "##### $(date '+%F %T') tk, ohne MTP, UD-IQ2_XXS, 1 Slot"
python3 bench/multiuser.py --engine tk --quant UD-IQ2_XXS --levels 1 --ctx-per-slot 16384 --ctx-tokens 8000 --max-tokens 512 --min-avail-gib 3.5 --budget-gib 105
echo "##### $(date '+%F %T') fertig"
