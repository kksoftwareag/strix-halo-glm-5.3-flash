#!/usr/bin/env bash
# Ein-Stream-Vergleiche mit/ohne MTP je Engine und Quant (8k Prompt, 512 Token, 16k Kontext, kein kv-unified).
set -uo pipefail
cd "$(dirname "$0")/.."
run() { echo "##### $(date '+%F %T') $1"; shift; python3 bench/multiuser.py --levels 1 --ctx-per-slot 16384 --ctx-tokens 8000 --max-tokens 512 --min-avail-gib 3.5 --budget-gib 105 "$@"; sleep 15; }
run "tk-mtp, MTP, UD-IQ2_XXS"      --engine tk-mtp    --mtp --quant UD-IQ2_XXS
run "unsloth, MTP, UD-IQ2_XXS"     --engine unsloth   --mtp --quant UD-IQ2_XXS
run "unsloth, ohne MTP, UD-IQ1_M"  --engine unsloth         --quant UD-IQ1_M
run "tk-mtp, MTP, UD-IQ1_M"        --engine tk-mtp    --mtp --quant UD-IQ1_M
run "tk-merged, MTP, UD-IQ2_XXS"   --engine tk-merged --mtp --quant UD-IQ2_XXS
echo "##### $(date '+%F %T') fertig"
