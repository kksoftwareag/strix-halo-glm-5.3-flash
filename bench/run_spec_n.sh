#!/usr/bin/env bash
# Draft-Tiefe 3 und 4 auf der unsloth-Engine, UD-IQ2_XXS, ein Stream, 8k Prompt (Vergleich: n2 = 14,3 t/s).
set -uo pipefail
cd "$(dirname "$0")/.."
run() { echo "##### $(date '+%F %T') $1"; shift; python3 bench/multiuser.py --engine unsloth --mtp --quant UD-IQ2_XXS --levels 1 --ctx-per-slot 16384 --ctx-tokens 8000 --max-tokens 512 --min-avail-gib 3.5 --budget-gib 105 "$@"; sleep 15; }
run "unsloth, MTP n3, UD-IQ2_XXS" --spec-n 3
run "unsloth, MTP n4, UD-IQ2_XXS" --spec-n 4
echo "##### $(date '+%F %T') fertig"
