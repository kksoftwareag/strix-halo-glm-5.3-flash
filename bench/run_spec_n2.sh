#!/usr/bin/env bash
# Draft-Tiefe 5–8 auf der unsloth-Engine, UD-IQ2_XXS, ein Stream, 8k Prompt (Vergleich: n4 = 15,2 t/s).
set -uo pipefail
cd "$(dirname "$0")/.."
run() { echo "##### $(date '+%F %T') $1"; shift; python3 bench/multiuser.py --engine unsloth --mtp --quant UD-IQ2_XXS --levels 1 --ctx-per-slot 16384 --ctx-tokens 8000 --max-tokens 512 --min-avail-gib 3.5 --budget-gib 105 "$@"; sleep 15; }
for n in 5 6 7 8; do run "unsloth, MTP n$n, UD-IQ2_XXS" --spec-n "$n"; done
echo "##### $(date '+%F %T') fertig"
