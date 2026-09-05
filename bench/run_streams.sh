#!/usr/bin/env bash
# Kleiner Stream-Benchmark: 1/2/4 gleichzeitige Nutzer, einmal ohne MTP (Engine tk), einmal mit MTP (tk-mtp).
# 16k Kontext je Slot, 8k Prompt, 512 Token Ausgabe, reasoning low. Läuft nacheinander (ein Modell zur Zeit).
set -uo pipefail
cd "$(dirname "$0")/.."
LEVELS="${LEVELS:-1,2,4}"
echo "##### $(date '+%F %T') tk ohne MTP"
python3 bench/multiuser.py --engine tk --levels "$LEVELS" --ctx-per-slot 16384 --ctx-tokens 8000 --max-tokens 512 --min-avail-gib 3.5
echo "##### $(date '+%F %T') tk-mtp mit MTP"
python3 bench/multiuser.py --engine tk-mtp --mtp --levels "$LEVELS" --ctx-per-slot 16384 --ctx-tokens 8000 --max-tokens 512 --min-avail-gib 3.5
echo "##### $(date '+%F %T') fertig"
