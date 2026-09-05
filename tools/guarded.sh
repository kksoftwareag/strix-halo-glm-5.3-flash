#!/usr/bin/env bash
# Führt ein Kommando so aus, dass es den laufenden Betrieb (anderer llama-Server) nicht gefährdet:
#   - eigene cgroup mit hartem Speicherdeckel (MemoryMax) und weichem (MemoryHigh): Page-Cache und
#     Compiler-Speicher werden innerhalb dieser cgroup zurückgeholt, nie beim Nachbarn
#   - niedrige CPU-/IO-Priorität (nice 19, ionice idle, CPUWeight/IOWeight klein)
#   - Wächter: fällt MemAvailable unter --pause-gib, wird die Prozessgruppe angehalten (SIGSTOP)
#     und erst wieder fortgesetzt, wenn --resume-gib überschritten ist
# Nutzung: tools/guarded.sh [--max GIB] [--high GIB] [--pause-gib N] [--resume-gib N] -- CMD ARGS…
set -euo pipefail
MAX=8; HIGH=6; PAUSE=12; RESUME=15
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max) MAX="$2"; shift 2 ;;
    --high) HIGH="$2"; shift 2 ;;
    --pause-gib) PAUSE="$2"; shift 2 ;;
    --resume-gib) RESUME="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "unbekannte Option: $1" >&2; exit 2 ;;
  esac
done
[[ $# -gt 0 ]] || { echo "Kommando fehlt" >&2; exit 2; }
avail_gib() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
setsid systemd-run --user --scope --quiet \
  -p "MemoryMax=${MAX}G" -p "MemoryHigh=${HIGH}G" -p IOWeight=20 -p CPUWeight=20 \
  -- nice -n 19 ionice -c 3 "$@" &
PID=$!
paused=0
trap 'kill -CONT -- -"$PID" 2>/dev/null; kill -TERM -- -"$PID" 2>/dev/null; exit 130' INT TERM
while kill -0 "$PID" 2>/dev/null; do
  a=$(avail_gib)
  if (( paused == 0 && a < PAUSE )); then
    echo "[guarded] MemAvailable ${a} GiB < ${PAUSE} -> pausiere $PID" >&2
    kill -STOP -- -"$PID" 2>/dev/null || kill -STOP "$PID"; paused=1
  elif (( paused == 1 && a > RESUME )); then
    echo "[guarded] MemAvailable ${a} GiB > ${RESUME} -> weiter" >&2
    kill -CONT -- -"$PID" 2>/dev/null || kill -CONT "$PID"; paused=0
  fi
  sleep 2
done
wait "$PID"
