#!/usr/bin/env bash
# Baut alle vier Engines nacheinander im Hintergrund (HIP), schonend für den laufenden Betrieb.
#   nohup engine/build-all.sh > engine/build-all.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export JOBS="${JOBS:-4}" MAXMEM="${MAXMEM:-8}" HIGHMEM="${HIGHMEM:-6}"
for e in tk-mtp unsloth tk tk-merged; do
  echo "##### $(date '+%F %T') Engine $e"
  engine/build.sh "$e" hip
  echo "##### $(date '+%F %T') exit $?"
done
echo "##### alle Builds fertig $(date '+%F %T')"
