#!/usr/bin/env bash
# Sweep 1: llama-bench (ohne MTP) über Engines × Quant × ubatch × KV-Typ × Tiefe.
# Misst pp512/tg128 bei Tiefe 0, 8k und 32k (-d), damit der Top-k-/Indexer-Pfad bei Tiefe sichtbar wird.
#   bench/llama_bench.sh                       # alle gebauten Engines, UD-IQ2_XXS
#   ENGINES="tk-mtp unsloth" QUANTS="UD-IQ2_XXS UD-Q2_K_XL" bench/llama_bench.sh
#   DEPTHS="0,16384" UBS="512 2048" bench/llama_bench.sh
# Läuft unter memguard (Abbruch bei < 8 GiB MemAvailable). Ergebnisse: bench/results/raw/*.json
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=bench/results/raw; mkdir -p "$OUT"
ENGINES="${ENGINES:-$(ls -d engine/build-*-hip 2>/dev/null | sed 's#engine/build-##; s#-hip##' | tr '\n' ' ')}"
QUANTS="${QUANTS:-UD-IQ2_XXS}"
UBS="${UBS:-512 1024 2048}"
KVS="${KVS:-q8_0}"
DEPTHS="${DEPTHS:-0,8192,32768}"
THREADS="${THREADS:-8}"
MIN_AVAIL="${MIN_AVAIL:-8}"
run() {  # run <name> <bin> <model> <args…>
  local name=$1 bin=$2 model=$3; shift 3
  echo "### $(date +%T) $name"
  local t0=$SECONDS
  python3 bench/memguard.py --min-avail-gib "$MIN_AVAIL" -- "$bin" -m "$model" -ngl 99 -t "$THREADS" -fa on -p 512 -n 128 -r 2 -d "$DEPTHS" "$@" -o json --progress > "$OUT/$name.json" 2> "$OUT/$name.err"
  echo "exit=$? dauer=$((SECONDS-t0))s"
  python3 - "$OUT/$name.json" <<'PY'
import json, sys
try:
    for r in json.load(open(sys.argv[1])):
        print(f"   {r['test']:>12}  {r['avg_ts']:8.2f} t/s ± {r['stddev_ts']:.2f}   (ub={r['n_ubatch']} ctk={r['type_k']} fa={r['flash_attn']})")
except Exception as e:
    print("   (kein JSON:", e, ")")
PY
}
for e in $ENGINES; do
  bin="engine/build-$e-hip/bin/llama-bench"; [ -x "$bin" ] || { echo "fehlt: $bin"; continue; }
  naming=glm5-next; case "$e" in unsloth*) naming=glm5next;; esac
  for q in $QUANTS; do
    model=$(ls models/variants/$q.$naming/*-00001-of-*.gguf 2>/dev/null | head -1)
    [ -n "$model" ] || { echo "Variante fehlt: models/variants/$q.$naming (./run.sh variants --create)"; continue; }
    for ub in $UBS; do for kv in $KVS; do
      run "$e-$q-ub$ub-$kv" "$bin" "$model" -ub "$ub" -b "$(( ub > 2048 ? ub : 2048 ))" -ctk "$kv" -ctv "$kv"
    done; done
  done
done
echo "### fertig $(date +%T)"
