#!/usr/bin/env bash
# Baut eine oder alle Engines als HIP- (gfx1151) oder Vulkan-Backend nach engine/build-<engine>-<backend>/bin.
# Der Build läuft über tools/guarded.sh: eigene cgroup mit Speicherdeckel, nice/ionice, Pause bei knappem RAM –
# damit ein parallel laufender llama-Server nicht gestört wird.
#
#   engine/build.sh [tk|tk-mtp|tk-merged|unsloth|all] [hip|vulkan|both]      (Default: all hip)
#   JOBS=6 MAXMEM=10 engine/build.sh tk-mtp hip
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JOBS="${JOBS:-6}"; MAXMEM="${MAXMEM:-10}"; HIGHMEM="${HIGHMEM:-8}"
GUARD="$HERE/../tools/guarded.sh --max $MAXMEM --high $HIGHMEM --pause-gib ${PAUSE_GIB:-12} --resume-gib ${RESUME_GIB:-15} --"
COMMON=(
  -G Ninja -DCMAKE_BUILD_TYPE=Release
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_MTMD=ON                      # Vision (mmproj) über llama-server/llama-mtmd-cli
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=ON
  -DGGML_NATIVE=ON -DGGML_CCACHE=ON
)
TARGETS=(llama-server llama-bench llama-cli llama-perplexity llama-gguf-split llama-mtmd-cli)

build_one() {  # build_one <engine> <backend>
  local src="$HERE/src-$1" out="$HERE/build-$1-$2" log="$HERE/build-$1-$2.log"
  [[ -d "$src" ]] || { echo "Quelle $src fehlt – erst engine/fetch.sh $1" >&2; return 1; }
  echo "== $1 / $2 -> $out (Log: $log)"
  case "$2" in
    hip)
      $GUARD env HIPCXX=/usr/lib64/rocm/llvm/bin/clang++ cmake -S "$src" -B "$out" "${COMMON[@]}" \
        -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 -DGPU_TARGETS=gfx1151 \
        -DCMAKE_HIP_COMPILER=/usr/lib64/rocm/llvm/bin/clang++ > "$log" 2>&1 ;;
    vulkan)
      if ! command -v glslc >/dev/null 2>&1 || [[ ! -f /usr/include/vulkan/vulkan.h ]]; then
        echo "   Vulkan übersprungen: glslc / vulkan-headers / vulkan-loader-devel fehlen (sudo dnf install glslc vulkan-headers vulkan-loader-devel)"
        return 0
      fi
      $GUARD cmake -S "$src" -B "$out" "${COMMON[@]}" -DGGML_VULKAN=ON > "$log" 2>&1 ;;
    *) echo "Backend $2 unbekannt" >&2; return 1 ;;
  esac
  $GUARD cmake --build "$out" -j"$JOBS" --target "${TARGETS[@]}" >> "$log" 2>&1 || { echo "   FEHLER, siehe $log"; tail -20 "$log"; return 1; }
  ls -la "$out/bin/" | grep -E "llama-(server|bench|cli|perplexity)$" | awk '{print "   " $NF, $5}'
  "$out/bin/llama-server" --version 2>&1 | head -2 | sed 's/^/   /'
}
ENG="${1:-all}"; BE="${2:-hip}"
[[ "$ENG" == all ]] && ENGS=(tk-mtp unsloth tk tk-merged) || ENGS=("$ENG")
[[ "$BE" == both ]] && BES=(hip vulkan) || BES=("$BE")
for e in "${ENGS[@]}"; do for b in "${BES[@]}"; do build_one "$e" "$b"; done; done
echo "== fertig $(date '+%F %T')"
