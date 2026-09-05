#!/usr/bin/env bash
# Holt llama.cpp mit den GLM-5.3-Flash-Branches und legt je Engine einen Arbeitsbaum an
# (git worktree, ein gemeinsames Objekt-Repo unter engine/src). Alle Commits sind in
# engine/patches/PINNED.env festgehalten.
#
#   tk        timkhronos GLM5.3-Flash      – PR #27773 (Basis-Architektur, Vision, kein MTP)
#   tk-mtp    timkhronos GLM-5.3-Flash-MTP – PR #27917 (MTP/NextN-Draft, Stand 2026-09-02, baut auf #27773 vom 01.09.)
#   tk-merged tk + Patch 0001               – #27917 auf den aktuellen #27773-HEAD gemergt (eigene Auflösung, experimentell)
#   unsloth   unslothai glm5next/upstream  – PR #27754 (eigenes Namensschema „glm5next“, MTP seit 30.08.)
#
# Patch 0002 (Issue #25992: keine gepinnten Host-Puffer auf integrierten HIP-GPUs) kommt auf jeden Baum;
# ohne ihn liefern mehrere Slots auf gfx1151 vertauschte Antworten (gemessen im Qwen3.8-Projekt).
#
# Nutzung: engine/fetch.sh [tk|tk-mtp|tk-merged|unsloth|all]   (Default: all)
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/patches/PINNED.env"
SRC="$HERE/src"

if [[ ! -d "$SRC/.git" ]]; then
  git clone https://github.com/ggml-org/llama.cpp "$SRC"
fi
remote() { git -C "$SRC" remote get-url "$1" >/dev/null 2>&1 || git -C "$SRC" remote add "$1" "$2"; }
remote timkhronos https://github.com/timkhronos/llama.cpp
remote unsloth    https://github.com/unslothai/llama.cpp
remote vcruz305   https://github.com/vcruz305/llama.cpp

have() { git -C "$SRC" cat-file -e "$1^{commit}" 2>/dev/null; }
fetch_commit() {  # fetch_commit <remote> <branch> <commit>
  have "$3" || git -C "$SRC" fetch --quiet "$1" "$2"
  have "$3" || { echo "FEHLER: Commit $3 nicht in $1/$2 – Branch wurde umgeschrieben? Siehe PINNED.env" >&2; exit 1; }
}

apply_patch() {  # apply_patch <dir> <patch>
  if git -C "$1" apply --reverse --check "$2" 2>/dev/null; then echo "  $(basename "$2"): bereits enthalten"
  elif git -C "$1" apply --check "$2" 2>/dev/null; then git -C "$1" apply "$2"; echo "  $(basename "$2"): angewendet"
  elif git -C "$1" apply -3 --check "$2" 2>/dev/null; then git -C "$1" apply -3 "$2"; echo "  $(basename "$2"): per Drei-Wege-Merge angewendet"
  else echo "FEHLER: $(basename "$2") passt nicht auf $1" >&2; exit 1; fi
}

worktree() {  # worktree <name> <commit>
  local dir="$HERE/src-$1"
  if [[ ! -d "$dir/.git" && ! -f "$dir/.git" ]]; then
    git -C "$SRC" worktree add --quiet --detach "$dir" "$2"
  else
    git -C "$dir" checkout --quiet --detach "$2"
    git -C "$dir" reset --quiet --hard
  fi
  echo "== $1: $(git -C "$dir" log -1 --format='%h %cs %s' | cut -c1-100)"
}

do_tk()      { fetch_commit timkhronos GLM5.3-Flash "$TK_COMMIT";          worktree tk "$TK_COMMIT";           apply_patch "$HERE/src-tk"      "$HERE/patches/0002-25992-rocm-igpu-host-buffer.patch"; }
do_tk_mtp()  { fetch_commit timkhronos GLM-5.3-Flash-MTP "$TK_MTP_COMMIT"; worktree tk-mtp "$TK_MTP_COMMIT";   apply_patch "$HERE/src-tk-mtp"  "$HERE/patches/0002-25992-rocm-igpu-host-buffer.patch"; }
do_unsloth() { fetch_commit unsloth glm5next/upstream "$UNSLOTH_COMMIT";   worktree unsloth "$UNSLOTH_COMMIT"; apply_patch "$HERE/src-unsloth" "$HERE/patches/0002-25992-rocm-igpu-host-buffer.patch"; }
do_merged()  {
  fetch_commit timkhronos GLM5.3-Flash "$TK_COMMIT"
  # Für den Drei-Wege-Merge des großen Patches braucht git die Blobs des MTP-Branches.
  fetch_commit timkhronos GLM-5.3-Flash-MTP "$TK_MTP_COMMIT"
  worktree tk-merged "$TK_COMMIT"
  apply_patch "$HERE/src-tk-merged" "$HERE/patches/0001-27917-mtp-on-27773-head.patch"
  apply_patch "$HERE/src-tk-merged" "$HERE/patches/0002-25992-rocm-igpu-host-buffer.patch"
}
case "${1:-all}" in
  tk) do_tk ;; tk-mtp) do_tk_mtp ;; tk-merged) do_merged ;; unsloth) do_unsloth ;;
  all) do_tk; do_tk_mtp; do_merged; do_unsloth ;;
  *) echo "usage: $0 [tk|tk-mtp|tk-merged|unsloth|all]" >&2; exit 1 ;;
esac
echo "Quellen bereit. Bauen: engine/build.sh all hip"
