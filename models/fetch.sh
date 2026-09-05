#!/usr/bin/env bash
# Lädt GLM-5.3-Flash-Quants und Zubehör in den Hugging-Face-Cache – speicherschonend:
# in einer cgroup mit 2–4 GiB Deckel (der Page-Cache des Downloads verdrängt nichts vom laufenden
# Server), zwei Verbindungen, niedrige IO-Priorität. Läuft im Vordergrund; für den Hintergrund:
#   nohup models/fetch.sh standard > state/logs/fetch.log 2>&1 &
#
#   models/fetch.sh standard        UD-Q2_K_XL + Shard_Rewrite-Header + mmproj (unsloth, AesSedai)
#   models/fetch.sh iq2xxs          UD-IQ2_XXS (liegt hier schon vor)
#   models/fetch.sh iq1s            UD-IQ1_S (86,7 GiB, mehr Platz für Kontext/Slots)
#   models/fetch.sh spark           sayyidfareed Spark-Q2XL-MTP (105 GiB, 2,80 bpw, nur mit kleinem VRAM-Carve-out)
#   models/fetch.sh aessedai-iq2s   AesSedai IQ2_S (105,8 GiB, braucht #27773-Namensschema, nur kleiner Carve-out)
#   models/fetch.sh aj-iq3xxs       aj9o9 AJ-IQ3_XXS (104,7 GiB, ohne NextN → kein MTP)
#   models/fetch.sh mmproj          nur die Vision-Projektoren
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
G="$HERE/../tools/guarded.sh --max 4 --high 2 --pause-gib 10 --resume-gib 13 --"
dl() { $G hf download "$@" --max-workers 2; }
what="${1:-standard}"
case "$what" in
  standard)
    dl unsloth/GLM-5.3-Flash-GGUF --include 'mmproj-F16.gguf' --include 'Shard_Rewrite/*UD-Q2_K_XL*' --include 'Shard_Rewrite/*UD-IQ2_XXS*' --include 'Shard_Rewrite/*UD-IQ1_S*'
    dl AesSedai/GLM-5.3-Flash-GGUF --include 'mmproj-GLM-5.3-Flash-F16.gguf' --include 'mmproj-GLM-5.3-Flash-Q8_0.gguf'
    dl unsloth/GLM-5.3-Flash-GGUF --include 'UD-Q2_K_XL/*';;
  iq2xxs) dl unsloth/GLM-5.3-Flash-GGUF --include 'UD-IQ2_XXS/*' --include 'Shard_Rewrite/*UD-IQ2_XXS*';;
  iq1s)   dl unsloth/GLM-5.3-Flash-GGUF --include 'UD-IQ1_S/*' --include 'Shard_Rewrite/*UD-IQ1_S*';;
  q2kxl)  dl unsloth/GLM-5.3-Flash-GGUF --include 'UD-Q2_K_XL/*' --include 'Shard_Rewrite/*UD-Q2_K_XL*';;
  spark)  dl sayyidfareed/GLM-5.3-Flash-Spark-Q2XL-MTP ;;
  aessedai-iq2s) dl AesSedai/GLM-5.3-Flash-GGUF --include 'IQ2_S/*' --include 'mmproj-GLM-5.3-Flash-Q8_0.gguf';;
  aj-iq3xxs) dl aj9o9/GLM-5.3-Flash-GGUF --include 'AJ-IQ3_XXS/*';;
  mmproj)
    dl unsloth/GLM-5.3-Flash-GGUF --include 'mmproj-F16.gguf'
    dl AesSedai/GLM-5.3-Flash-GGUF --include 'mmproj-GLM-5.3-Flash-F16.gguf' --include 'mmproj-GLM-5.3-Flash-Q8_0.gguf';;
  *) echo "unbekannt: $what" >&2; exit 2 ;;
esac
echo "fertig: $what"
