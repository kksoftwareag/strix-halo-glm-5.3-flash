#!/usr/bin/env bash
# Startet llama-server unter dem Speicher-Wächter. Beispiele:
#   ./serve.sh                                   # Preset tkmtp-agent
#   ./serve.sh --preset unsloth-agent --ctx 65536
#   ./serve.sh --preset chat-vision --reasoning low --webui
#   ./serve.sh --preset tk-plain --np 4 --kv-unified
# Vorher prüfen: ./run.sh show --preset tkmtp-agent
cd "$(dirname "$0")"
exec python3 -m glm53 run "$@"
