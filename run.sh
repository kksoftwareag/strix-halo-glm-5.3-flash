#!/usr/bin/env bash
# Einstieg: ./run.sh presets | models | show --preset … | cmd … | run … | variants --create | gguf-info DATEI
cd "$(dirname "$0")"
exec python3 -m glm53 "$@"
