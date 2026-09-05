# Agenten-Benchmark: Terminal-Bench-Mini-20

Anbindung an **Terminal-Bench-Mini-20** (20 Aufgaben aus Terminal-Bench 2.1, Agent Terminus-2 über Harbor in Docker).
Der Benchmark selbst kommt aus <https://github.com/kyuz0/terminal-bench-mini> (Apache-2.0) und wird geholt, nicht versioniert:

```bash
bench/quality/fetch.sh            # klont den Benchmark, prüft docker, uv, Speicher, Platz
```

## Läufe

```bash
bench/quality/tbench.py --dry-run                                  # Speicherbilanz und Kommandos
bench/quality/tbench.py --tier smoke --attempts 1                  # Rauchtest, eine Aufgabe
bench/quality/run-quants.sh                                        # UD-IQ2_XXS mit tkmtp-agent, 20 Aufgaben, pass@1
TB_PRESET=unsloth-agent bench/quality/run-quants.sh                # Engine-Vergleich
bench/quality/run-quants.sh UD-IQ2_XXS UD-IQ1_S                    # mehrere Quants
python3 bench/quality/report.py                                    # docs/TERMINAL-BENCH.md und docs/tbmini-data.js
```

Die Denkstufe (`TB_EFFORT`, Default `high`; GLM kennt `low`, `high`, `max`) und das Preset stecken in der Kennung jedes Laufs.
Server-Overrides wie im Programm (`--ctx`, `--spec-n`, `--no-mtp`, `--np` …). Der Server läuft unter `bench/memguard.py`;
die Docker-Container brauchen zusätzlich rund 8 GiB Spielraum – bei UD-IQ2_XXS mit MTP und 16 GiB Carve-out ist das knapp;
notfalls `--ctx 65536` oder `--cache-ram 256`.

`dockershim/docker` schiebt den Containern bei Bedarf einen schnellen Ubuntu-Spiegel unter (`--apt-mirror`, Default `auto`).
