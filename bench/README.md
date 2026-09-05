# bench/ – Messwerkzeuge und Ergebnisse

Alle Läufe starten den Server **unter `memguard.py`** (SIGKILL bei zu wenig `MemAvailable`), weil GTT-Speicher
nicht im RSS auftaucht und der Kernel-OOM-Killer sonst die ganze Sitzung mitnimmt. Nichts davon läuft, solange
ein anderer llama-Server die Maschine belegt – die Skripte prüfen den Port, nicht den Nachbarn; vorher
`free -g` ansehen.

| Skript | Zweck |
| --- | --- |
| `memguard.py --min-avail-gib N -- CMD…` | Wächter + CSV-Mitschrieb (MemAvailable, GTT, VRAM, RSS) |
| `mem_probe.py NAME --preset P …` | Server starten, eine Anfrage, hart beenden; JSON mit Ladezeit, pp/tg, Draft-Akzeptanz, Peak-Verbrauch, Puffergrößen – **Kalibrierung des Speichermodells** |
| `llama_bench.sh` | Sweep 1: llama-bench pp512/tg128 bei Tiefe 0/8k/32k über Engines × ubatch × KV-Typ (ohne MTP) |
| `mtp_sweep.py --engine E --quant Q` | Sweep 2: MTP-Feintuning (Draft-Tiefe 1–4, p_min, Temperatur, ngram-mod) mit drei Prompt-Typen |
| `multiuser.py --engine E` | Sweep 3: 1/2/4/8 gleichzeitige Nutzer, Codewort-Prüfung auf vertauschte Slots |
| `context_limits.py` | Wie viele Slots welcher Größe passen (Speichermodell), für beide Carve-out-Szenarien |

Ergebnisse: `results/raw/*.json` (llama-bench), `results/mem/*.json|csv|log` (Footprints), `results/mtp/summary.jsonl`,
`results/multi/*.json`. Auswertung in `../docs/RESEARCH.md` und auf den Doku-Seiten.

## Reihenfolge nach dem „Go“

1. `python3 bench/mem_probe.py probe-tkmtp-32k --preset tkmtp-agent --ctx 32768` – passt der Standard? Puffergrößen ins Speichermodell übernehmen (`glm53/memory.py`).
2. Dasselbe für `unsloth-agent`, `tk-plain`, `merged-agent`; dann 131k Kontext.
3. `bench/llama_bench.sh` (Engines × ubatch × Tiefe) – zeigt, ob der HIP-Top-k-Pfad bei Tiefe hält.
4. `python3 bench/mtp_sweep.py --engine tk-mtp` und `--engine unsloth` (je ~1 h).
5. `python3 bench/multiuser.py --engine tk` und mit `--mtp`.
6. Qualität: `bench/quality/` (Terminal-Bench-Mini-20), siehe dort.

Wichtig beim Beenden von Probe-Servern: SIGINT löst einen minutenlangen Teardown (GTT-Freigabe) aus – die Skripte
senden deshalb SIGKILL.
