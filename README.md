# strix-halo-glm-5.3-flash

> **Hinweis:** Der Großteil dieses Repositorys (Skripte, Werkzeug, Recherche, Dokumentation und Website) wurde von
> Claude Fable 5.1 (Anthropic) erstellt, gesteuert und geprüft durch den Autor. Messwerte stammen – sobald vorhanden –
> von echten Läufen auf der beschriebenen Hardware.

Werkzeuge und Dokumentation, um **GLM-5.3-Flash** (320 B Parameter, 18 B aktiv, MTP, Vision) auf einem **AMD Strix Halo**
Rechner (Ryzen AI MAX+ 395 / Radeon 8060S, 128 GB Unified Memory) mit **llama.cpp** einzurichten, zu starten, zu überwachen
und zu messen. Das Modell ist in llama.cpp noch nicht gemerged; dieses Projekt kombiniert die passenden Pull-Requests
(#27773 + #27917 für MTP, #27754 als Vergleich), löst den Konflikt der zwei GGUF-Namensschemata und schützt den Speicher.

Alle Entscheidungen, die Recherche mit Quellen und der Messplan stehen in `docs/` (Website) und `docs/RESEARCH.md`.

**Stand 5. September 2026, abends:** Setup steht, erste Messungen liegen vor (Footprint-Probe, Stream-Benchmark mit 1/2/4 Nutzern
über drei Engines und zwei Quants). Ergebnis in Kürze: unsloth-Engine mit MTP 15,5 t/s (UD-IQ2_XXS, Draft-Tiefe 5; Optimum n4–n5) bzw. 15,7 t/s (UD-IQ1_M) bei
8k Prompt-Tiefe, Prefill 150 t/s, mehrere Streams erhöhen den Durchsatz nicht; der MTP-PR #27917 stürzt ab etwa 2k Token Kontext ab.
Details: `docs/benchmarks.html`, `docs/RESEARCH.md` Abschnitt 8. Offene Messungen: `bench/README.md`.

## Was hier drin ist

- **`glm53/`** – Python-Werkzeug (nur Standardbibliothek): GGUF-Header lesen und umschreiben, Modelle/Engines/Projektoren
  finden, Speicherbedarf schätzen, Presets, fertige Kommandozeile, Start unter dem Speicher-Wächter.
- **`engine/`** – `fetch.sh` (llama.cpp mit vier Branches an festgehaltenen Commits, Patches), `build.sh` (HIP für gfx1151,
  Vulkan vorbereitet), `patches/` (0001: #27917 auf #27773-HEAD gemergt, 0002: iGPU-Host-Puffer #25992, `PINNED.env`).
- **`models/`** – `fetch.sh` (speicherschonender Download in einer cgroup), `variants/` (Shard-Sätze je Namensschema).
- **`bench/`** – Speicher-Wächter, Footprint-Probe, llama-bench-Sweep, MTP-Feintuning, Mehrnutzer-Test, Slot-Rechner;
  `bench/quality/` Terminal-Bench-Mini-20.
- **`docs/`** – Website (GitHub Pages aus `docs/`), `RESEARCH.md`, `QUALITAETS-BENCHMARKS.md`, `TERMINAL-BENCH.md`.
- **`tools/`** – `guarded.sh` (cgroup-Deckel, nice/ionice, Pause bei knappem RAM), `mkpage.sh` (Doku-Seiten).

## Die wichtigsten Erkenntnisse der Recherche

1. **Branches.** PR #27773 (timkhronos, Schema `glm5-next`) ist der Kandidat für den Merge (zwei Approvals); MTP kommt aus
   PR #27917 (`--spec-type draft-mtp`, NextN im Haupt-GGUF) – auf dieser Maschine stürzt #27917 aber ab etwa 2k Token Kontext
   mit `GGML_ASSERT(width == mtp_dsa_sel_width)` ab. Der unsloth-Branch #27754 (`glm5next`) hat eine eigene, funktionierende
   MTP-Implementierung und ist deshalb die Standard-Engine. Alle enthalten den ROCm-Radix-Top-k (#27466).
2. **GGUF-Schema.** Die unsloth-Quants liegen im Schema `glm5next`; für #27773 braucht nur Shard 1 (9 MB) ein anderes Header
   (unsloth `Shard_Rewrite/` oder `./run.sh rename-shard1`). `./run.sh variants --create` legt beide Sätze per Symlink an.
3. **Speicher.** llama.cpp/HIP nutzt auf gfx1151 den BIOS-VRAM-Carve-out (16 GiB) nicht – alles liegt in GTT. Mit 16 GiB
   Carve-out passt UD-IQ2_XXS (94,8 GiB) mit MTP knapp; UD-Q2_K_XL (101,3 GiB) und die 2,8-bpw-Community-Quants passen nur,
   wenn der Carve-out im BIOS auf 512 MiB–1 GiB gestellt wird (+15 GiB).
4. **Flags.** `-fa on -ctk q8_0 -ctv q8_0 -ub 1024 --load-mode none --cache-ram 512`; `reasoning_effort` kennt nur `low`,
   `high`, `max` (alles andere wird `max`); mehrere Slots brauchen `--kv-unified`.

## Schnellstart

```bash
models/fetch.sh iq2xxs            # UD-IQ2_XXS (liegt hier bereits vor)
./run.sh variants --create        # Shard-Sätze je Schema
engine/fetch.sh && engine/build.sh all hip
./run.sh models                   # Inventar und Maschine
./run.sh show --preset unsloth-agent
python3 bench/mem_probe.py probe-unsloth --preset unsloth-agent --ctx 32768   # erster Lauf: Footprint messen
./serve.sh --preset unsloth-agent   # Server unter dem Speicher-Wächter
uv run --group dev pytest -q      # Tests
```

Engines: `unsloth` (Standard, MTP), `tk` (ohne MTP), `tk-mtp` und `tk-merged` (MTP-Assert, siehe Recherche). Presets: `./run.sh presets`.

## Ordner

| Pfad | Inhalt |
| --- | --- |
| `glm53/` | `gguf.py` (Reader/Umschreiber), `discovery.py`, `memory.py`, `config.py` (Presets, Kommando), `hardware.py`, `__main__.py` (CLI) |
| `engine/` | `fetch.sh`, `build.sh`, `build-all.sh`, `patches/`; nach dem Bauen `build-<engine>-hip/bin/` |
| `models/` | `fetch.sh`, `variants/` (nicht versioniert) |
| `bench/` | Messskripte und Ergebnisse (`results/`), `quality/` (Terminal-Bench-Mini-20: `fetch.sh`, `tbench.py`, `run-quants.sh`, `report.py`) |
| `docs/` | Website und Recherche |
| `tests/` | `uv run --group dev pytest -q` |
| `state/` | Logs, Profile, Messergebnisse (nicht versioniert) |

## Dokumentations-Website

`docs/` enthält eine statische Website (`index.html`, `entscheidungen.html`, `speicher.html`, `benchmarks.html`,
`terminal-bench.html`, `anleitung.html`, `recherche.html`). Veröffentlichen: in den Repository-Einstellungen unter „Pages“
den Branch `main` mit dem Ordner `/docs` wählen; `.nojekyll` liegt bei. Seiten neu bauen: `tools/mkpage.sh`.

## Dank

- [llama.cpp](https://github.com/ggml-org/llama.cpp), timkhronos (PR #27773/#27917), unsloth (PR #27754, Quants), vcruz305
- [Zhipu / Z.ai](https://huggingface.co/zai-org/GLM-5.3-Flash) für das Modell
- sayyidfareed, AesSedai, aj9o9, avar6 für Community-Quants und Projektoren
- Das Schwesterprojekt [strix-halo-qwen-3.8-flash-next](https://github.com/kksoftwareag/strix-halo-qwen-3.8-flash-next)

## Lizenz

EUPL-1.2, siehe `LICENSE`. Modelle, llama.cpp und Patches Dritter stehen unter ihren eigenen Lizenzen.
