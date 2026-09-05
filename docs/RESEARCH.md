# GLM-5.3-Flash auf Strix Halo – Recherche (Stand 2026-09-05)

Zusammengetragen aus der Modellkarte, der unsloth-Dokumentation, den offenen llama.cpp-Pull-Requests samt Diskussionen,
Community-Quantisierungen auf Hugging Face, Strix-Halo-Berichten und dem eigenen Vorlauf vom 29.08.2026 auf dieser Maschine
(Ryzen AI MAX+ 395, 109,7 GiB nutzbarer RAM, ROCm/HIP 7.1, Fedora 44). **Eigene Messwerte sind als solche markiert; die
eigentlichen Messreihen stehen noch aus** (Abschnitt 8). Quellen: nummerierte Liste am Ende.

## 1. Modell-Fakten (aus dem GGUF-Header gelesen, eigene Prüfung)

- **Architektur** `Glm5NextForConditionalGeneration`, in llama.cpp `glm5-next` (#27773) bzw. `glm5next` (#27754): 320 B Parameter,
  18 B aktiv, 46 Blöcke = 45 Trunk-Layer + 1 NextN-Layer (MTP). 34 Layer KDA (Kimi Delta Attention, gated delta-net, rekurrenter
  Zustand, head_dim 128) und 11 Layer DSA/MLA (nur diese haben einen KV-Cache: kv_lora_rank 512, keine RoPE-Dimensionen).
  DSA-Indexer: 32 Heads × 128, **Top-k 2048 über Pools von 4 Tokens** (`indexer.kpool = 4`). 288 geroutete Experten, 8 aktiv + 1
  gemeinsamer, Experten-FFN 2048, 3 dichte Leading-Blöcke. Hyper-Connections (mHC, 4 Streams, Sinkhorn 20 Iterationen).
  Vokabular 154 880. Kontext 1 048 576 [1][2][8].
- **Vision**: nativ multimodal; in llama.cpp als separater Projektor (`mmproj`, glm5v-Encoder mit Clamped-SwiGLU) [8].
- **Thinking / Chat-Template** (aus dem GGUF): `reasoning_effort` akzeptiert **genau `low` und `high`; jeder andere Wert
  (auch `medium`) wird still zu `max`** (Template-Default). `enable_thinking` schaltet Thinking ab, `clear_thinking` (Default
  false) steuert, ob Reasoning früherer Runden im Verlauf bleibt. Tool-Calls im XML-Format
  `<tool_call>{name}<arg_key>…</arg_key><arg_value>…</arg_value>`.
- **Sampling** (im GGUF hinterlegt, unsloth-Doku): temp 1.0, top_p 0.95; reasoning_effort „max“ empfohlen [2].
- **NextN/MTP**: Layer 45 (`blk.45.*`, 2,60 GiB in UD-IQ2_XXS) steckt in jedem unsloth-Quant; ohne MTP wird er ignoriert
  („unused tensor … ignoring“, eigenes Log vom 29.08.), mit MTP geladen. `index_share_for_mtp_iteration = true` (der Draft-Head
  übernimmt die Indexer-Auswahl des Trunks) [8][9].

## 2. Quants und was auf 128 GB passt

Größen als GiB (HF-API, eigene GGUF-Analyse); unsloth gibt dezimale GB an. „Accuracy“ = unsloth-Angabe Top-1 vs. BF16 [2].

| Quant | Datei | Experten | NextN | Qualität | Passt mit 16 GiB Carve-out? | mit 1 GiB Carve-out? |
| --- | --- | --- | --- | --- | --- | --- |
| aj9o9 AJ-IQ2_XXS | 81,4 | – | **fehlt** (kein MTP) | Top-1 73,2 %, KLD 0,71 | ja, viel Platz | ja |
| UD-IQ1_S | 86,7 | | ja | 71 % | ja | ja |
| UD-IQ1_M | 90,9 | | ja | 73 % (IQ1_M lt. eauchs „garbage“-anfällig [6]) | ja | ja |
| **UD-IQ2_XXS** | **94,8** | 86,2 | 2,6 | **76 %** | **ja** (eigener Lauf 29.08.: ~12 GiB frei bei 131k) | ja |
| UD-Q2_K_XL | 101,3 | | ja | 78 % | **nein** mit MTP (Schätzung 106 GiB Bedarf) | ja |
| sayyidfareed Spark-Q2XL-MTP | 104,5 | IQ2_XS/IQ3_XXS, Attn Q8_0 | ja | HumanEval+ 153/164 vs. UD-Q2_K_XL 148/164 [12] | nein | ja (~15 GiB frei) |
| aj9o9 AJ-IQ3_XXS | 104,7 | | fehlt | Top-1 81,8 %, KLD 0,36 | nein | ja, ohne MTP |
| AesSedai IQ2_S | 105,8 | Q6_K / IQ2_XS / IQ3_XXS | ja | KLD 0,375, PPL +33 % [13] | nein | ja |
| UD-IQ3_XXS | 112,1 | | ja | 82 % | nein | nein (zu knapp) |

**Zentrale Erkenntnis zum Speicher** (eigene Messung am laufenden Qwen-Server und am Log vom 29.08.): llama.cpp/HIP legt
auf gfx1151 *alle* Puffer (Gewichte, KV, Compute) in **GTT**, nicht in den BIOS-VRAM-Carve-out. Bei 16 GiB Carve-out sind
16 GiB RAM für llama.cpp unsichtbar (`mem_info_vram_used` 0,2 GiB, während 77 GiB GTT belegt sind). **Den UMA Frame Buffer im
BIOS auf 512 MiB–1 GiB zu stellen bringt ~15 GiB** und macht UD-Q2_K_XL, Spark-Q2XL-MTP und AesSedai-IQ2_S mit MTP nutzbar –
AMD empfiehlt das ohnehin für Strix Halo [17]. Ohne diese Änderung ist **UD-IQ2_XXS der beste Quant mit MTP**, der passt.

## 3. llama.cpp: drei konkurrierende PRs und der MTP-PR

| PR | Autor / Branch | Stand 05.09. | Schema | Besonderheiten |
| --- | --- | --- | --- | --- |
| **#27773** | timkhronos `GLM5.3-Flash` @ 8134115 (04.09.) | offen, **von CISC und ngxson approved**, ggerganov-Review angefragt; 41 Commits, 5 hinter master | `glm5-next` | Pooled-Index-Cache (Decode tiefen-invariant: 25,3 → 23,7 t/s bei 29k, #27754: 24,8 → 14,6), `-fa on` korrekt und speicherflach, Sparse-FA für DSA-Prefill (03.09.), Multi-Stream (`--kv-unified` und `--no-kv-unified`), Vision (glm5v), ~1 GB präzisionsempfindliche Tensoren unquantisiert; NextN-Tensoren erhalten, aber ungenutzt [8][10] |
| **#27917** | timkhronos `GLM-5.3-Flash-MTP` @ 5b8593b (02.09.) | Draft, baut auf #27773 vom 01.09.; 70 hinter master, Merge-Konflikte mit master | `glm5-next` | `--spec-type draft-mtp`, kein separates Draft-GGUF, `index_share_for_mtp_iteration`; Akzeptanz 0,745 (Positionen 0,92/0,81/0,60/0,44/0,35), +15–30 % bei CPU-Offload; auf 4×Ampere (alles in VRAM) **langsamer** (25,0 → 19,0 t/s n1, 16,4 t/s n3) – MTP lohnt sich nur, wo Bandbreite der Engpass ist (Strix Halo: ja) [9] |
| #27754 | unsloth `glm5next/upstream` @ 629b505 (04.09.) | offen, 40 Commits, Konflikte mit master | `glm5next` | MTP seit 30.08. („Add MTP support“), „Faster inference“, Indexer-Key-Cache; kein `can_reuse` für den kpool-Input → Graph wird jeden Token neu gebaut (~7 % Decode-Verlust); die Metal-`@@@`-Kollaps-Meldungen sind ein int32-Überlauf in Metal `mul_mm` (#28210), betreffen HIP nicht [7] |
| #27752 | eauchs `glm5next/add-glm-5.3-flash` @ 1d0c76f (03.09.) | offen, textonly | – | nicht weiter verfolgt (kein Vision, kein MTP) [6] |
| (Referenz) | vcruz305 `glm5next-mtp` @ 4a06ec6 (28.08.) | 180 hinter master | `glm5next` | Grundlage des eigenen Versuchs vom 29.08.; **ohne** den HIP-Top-k-Fix [11] |

**Kompatibilität der GGUFs**: unsloth liefert für #27773 den Ordner `Shard_Rewrite/` – nur Shard 1 (9 MB, Metadaten und Vokabular,
keine Tensoren) unterscheidet sich: Architekturname, Schlüsselpräfix und ein zusätzlicher Schlüssel
`attention.indexer.index_share_mtp = true`. Eigene Prüfung: der Umschreiber in `glm53/gguf.py` erzeugt aus dem unsloth-Shard 1
byteidentische Metadaten zum Shard_Rewrite-Header. Die Meldung „head_count_kv has wrong array length; expected 45, got 46“
(miminashi, 02.09.) ist seit #28173 (01.09., in #27773 enthalten) behoben: Arrays werden mit `n_layer_all` = 46 gelesen [10].
Der unsloth-`mmproj` in `Shard_Rewrite/` ist veraltet (vor der `swiglu_clamp`-Umbenennung) – für #27773 den Projektor von
avar6 oder AesSedai nehmen [10]. Tensornamen sind seit 28.08. angeglichen (`indexer_compressor_ape/gate`).

**HIP-relevante Fixes in master** (eigene Prüfung per `git merge-base`): `ROCm: add radix TOP_K for long rows` (#27466,
31.08.) ist in #27773, #27917 und #27754 enthalten, **nicht** in vcruz305. Das ist der Fix für den CPU-Fallback von
`ggml_top_k` ab 1024 Zeilen, der im Qwen3.8-Projekt den Decode mit der Tiefe einbrechen ließ [15]. GLM-5.3-Flash sortiert je
Token 11× Top-2048 über die Pool-Zeilen (Kontext/4) – bei 131k Kontext also 32k Zeilen, weit über der 1024er-Schwelle. Der
Einbruch von 14 auf 7,3 t/s bei 42k Tiefe im eigenen Log vom 29.08. (vcruz305-Fork ohne den Fix) passt zu diesem Bild.
`#26592` (hipCUB) ist weiter offen; `#27311` (UMA-Ring-Puffer, Qwen-Patch 0003) und `#25863` (iGPU-Host-Puffer, Patch 0002)
ebenfalls – Patch 0002 wird hier wie im Qwen-Projekt auf alle Engines angewendet (Multi-Slot-Mix-ups auf gfx1151).

**Entscheidung** (bis Messungen etwas anderes zeigen): vier Engines bauen und vergleichen –
1. `tk-mtp`: #27917 (MTP auf #27773, Stand 02.09.) – Hauptkandidat.
2. `unsloth`: #27754 mit MTP – A/B-Vergleich, braucht Shard 1 im `glm5next`-Schema (Original).
3. `tk`: #27773-HEAD ohne MTP – Basislinie, neueste Optimierungen (Sparse-FA-Prefill, Multi-Stream).
4. `tk-merged`: #27917 auf den #27773-HEAD gemergt (eigene Auflösung von zwei Konflikten in
   `llama-memory-hybrid-idx`: `kpool_dirty` → `mem_idx_stale`, `mtp_dsa_selection.clear()` beibehalten; dazu das vom Merge
   verlorene `#include "llama-memory-hybrid-idx.h"` in `llama-context.cpp`; Patch 0001) – experimentell; nur wenn er
   korrekt antwortet (Ausgabenvergleich mit `tk-mtp`) und schneller ist als 1. Build-Protokoll: `engine/build-tk-merged-hip.log`.

## 4. Strix-Halo-spezifisch

- **Backend**: HIP zuerst (Qwen-Erfahrung: MTP auf Vulkan brach ein; Vulkan-Entwicklungspakete fehlen auf dem System).
  Ein Vulkan-Build ist in `engine/build.sh` vorbereitet (`sudo dnf install glslc vulkan-headers vulkan-loader-devel`).
  Vulkan könnte den VRAM-Carve-out mitnutzen (device-local heap) und hat einen eigenen Radix-Top-k (#28032) – als
  Vergleichsmessung sinnvoll [14][15].
- **Kernel-Parameter** gesetzt: `amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856` (GTT 110 GiB).
- **Lade-Modus**: `--load-mode none` (Deniz-Eren: `mmap+mlock` stürzte bei der Generierung ab; `none` stabil bis 322k) [3].
  ROCm meldet ohnehin „kein mmap“; lazy-mmap ist im Qwen-Projekt am fehlenden Readahead gescheitert.
- **Flags**: `-fa on` (bei #27773 korrekt und speicherflach; `-fa off` war bei #27754 als „required“ gelistet, ist dort aber
  6,4× langsamer bei 131k [7][10]); `-ctk/-ctv q8_0` (Indexer-Key-Cache bleibt ohnehin f16, Log 29.08.); `-ub 1024` (bei
  #27773 bis 4096 ohne Fehler; MoE-Prefill profitiert von großem ubatch, Speicher wächst mit); mehrere Slots:
  `--kv-unified` Pflicht auf #27773; `--cache-ram 512` (Spark-README: 8 GiB Default kostete die Reserve) [12].
- **Umgebungsvariablen zum Testen**: `ROCBLAS_USE_HIPBLASLT=1` (EngramHalo-Empfehlung für Qwen), `GGML_CUDA_DISABLE_GRAPHS=1`
  (drluoto-Stack: „required on ROCm < 7.13“ – Bedeutung unklar, hier ROCm 7.1.1; nur bei Abstürzen probieren) [15].
- **MTP-Draft-Tiefe**: Akzeptanz je Position 0,92/0,81/0,60/0,44/0,35 [9] → n_max 2–3 auf bandbreitenlimitierter Hardware;
  unsloth nennt n = 2 als Optimum [2]. Temperatur senkt die Akzeptanz (Verifikation sampelt mit dem vollen Sampler).
- **Alternativen**: DFlash2-Drafter für GLM-5.3-Flash (Inco AI, Anbeeld/vcruz305-GGUFs, 0,4–2,2 GB; Block-Diffusion, in
  SGLang 2,4× bei Concurrency 1; Lizenz CC BY-NC-ND) – DFlash2 ist in llama.cpp master (#27342), ob der Drafter mit den
  GLM-Branches läuft, ist ungetestet [16]. ik_llama.cpp: kein GLM-5.3-Flash, kein ROCm.

## 5. Speicher auf dieser Maschine

- MemTotal 109,7 GiB; MemAvailable leer ≈ 105 GiB (Log 29.08.). Alles teilt sich diesen RAM; GTT taucht **nicht** im RSS auf;
  `MemAvailable` ist die einzige verlässliche Größe → `bench/memguard.py` (SIGKILL bei 5 GiB).
- Footprint-Modell (`glm53/memory.py`, zu kalibrieren): Gewichte (Datei minus NextN, bei MTP plus NextN) + KV der 11 MLA-Layer
  (≈ 8 KiB/Token bei q8_0 plus Indexer ≈ 3 KiB) + KDA-Zustand (~0,1 GiB je Slot) + Compute (1–2,5 GiB je nach ubatch) + Host 1,2 GiB
  + MTP-Draft-Kontext 0,8 GiB + Prompt-Cache-Obergrenze; Reserve 4 GiB. Eigener Messpunkt 29.08.: UD-IQ2_XXS ohne MTP, 131k,
  ≈ 93 GiB Bedarf (105 → ~12 GiB frei), Decode 14,0 t/s (kurz) → 9,5 t/s Mittel über 42k Tokens, Prefill 71 t/s bei 290 Tokens.
- Kontext ist billig (11 KV-Layer, MLA): 256k ≈ 2,8 GiB. Die Grenze ist das Prompt-Processing.

## 6. Qualitäts-Benchmark

Wie im Qwen-Projekt: Terminal-Bench-Mini-20 (Terminus-2 über Harbor, Docker) je Quant/Engine, ein Slot, MTP an, pass@1,
Denkstufe `high` (statt `medium`, das GLM nicht kennt). Vergleichswerte: Spark-README HumanEval+ 153/164 (Spark) vs. 148/164
(UD-Q2_K_XL) auf DGX Spark [12]. `docs/QUALITAETS-BENCHMARKS.md` (aus dem Qwen-Projekt übernommen) beschreibt die Alternativen.

## 7. Offene Punkte

- `tk-merged` erbt den MTP-Assert von #27917; Vergleich erst nach einem Fix sinnvoll.
- MTP auf #27917 ab ~2k Token Kontext: Assert `width == mtp_dsa_sel_width` – upstream melden (Logs liegen vor) oder Patch.
- Reproduziert sich der Decode-Einbruch mit der Tiefe auf HIP mit #27466? (`bench/llama_bench.sh -d 0,8192,32768`)
- Lohnt MTP auf Strix Halo bei GLM (Akzeptanz, Draft-Tiefe, Temperatur)? (`bench/mtp_sweep.py`)
- Graph-Reuse bei #27773 (`graphs reused = 0` im Log vom 29.08. auf dem unsloth-Fork).
- Vision: AesSedai-mmproj (Q8_0/F16) mit `tk-mtp`; Bildtokenzahl und Speicher.
- BIOS-Carve-out: Entscheidung des Betreibers; danach Q2_K_XL/Spark/AesSedai-IQ2_S messen.
- Vulkan-Build (Pakete fehlen).

## 8. Eigene Messungen (2026-09-05, nach dem Go)

**Footprint-Probe** `bench/mem_probe.py probe-tkmtp-32k --preset tkmtp-agent --ctx 32768` – Engine `tk-mtp` (#27917 auf #27773),
UD-IQ2_XXS im Schema glm5-next, MTP n2/p0.75, KV q8_0, ub 1024, ein Slot, reasoning high, Prompt 39 Token, 300 Token Ausgabe:

| Größe | Wert |
| --- | --- |
| Ladezeit bis /health | 77 s (Modell aus dem Page-Cache/NVMe, `--load-mode none`) |
| Decode | **17,3 t/s** (300 Token; Verlauf 15,8 → 18,1 t/s) |
| Draft-Akzeptanz | **0,87** (162 von 187), mittlere Draft-Länge 2,6 |
| Graph-Reuse | 22 Wiederverwendungen (auf #27773 funktioniert `can_reuse`, anders als auf #27754) |
| Speicherbedarf | **99,9 GiB** (MemAvailable 105,0 → 5,2 GiB), Peak GTT 97,2 GiB, RSS 1,6 GiB |
| Schätzung des Programms | 99,4 GiB (Abweichung 0,5 GiB) |

Folgerungen: Die Engine-Kombination lädt die unsloth-Shards mit Shard_Rewrite-Header ohne Änderung; MTP funktioniert ohne
separates Draft-GGUF; die Speicherschätzung stimmt auf ein halbes GiB. Mit 16-GiB-Carve-out bleiben bei 32k Kontext nur 5 GiB
frei – für 131k Kontext oder Docker-Container daneben ist das zu wenig; **der kleine BIOS-Carve-out ist damit nicht optional,
sondern nötig**, sobald mehr als der Server allein laufen soll.

**MTP auf #27917 stürzt ab, sobald der Kontext etwa 2k Token übersteigt** (eigener Befund, 05.09., 8 Läufe):
`tk-mtp` und `tk-merged` mit `--spec-type draft-mtp` brechen bei der ersten Anfrage mit
`GGML_ASSERT(width == mtp_dsa_sel_width)` (`llama-context.cpp:2062`) ab. Eingegrenzt mit UD-IQ1_M, `-ub 1024`:

| Prompt (Token gesamt) | Ergebnis |
| --- | --- |
| 39 (Probe), 863, 2089 | läuft: 17,3 / 16,5 / 10,7 t/s, Akzeptanz 87 / 74 / 78 % |
| ~3200, ~5200, ~8200 | Assert vor der ersten Fortschrittsmeldung – unabhängig von `-np 1` oder `-np 4 --kv-unified`, von UD-IQ2_XXS oder UD-IQ1_M, und von `-b 4096` oder `-b 16384` |

Im Code: die Auswahlbreite für den Draft ist `4 × min(Pools, 512) + 3` und sättigt bei 512 Pools = 2048 Token; die Breite wird
beim ersten Ubatch mit Gather-Pfad festgeschrieben und muss danach gleich bleiben. Warum 2089 Token noch laufen und ~3200
nicht, ist ungeklärt (vermutlich Trunk- gegen Draft-Kontext beim Sättigen). **Für Agenten-Prompts ist MTP auf #27917 derzeit
nicht nutzbar.** Die unsloth-Engine (#27754) hat eine eigene MTP-Implementierung, die mit 8k Prompts und mit vier Slots läuft.
Nicht upstream gemeldet; Logs: `bench/results/multi/tk-mtp-*.log`, `tk-merged-*.log`.

**Stream-Benchmark** (`bench/multiuser.py`, 8k Prompt mit eigenem Fülltext je Nutzer, 512 Token Ausgabe, 16k Kontext je Slot,
`--kv-unified` ab zwei Slots, reasoning low, Codewort-Prüfung; „Σ Decode“ = Summe der Decode-Raten der Streams):

| Engine | Quant | MTP | Slots | 1 Stream | 2 Streams Σ (je) | 4 Streams Σ (je) | Prefill je Stream | TTFT 8k | Akzeptanz | Bedarf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tk (#27773) | UD-IQ2_XXS | – | 4 | 10,5 t/s | 11,5 (5,8) | 9,1 (2,3) | 150 / 66 / 40 t/s | 72 / 162 / 311 s | – | 97,9 GiB |
| tk (#27773) | UD-IQ2_XXS | – | 1 | 10,1 t/s | – | – | 151 t/s | 72 s | – | – |
| unsloth (#27754) | UD-IQ1_M | n2 | 4 | **15,7 t/s** | 15,2 (7,6) | 9,8 (2,5) | 144 / 62 / 38 t/s | 75 / 171 / 330 s | 82–84 % | 100,0 GiB |
| unsloth (#27754) | UD-IQ1_M | – | 1 | 13,4 t/s | – | – | 149 t/s | 72 s | – | – |
| unsloth (#27754) | UD-IQ2_XXS | n2 | 1 | **14,3 t/s** | – | – | 147 t/s | 74 s | 79 % | 97,5 GiB |
| unsloth (#27754) | UD-IQ2_XXS | – | 1 | 13,0 t/s | – | – | 152 t/s | 71 s | – | – |
| tk-mtp / tk-merged | beide | n2 | 1 und 4 | Absturz (siehe oben) | | | | | | |

Folgerungen (Stand 05.09., ohne weiteres Tuning):
1. **Prefill ist der Engpass**: rund 150 t/s Gesamtdurchsatz bei jeder Slot-Zahl, also 72 s bis zum ersten Token für einen
   8k-Prompt – bei vier gleichzeitigen Streams über fünf Minuten. Für Agenten mit 50–100k Kontext ist das die dominante Zeit
   (Prompt-Cache vorausgesetzt, sonst unbrauchbar).
2. **Continuous Batching bringt beim Decode nichts**: Σ Decode bleibt bei 1–2 Streams um 10–15 t/s und fällt bei 4 Streams.
   Anders als bei Qwen3.8 (20 → 50 t/s) amortisiert sich das Gewichte-Lesen nicht; der Aufwand je Sequenz (KDA-Zustand,
   Indexer-Top-k, mHC) dominiert. Mehrere Slots lohnen nur für die Latenzverteilung, nicht für den Durchsatz.
3. **MTP bringt auf der unsloth-Engine +10 bis +17 %** (IQ2_XXS 13,0 → 14,3, IQ1_M 13,4 → 15,7 t/s, Akzeptanz 79–84 %),
   deutlich weniger als bei Qwen3.8 (+70 %); mit vier Streams kein Nachteil (9,8 gegen 9,1 t/s ohne MTP auf tk).
4. **Engine-Vergleich ohne MTP, gleicher Quant (UD-IQ2_XXS)**: unsloth 13,0 t/s gegen #27773-HEAD 10,1 t/s (+29 %).
   Prefill ist gleich (150 t/s). Der Quant macht wenig aus: IQ1_M ist nur 3 % schneller als IQ2_XXS (13,4 gegen 13,0).
   MTP auf unsloth: +10 % bei IQ2_XXS (13,0 → 14,3), +17 % bei IQ1_M (13,4 → 15,7).
5. **Speicher**: alle Konfigurationen liegen bei 98–100 GiB Bedarf, es bleiben 5–8 GiB. Die Docker-Container des
   Qualitäts-Benchmarks brauchen zusätzlich Platz – der kleine BIOS-Carve-out ist dafür nötig.

**Draft-Tiefe** (unsloth, UD-IQ2_XXS, ein Stream, 8k Prompt, 512 Token, p_min 0,75, 05.09. 23:13):

| Draft-Tiefe | Decode | Akzeptanz |
| --- | --- | --- |
| ohne MTP | 13,0 t/s | – |
| n2 | 14,3 t/s | 79 % |
| n3 | 14,8 t/s | 78 % |
| **n4** | **15,2 t/s** | 82 % |

Jede Stufe bringt etwa 3 %, die Akzeptanz bleibt hoch – die Verifikation des längeren Drafts kostet auf der bandbreiten-
limitierten Hardware kaum etwas. Messrauschen liegt bei etwa 3 %; n4 ist deshalb nur „mindestens so gut wie n3“. Größere
Tiefen (n5/n6), p_min 0 und Temperatur folgen mit `bench/mtp_sweep.py`.

**Empfehlung nach diesen Messungen**: Preset `unsloth-agent` (unsloth-Engine, UD-IQ2_XXS, MTP n4) für Agenten; `tk-plain`
als Fallback; `tk-mtp` erst nach einem Fix des Asserts. Feintuning (p_min, ubatch, `ROCBLAS_USE_HIPBLASLT`) steht aus.

## Quellen

1. [zai-org/GLM-5.3-Flash – Modellkarte](https://huggingface.co/zai-org/GLM-5.3-Flash)
2. [unsloth: GLM-5.3-Flash lokal ausführen](https://unsloth.ai/docs/models/glm-5.3-flash), [unsloth GGUF-Repo](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)
3. [HF-Diskussion #4 (unsloth): Branches, `--load-mode none`, Berichte](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/discussions/4)
4. [llama.cpp Discussion #28203: GLM-5.3 Status](https://github.com/ggml-org/llama.cpp/discussions/28203), [Issue #27922](https://github.com/ggml-org/llama.cpp/issues/27922)
5. [llama.cpp PR #27754 (unsloth, glm5next)](https://github.com/ggml-org/llama.cpp/pull/27754)
6. [llama.cpp PR #27752 (eauchs)](https://github.com/ggml-org/llama.cpp/pull/27752)
7. Kommentare in #27754: feni6 (Metal-Kollaps, `-ub 128`, Ursache #28210), Suaroman (fa on/off, MTP n2 84 t/s auf SM120), noonghunna (`can_reuse`, ~7 %)
8. [llama.cpp PR #27773 (timkhronos, glm5-next)](https://github.com/ggml-org/llama.cpp/pull/27773)
9. [llama.cpp PR #27917 (timkhronos, MTP)](https://github.com/ggml-org/llama.cpp/pull/27917), Kommentar borakilo (4×Ampere)
10. Kommentare in #27773: miminashi (head_count_kv, Workaround), nicholasshirley (Decode vs. Tiefe, fa on/off, 524k-Agentensitzung, Vision, mmproj-Hinweis), Neresco (Strix Halo + RPC, `-b/-ub` 128)
11. [vcruz305/llama.cpp glm5next-mtp](https://github.com/vcruz305/llama.cpp), [vcruz305 GGUFs](https://huggingface.co/vcruz305/GLM-5.3-Flash-GGUF), [Ankündigung](https://x.com/ViC305/status/2093016892537737616)
12. [sayyidfareed/GLM-5.3-Flash-Spark-Q2XL-MTP](https://huggingface.co/sayyidfareed/GLM-5.3-Flash-Spark-Q2XL-MTP)
13. [AesSedai/GLM-5.3-Flash-GGUF (KLD-Tabelle, mmproj)](https://huggingface.co/AesSedai/GLM-5.3-Flash-GGUF), [aj9o9/GLM-5.3-Flash-GGUF](https://huggingface.co/aj9o9/GLM-5.3-Flash-GGUF)
14. [Nathanw1014/strix-halo-llamacpp (FA/MoE-Fixes, Vulkan+HIP)](https://github.com/Nathanw1014/strix-halo-llamacpp), [soothill.io: ROCm vs. Vulkan](https://www.soothill.io/blog/2026/08/27/qwen38-flash-next-rocm-vulkan-strix-halo/), [kyuz0 Toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes)
15. [Discussion #27950: Qwen3.8-Flash-Next auf Strix Halo, TOP_K-CPU-Fallback, #26592/#27466](https://github.com/ggml-org/llama.cpp/discussions/27950), [PR #27466](https://github.com/ggml-org/llama.cpp/pull/27466), [PR #26592](https://github.com/ggml-org/llama.cpp/pull/26592)
16. [Anbeeld/GLM-5.3-Flash-DFlash2-GGUF](https://huggingface.co/Anbeeld/GLM-5.3-Flash-DFlash2-GGUF), [PR #27342 DFlash2](https://github.com/ggml-org/llama.cpp/pull/27342)
17. [AMD: Strix-Halo-Systemoptimierung](https://rocmdocs.amd.com/en/develop/how-to/system-optimization/strixhalo.html), [LucRoot Strix-Halo-ROCm-Rezept](https://github.com/LucRoot/Strix-Halo-Linux-Llama_cpp-ROCm)
18. Qwen3.8-Flash-Next-Projekt auf derselben Maschine: `../qwen38-flash/docs/RESEARCH.md` (Patch 0002, memguard, Vulkan/MTP-Erfahrung)
